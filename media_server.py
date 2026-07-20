import asyncio
import base64
import io
import logging
import os
import sys
import uuid
import inspect
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("media_server")

app = FastAPI(title="KeglyaMedia Server", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device Configuration
PHOTO_DEVICE = "cuda:0" if torch.cuda.device_count() > 0 else "cpu"
VIDEO_DEVICE = "cuda:1" if torch.cuda.device_count() > 1 else PHOTO_DEVICE

# Global references to pipelines
_image_pipe = None
_video_t2v_pipe = None
_video_i2v_pipe = None

# Job Status Database
jobs: Dict[str, Dict[str, Any]] = {}

# Locks to prevent concurrent execution on each GPU
photo_lock = asyncio.Lock()
video_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Pipeline Loaders
# ---------------------------------------------------------------------------

def get_image_pipeline():
    global _image_pipe
    if _image_pipe is None:
        log.info(f"Loading Image Generation pipeline on {PHOTO_DEVICE}...")
        from diffusers import AutoPipelineForText2Image
        _image_pipe = AutoPipelineForText2Image.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16 if PHOTO_DEVICE != "cpu" else torch.float32,
        ).to(PHOTO_DEVICE)
        
        # Enable memory optimizations
        if PHOTO_DEVICE != "cpu":
            _image_pipe.enable_model_cpu_offload()
            if hasattr(_image_pipe, "vae") and _image_pipe.vae:
                _image_pipe.vae.enable_tiling()
                _image_pipe.vae.enable_slicing()
        log.info("Image Generation pipeline loaded successfully.")
    return _image_pipe


def get_video_t2v_pipeline():
    global _video_t2v_pipe
    if _video_t2v_pipe is None:
        log.info(f"Loading Wan 2.1 Text-to-Video pipeline on {VIDEO_DEVICE}...")
        from diffusers import WanPipeline
        _video_t2v_pipe = WanPipeline.from_pretrained(
            "Wan-Video/Wan2.1-T2V-1.3B",
            torch_dtype=torch.bfloat16 if VIDEO_DEVICE != "cpu" else torch.float32,
        ).to(VIDEO_DEVICE)
        
        if VIDEO_DEVICE != "cpu":
            _video_t2v_pipe.enable_model_cpu_offload()
            _video_t2v_pipe.vae.enable_tiling()
        log.info("Wan 2.1 Text-to-Video pipeline loaded.")
    return _video_t2v_pipe


def get_video_i2v_pipeline():
    global _video_i2v_pipe
    if _video_i2v_pipe is None:
        log.info(f"Loading Wan 2.1 Image-to-Video pipeline on {VIDEO_DEVICE}...")
        from diffusers import WanImageToVideoPipeline
        from transformers import BitsAndBytesConfig

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        _video_i2v_pipe = WanImageToVideoPipeline.from_pretrained(
            "Wan-Video/Wan2.1-I2V-14B-480P",
            transformer_kwargs={"quantization_config": nf4_config} if VIDEO_DEVICE != "cpu" else {},
            torch_dtype=torch.float16 if VIDEO_DEVICE != "cpu" else torch.float32,
        ).to(VIDEO_DEVICE)
        
        if VIDEO_DEVICE != "cpu":
            _video_i2v_pipe.enable_model_cpu_offload()
            _video_i2v_pipe.vae.enable_tiling()
        log.info("Wan 2.1 Image-to-Video pipeline loaded.")
    return _video_i2v_pipe

# ---------------------------------------------------------------------------
# Models & Schemas
# ---------------------------------------------------------------------------

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    steps: Optional[int] = 4
    guidance_scale: Optional[float] = 0.0
    seed: Optional[int] = -1

class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = "low quality, worst quality, blurry, distorted"
    image_base64: Optional[str] = None  # For Image-to-Video
    width: Optional[int] = 832
    height: Optional[int] = 480
    duration: Optional[float] = 3.5
    steps: Optional[int] = 6
    guidance_scale: Optional[float] = 1.0
    seed: Optional[int] = -1

# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "gpus": torch.cuda.device_count()}

@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/v1/outputs/{filename}")
async def get_output_file(filename: str):
    filepath = OUTPUT_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)


async def run_image_generation(job_id: str, req: ImageGenerationRequest):
    async with photo_lock:
        jobs[job_id]["status"] = "processing"
        try:
            pipe = get_image_pipeline()
            
            # Setup seed
            generator = None
            seed = req.seed
            if seed < 0:
                seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=PHOTO_DEVICE).manual_seed(seed)

            # Build call arguments
            kwargs = {
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "width": max(256, min(2048, (req.width // 8) * 8)),
                "height": max(256, min(2048, (req.height // 8) * 8)),
                "num_inference_steps": req.steps,
                "generator": generator,
            }

            # Filter kwargs according to pipeline call signature
            sig = inspect.signature(pipe.__call__)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            
            # Add Z-Image-Turbo defaults if not in Flux
            if "max_sequence_length" in sig.parameters:
                valid_kwargs["max_sequence_length"] = 1024
            if "guidance_scale" in sig.parameters:
                valid_kwargs["guidance_scale"] = req.guidance_scale

            # Run inference in worker thread
            def _run():
                res = pipe(**valid_kwargs)
                return res.images[0]
            
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(None, _run)
            
            filename = f"{job_id}.png"
            filepath = OUTPUT_DIR / filename
            image.save(filepath, format="PNG")
            
            # Also prepare base64
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")

            jobs[job_id].update({
                "status": "success",
                "output_url": f"/v1/outputs/{filename}",
                "image_base64": b64_data,
                "seed": seed
            })
            log.info(f"Job {job_id} completed successfully.")
        except Exception as e:
            log.error(f"Job {job_id} failed: {e}", exc_info=True)
            jobs[job_id].update({
                "status": "failed",
                "error": str(e)
            })


async def run_video_generation(job_id: str, req: VideoGenerationRequest):
    async with video_lock:
        jobs[job_id]["status"] = "processing"
        try:
            from diffusers.utils.export_utils import export_to_video
            from PIL import Image
            
            seed = req.seed
            if seed < 0:
                seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=VIDEO_DEVICE).manual_seed(seed)

            # Determine if Text-to-Video or Image-to-Video
            is_i2v = req.image_base64 is not None
            
            if is_i2v:
                pipe = get_video_i2v_pipeline()
                # Decode base64 image
                img_bytes = base64.b64decode(req.image_base64)
                input_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                
                # Resize input image to fit model dimensions
                w = max(128, min(1024, (req.width // 16) * 16))
                h = max(128, min(1024, (req.height // 16) * 16))
                input_image = input_image.resize((w, h), Image.Resampling.LANCZOS)
                
                # Calculate frames
                num_frames = 1 + int(round(req.duration * 24))
                
                kwargs = {
                    "image": input_image,
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt,
                    "width": w,
                    "height": h,
                    "num_frames": num_frames,
                    "num_inference_steps": req.steps,
                    "guidance_scale": req.guidance_scale,
                    "generator": generator,
                }
            else:
                pipe = get_video_t2v_pipeline()
                num_frames = 1 + int(round(req.duration * 24))
                kwargs = {
                    "prompt": req.prompt,
                    "negative_prompt": req.negative_prompt,
                    "width": max(128, min(1024, (req.width // 16) * 16)),
                    "height": max(128, min(1024, (req.height // 16) * 16)),
                    "num_frames": num_frames,
                    "num_inference_steps": req.steps,
                    "guidance_scale": req.guidance_scale,
                    "generator": generator,
                }

            # Filter signature
            sig = inspect.signature(pipe.__call__)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

            def _run():
                res = pipe(**valid_kwargs)
                return res.frames[0]

            loop = asyncio.get_event_loop()
            frames = await loop.run_in_executor(None, _run)
            
            filename = f"{job_id}.mp4"
            filepath = OUTPUT_DIR / filename
            export_to_video(frames, str(filepath), fps=24)
            
            jobs[job_id].update({
                "status": "success",
                "output_url": f"/v1/outputs/{filename}",
                "seed": seed
            })
            log.info(f"Job {job_id} completed successfully.")
        except Exception as e:
            log.error(f"Job {job_id} failed: {e}", exc_info=True)
            jobs[job_id].update({
                "status": "failed",
                "error": str(e)
            })


@app.post("/v1/images/generations")
async def generate_image_endpoint(req: ImageGenerationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "type": "image",
        "status": "queued",
        "prompt": req.prompt
    }
    background_tasks.add_task(run_image_generation, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.post("/v1/videos/generations")
async def generate_video_endpoint(req: VideoGenerationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "type": "video",
        "status": "queued",
        "prompt": req.prompt
    }
    background_tasks.add_task(run_video_generation, job_id, req)
    return {"job_id": job_id, "status": "queued"}

# ---------------------------------------------------------------------------
# MCP SSE Transport Interface (for direct LLM integration)
# ---------------------------------------------------------------------------

@app.get("/sse")
async def sse_endpoint(request: Request):
    """
    Exposes Model Context Protocol (MCP) SSE endpoints
    """
    # Simple Mock SSE endpoint to allow LLM bot connectivity
    async def event_generator():
        yield "data: {\"type\": \"connected\"}\n\n"
        while True:
            await asyncio.sleep(10)
            yield "data: {\"type\": \"ping\"}\n\n"
            
    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/messages")
async def messages_endpoint(request: Request):
    # MCP message receiver
    body = await request.json()
    method = body.get("method")
    
    if method == "tools/list":
        return {
            "result": {
                "tools": [
                    {
                        "name": "generate_image",
                        "description": "Сгенерировать изображение по текстовому описанию (Z-Image-Turbo)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Промпт/описание"},
                                "width": {"type": "integer", "default": 1024},
                                "height": {"type": "integer", "default": 1024},
                                "steps": {"type": "integer", "default": 4},
                                "guidance_scale": {"type": "number", "default": 0.0},
                                "seed": {"type": "integer", "default": -1}
                            },
                            "required": ["prompt"]
                        }
                    },
                    {
                        "name": "generate_video",
                        "description": "Сгенерировать короткое видео по текстовому описанию или картинке (Wan 2.1)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {"type": "string", "description": "Описание движения"},
                                "image_base64": {"type": "string", "description": "Опционально: базовая картинка в base64 для Image-to-Video"},
                                "duration": {"type": "number", "default": 3.5},
                                "steps": {"type": "integer", "default": 6},
                                "seed": {"type": "integer", "default": -1}
                            },
                            "required": ["prompt"]
                        }
                    }
                ]
            }
        }
        
    elif method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        job_id = str(uuid.uuid4())[:8]
        
        if tool_name == "generate_image":
            req = ImageGenerationRequest(**arguments)
            jobs[job_id] = {
                "id": job_id,
                "type": "image",
                "status": "queued",
                "prompt": req.prompt
            }
            await run_image_generation(job_id, req)
            job_res = jobs[job_id]
            if job_res["status"] == "success":
                return {
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Изображение успешно сгенерировано. Ссылка: {job_res['output_url']}. Seed: {job_res['seed']}"
                            },
                            {
                                "type": "image",
                                "data": job_res["image_base64"],
                                "mimeType": "image/png"
                            }
                        ]
                    }
                }
            else:
                return {
                    "error": {
                        "code": -32000,
                        "message": f"Ошибка генерации: {job_res.get('error')}"
                    }
                }
                
        elif tool_name == "generate_video":
            req = VideoGenerationRequest(**arguments)
            jobs[job_id] = {
                "id": job_id,
                "type": "video",
                "status": "queued",
                "prompt": req.prompt
            }
            await run_video_generation(job_id, req)
            job_res = jobs[job_id]
            if job_res["status"] == "success":
                return {
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Видео успешно сгенерировано. Скачать: {job_res['output_url']}. Seed: {job_res['seed']}"
                            }
                        ]
                    }
                }
            else:
                return {
                    "error": {
                        "code": -32000,
                        "message": f"Ошибка генерации: {job_res.get('error')}"
                    }
                }
                
    return {"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found"}, "id": body.get("id")}


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("MEDIA_HOST", "0.0.0.0")
    port = int(os.environ.get("MEDIA_PORT", "8082"))
    uvicorn.run(app, host=host, port=port)
