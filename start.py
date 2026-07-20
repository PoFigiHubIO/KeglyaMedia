#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import json
import urllib.request
from pathlib import Path

def step(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78 + "\n")

def run(cmd, check=True):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def load_kaggle_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        # Automatically export all secrets as environment variables
        # This will export HF_TOKEN, TELEGRAM_BOT_TOKEN, NGROK_AUTHTOKEN, etc.
        import json
        secrets_dict = client.list_secrets()
        for secret_name in secrets_dict:
            try:
                val = client.get_secret(secret_name)
                if val:
                    os.environ[secret_name] = val
                    # Mask value in log
                    masked = val[:5] + "..." + val[-5:] if len(val) > 10 else "..."
                    print(f"   ✅ Успешно загружен секрет: {secret_name} ({masked})")
            except Exception:
                pass
    except Exception:
        print("   ⚠️ Запуск вне Kaggle или Kaggle Secrets недоступны.")

def main():
    load_kaggle_secrets()

    # Isolated cache dir
    os.environ["HF_HOME"] = "/tmp/.cache"
    
    # Establish directory structure
    Path("./logs").mkdir(exist_ok=True)
    Path("./bin").mkdir(exist_ok=True)
    
    # Step 1: Install dependencies
    step("ЭТАП 1/4 — Установка Python-зависимостей")
    run(["pip", "install", "-r", "requirements.txt", "-q"])

    # Step 2: Pre-download model weights
    step("ЭТАП 2/4 — Предварительная загрузка весов моделей (Z-Image-Turbo и Wan)")
    print("[start.py] Это запустит скачивание с отображением прогресс-бара...")
    try:
        from huggingface_hub import snapshot_download
        
        # 1. Z-Image-Turbo (Photo)
        print("\n[start.py] [1/3] Загрузка модели Tongyi-MAI/Z-Image-Turbo...")
        snapshot_download(repo_id="Tongyi-MAI/Z-Image-Turbo", resume_download=True)
        
        # 2. Wan 2.1 T2V (Video Text-to-Video)
        print("\n[start.py] [2/3] Загрузка модели Wan-Video/Wan2.1-T2V-1.3B...")
        snapshot_download(repo_id="Wan-Video/Wan2.1-T2V-1.3B", resume_download=True)

        # 3. Wan 2.1 I2V (Video Image-to-Video)
        print("\n[start.py] [3/3] Загрузка модели Wan-Video/Wan2.1-I2V-14B-480P...")
        snapshot_download(repo_id="Wan-Video/Wan2.1-I2V-14B-480P", resume_download=True)
        
        print("\n[start.py] ✅ Все веса моделей успешно загружены!")
    except Exception as e:
        print(f"\n[start.py][warn] Ошибка при предзагрузке моделей: {e}")
        print("[start.py] Попробуем продолжить запуск, медиа-сервер скачает модели сам.")

    # Step 3: Start FastAPI media server
    step("ЭТАП 3/4 — Запуск FastAPI Media Server")
    os.system("pkill -f media_server.py")
    
    port = int(os.environ.get("MEDIA_PORT", "8082"))
    log_f = open("logs/media_server.log", "w")
    
    media_proc = subprocess.Popen(
        [sys.executable, "media_server.py"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    with open("logs/media_server.pid", "w") as f:
        f.write(str(media_proc.pid))
    print(f"[start.py] Media Server запущен (PID={media_proc.pid})")
    
    # Wait for status response
    ready = False
    for i in range(30):
        if media_proc.poll() is not None:
            print("[start.py][error] Процесс Media Server аварийно завершился!")
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    ready = True
                    print(f"[start.py] ✅ Media Server успешно запущен и слушает порт {port}!")
                    break
        except Exception:
            pass
        time.sleep(2)

    if not ready:
        print("[start.py][error] Media Server не ответил на /health. Логи запуска:")
        if os.path.exists("logs/media_server.log"):
            with open("logs/media_server.log", "r") as lf:
                for line in lf.readlines()[-40:]:
                    print(line, end="")
        sys.exit(1)

    # Step 4: Expose Tunnel
    step("ЭТАП 4/4 — Публикация через туннель")
    from scripts.tunnel import start_tunnel
    
    # Get tunnel config
    provider = os.environ.get("TUNNEL_PROVIDER", "cloudflared")
    cloudflare_token = os.environ.get("CLOUDFLARE_TUNNEL_TOKEN", "")
    cloudflare_domain = os.environ.get("CLOUDFLARE_TUNNEL_DOMAIN", "")
    ngrok_domain = os.environ.get("NGROK_DOMAIN", "")
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN", "")

    tunnel_pid_file = "logs/tunnel.pid"
    if os.path.exists(tunnel_pid_file):
        try:
            with open(tunnel_pid_file, "r") as f:
                old_pid = int(f.read().strip())
            os.kill(old_pid, 15)
            time.sleep(1)
        except Exception:
            pass

    proc, public_url = start_tunnel(
        provider,
        port,
        cloudflare_token=cloudflare_token,
        cloudflare_domain=cloudflare_domain,
        ngrok_domain=ngrok_domain,
        ngrok_token=ngrok_token
    )
    with open(tunnel_pid_file, "w") as f:
        f.write(str(proc.pid))

    if not public_url:
        public_url = f"http://127.0.0.1:{port}"
    print(f"[start.py] Публичный URL Медиа-сервера: {public_url}")

    # Warm-up / validation test
    print("\n" + "-" * 70)
    print("[start.py] Тест генерации изображения для прогрева VRAM и кэша...")
    print("[start.py] (При первом запуске это загрузит веса в VRAM)")
    print("-" * 70)
    
    # Start thread to log media_server output
    import threading
    stop_tail = False
    def tail_logs():
        try:
            with open("logs/media_server.log", "r") as lf:
                lf.seek(0, 2)
                while not stop_tail:
                    line = lf.readline()
                    if line:
                        print(f"[server] {line.strip()}")
                    else:
                        time.sleep(0.5)
        except Exception:
            pass

    tail_thread = threading.Thread(target=tail_logs, daemon=True)
    tail_thread.start()

    # Call REST endpoint to generate image
    req_data = json.dumps({
        "prompt": "Test warmup: red ball on blue table",
        "width": 512,
        "height": 512,
        "steps": 1,
        "guidance_scale": 0.0,
        "seed": 42
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/images/generations",
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status == 200:
                res_json = json.loads(resp.read().decode())
                job_id = res_json.get("job_id")
                print(f"[start.py] Задача отправлена в очередь. ID: {job_id}. Ждем завершения...")
                
                # Poll status
                success = False
                for _ in range(30):
                    time.sleep(2)
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/jobs/{job_id}") as poll_resp:
                        job_status = json.loads(poll_resp.read().decode())
                        status = job_status.get("status")
                        if status == "success":
                            success = True
                            print("[start.py] ✅ Тестовая генерация на GPU 0 завершена успешно!")
                            break
                        elif status == "failed":
                            print(f"[start.py][error] Ошибка в задаче: {job_status.get('error')}")
                            break
                if not success:
                    print("[start.py][error] Таймаут ожидания прогревочной генерации.")
            else:
                print(f"[start.py][error] HTTP-статус запроса генерации: {resp.status}")
    except Exception as e:
        print(f"[start.py][error] Ошибка при тестировании генерации: {e}")
    finally:
        stop_tail = True

    print("\n" + "#" * 78)
    print("#  KeglyaMedia — ГОТОВО")
    print("#" * 78)
    print(f"""
  Локальный URL:            http://127.0.0.1:{port}/
  Публичный URL:            {public_url}/
  MCP SSE Endpoint:         {public_url}/sse (Сообщения: {public_url}/messages)
  
  Доступные REST API:
    - POST /v1/images/generations
    - POST /v1/videos/generations
    - GET  /v1/jobs/{{job_id}}
    - GET  /v1/outputs/{{filename}}
    
  GPU Allocation:
    - GPU 0 (cuda:0): Photo (Z-Image-Turbo)
    - GPU 1 (cuda:1): Video (Wan 2.1)
""")

if __name__ == "__main__":
    main()
