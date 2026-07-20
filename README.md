# KeglyaMedia

Репозиторий медиа-сервера для генерации фото и видео на Kaggle GPU (2x T4). 
В проекте разделены ресурсы: текстовая часть (LLM) полностью вынесена, а этот репозиторий сфокусирован только на медиа-генерации.

## Архитектура
* **GPU 0 (`cuda:0`):** Генерация изображений (модель `Tongyi-MAI/Z-Image-Turbo` с быстрой T2I/I2I генерацией за 4 шага).
* **GPU 1 (`cuda:1`):** Генерация видео (Wan 2.1: `Wan2.1-T2V-1.3B` для генерации видео из текста и `Wan2.1-I2V-14B-480P` с NF4 квантованием для Image-to-Video).
* **Туннелирование:** cloudflared / ngrok для проброса портов наружу.
* **Интерфейс:** Полный REST API + встроенный MCP-сервер (sse/messages) для мгновенной интеграции с LLM ботами (включая текстовый бот из репозитория `KeglaAI`).

## Быстрый запуск на Kaggle
1. Создайте Kaggle Notebook.
2. В настройках (Settings) выберите:
   * **Accelerator:** GPU T4 x2
   * **Internet:** On
3. Задайте секреты в Add-ons -> Secrets:
   * `HF_TOKEN` — ваш токен Hugging Face.
   * `NGROK_AUTHTOKEN` — (опционально) ваш токен Ngrok.
4. Выполните клонирование и запуск:
   ```bash
   git clone https://github.com/PoFigiHubIO/KeglyaMedia.git
   cd KeglyaMedia
   python start.py
   ```
5. В консоли отобразится интерактивный прогресс-бар скачивания весов моделей и публичные ссылки на REST API и MCP SSE endpoint.

## REST API Эндпоинты

### 1. Здоровье сервера
* **GET `/health`** — возвращает состояние сервера и количество GPU.

### 2. Генерация фото (GPU 0)
* **POST `/v1/images/generations`**
  ```json
  {
    "prompt": "красивый пушистый кот на подоконнике",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance_scale": 0.0,
    "seed": -1
  }
  ```
  Возвращает `{"job_id": "xxxx", "status": "queued"}`.

### 3. Генерация видео (GPU 1)
* **POST `/v1/videos/generations`**
  ```json
  {
    "prompt": "кошка лениво потягивается, кинематографичное освещение",
    "image_base64": null, // или строка base64 для Image-to-Video
    "duration": 3.5,
    "steps": 6,
    "seed": -1
  }
  ```
  Возвращает `{"job_id": "xxxx", "status": "queued"}`.

### 4. Статус задачи
* **GET `/v1/jobs/{job_id}`** — возвращает статус задачи (`queued`, `processing`, `success`, `failed`). В случае успеха возвращает `output_url` для скачивания файла и base64-строку изображения.

### 5. Получение файлов
* **GET `/v1/outputs/{filename}`** — скачивание сгенерированного файла (PNG или MP4).
