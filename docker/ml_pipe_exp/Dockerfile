# --- Stage 1: Builder (Компиляция) ---
FROM python:3.10-slim as builder

WORKDIR /app

# Устанавливаем системные пакеты для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Создаем виртуальное окружение
RUN python -m venv /opt/venv
# Активируем его для последующих команд
ENV PATH="/opt/venv/bin:$PATH"

# Устанавливаем зависимости
COPY requirements.txt .
# Объединяем все pip install в один слой для кэширования
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "feast[redis,s3]" pyarrow fastparquet boto3 pandas s3fs

# --- Stage 2: Runtime (Финальный образ) ---
FROM python:3.10-slim as runtime

WORKDIR /app

# Копируем виртуальное окружение из билдера
COPY --from=builder /opt/venv /opt/venv

# Активируем окружение
ENV PATH="/opt/venv/bin:$PATH"

# Копируем код проекта
# COPY feature_repo ./feature_repo  <-- ЭТО УБИРАЕМ! Конфиг будем монтировать через K8s
COPY feature_repo/definitions.py ./feature_repo/definitions.py 
# feature_store.yaml скопируем как "заглушку", но в K8s перекроем его
COPY feature_repo/feature_store.yaml ./feature_repo/feature_store.yaml

COPY train2.py .

# Настройки python для контейнера
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["python", "train2.py"]