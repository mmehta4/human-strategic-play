# Arena web app. Build from the stage1/ folder:  docker build -t arena .
FROM python:3.12-slim
WORKDIR /app
COPY arena/requirements.txt /app/arena/requirements.txt
RUN pip install --no-cache-dir -r arena/requirements.txt
COPY arena /app/arena
ENV ARENA_DB=/data/arena.db ARENA_HTTPS=1 PORT=8080
EXPOSE 8080
# One worker: SQLite is happiest with a single writer. Threads handle concurrency.
CMD gunicorn --workers 1 --threads 8 --bind 0.0.0.0:${PORT} --access-logfile - arena.app:app
