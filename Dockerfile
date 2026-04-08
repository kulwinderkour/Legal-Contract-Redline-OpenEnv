FROM python:3.11-slim

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

CMD ["python", "server.py"]
