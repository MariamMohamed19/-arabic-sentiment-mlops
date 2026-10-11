FROM python:3.11-slim
WORKDIR /app

# CPU-only torch keeps the image far smaller than the CUDA build
RUN pip install --no-cache-dir --retries 10 --timeout 120 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --retries 10 --timeout 120 transformers

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
