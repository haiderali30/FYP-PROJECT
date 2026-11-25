# Image for the federated learning server (build context: trainer/)
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY model/ model/
COPY server/server.py server/server.py

EXPOSE 8080
CMD ["python", "server/server.py"]
