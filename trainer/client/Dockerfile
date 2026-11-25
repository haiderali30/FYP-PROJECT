# Image for a federated learning client (build context: trainer/)
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONPATH=/app

COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py .
COPY model/ model/
COPY client/client.py client/client.py

# docker-compose passes the building CSV and client id as arguments
CMD ["python", "client/client.py"]
