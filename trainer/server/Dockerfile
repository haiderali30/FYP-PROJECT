# server/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy global configuration and model files
COPY config.py .
COPY model/ model/

# Copy the server folder contents (server.py and requirements.txt)
COPY server/requirements.txt .
COPY server/server.py .

# Install server dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "server.py"]
