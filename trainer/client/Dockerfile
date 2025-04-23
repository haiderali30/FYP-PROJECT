# client/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy global configuration and model files from the root
COPY config.py .
COPY model/ model/

# Copy the client folder contents (client.py and requirements.txt)
# Assuming your Docker Compose context is the project root.
COPY client/requirements.txt .
COPY client/client.py .

# Install client dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the default command.
# Note: We won't set CMD here since we'll override it with a command in docker-compose.yml.
CMD ["python", "client.py"]
