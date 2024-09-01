FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y cmake libboost-all-dev g++ && \
    apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application (replace with your actual command)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
