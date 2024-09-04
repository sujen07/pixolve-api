FROM python:3.12

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install system dependencies
RUN ls 
    apt-get update && \
    apt-get install -y \
    cmake \
    libboost-all-dev \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Set environment variable for OpenCV
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Command to run the application (replace with your actual command)
CMD ["fastapi", "run", "/app/app.py", "--port", "8000"]