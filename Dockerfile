FROM python:3.12

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the application (replace with your actual command)
CMD ["fastapi", "run", "app.py", "--port", "8000"]