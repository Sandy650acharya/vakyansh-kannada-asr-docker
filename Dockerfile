# Use Python 3.9 slim image
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Install audio dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 git

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Start server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
