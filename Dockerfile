# Use Python 3.9 slim image
FROM python:3.9-slim

# Set workdir
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsndfile1 git wget

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download Kannada ASR model from Vakyansh official source
RUN wget https://storage.googleapis.com/vakyansh-open-models/models/kannada/kn-IN/kannada_infer.pt -O kannada_infer.pt

# Expose Flask port
EXPOSE 5000

# Start Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
