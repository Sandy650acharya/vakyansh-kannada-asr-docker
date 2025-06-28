FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install OS dependencies including g++
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    git \
    g++ \
    && apt-get clean

# Copy project files
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip==23.3.2 \
 && pip install --no-cache-dir -r requirements.txt

# Download Kannada model from Vakyansh official storage
RUN wget https://storage.googleapis.com/vakyansh-open-models/models/kannada/kn-IN/kannada_infer.pt -O kannada_infer.pt

# Expose the Flask app port
EXPOSE 5000

# Start the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]