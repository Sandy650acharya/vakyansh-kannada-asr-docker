FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    git \
    g++ \
    && apt-get clean

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip==23.3.2 \
 && pip install --no-cache-dir -r requirements.txt

# Create kn_model folder and download the Kannada model
RUN mkdir -p kn_model \
 && wget https://storage.googleapis.com/vakyansh-open-models/models/kannada/kn-IN/kannada_infer.pt -O kannada_infer.pt

# Expose port
EXPOSE 5000

# Start the app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]