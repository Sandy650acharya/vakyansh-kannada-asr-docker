FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    git \
    g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Install Python packages
RUN pip install --upgrade pip==23.3.2 \
 && pip install --no-cache-dir -r requirements.txt

# Download Kannada ASR model and dictionary
RUN mkdir -p kn_model \
 && wget https://storage.googleapis.com/vakyansh-open-models/models/kannada/kn-IN/kannada_infer.pt -O kn_model/kannada_infer.pt \
 && wget https://storage.googleapis.com/vakyansh-open-models/models/kannada/kn-IN/dict.ltr.txt -O kn_model/dict.ltr.txt

# Expose the default Flask port
EXPOSE 5000

# Run using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
