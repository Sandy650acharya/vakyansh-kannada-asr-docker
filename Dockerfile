FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies (ffmpeg, sound libs, git)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip safely and install Python dependencies
RUN pip install --upgrade pip==23.3.2 \
 && pip install --no-cache-dir -r requirements.txt

# Download Kannada model with retry and fallback check
RUN wget -q --show-progress --tries=3 --timeout=30 \
  https://storage.googleapis.com/vakyansh-open-models/models/kannada/kn-IN/kannada_infer.pt \
  -O /app/kannada_infer.pt || (echo "❌ Model download failed" && exit 1)

# Expose the port used by the Flask app
EXPOSE 5000

# Run the app using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
