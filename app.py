from flask import Flask, request, jsonify
from inference import transcribe

app = Flask(__name__)

@app.route('/')
def index():
    return '✅ Vakyansh Kannada ASR is running! Use /transcribe endpoint.'

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio_file = request.files['audio']
    audio_path = "temp.wav"
    audio_file.save(audio_path)

    try:
        transcription = transcribe(audio_path)
        return jsonify({'transcription': transcription})
    except Exception as e:
        return jsonify({'error': str(e)}), 500