import torch
import torchaudio
import os

from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc

# Register the safe class so PyTorch knows how to unpickle the model
torch.serialization.add_safe_class(Wav2VecCtc)

# Load model path
MODEL_PATH = "kn_model/kannada_infer.pt"
DICT_PATH = "kn_model/dict.ltr.txt"

print("🔁 Loading Kannada ASR model...")

# Safely load full model checkpoint
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model = checkpoint["model"]
model.eval()

# Load dictionary
with open(DICT_PATH, "r", encoding="utf-8") as f:
    index_to_char = [line.split()[0] for line in f]

def transcribe(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    with torch.no_grad():
        logits = model(waveform.squeeze(0))["encoder_out"][0]
        pred_ids = torch.argmax(logits, dim=-1)

    transcript = ''.join([index_to_char[i] for i in pred_ids if i < len(index_to_char)])
    return transcript.replace("|", " ").strip()
