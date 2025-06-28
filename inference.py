import torch
import torchaudio
import os

# 🛠 Dummy class to support model unpickling
class Wav2VecCtc:
    def __init__(self, *args, **kwargs):
        pass

# 🛠 Make the class visible to pickle loader
import builtins
builtins.Wav2VecCtc = Wav2VecCtc

# 🗂️ Model & dictionary paths
MODEL_PATH = "kn_model/kannada_infer.pt"
DICT_PATH = "kn_model/dict.ltr.txt"

print("🔁 Loading Kannada ASR model...")

# ✅ Safe model load
checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
model_state = checkpoint["model"]
model_args = checkpoint["args"]

from fairseq.models.wav2vec import Wav2Vec2Model
model = Wav2Vec2Model.build_model(model_args, task=None)
model.load_state_dict(model_state)
model.eval()

# 📖 Load char dictionary
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
