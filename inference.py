import torch
import torchaudio
import os

from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq import tasks

# Model and dictionary paths
MODEL_PATH = "kn_model/kannada_infer.pt"
DICT_PATH = "kn_model/dict.ltr.txt"

print("🔁 Loading Kannada ASR model...")

# Load model using fairseq utilities (safe method)
models, cfg, task = load_model_ensemble_and_task([MODEL_PATH])
model = models[0]
model.eval()

# Load dictionary
with open(DICT_PATH, "r", encoding="utf-8") as f:
    index_to_char = [line.split()[0] for line in f]

def transcribe(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    with torch.no_grad():
        logits = model(waveform.squeeze(0))["encoder_out"][0]
        pred_ids = torch.argmax(logits, dim=-1)

    # Greedy decode
    transcript = ''.join([index_to_char[i] for i in pred_ids if i < len(index_to_char)])
    return transcript.replace("|", " ").strip()
