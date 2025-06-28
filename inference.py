import torch
import torchaudio
import os

# Dummy class required for loading the checkpoint
class Wav2VecCtc(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

# Model and dictionary paths
MODEL_PATH = "kn_model/kannada_infer.pt"
DICT_PATH = "kn_model/dict.ltr.txt"

print("🔁 Loading Kannada ASR model...")

# Load model safely
cp = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
model = cp['model']
model.eval()

# Load dictionary
with open(DICT_PATH, "r", encoding="utf-8") as f:
    index_to_char = [line.split()[0] for line in f]

# Transcription function
def transcribe(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    with torch.no_grad():
        logits = model(waveform.squeeze(0))["encoder_out"][0]
        pred_ids = torch.argmax(logits, dim=-1)

    # Greedy decode
    transcript = ''.join([index_to_char[i] for i in pred_ids if i < len(index_to_char)])
    return transcript.replace("|", " ").strip()
