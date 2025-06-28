import torch
import torchaudio
import os

from fairseq.models.wav2vec import Wav2Vec2Model

# Load model
MODEL_PATH = "kn_model/kannada_infer.pt"
DICT_PATH = "kn_model/dict.ltr.txt"

print("🔁 Loading Kannada ASR model...")
cp = torch.load(MODEL_PATH)
model = Wav2Vec2Model.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

# Load dictionary
with open(DICT_PATH, "r", encoding="utf-8") as f:
    index_to_char = [line.split()[0] for line in f]

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