from glob import glob
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import librosa

device = "cuda"
model = Wav2Vec2Model.from_pretrained("wav2vec2_files")
model.to(device)
processor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2_files")
sampling_rate = 16000
paths = glob("./audios/*")
lens = sampling_rate * 22

for path in paths:
    name = path.split("\\")[-1][:-3]
    print(name)
    audio_ar, rate = librosa.load(path, sr=sampling_rate)  # 读取音频数据
    audio_ar = audio_ar[-lens:]
    values = processor([audio_ar], sampling_rate=sampling_rate, return_tensors="pt",
                               padding="max_length", truncation=True, max_length=lens)  # 填充成相同长度，因为音频时间有长有短，进行0填充

    values = {k:v.to(device) for k, v in values.items()}

    with torch.no_grad():
        output = model(**values)[0]

    output = output.cpu()
    torch.save(output, f"./audio_tensor/{name}pkl")
