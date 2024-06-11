from glob import glob
import torch
from transformers import TimesformerModel, VideoMAEFeatureExtractor
import av
import numpy as np

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


device = "cuda"
model = TimesformerModel.from_pretrained("timesformer_files")
frame_size = model.config.num_frames
model.to(device)
processor = VideoMAEFeatureExtractor.from_pretrained("timesformer_files")
paths = glob("./videos/*")


for path in paths:
    name = path.split("\\")[-1][:-3]
    print(name)
    try:
        container = av.open(path)
        indices = sample_frame_indices(clip_len=frame_size, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices)

        values = processor(list(video), return_tensors="pt")
        values = {k:v.to(device) for k, v in values.items()}

        with torch.no_grad():
            output = model(**values)[0]
    except:
        continue
    output = output.cpu()
    torch.save(output, f"./video_tensor/{name}pkl")
