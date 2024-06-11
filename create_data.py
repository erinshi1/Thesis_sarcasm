from moviepy.editor import *
import pandas as pd
import numpy as np
from glob import glob
import shutil


def extract_audio_from_mp4(video_file, output_audio_file):
    # 加载视频文件
    video = VideoFileClip(video_file)

    # 提取音频
    audio = video.audio

    # 保存音频为MP3格式
    audio.write_audiofile(output_audio_file, codec="libmp3lame")

    # 关闭视频文件和音频文件
    video.close()


# extract_audio_from_mp4(video_file_path, output_audio_path)

datas = pd.read_csv("./final_context_videos/SE-MUStARD-Final.csv")

datas = datas.values

paths = glob("./final_context_videos/final_context_videos/*")

for path in paths:
    print(path)
    try:
        name = path.split("\\")[-1][:-6]

        index = np.where(datas[:, 0] == name)[0]

        value = datas[index]

        sentence = list(value[:, 2])
        label = value[-1, 4]
    except:
        continue

    sentence = " ".join(sentence)

    sentence += "\t"+str(label)

    extract_audio_from_mp4(path, f"./audios/{name}.mp3")

    shutil.copy(path, f"./videos/{name}.mp4")

    with open(f"./texts/{name}.txt", "w", encoding="utf-8") as f:
        f.write(sentence)
