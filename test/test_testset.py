import torch, imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.pipeline import EeveePipeline



class LowMemoryVideo:
    def __init__(self, file_name):
        self.reader = imageio.get_reader(file_name)
    
    def __len__(self):
        return self.reader.count_frames()

    def __getitem__(self, item):
        return Image.fromarray(np.array(self.reader.get_data(item))).convert("RGB")

    def __del__(self):
        self.reader.close()

class VideoData:
    def __init__(self, video_file=None, image_folder=None, height=None, width=None, **kwargs):
        if video_file is not None:
            self.data_type = "video"
            self.data = LowMemoryVideo(video_file, **kwargs)
        elif image_folder is not None:
            self.data_type = "images"
            self.data = LowMemoryImageFolder(image_folder, **kwargs)
        else:
            raise ValueError("Cannot open video or image folder")
        self.length = None
        self.set_shape(height, width)

    def raw_data(self):
        frames = []
        for i in range(self.__len__()):
            frames.append(self.__getitem__(i))
        return frames

    def set_length(self, length):
        self.length = length

    def set_shape(self, height, width):
        self.height = height
        self.width = width

    def __len__(self):
        if self.length is None:
            return len(self.data)
        else:
            return self.length

    def shape(self):
        if self.height is not None and self.width is not None:
            return self.height, self.width
        else:
            height, width, _ = self.__getitem__(0).shape
            return height, width

    def __getitem__(self, item):
        frame = self.data.__getitem__(item)
        width, height = frame.size
        if self.height is not None and self.width is not None:
            if self.height != height or self.width != width:
                frame = crop_and_resize(frame, self.height, self.width)
        return frame

    def __del__(self):
        pass

    def save_images(self, folder):
        os.makedirs(folder, exist_ok=True)
        for i in tqdm(range(self.__len__()), desc="Saving images"):
            frame = self.__getitem__(i)
            frame.save(os.path.join(folder, f"{i}.png"))

def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


NUM_FRAMES = 21

pipe = EeveePipeline.from_pretrained(
    torch_dtype = torch.bfloat16,
    device = "cpu",
    vae_model_path = "./checkpoints/Wan2.1-VACE-14B/Wan2.1_VAE.pth",
    text_encoder_model_path = "./checkpoints/Wan2.1-VACE-14B/models_t5_umt5-xxl-enc-bf16.pth",
    dit_model_path = [
        "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00001-of-00007.safetensors",
        "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00002-of-00007.safetensors",
        "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00003-of-00007.safetensors",
        "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00004-of-00007.safetensors",
        "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00005-of-00007.safetensors",
        "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00006-of-00007.safetensors",
        "./checkpoints/Wan2.1-VACE-14B/diffusion_pytorch_model-00007-of-00007.safetensors",
    ],
    tokenizer_path = "./checkpoints/Wan2.1-VACE-14B/google/umt5-xxl",
)
pipe.device = "cuda"
pipe.vram_management_enabled = True

pipe.load_lora(pipe.vace, "./checkpoints/step-10000.safetensors", alpha=1)
torch.cuda.empty_cache()

import csv

# dresses_test.csv 파일에서 garment와 model 인덱스 읽기
test_indices = []
with open("./data/Eevee/dresses_test.csv", mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num == 1:
            continue
   
        if len(row) >= 2:
            model_idx = int(row[0])
            garment_idx = int(row[1])
            test_indices.append((model_idx, garment_idx))

for i in range(len(test_indices)):
    test_model_idx = test_indices[i][0]
    test_garment_idx = test_indices[i][1]
    video = VideoData(f"./data/Eevee/dresses/{test_model_idx:05d}/video_0_agnostic.mp4", height=1088, width=816)
    video = [video[i] for i in range(NUM_FRAMES)]
    video_mask = VideoData(f"./data/Eevee/dresses/{test_model_idx:05d}/video_0_mask.mp4", height=1088, width=816)
    video_mask = [video_mask[i] for i in range(NUM_FRAMES)]

    vace_reference_image=Image.open(f"./data/Eevee/dresses/{test_garment_idx:05d}/garment.png").resize((816, 1088))

    # # INSERT_YOUR_CODE
    # # video의 첫 프레임, video_mask의 첫프레임, vace_reference_img를 이미지로 출력하는 테스트 코드
    # # 파일로 저장하는 코드로 변경
    # import os
    # print(test_model_idx, test_garment_idx)
    # os.makedirs("./debug", exist_ok=True)
    # video[0].save("./debug/video_0.png")
    # video_mask[0].save("./debug/video_mask_0.png")
    # vace_reference_image.save("./debug/vace_reference_image.png")
    # print('Image files saved in ./debug/')
    # exit(0)

    video = pipe(
        prompt="Model is wearing A pale yellow sleeveless dress features a simple yet elegant design. A V-neckline gracefully frames the upper part, leading down to a fitted bodice that smoothly transitions into a gently flared skirt. A subtle seam runs horizontally across the waist, accentuating the silhouette. A small, rectangular tag is visible at the neckline, bearing text too faint to discern clearly. A smooth texture suggests a soft fabric, likely designed for comfortable wear. A clean and minimalist aesthetic defines the overall style, with no additional patterns or graphics adorning the surface.",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        vace_video=video,
        vace_reference_image=vace_reference_image,
        vace_video_mask = video_mask,
        width=816,
        height=1088,
        num_frames=NUM_FRAMES,
        num_inference_steps=30,
        seed=1,
        tiled=True,
        tile_size=(15, 26),
        tile_stride=(8, 13),
    )
    save_video(video, f"./results/{test_model_idx}_{test_garment_idx}.mp4", fps=25, quality=5)  



# video = VideoData("./data/Eevee/dresses/00137/video_0_agnostic.mp4", height=1088, width=816)
# video = [video[i] for i in range(NUM_FRAMES)]
# video_mask = VideoData("./data/Eevee/dresses/00137/video_0_mask.mp4", height=1088, width=816)
# video_mask = [video_mask[i] for i in range(NUM_FRAMES)]

# vace_reference_image=Image.open("./data/Eevee/dresses/00137/garment.png").resize((816, 1088))


# video = pipe(
#     prompt="Model is wearing A pale yellow sleeveless dress features a simple yet elegant design. A V-neckline gracefully frames the upper part, leading down to a fitted bodice that smoothly transitions into a gently flared skirt. A subtle seam runs horizontally across the waist, accentuating the silhouette. A small, rectangular tag is visible at the neckline, bearing text too faint to discern clearly. A smooth texture suggests a soft fabric, likely designed for comfortable wear. A clean and minimalist aesthetic defines the overall style, with no additional patterns or graphics adorning the surface.",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=video,
#     vace_reference_image=vace_reference_image,
#     vace_video_mask = video_mask,
#     width=816,
#     height=1088,
#     num_frames=NUM_FRAMES,
#     num_inference_steps=30,
#     seed=1,
#     tiled=True,
#     tile_size=(15, 26),
#     tile_stride=(8, 13),
# )
# save_video(video, "1.mp4", fps=25, quality=5)