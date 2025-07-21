#%%
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
import json
from PIL import Image
import io
import os
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
)

#%%
# Load HumanEdit dataset
hf_dataset = load_dataset("dataset/instructpix2pix-1000-samples", split="train", trust_remote_code=True)

# Load your local tar_desc mapping
with open("downloaded_datatset/target_description.json", "r") as f:
    desc_data = json.load(f)
imgid2desc = {item["img_id"]: item["tar_desc"] for item in desc_data} if isinstance(desc_data, list) else desc_data

#%%
model_id = "pretrained_frameworks/pretrained_IEDMs/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# adjust the path with your trained unet model
pipe.unet = UNet2DConditionModel.from_pretrained("pretrained_models/instruct-pix2pix/unet", torch_dtype=torch.float16).to("cuda")

# %% 
class StreamingDataset(IterableDataset):
    def __init__(self, hf_dataset, processor=None,
                 input_key="original_image", output_key="edited_image",
                 instruction_key="edit_prompt"):
        self.dataset = hf_dataset
        self.imgid2desc = imgid2desc
        self.input_key = input_key
        self.output_key = output_key
        self.instruction_key = instruction_key
        self.processor = processor

    def __iter__(self):
        for sample in self.dataset:
            try:
                # Load input image
                input_img = sample[self.input_key]
                input_img = Image.open(io.BytesIO(input_img)) if isinstance(input_img, bytes) else input_img
                input_img = input_img.convert("RGB").resize((512, 512))

                # Load output image
                output_img = sample[self.output_key]
                output_img = Image.open(io.BytesIO(output_img)) if isinstance(output_img, bytes) else output_img
                output_img = output_img.convert("RGB").resize((512, 512))

                instruction = sample[self.instruction_key]

                yield {
                    "input_image": input_img,
                    "output_image": output_img,
                    "text": instruction,
                }
            except Exception:
                continue
            
# %%
output_dir = "generated_results/ip2p"
model_name = "ip2p_nogated_fusio"
save_dir = os.path.join(output_dir, model_name)
os.makedirs(save_dir, exist_ok=True)

num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10
generator = torch.Generator("cuda").manual_seed(42)

dataset = StreamingDataset(hf_dataset, imgid2desc)

# === Run inference and save results
for sample in dataset:
    prompt = sample["text"]
    pil_input_img = sample["input_image"]
    img_id = sample["img_id"]

    edited_image = pipe(
        prompt,
        image=pil_input_img,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    save_path = os.path.join(save_dir, f"{img_id}_edited.jpg")
    edited_image.save(save_path)

# %%
