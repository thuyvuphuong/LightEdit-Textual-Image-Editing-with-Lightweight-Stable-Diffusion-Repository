#%%
import PIL
import torch
import matplotlib.pyplot as plt
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler
)

#%%
# torch.cuda.set_device(6)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load main pipeline
model_id = "pretrained_models/instruct-pix2pix"
# pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.unet = UNet2DConditionModel.from_pretrained("pretrained_models/instruct-pix2pix/unet", 
                                                 torch_dtype=torch.float32).to(device)

#%%
# Image preparation
# generator = torch.Generator("cuda").manual_seed(0)
image_path = "imgs/test.jpg"
target_resolution = (512, 512)

def load_image(img_path):
    image = PIL.Image.open(img_path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    resized_image = image.resize(target_resolution, resample=PIL.Image.LANCZOS)
    return resized_image

image = load_image(image_path)

#%%
# Inference
prompt = "Remove the sun"
num_inference_steps = 20
image_guidance_scale = 1.5
guidance_scale = 10

edited_image = pipe(
    prompt,
    image=image,
    num_inference_steps=num_inference_steps,
    image_guidance_scale=image_guidance_scale,
    guidance_scale=guidance_scale,
    # generator=generator,
).images[0]
edited_image
# edited_image.save("edited_image1.png")

# %%
