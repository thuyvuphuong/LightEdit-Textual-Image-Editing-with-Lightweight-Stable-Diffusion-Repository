import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.models.inception import inception_v3
import torch.nn.functional as F
import lpips
import timm
import torchvision.transforms as T
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoImageProcessor
from transformers import BlipProcessor, BlipForImageTextRetrieval
import json

def load_image(path):
    image = Image.open(path).convert("RGB")
    return image

def load_images_from_folder(folder, transform):
    images = []
    for fname in os.listdir(folder):
        if fname.endswith((".jpg", ".png", ".jpeg")):
            img = load_image(os.path.join(folder, fname))
            img = transform(img)
            images.append(img)
    return torch.stack(images)


def calculate_clip_image_similarity(folder_a, folder_b, verbose=True):
    """
    Compute CLIP cosine similarity between image pairs from two folders.
    
    Args:
        folder_a (str): Path to folder A (e.g., generated images).
        folder_b (str): Path to folder B (e.g., original images).
        verbose (bool): Whether to print results.
        
    Returns:
        similarities (list of float): CLIP cosine similarities per pair.
        avg_similarity (float): Mean CLIP similarity across pairs.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("pretrained_models/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("pretrained_models/clip-vit-base-patch32")

    # Get matching IDs from folder A
    image_ids = [
        fname.split("_")[0]
        for fname in os.listdir(folder_a)
        if fname.endswith(".jpg")
    ]

    similarities = []
    for img_id in tqdm(image_ids, desc="Computing CLIP similarity"):
        path_a = os.path.join(folder_a, f"{img_id}_input_edited.jpg")
        path_b = os.path.join(folder_b, f"{img_id}_input.jpg")

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue

        img_a = load_image(path_a)
        img_b = load_image(path_b)

        inputs = processor(images=[img_a, img_b], return_tensors="pt").to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)

            sim = cosine_similarity(features[0], features[1], dim=0).item()
            similarities.append(sim)

    avg_similarity = float(np.mean(similarities)) if similarities else 0.0

    if verbose:
        print(f"Compared {len(similarities)} image pairs.")
        print(f"Average CLIP similarity: {avg_similarity:.4f}")

    return similarities, avg_similarity

def calculate_fid(real_folder, gen_folder, device="cuda"):
    fid = FrechetInceptionDistance(normalize=True).to(device)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    real_images = load_images_from_folder(real_folder, transform).to(device)
    gen_images = load_images_from_folder(gen_folder, transform).to(device)

    # Update FID state
    fid.update(real_images, real=True)
    fid.update(gen_images, real=False)

    score = fid.compute().item()
    print(f"FID score: {score:.4f}")
    return score

def calculate_inception_score(image_folder, device="cuda", batch_size=32, splits=10):
    """
    Compute the Inception Score (IS) for a folder of images.
    
    Args:
        image_folder (str): Path to the folder containing images.
        device (str): "cuda" or "cpu".
        batch_size (int): Batch size for feeding images into Inception.
        splits (int): Number of splits to calculate IS variance.
    
    Returns:
        Tuple: (mean IS, std IS)
    """
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Load and preprocess all images
    all_images = load_images_from_folder(image_folder, transform).to(device)

    preds = []
    with torch.no_grad():
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i+batch_size]
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
            preds.append(probs.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    scores = []

    N = preds.shape[0]
    split_size = N // splits

    for i in range(splits):
        part = preds[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl_sum = np.sum(kl, axis=1)
        scores.append(np.exp(np.mean(kl_sum)))

    mean_is = float(np.mean(scores))

    print(f"Inception Score: {mean_is:.4f}")
    return mean_is

loss_fn = lpips.LPIPS(net='vgg').cuda()
def preprocess(image_path):
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    tensor = transforms.ToTensor()(image).unsqueeze(0) * 2 - 1  # scale to [-1, 1]
    return tensor.cuda()

# Compare images from two folders
def calculate_lpips(folder_a, folder_b):
    scores = []
    image_ids = [f.split("_")[0] for f in os.listdir(folder_a) if f.endswith(".jpg")]

    for img_id in tqdm(image_ids):
        path_a = os.path.join(folder_a, f"{img_id}_input_edited.jpg")
        path_b = os.path.join(folder_b, f"{img_id}_input.jpg")

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue

        img0 = preprocess(path_a)
        img1 = preprocess(path_b)

        with torch.no_grad():
            d = loss_fn(img0, img1).item()
        scores.append(d)

    avg_score = sum(scores) / len(scores)
    print(f"Average LPIPS: {avg_score:.4f}")
    return avg_score


def load_hf_dino_model(local_path="pretrained_models/dino-vitb16", device="cuda"):
    """
    Load the locally saved Hugging Face DINO model and processor.
    """
    model = AutoModel.from_pretrained(local_path).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(local_path)
    return model, processor


def extract_dino_features_hf(image_path, model, processor, device):
    """
    Extract DINO ViT features from an image using Hugging Face model and processor.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).squeeze()  # Mean pool [CLS] patch

    return features


def calculate_dino_similarity_score(folder_a, folder_b, local_model_path="pretrained_models/dino-vitb16", verbose=True):
    """
    Compute the average cosine similarity of DINO features between two folders.
    
    Assumes files are named: <id>_edited.jpg in folder_a and <id>_input.jpg in folder_b.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_hf_dino_model(local_model_path, device)

    image_ids = [fname.split("_")[0] for fname in os.listdir(folder_a) if fname.endswith(".jpg")]
    similarities = []

    for img_id in tqdm(image_ids, desc="Calculating DINO similarity"):
        path_a = os.path.join(folder_a, f"{img_id}_input_edited.jpg")
        path_b = os.path.join(folder_b, f"{img_id}_input.jpg")

        if not os.path.exists(path_a) or not os.path.exists(path_b):
            continue

        feat_a = extract_dino_features_hf(path_a, model, processor, device)
        feat_b = extract_dino_features_hf(path_b, model, processor, device)

        sim = cosine_similarity(feat_a, feat_b, dim=0).item()
        similarities.append(sim)

    avg_score = float(np.mean(similarities)) if similarities else 0.0

    print(f"Average DINO similarity: {avg_score:.4f}")

    return avg_score

