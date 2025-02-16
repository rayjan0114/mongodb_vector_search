import glob
import json
import os
import shutil
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_V2_M_Weights, efficientnet_v2_m
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import requests

DATA_DIR = "animals10/raw-img"

def test_server(file_path):
    url = "http://0.0.0.0:1234/query"

    with open(file_path, "r") as f:
        print(file_path)
        data = json.load(f)

    response = requests.post(url, json={"embedding": data})

    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")


def load_embeddings_and_file_paths(filepath):
    embeddings = []
    file_paths = []
    for file in glob.glob(filepath):
        with open(file, "r") as f:
            embedding = json.load(f)
            embeddings.append(embedding)
            file_paths.append(file)
    return np.array(embeddings), file_paths


def save_embedding(file_path, embedding):
    """Save the embedding to a .txt file in the corresponding class folder."""
    file_path = file_path.replace("raw-img", "embedding")
    file_path = os.path.splitext(file_path)[0] + ".json"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    embedding_list = embedding.cpu().numpy().tolist()

    with open(file_path, "w") as f:
        json.dump(embedding_list, f)


def convert_images_to_jpg():
    for file_path in glob.glob(f"{DATA_DIR}/*/*"):
        if not file_path.endswith((".jpeg", ".png")):
            continue
        with Image.open(file_path) as img:
            new_path = Path(file_path).with_suffix(".jpg")

            if img.mode in ("RGBA", "LA"):
                img = img.convert("RGB")

            img.save(new_path, "JPEG", quality=100)
            print(f"Converted: {file_path} -> {new_path}")

        Path(file_path).unlink()
        print(f"Deleted original file: {file_path}")


def main():
    convert_images_to_jpg()
    torch.hub.set_dir(os.getcwd())  # Sets cache directory for models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    model = efficientnet_v2_m(weights=weights).to(device)
    preprocess = weights.transforms()
    model.eval()

    # Modify model to extract features before the softmax layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier (softmax layer)

    dataset = datasets.ImageFolder(root=DATA_DIR, transform=preprocess)

    image_paths = [dataset.samples[i][0] for i in range(len(dataset))]

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in tqdm(enumerate(dataloader)):
            images = images.to(device)

            embeddings = feature_extractor(images).squeeze(-1).squeeze(-1)  # Shape: (batch_size, 1280)

            batch_file_paths = image_paths[batch_idx * len(images) : (batch_idx + 1) * len(images)]

            for i, (file_path, embedding) in enumerate(zip(batch_file_paths, embeddings)):
                assert dataset.class_to_idx[file_path.split("/")[2]] == labels[i].item()
                save_embedding(file_path, embedding)


if __name__ == "__main__":
    # python3 -m prepare_data test_server --file_path 'animals10/embedding/cane/OIP-70DOwg3RVrsbHXBWgGH52QHaGD.json'
    # python3 -m prepare_data main
    from fire import Fire
    Fire()
