import os
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
from tqdm import tqdm
import json


def save_embedding(file_path, embedding):
    """Save the embedding to a .txt file in the corresponding class folder."""
    file_path = file_path.replace("raw-img", "embedding")
    file_path = os.path.splitext(file_path)[0] + ".json"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    embedding_list = embedding.cpu().numpy().tolist()

    with open(file_path, "w") as f:
        json.dump(embedding_list, f)


def main():
    data_dir = 'animals10/raw-img'
    torch.hub.set_dir(os.getcwd())  # Sets cache directory for models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
    model = efficientnet_v2_m(weights=weights).to(device)
    preprocess = weights.transforms()
    model.eval()

    # Modify model to extract features before the softmax layer
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier (softmax layer)

    dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)

    image_paths = [dataset.samples[i][0] for i in range(len(dataset))]

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        for batch_idx, (images, labels) in tqdm(enumerate(dataloader)):
            images = images.to(device)

            embeddings = feature_extractor(images).squeeze(-1).squeeze(-1)  # Shape: (batch_size, 1280)

            batch_file_paths = image_paths[batch_idx * len(images):(batch_idx + 1) * len(images)]

            for file_path, embedding in zip(batch_file_paths, embeddings):
                assert dataset.class_to_idx[file_path.split("/")[2]] == labels[0].item()
                save_embedding(file_path, embedding)


if __name__ == "__main__":
    main()
