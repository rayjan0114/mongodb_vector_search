import glob
import json
import numpy as np


def load_embeddings_and_file_paths(filepath):
    embeddings = []
    file_paths = []
    for file in glob.glob(filepath):
        with open(file, "r") as f:
            embedding = json.load(f)
            embeddings.append(embedding)
            file_paths.append(file)
    return np.array(embeddings), file_paths


if __name__ == "__main__":
    embeddings, file_paths = load_embeddings_and_file_paths("animals10/embedding/*/*.json")
    print(embeddings.shape)
    print(file_paths)
