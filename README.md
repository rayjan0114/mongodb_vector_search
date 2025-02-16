# MongoDB Vector Search

![Demo](assets/demo.gif)

<b>This repo is for a job interview showing I am able to quickly learn C++ and build a simple vector search server in a day.</b>

## Features
- A C++ vector search server
- Python script for generating embeddings & index.html for a simple client
- TODO: upsert
- TODO: HNSW algorithm
- TODO: embedding with docs

## Quick Start
```bash
pip install -r requirements.txt
git clone https://gitlab.com/libeigen/eigen.git
```

### Download data
```bash
kaggle datasets download alessiocorrado99/animals10
unzip animals10.zip
```

### Prepare data
- Running EfficientNet for generating embeddings might take a while.
```bash
python -m prepare_data main
```

### Run server
```bash
make mrun
```

### Run client
```bash
open index.html
```
