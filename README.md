      
# BV-DA-CNN

This repository contains the official PyTorch implementation for **BV-DA-CNN**, the fully supervised domain adaptation method for object detection presented in our paper:

[**"An artificial intelligence cloud platform for OCT-based retinal anomalies screening system in real clinical environments"**](https://www.nature.com/articles/s41746-025-01959-7)
## Introduction

BV-DA-CNN is designed for tasks where a model trained on a source domain dataset (typically larger) needs to be adapted to perform well on a target domain dataset (typically smaller and with different data characteristics), specifically in the context of OCT-based retinal disease screening. This implementation focuses on object detection tasks.

## Prerequisites

1.  **Clone this repository:**
    ```bash
    git clone https://github.com/wRuanMing/BV-DA-CNN.git
    cd BV-DA-CNN
    ```
2.  **Install Dependencies:** This project is built upon [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Please follow the installation instructions provided in the `maskrcnn-benchmark` repository to set up the necessary environment and dependencies. "Note on Directory Paths: The maskrcnn_benchmark and tools directories included in this repository may conflict with identically named directories from the base maskrcnn-benchmark framework if it's also in your Python path or project structure. Please ensure that scripts and import statements correctly reference the maskrcnn_benchmark and tools directories local to this project."

## Dataset Preparation

You will need a source domain dataset and a target domain dataset, **both with bounding box annotations for object detection**.

1.  **Dataset Guidelines:**
    *   The **source domain dataset** should ideally have a larger number of images.
    *   The **target domain dataset** can have a smaller number of images.

2.  **Symlink Datasets:**
    Create symbolic links from your dataset locations to the `datasets/` directory within this project:
    ```bash
    # Create the datasets directory if it doesn't exist
    mkdir -p datasets

    # Symlink your source domain dataset
    ln -s /path/to/your/source_domain_dataset datasets/source_dataset_name

    # Symlink your target domain dataset
    ln -s /path/to/your/target_domain_dataset datasets/target_dataset_name
    ```
    Replace `/path/to/your/source_domain_dataset` and `source_dataset_name` with the actual path to your source data and a chosen name for the symlink (e.g., `oct_source_data`). Do the same for the target domain. The names `source_dataset_name` and `target_dataset_name` should correspond to how they are referenced in your configuration files.

## Training

To train the BV-DA-CNN model, run the following command:
```bash
python tools/train_net.py --config-file "configs/BV-DA-CNN.yaml"
```

## Testing (Evaluation)

To test a trained model, use the following command:

```bash
python tools/test_net.py \
    --config-file "configs/BV-DA-CNN.yaml" \
    MODEL.WEIGHT /path/to/your/trained_model_checkpoint/model_final.pth
```
    
## Acknowledgements
This project is built upon the maskrcnn-benchmark object detection framework. And our code was inspired by and partially from krumo/Domain-Adaptive-Faster-RCNN-PyTorch as a valuable reference for domain adaptation techniques in object detection.