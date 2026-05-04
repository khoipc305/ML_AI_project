# Skin Cancer Classifier (CS4210 Project)

Group project to classify skin lesion pictures from the
HAM10000 dataset into 7 types, and try semi-supervised learning.

## The 7 classes

`nv` (normal mole), `mel` (melanoma), `bkl`, `bcc`, `akiec`,
`vasc`, `df`. Most pictures are `nv`, so the data is imbalanced.

## What we do

1. Prepare the data (this folder).
2. Train models on the prepared CSVs.
3. Compare scores and write the report.

## Data prep steps

- Check every image file exists.
- Remove duplicate pictures of the same lesion.
- Resize images to 224x224 and normalize.
- Turn `dx` labels into numbers 0-6.
- Split into train / val / test = 70 / 15 / 15.
- Inside train, split into labeled and unlabeled for semi-supervised learning.

## Models to try

- **KNN** and **Decision Tree** as simple baselines.
- **CNN (ResNet-18)** fine-tuned on the labeled data.
- **CNN + self-training** that also learns from unlabeled images.

## How we score

Main score is **macro F1**. Also report accuracy, ROC-AUC per class,
and a confusion matrix.

## Files

- `config.py` - settings (paths, split sizes, image size).
- `utils.py` - small helper functions.
- `data_preparation.py` - makes the CSV files.
- `example_usage.py` - tiny example of loading an image.
- `HAM10000/` - raw images and metadata.
- `processed_data/` - CSV files created by the script.

## Get the dataset

The HAM10000 images are too large to keep in this repo, so download them
from Kaggle and put them in a local `HAM10000/` folder.

Option A - manual download:

1. Go to https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click "Download" and unzip the archive.
3. Copy all `.jpg` files into `HAM10000/`.
4. Copy the `HAM10000_metadata` file (no extension) into `HAM10000/` too.

Option B - Kaggle CLI (faster):

```powershell
py -m pip install kaggle
# Make sure your Kaggle API token is saved at %USERPROFILE%\.kaggle\kaggle.json
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p HAM10000 --unzip
```

After this, `HAM10000/` should contain ~10,000 `.jpg` files plus
`HAM10000_metadata`.

## How to run

In PowerShell, from this folder:

```powershell
py -m pip install -r requirements.txt
py data_preparation.py
```

Optional quick check:

```powershell
py example_usage.py
```

## Source

Based on `CS4210_SkinCancer_ML_Proposal.pdf` (see it for references).
