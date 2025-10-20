# 🌱 GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction

This repository provides training scripts and baseline experiments for semi- and self-supervised learning on hyperspectral reflectance data described in this [paper](https://arxiv.org/abs/2507.06806). Implemented methods include:

- Masked Autoencoders (MAE)
- Generative Adversarial Networks (SR-GAN)
- Autoencoders with Radiative Transfer Models (RTM-AE)
- Supervised Multi-trait learning framework

The goal is to benchmark various semi- and self- supervised learning strategies for plant trait prediction using hyperspectral data.

---

## 📂 Dataset

Dataset available at:  
👉 [Hugging Face – GreenHyperSpectra](https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra)

<p align="center">
  <img src="./MultiSourceIcon.png" alt="DatasetIcon" width="40%"/>
</p>

Place the downloaded complete dataset under `Datasets/`. 
1. You can run `scripts/Split_data.py` to download the complete directories of the dataset + create unlabeled splits for the experiements (for this option intall git lfs [sudo apt-get install git-lfs, git lfs install])
2. You can check `notebooks/DataLoad_chunks.ipynb`
3. Check the data with Hugging Face datasets library, as follows:
```
from datasets import load_dataset

### labeled_all ###
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("Avatarr05/GreenHyperSpectra", "labeled_all")

df = ds['train'].to_pandas().drop(['Unnamed: 0'], axis=1)
display(df.head())

### labeled_splits: train ###
train_dataset = load_dataset("Avatarr05/GreenHyperSpectra", 'labeled_splits', split="train")

train_dataset = train_dataset.to_pandas().drop(['Unnamed: 0'], axis=1)
display(df.head())

```

---

## ⚙️ Requirements

Tested on:  
- Python 3.8.2  
- PyTorch 1.8.1+cu111  

To set up the environment:

```bash
conda create -n greenhyperspectra python=3.8.2
conda activate greenhyperspectra
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

---

## 🗂️ Repository Structure

```
HyspecSSL/
├── Datasets/              # Contains labeled and unlabeled hyperspectral data: To be downloaded from Hugging Face
├── notebooks/             # Evaluation and visualization notebooks
├── scripts/               # Training scripts for all models
├── Splits/                # Contains the stratfied splits of unlabeled data: to be created with scripts/Split_data.py
├── src/                   # Supporting modules/utilities
├── README.md
└── requirements.txt
```

### `scripts/` includes:

- **(a) Sensitivity Analysis**:  
  `*_variation_UnLb.py`, `*_variation_Lb.py`, etc.  # Train benchmark models with variable labled (Lb) and unlabeled (UnLb) data samples

- **(b) Full-range Trait Prediction**:  
  `Gan_main_unlabeled.py`, `AE_RTM_main_unlabeled.py`, `multi_main.py`

- **(c) Half-range Trait Prediction**:  
  Same as above with `--type_s 'half'` and `--input_shape 500`

- **(d) Out-of-Distribution (OOD) Evaluation**:  
  `*_main_unlabeled_TransCV.py`, `multi_main_Trans.py`

- **(e) MAE Ablation Studies**:  
  `MAE_grid_search_*.py`

---

## 🚀 Training

Each script accepts the following arguments:

| Argument              | Description |
|-----------------------|-------------|
| `--seed`              | Random seed for reproducibility |
| `--path_data_lb`      | Path to labeled dataset (CSV) |
| `--directory_path`    | Path to directory with unlabeled splits |
| `--input_shape`       | Input dimensionality (e.g. 1720 or 500) |
| `--type_s`            | Training subset type (`full` or `half`) |
| `--n_epochs`          | Number of training epochs (default: 300) |
| `--batch_size`        | Batch size (default: 128) |
| `--lr`                | Learning rate (varies by method) |
| `--mask_ratio`        | For MAE: proportion of masked features |
| `--name_experiment`   | Identifier for the experiment |
| `--project_wandb`     | (Optional) Weights & Biases project name |
| `--path_save`         | Directory to save outputs |

### Example Training Commands

**GAN:**
```bash
python scripts/Gan_main_unlabeled.py \
  --seed 42 \
  --path_data_lb Datasets/annotated.csv \
  --directory_path Splits/ \
  --input_shape 1720 \
  --type_s full \
  --name_experiment gan_full_run \
  --path_save checkpoints/
```

**AE-RTM:**
```bash
python scripts/AE_RTM_main_unlabeled.py \
  --seed 42 \
  --path_data_lb Datasets/annotated.csv \
  --directory_path Splits/ \
  --input_shape 1721 \
  --type_s full \
  --name_experiment aertm_full_run \
  --path_save checkpoints/
```

**Multi-Trait Supervised:**
```bash
python scripts/multi_main.py \
  --seed 42 \
  --path_data_lb Datasets/annotated.csv \
  --directory_path Splits/ \
  --input_shape 1720 \
  --type_s full \
  --name_experiment multi_supervised \
  --path_save checkpoints/
```

**MAE:**
The traits' prediction model with MAE is trained in two steps:
**1. MAE pretaining:**
```bash
python scripts/mae_unlabeled.py \
  --seed 42 \
  --directory_path Splits/ \
  --input_shape 1720 \
  --type_s full \
  --name_experiment mae_full \
  --path_save checkpoints/
```
**2. MAE downstream task:**
This requires the reference to the pre-trained MAE model
```bash
python scripts/MAE_downstreamReg.py \
  --seed 42 \
  --path_data_lb Datasets/annotated.csv \
  --input_shape 1720 \
  --type_s full
```
---

## 💾 Pretrained Models

Pretrained models can be found here:  
👉 [GreenHyperSpectra Pretrained Checkpoints](https://huggingface.co/Avatarr05/Multi-trait_SSL/tree/main)

---

## ✅ Evaluation

Run the corresponding Jupyter notebooks in `notebooks/` to evaluate the models and visualize results. Make sure to update paths to match your local setup or use data from Hugging Face, both options are included in the notebooks.

---

## 📣 Citation

If you use this work, please cite the corresponding publication:
```bibtex
@article{cherif2025greenhyperspectra,
  title={GreenHyperSpectra: A multi-source hyperspectral dataset for global vegetation trait prediction},
  author={Cherif, Eya and Ouaknine, Arthur and Brown, Luke A and Dao, Phuong D and Kovach, Kyle R and Lu, Bing and Mederer, Daniel and Feilhauer, Hannes and Kattenborn, Teja and Rolnick, David},
  journal={arXiv preprint arXiv:2507.06806},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the **CC BY-NC 4.0 License**. See the [LICENSE](LICENSE) file for more information.
