# 🌍 Land Cover Semantic Segmentation from Satellite Imagery

This research focuses on multi-class land cover segmentation from high-resolution satellite imagery using deep learning. The study leverages the **LandCover.ai** dataset and implements a **DeepLabV3+** architecture for pixel-wise semantic classification.

The objective is to accurately segment satellite images into five land cover categories using a supervised deep learning framework.

---


<!-- ## 🧠 Model Architecture

- **Model:** DeepLabV3+  
- **Backbone:** ResNet101  
- **Framework:** PyTorch  
- **Dataset:** LandCover.ai  

DeepLabV3+ is selected for its ability to capture multi-scale contextual information using atrous convolution and encoder-decoder refinement.

--- -->

## 🖼 Example Results

### Original Satellite Image
![Original Image](images/image.png)

### Segmented Overlay Prediction
![Segmented Overlay](images/overlay.png)



# ⚙️ Environment Setup

## 1️⃣ Create Virtual Environment






## 1️⃣ Create Virtual Environment

```bash
conda create -n landcover_env python=3.10 -y
conda activate landcover_env
```


## 2️⃣ Install Dependencies


Install required libraries:

```bash
pip install torch torchvision
pip install segmentation-models-pytorch timm
pip install opencv-python
pip install numpy matplotlib tqdm albumentations
```

---

# 📥 Download Dataset

This project uses the **LandCover.ai** dataset.

To download and prepare the dataset:

```bash
python download_data.py
cd data/raw/landcoverai/
python python split.py
```

Dataset structure:

```
data/
└── raw/
    └── landcoverai/
        ├── images/
        ├── masks/
        ├── output/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

---

# 🏋️ Training

Run the training script:

```bash
python train.py \
--data_dir data/raw/landcoverai \
--train_split data/raw/landcoverai/train.txt \
--val_split data/raw/landcoverai/val.txt \
--epochs 50 \
--batch_size 8 \
--lr 1e-4 \
--val_interval 5
```

# 🧪 Testing

Evaluate model on the test dataset:

```bash
python test.py \
--data_dir data/raw/landcoverai \
--test_split data/raw/landcoverai/test.txt \
--model_path outputs/models/model_name.pth
```



# 🔎 Inference on Single Image

Run inference on a `.tif` satellite image:

```bash
python inference.py \
--model_path outputs/models/model_name \
--image_path path/to/image.tif
```




# 🎨 Color Mapping

| Class | Category | Color |
|--------|----------|--------|
| 0 | Background / Ground | Grey |
| 1 | Urban / Buildings | Red |
| 2 | Vegetation / Forest | Green |
| 3 | Water | Blue |
| 4 | Roads | Yellow |

---

# 📜 License

This project is released under the MIT License.