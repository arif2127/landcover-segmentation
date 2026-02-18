# 🌍 Land Cover Semantic Segmentation from Satellite Imagery

This research focuses on multi-class land cover segmentation from high-resolution satellite imagery using deep learning. The study leverages the **LandCover.ai** dataset and implements a **DeepLabV3+** architecture for pixel-wise semantic classification.

The objective is to accurately segment satellite images into five land cover categories using a supervised deep learning framework.

---

## 📌 Land Cover Classes (5)

| Class ID | Category | Description |
|----------|----------|-------------|
| 0 | Background / Ground | Bare soil and non-structured land |
| 1 | Water | Rivers, lakes, water bodies |
| 2 | Urban / Buildings | Residential and constructed areas |
| 3 | Vegetation / Forest | Trees and dense vegetation |
| 4 | Roads | Transportation infrastructure |

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

```bash
python -m venv landcover_env
source landcover_env/bin/activate      # Linux / Mac
landcover_env\Scripts\activate         # Windows
