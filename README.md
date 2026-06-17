# 🌳 Satellite Tree Instance Segmentation

Instance segmentation of individual trees from high-resolution satellite imagery using a **Mask R-CNN** model with a **Swin Transformer V2** backbone, trained on LabelMe polygon annotations.

---

## Overview

This Kaggle notebook implements an end-to-end pipeline for detecting and segmenting individual tree crowns in large `.tif` satellite images. The pipeline covers everything from reading raw georeferenced imagery and polygon annotations, through patch-based preprocessing, to training, inference, and evaluation.

**Key design choices:**
- Swin Transformer V2 (`swinv2_tiny_window16_256`) replaces the default ResNet-50 backbone in Mask R-CNN, providing stronger hierarchical feature extraction for high-resolution aerial data.
- Large images are sliced into overlapping 1024×1024 patches during both preprocessing and inference, with NMS-based merging to reassemble full-image predictions.
- A tracking JSON prevents duplicate processing when iteratively adding new images to the dataset.

---

## Pipeline

```
Raw .tif + LabelMe .json
         │
         ▼
  Patch Extraction & Preprocessing
  (resize 0.6×, 1024px patches, 50% stride)
         │
         ▼
  TreeDataset  ──►  DataLoader (train / val split)
         │
         ▼
  Swin V2 + FPN + Mask R-CNN
         │
         ▼
  Training (AdamW, ReduceLROnPlateau, Early Stopping)
         │
         ▼
  Sliding-Window Inference on Full .tif
         │
         ▼
  NMS Merge  ──►  Evaluation (Precision / Recall / F1 / IoU)
```

---

## Dataset Format

The dataset consists of pairs of files with matching names:

| File | Description |
|------|-------------|
| `<name>.tif` | Multi-band GeoTIFF satellite image |
| `<name>.json` | LabelMe polygon annotations (one polygon per tree) |

Annotations are created with the [LabelMe](https://github.com/wkentaro/labelme) tool. Each `shapes` entry contains a polygon with `points` in `[x, y]` format.

---

## Preprocessing

The preprocessing cell converts full-size `.tif` images into model-ready patches:

- **Resize** — images are scaled to 60% of original resolution (`RESIZE_FACTOR = 0.60`)
- **Pad** — padded to the nearest multiple of `PATCH_SIZE` using reflect mode
- **Patch** — extracted with `PATCH_SIZE = 1024` and `STRIDE = 512` (50% overlap)
- **Filter** — objects are kept in a patch if ≥ 50% of their area falls inside (`KEEP_THRESHOLD = 0.50`)
- **Track** — a `TRACKING.json` records which source images have been processed to avoid re-running

For each patch with at least one tree, three outputs are saved:

```
preprocessed/
├── train_test_PNG/     # <patch_id>.png  +  <patch_id>.json  (LabelMe format)
├── train_test_MASK/    # <patch_id>_mask.png  (binary, 0/255)
└── train_test_BBOX/    # <patch_id>_bbox.png  (preview with green bounding boxes)
```

---

## Model Architecture

**Backbone:** Swin Transformer V2 (`swinv2_tiny_window16_256`, ~28M params) loaded via [timm](https://github.com/huggingface/pytorch-image-models), with `features_only=True` to expose 4 hierarchical feature maps.

**Neck:** Feature Pyramid Network (FPN) with `out_channels = 256` and `LastLevelMaxPool` to produce 5 scales, matching the RPN's anchor expectations.

**Head:** Standard Mask R-CNN box and mask predictors, replaced with `FastRCNNPredictor` and `MaskRCNNPredictor` for `num_classes = 2` (background + tree).

Available Swin V2 variants (swap via `MODEL_NAME`):

| Variant | Params | Notes |
|---------|--------|-------|
| `swinv2_tiny_window16_256` | ~28M | Default — fastest |
| `swinv2_small_window16_256` | ~50M | Balanced |
| `swinv2_base_window16_256` | ~88M | Best accuracy |

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW (`lr=1e-4`, `weight_decay=0.1`) |
| LR Scheduler | ReduceLROnPlateau (`factor=0.5`, `patience=3`) |
| Early Stopping | `patience=5`, `min_delta=0.001` |
| Max Epochs | 80 |
| Batch Size | 6 |
| Train / Val Split | 90 / 10 |
| Input Size | 1024 × 1024 |

**Augmentations** (training only):
- Random horizontal & vertical flip (p=0.5 each)
- Random brightness ±20% (p=0.5)
- Random 90°/180°/270° rotation (p=0.5)

Checkpoints are saved to `./swin_checkpoints/`:
- `best_model.pth` — lowest validation loss
- `final_model.pth` — last epoch
- `history.json` — per-epoch train/val loss
- `training_curves.png` — loss plot

---

## Inference

Full `.tif` images are processed with a sliding-window approach:

1. Resize and pad the full image (same parameters as preprocessing)
2. Extract 1024×1024 patches with 50% stride
3. Run the model on each patch; filter detections by `SCORE_THRESHOLD = 0.9`
4. Shift box/mask coordinates back to full-image space
5. Apply NMS (`iou_threshold = 0.5`) across all patches to remove duplicates
6. Scale results back to original image resolution

---

## Evaluation

The evaluation cell computes the following metrics over the entire validation set:

| Metric | Description |
|--------|-------------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | Harmonic mean of precision and recall |
| Avg Box IoU | Mean IoU of matched bounding boxes |
| Avg Mask IoU | Mean IoU of matched instance masks |

Matching uses greedy assignment sorted by IoU descending, with a configurable `iou_threshold` (default 0.5).

Visualizations are saved as 6-panel figures (original image / boxes / masks for both GT and predictions).

---

## Requirements

```bash
pip install rasterio shapely timm
```

Core dependencies (available by default in Kaggle):

```
torch
torchvision
opencv-python
Pillow
matplotlib
numpy
pandas
tqdm
```

---

## Usage

### 1. Preprocess

Update the config block at the top of the preprocessing cell:

```python
image_folder    = "/kaggle/input/your-dataset/..."
annotation_folder = "/kaggle/input/your-dataset/..."
preprocessed_folder = "/kaggle/working/preprocessed/"
tracking_json_path  = "/kaggle/working/TRACKING.json"
```

### 2. Create DataLoaders

```python
train_loader, val_loader = get_dataloaders(
    preprocessed_folder="/kaggle/working/preprocessed/",
    batch_size=6,
    train_split=0.9
)
```

### 3. Build and Train the Model

```python
model = get_swin_maskrcnn_model(
    num_classes=2,
    model_name='swinv2_tiny_window16_256',
    pretrained_backbone=True,
    min_size=1024,
    max_size=1024
)

trained_model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=80,
    lr=1e-4,
    device="cuda",
    save_dir='./swin_checkpoints',
    patience=5
)
```

### 4. Run Full-Image Inference

```python
boxes, scores, masks, orig_img = infer_on_full_tiff(tiff_path, model, device)

visualize_full_image_comparison(
    orig_img, boxes, scores, masks,
    gt_boxes, gt_polygons,
    save_path="./prediction_result.png"
)
```

---

## Output Structure

```
kaggle/working/
├── preprocessed/
│   ├── train_test_PNG/      # Patch images + LabelMe JSONs
│   ├── train_test_MASK/     # Binary instance masks
│   └── train_test_BBOX/     # Bounding box preview images
├── swin_checkpoints/
│   ├── best_model.pth
│   ├── final_model.pth
│   ├── history.json
│   └── training_curves.png
├── val_visualizations/      # Per-sample GT vs. prediction panels
└── TRACKING.json
```

---

## References

- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870)
- [Swin Transformer V2 (Liu et al., 2022)](https://arxiv.org/abs/2111.09883)
- [LabelMe Annotation Tool](https://github.com/wkentaro/labelme)
- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [rasterio — Geospatial Raster I/O](https://rasterio.readthedocs.io/)
