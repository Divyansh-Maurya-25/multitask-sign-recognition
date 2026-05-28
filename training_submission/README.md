# Weather- and Lighting-Robust Traffic Sign Classification — Demo

## Overview

Multi-task ResNet-18 trained on GTSRB plus a synthetic adverse-condition dataset.
The model produces **two outputs per image**:

1. **Traffic sign class** — 43-way GTSRB classification (primary task)
2. **Adverse condition type** — 8-way (clean / low_brightness / overexposure / shadow / fog / rain / motion_blur / low_contrast) (auxiliary task)

The auxiliary condition-prediction head was trained jointly with the sign-classification head to encourage degradation-aware feature learning, which improves robustness on the primary task.

> **No training is performed in this demo.** It only loads a pretrained model and runs inference.

---

## Files

```
demo/
├── demo.ipynb                   # Google Colab inference notebook
├── README.md
├── multitask_resnet18.pth       # Pretrained dual-head ResNet-18
├── real_adverse_converted.zip   # Uncropped real adverse-condition scenes
└── real_cropped.zip             # Cropped + labeled real signs (for accuracy eval)
```

- **`multitask_resnet18.pth`** — ResNet-18 backbone with two heads (43 sign classes + 8 conditions), trained on GTSRB + synthetic corruptions.
- **`real_adverse_converted.zip`** — 24 hand-collected real-world adverse-condition images organized by folder: `rain/`, `fog/`, `glare/`, `night/`. Used for qualitative inspection.
- **`real_cropped.zip`** — Hand-cropped GTSRB-class signs from the above, with class IDs encoded in filenames (e.g. `14_stop_rain_01.jpg` → class 14 Stop). Used for **real-world accuracy** evaluation.

---

## How to Run (Google Colab)

1. Open `demo.ipynb` in Google Colab.
2. Set runtime to GPU (Runtime → Change runtime type → GPU).
3. Run cells top to bottom.
4. When prompted, upload:
   - `multitask_resnet18.pth` (Cell 4)
   - `real_adverse_converted.zip` (Cell 7)
   - `real_cropped.zip` (Cell 11) — *optional, enables real-world accuracy*

The notebook will:
- Load the multi-task model
- Run inference on each real-world image
- Display each image with its predicted **sign**, **condition**, and confidence scores
- Print per-folder confidence summaries
- *(If cropped zip uploaded)* Compute and display real-world classification **accuracy**

---

## Expected Output

For each image:
- **Folder** (rain / fog / glare / night)
- **Predicted sign class** + confidence
- **Predicted condition type** + confidence

For cropped images, an accuracy report:

```
Real-world cropped accuracy: 0.XXX  (correct: K/N)

By condition folder:
            sum  count   mean
folder
fog           ?      2    ?
rain          ?      3    ?
```

---

## Architecture

```
Input image (3, 64, 64)
         │
ResNet-18 backbone (ImageNet-pretrained)
         │
   Pooled features (512)
         │
    ┌────┴────────────┐
    ↓                 ↓
sign_head        condition_head
(Linear 512→43)  (Linear 512→8)
```

Joint training loss:
```
L = L_sign  +  λ * L_condition,    λ = 0.3
```

---

## Dataset

- **Training base:** GTSRB (German Traffic Sign Recognition Benchmark, 43 classes).
- **Training corruptions (synthetic):** seven OpenCV-generated degradations applied at 70% probability per training sample — `low_brightness`, `overexposure`, `shadow`, `fog`, `rain`, `motion_blur`, `low_contrast`.
- **Real-world test images:** 24 hand-collected adverse-condition scenes, plus a small subset cropped + labeled with GTSRB classes for real accuracy measurement.

The synthetic corruption pipeline is the project's primary novelty (unique dataset);
the multi-task dual-head architecture is the secondary novelty (architectural adaptation).

---

## Quantitative Evaluation

Full per-condition accuracy comparison (Baseline vs Augmented vs Multi-Task) and the
corruption-probability ablation are in the **training notebook**, not this demo.
See `metrics.csv` and `ablation_corruption_prob.csv` in the project bundle.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Model file not found | Upload `multitask_resnet18.pth` when Cell 4 prompts |
| `RuntimeError: Error(s) in loading state_dict` | You uploaded a single-head model — must be the multi-task model |
| Image folder not found | Upload `real_adverse_converted.zip` when Cell 7 prompts |
| Slow inference | Set runtime to GPU |
