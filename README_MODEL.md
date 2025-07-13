# Training & Converting a Body-Part Detection Model

This document outlines a **reference pipeline** for building a custom detector that recognises the eight body parts used by the VR Body-Part Analyzer and deploying it as a TensorRT engine used by the application.

---

## 1. Dataset Preparation

1. Record or collect video frames / images of the target domain (adult 3-D/VR content).
2. Label the following classes using a tool like *Label Studio* or *CVAT*:

   * `head`
   * `mouth`
   * `hand_1`  (left hand)
   * `hand_2`  (right hand)
   * `breasts`
   * `pelvis`
   * `vagina`
   * `penis`

3. Export annotations as **YOLOv7 format** (`images/` + `labels/` with one text file per image).

> Why YOLOv7? Any single-stage detector works.  YOLO formats are ubiquitous and have straightforward ONNX/TensorRT conversion tooling.

## 2. Training (YOLOv7 example)

```bash
# Clone repo + install requirements
$ git clone https://github.com/WongKinYiu/yolov7 && cd yolov7
$ pip install -r requirements.txt

# Train (tune hyper-parameters as needed)
$ python train.py \
    --weights yolov7.pt \
    --data ../bodypart.yaml \
    --epochs 50 --batch 16 --img 960 960 \
    --device 0

# Best checkpoint will be in runs/train/…/weights/best.pt
```

## 3. Export to ONNX

```bash
$ python export.py --weights best.pt --img 960 960 --simplify --dynamic-batch
# Produces best.onnx
```

## 4. Convert ONNX → TensorRT

Install NVIDIA’s TensorRT *and* the Python wheel that matches your CUDA driver.

```bash
$ /usr/src/tensorrt/bin/trtexec \
    --onnx=best.onnx \
    --saveEngine=body_parts.engine \
    --explicitBatch --workspace=4096 --fp16

# The resulting engine (~MBs) should be copied to
#    models/body_parts.engine
```

* `--fp16` can be dropped if your GPU lacks FP16 support.
* Increase `--workspace` for large models.

## 5. Verify the Engine

Run the analyzer and ensure the log prints

```
[INFO] TensorRT engine loaded from models/body_parts.engine
```

## 6. Troubleshooting

| Problem | Hint |
|---------|------|
| "module tensorrt not found" | Install the official wheel: `pip install tensorrt==X.X.X` matching the system’s TensorRT version |
| Inference crashes with shape errors | Ensure `--img` size used in export matches **exactly** the size passed during training/inference |
| CUDA out of memory | Reduce `--batch` or switch to `--fp16` / `--int8` quantisation |

## 7. Alternative Frameworks

• **Ultralytics YOLOv8** → `yolo export onnx` then `trtexec`
• **MMDetection / Detectron2** → Export to ONNX via script, then follow step 4.

---
Once `body_parts.engine` is present, the app automatically switches from CPU-mock detections to live TensorRT inference.