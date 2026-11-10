# Ground-Penetrating Radar Image Processing & Anomaly Analysis

A notebook-based experiment for preprocessing a ground-penetrating radar (GPR) dataset, preparing data for anomaly detection and segmentation, training models, and exporting inference artifacts.

## Recognition

This project is patented. The repository documents the associated GPR image-processing and anomaly-analysis workflow; patent identifiers and legal documentation are intentionally not reproduced here.

## Workflow

The notebook `gpr-analysis-system.ipynb` is the main entry point. It is written for a Kaggle-style environment and uses paths under `/kaggle/working`.

1. Download and extract the MCG GPR dataset from [Zenodo record 14270869](https://zenodo.org/records/14270869).
2. Preprocess the input scans and generate image, mask, thumbnail, and metadata outputs.
3. Build a hybrid dataset with train, validation, and test splits for segmentation and detection.
4. Train a YOLOv8-based object detector.
5. Train a small U-Net segmentation model.
6. Export model formats for later inference (TorchScript for U-Net and ONNX for YOLO).

Run the notebook cells in order in an environment with internet access enabled. The notebook imports NumPy, OpenCV, PyTorch, and supporting utilities; it installs Ultralytics when it is not already available.

## Repository contents

| File | Description |
|---|---|
| `gpr-analysis-system.ipynb` | Dataset download, preprocessing, dataset construction, training, and export workflow |
| `unet_gpr_best_ts.pt.svg` | SVG file associated with the U-Net TorchScript export |
| `yolo_gpr_best.onnx.svg` | SVG file associated with the YOLO ONNX export |

The actual `.pt`, `.pth`, and `.onnx` model binaries are generated in the notebook's working `models/` directory; they are not stored in this repository.

## Important validation note

The checked-in notebook output for the YOLO training run reports **0 labeled instances** and warns that no labels were found in the detection set. That run does not establish valid detector performance. Verify the generated detection labels and dataset YAML, then rerun training and evaluation before reporting YOLO metrics or treating the exported detector as usable.

The notebook also includes U-Net training logs, but this repository does not currently include a separate benchmark report or a standalone inference application.

## Reproducibility

- Use the dataset from its original Zenodo record and review its terms before redistribution.
- Run the notebook from top to bottom so that preprocessing and generated dataset paths are created.
- Keep large source datasets, intermediate outputs, and trained model binaries out of Git unless there is a deliberate release plan.
- Record the environment and dataset version when comparing experiments.

## Scope

This repository currently provides an experimental notebook workflow. A future cleanup could split preprocessing, training, evaluation, and inference into reusable scripts, add a pinned dependency file, and publish validated quantitative results.
