# Efficient ResNet for CIFAR-10 (under 5M parameters)

Deep Learning (Fall 2026), Project 1: a modified ResNet trained from scratch on CIFAR-10 with at most 5 million trainable parameters, evaluated on the held-out Kaggle test set of the *Efficient ResNet Architecture Challenge*.

## Results

Final model: 3-stage ResNet with basic residual blocks, stage widths 64 / 128 / 256, 3 blocks per stage.

| | |
|---|---|
| Trainable parameters | **4,327,754** (limit 5,000,000) |
| Accuracy on 5,000 hold-out training images | **96.42%** (96.80% with horizontal-flip TTA) |
| Kaggle public / private score (late submission) | **0.867 / 0.840** |

The hold-out images come from the provided training set and were never trained on. The Kaggle test set is a separate collection of images, and all our models score about 10 to 13 points lower on it than on the hold-out set.

### Experiments

All runs use the same 45,000 / 5,000 stratified split and the same optimizer settings.

| Experiment | Model | Parameters | Epochs | Augmentation / regularization | Hold-out acc. |
|---|---|---|---|---|---|
| 1 Baseline | A (2-2-2 blocks) | 2,777,674 | 60 | crop + flip | 94.64% |
| 2 Regularized | A | 2,777,674 | 60 | + Cutout 8x8, label smoothing 0.1 | 95.40% |
| 3 Deeper | B (3-3-3 blocks) | 4,327,754 | 60 | same as 2 | 95.82% |
| Longer training | B | 4,327,754 | 200 | same as 2 | 96.46% |
| 4 Strong augmentation (final) | B | 4,327,754 | 200 | + rotation, zoom, color jitter, grayscale, blur, noise | 96.42% |

Hold-out accuracy is measured at the last epoch, without test-time augmentation.

### Kaggle scores

| Submission | Public | Private |
|---|---|---|
| Model B, 200 epochs, standard augmentation | 0.867 | 0.827 |
| **Experiment 4: strong augmentation (final)** | **0.867** | **0.840** |
| Experiment 4 + test-time BatchNorm adaptation | 0.859 | 0.836 |
| Experiment 4 + BatchNorm adaptation + class-balanced prediction | 0.861 | 0.838 |

**Disclosure.** The last two rows use the unlabeled test images at prediction time (BatchNorm statistics recomputed from the test images; class probabilities rescaled to be balanced). Both techniques were explicitly allowed by the instructor. They did not improve the score, and the final result does not use them.

## Repository contents

| File | Description |
|---|---|
| `resnet_cifar10_v3.ipynb` | Complete, executed notebook: data loading, model, training, all experiments, plots, final accuracy and parameter count, Kaggle submission files |
| `README.md` | This file |

## How to reproduce

The notebook is written for Google Colab with a T4 GPU.

1. Download the competition data from Kaggle and upload the zip file (`data-2.zip`, containing `train_images/`, `test_images/`, `train_labels.csv`, `test.csv`, `sample_submission.csv`) anywhere in your Google Drive.
2. Open `resnet_cifar10_v3.ipynb` in Colab and select `Runtime > Change runtime type > T4 GPU`.
3. Select `Runtime > Run all` and allow access to Google Drive when asked.

The full notebook takes about 2.5 hours on a T4 (about 10 s per epoch for model A and 16 s for model B). A checkpoint is saved to `MyDrive/dl-project1/runs/` after every epoch; if the session disconnects, run all cells again and training resumes where it stopped. Set `SMOKE_TEST = True` in the first code cell for a 2-epoch run that only checks the pipeline.

Outside Colab, set the environment variables `DATA_DIR` (folder containing the unzipped data) and `OUT_DIR` (folder for checkpoints and results). Requirements: Python 3, PyTorch 2.3 or newer, NumPy, pandas, Matplotlib, Pillow.

Outputs written to the results folder:

* `<experiment>/checkpoint.pt` and `<experiment>/history.json` for every run
* `submission.csv` (standard augmentation), `submission_exp4.csv` (final), and the test-time adaptation variants

## Training configuration

| Setting | Value |
|---|---|
| Optimizer | SGD, Nesterov momentum 0.9 |
| Batch size | 128 |
| Learning rate | peak 0.1, linear warm-up for 2 epochs, cosine decay to 0 |
| Weight decay | 5e-4 |
| Label smoothing | 0.1 (experiments 2 to 4) |
| Epochs | 60 for the comparison runs, 200 for the final runs |
| Precision | mixed precision (AMP) |
| Seed | 0 (data split and training) |
| Pretrained weights / external data | none |

## References

* K. He, X. Zhang, S. Ren, J. Sun. Deep Residual Learning for Image Recognition. CVPR 2016.
* T. DeVries, G. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. arXiv:1708.04552, 2017.
* C. Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
* I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017.
* D. Hendrycks, T. Dietterich. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019.
* B. Recht, R. Roelofs, L. Schmidt, V. Shankar. Do CIFAR-10 Classifiers Generalize to CIFAR-10? arXiv:1806.00451, 2018.
* Z. Nado et al. Evaluating Prediction-Time Batch Normalization for Robustness under Covariate Shift. arXiv:2006.10963, 2020.
* S. Schneider et al. Improving Robustness against Common Corruptions by Covariate Shift Adaptation. NeurIPS 2020.
* A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. 2009.
