# FSL-105 Sign Language Recognition

A small project that trains and serves a Filipino Sign Language recognizer using MediaPipe hand landmarks and a PyTorch LSTM. Use the notebook for training on the FSL-105 dataset and the script for real-time webcam inference.

## Project contents

- Training notebook: [fsl_video_training.ipynb](fsl_video_training.ipynb)
- Webcam inference script: [webcam_video_inference.py](webcam_video_inference.py#L1)
- Trained weights (v4, best checkpoint): [fsl_video_model_v4_best.pth](fsl_video_model_v4_best.pth)

## Dataset used

-https://data.mendeley.com/datasets/48y2y99mb9/2

## Environment setup

1. Create/activate a Python 3.9+ environment.
2. Install the core dependencies:

```bash
pip install torch torchvision torchaudio opencv-python mediapipe pandas numpy matplotlib tqdm seaborn scikit-learn
```

3. (Optional) Install CUDA-enabled PyTorch if you have a GPU for faster training/inference.

## Dataset preparation (FSL-105)

- Download the FSL-105 dataset and place it alongside this folder so the notebook paths resolve to:
  - `FSL-105 A dataset for recognizing 105 Filipino sign language videos/`
- Expected files inside that directory: `clips.zip`, `train.csv`, `test.csv`, `labels.csv`, and the `clips/` folder (unzipped).
- The notebook will unzip `clips.zip` into `clips/` if it does not already exist.
- If you store the dataset elsewhere, update the `DATA_ROOT` and related paths near the top of [fsl_video_training.ipynb](fsl_video_training.ipynb#L29).

## Training workflow

1. Open [fsl_video_training.ipynb](fsl_video_training.ipynb) and run cells in order.
2. The pipeline:
   - Extracts MediaPipe hand landmarks per frame (120 frames per clip, zero-padded).
   - Applies keypoint augmentations, motion features (velocity/acceleration), and smoothing.
   - Trains the `LandmarkLSTM` with mixup, focal loss, triplet loss, OneCycleLR, and optional SWA.
   - Saves checkpoints such as `fsl_video_model_v4_best.pth`, `fsl_video_model_v4_final.pth`, and `fsl_video_model_v4_swa.pth`.
3. Evaluation cells generate confusion matrices, per-class metrics, calibration plots, and a classification report saved as files in this folder.

## Real-time inference (webcam)

1. Ensure a model file is available. The script expects `fsl_video_model_best.pth` by default; either
   - rename `fsl_video_model_v4_best.pth` to `fsl_video_model_best.pth`, or
   - edit `MODEL_PATH` near the top of [webcam_video_inference.py](webcam_video_inference.py#L58) to match your filename.
2. Set `LABELS_PATH` in the same file to point to your `labels.csv` from the FSL-105 dataset (current path assumes the dataset folder sits next to this project).
3. Run the webcam app:

```bash
python webcam_video_inference.py
```

4. Controls: press SPACE to start/stop a recording window (120 frames). The predicted label and confidence appear once a clip is captured. Press `q` to quit.

## Tips

- If MediaPipe cannot find a camera, check that another app is not using it and that `cv2.VideoCapture(0)` works in a short test script.
- Lower `SEQUENCE_LENGTH` or `CONFIDENCE_THRESHOLD` in [webcam_video_inference.py](webcam_video_inference.py#L53) if you want faster but potentially less stable predictions.
- For reproducibility in training, set random seeds for NumPy and PyTorch near the imports in the notebook.
