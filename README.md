
---

## 📸 1. Face Extraction

We use [MTCNN](https://github.com/ipazc/mtcnn) to detect and crop faces from the input video.

### ✅ Features:
- Skips frames for efficiency (`frame_skip=10`)
- Ignores faces smaller than `80x80`
- Saves resized `128x128` faces
- Stops when it reaches `4000` faces

### 🔧 Parameters (configurable inside script):
- `input_path`: Path to the input video (default: `input.webm`)
- `output_dir`: Directory where extracted faces are saved
- `target_faces`: Number of faces to extract
- `min_face_size`: Minimum width/height of a detected face
- `frame_skip`: Number of frames to skip for speed-up

### ▶️ Run:

```bash
python face_extraction.py
```
## 🤖 2. Face Generation with GAN
A Deep Convolutional GAN (DCGAN) is used to generate 128x128 realistic faces from random noise vectors.

### 🧠 Components:
- Generator: Upsamples 100-dim noise → 128×128×3 image using Conv2DTranspose

- Discriminator: Downsamples image → single logit for real/fake classification

### 🔧 Key Settings:
- Batch size: `16`

- Noise dimension: `100`

- Epochs: `100`

- Images normalized to [-1, 1] before training

### ▶️ Run:
```bash
python face_gan_train.py
```
## 🛠 Requirements
```bash
pip install -r requirements.txt
```
###  `requirements.txt`
```bash
opencv-python
numpy
mtcnn
tqdm
tensorflow
pillow
matplotlib

```