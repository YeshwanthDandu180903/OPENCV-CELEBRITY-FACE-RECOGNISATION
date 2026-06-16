<div align="center">

# 🎭 OpenCV Celebrity Face Recognition

**A Python + OpenCV project that trains an LBPH face recogniser on celebrity images and identifies faces in real-time using Haar Cascade detection.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/YeshwanthDandu180903/OPENCV-CELEBRITY-FACE-RECOGNISATION?style=for-the-badge)](https://github.com/YeshwanthDandu180903/OPENCV-CELEBRITY-FACE-RECOGNISATION/stargazers)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [How It Works](#-how-it-works)
- [Celebrities Recognised](#-celebrities-recognised)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Step 1 — Training the Model](#step-1--training-the-model)
  - [Step 2 — Recognising a Face](#step-2--recognising-a-face)
- [Configuration](#️-configuration)
- [Sample Output](#-sample-output)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

This project demonstrates a complete **face recognition pipeline** built entirely with OpenCV and Python — no deep-learning frameworks required. It uses:

| Step | Technique | Purpose |
|------|-----------|---------|
| Face Detection | **Haar Cascade Classifier** | Locate face regions in an image |
| Feature Extraction | Grayscale crop of face ROI | Prepare data for the recogniser |
| Model Training | **LBPH Face Recogniser** | Learn each celebrity's texture patterns |
| Prediction | `face_recognizer.predict()` | Return name + confidence score |

The model is trained on a custom dataset of **4 South Indian celebrities** and can identify them from any test image, drawing a bounding box and name label overlay on the detected face.

---

## ⚙️ How It Works

```
Training Phase
──────────────
Celebrity Images  ─►  Haar Cascade (detect face)  ─►  Grayscale ROI crop
                                                         │
                                              LBPH Face Recognizer .train()
                                                         │
                               trained_model.yml  +  features.npy  +  labels.npy

Recognition Phase
─────────────────
Test Image  ─►  Resize  ─►  Grayscale  ─►  Haar Cascade (detect face)
                                                         │
                                              LBPH .predict(face_region)
                                                         │
                               Celebrity Name  +  Confidence Score  ─►  Annotated Image
```

> **LBPH (Local Binary Pattern Histogram)** is a classic, lightweight algorithm that encodes local texture patterns in a face image. It is rotation-tolerant, fast to train, and performs well on small datasets — making it ideal for a project like this without needing a GPU.

---

## 🌟 Celebrities Recognised

| Label | Celebrity | Domain |
|-------|-----------|--------|
| 0 | **Fahad Fazil** | South Indian (Malayalam) actor |
| 1 | **Nazriya Nazim** | South Indian (Malayalam/Tamil) actress |
| 2 | **Rajinikanth** | South Indian (Tamil) superstar |
| 3 | **Vijay (Thalapathy)** | South Indian (Tamil) actor |

Each celebrity folder contains **6–8 curated JPEG images** covering varied poses and lighting conditions.

---

## 📁 Project Structure

```
OPENCV-CELEBRITY-FACE-RECOGNISATION/
│
├── face_project/
│   ├── features.npy            # Saved face-region features (numpy array)
│   ├── labels.npy              # Corresponding integer labels
│   ├── trained_model.yml       # Pre-trained LBPH model weights (~4.8 MB)
│   │
│   └── faces/
│       ├── train.py            # ← Run this first to train the model
│       ├── recognize.py        # ← Run this to recognise a face
│       │
│       ├── face Recognize/     # Training dataset
│       │   ├── Rajini_kanth/   # 8 images of Rajinikanth
│       │   ├── fahad/          # Images of Fahad Fazil
│       │   ├── nazariya/       # Images of Nazriya Nazim
│       │   └── thalapathy/     # Images of Vijay (Thalapathy)
│       │
│       └── face_validation/    # Test / validation images
│           ├── Rajini_kanth/
│           ├── fahad/
│           ├── nazariya/
│           └── thalapathy/
│
└── README.md
```

---

## 🛠️ Prerequisites

| Requirement | Minimum Version | Install Command |
|-------------|-----------------|----------------|
| Python | 3.8+ | [python.org](https://www.python.org/downloads/) |
| OpenCV with contrib | 4.x | `pip install opencv-contrib-python` |
| NumPy | 1.21+ | `pip install numpy` |
| `haar_face.xml` | — | Bundled with OpenCV install |

> **Why `opencv-contrib-python`?**  
> The `cv2.face` module (which contains `LBPHFaceRecognizer`) lives in OpenCV's extra/contrib packages and is **not** included in the standard `opencv-python` wheel.

**Where to find `haar_face.xml`:**
```
# After installing opencv-contrib-python, the file is at:
<your-python-env>/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml
```

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/YeshwanthDandu180903/OPENCV-CELEBRITY-FACE-RECOGNISATION.git
cd OPENCV-CELEBRITY-FACE-RECOGNISATION

# 2. (Recommended) Create a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install opencv-contrib-python numpy
```

---

## 🚀 Usage

### Step 1 — Training the Model

> **Skip this step** if you want to use the pre-trained model already committed to the repository (`trained_model.yml`, `features.npy`, `labels.npy`).

Open `face_project/faces/train.py` and update the two absolute paths to match your system:

```python
# train.py — update these lines
haar_face = cv2.CascadeClassifier("PATH/TO/haarcascade_frontalface_default.xml")
DIR       = r"PATH/TO/face_project/faces/face Recognize"
```

Then run:

```bash
python face_project/faces/train.py
```

**Expected console output:**
```
Training done..................
```

This saves three files: `trained_model.yml`, `features.npy`, and `labels.npy`.

---

### Step 2 — Recognising a Face

Open `face_project/faces/recognize.py` and update the absolute paths:

```python
# recognize.py — update these lines
input_image    = cv2.imread("PATH/TO/your_test_image.jpeg")
haar_face      = cv2.CascadeClassifier("PATH/TO/haarcascade_frontalface_default.xml")
features       = np.load("PATH/TO/face_project/features.npy", allow_pickle=True)
labels         = np.load("PATH/TO/face_project/labels.npy",   allow_pickle=True)
face_recognizer.read("PATH/TO/face_project/trained_model.yml")
```

Then run:

```bash
python face_project/faces/recognize.py
```

**Expected console output:**
```
thalapathy
 Person name is thalapathy confidence level is 47.83
```

**Expected visual output:** Two OpenCV windows open — a grayscale preview and the annotated colour image with a green bounding box and name label drawn above the face.

Press any key to close the windows.

---

## ⚙️ Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `scale` | `resize_image()` in `recognize.py` | `200` | Resize multiplier (% of original size) |
| `scaleFactor` | `detectMultiScale()` | `1.3` | Image pyramid scale step |
| `minNeighbors` | `detectMultiScale()` | `2` | Minimum rectangles required for a valid detection |
| `people` | Both scripts | `['fahad','nazariya','Rajini_kanth','thalapathy']` | Ordered list — index matches the integer label |

### Adding a New Celebrity

1. Create a new folder inside `face Recognize/` named after the celebrity.
2. Add **at least 6–8** clear, front-facing face images (JPEG/PNG).
3. Append the folder name to the `people` list in **both** `train.py` and `recognize.py` (order must match).
4. Re-run `train.py` to regenerate all three model files.

---

## 🖼️ Sample Output

```
Input image  ──►  Haar Cascade detects face  ──►  LBPH predicts label

╔═══════════════════════════════════════╗
║  thalapathy                           ║  ← Name drawn above bounding box
║  ┌─────────────────────┐              ║
║  │                     │              ║
║  │    [ face ROI ]     │  ← Green     ║
║  │                     │    rectangle ║
║  └─────────────────────┘              ║
║                                       ║
║  Confidence: 47.83  (lower = better)  ║
╚═══════════════════════════════════════╝
```

> **About the Confidence Score:** In LBPH, a **lower** value means a **better** match (it is a distance measure, not a probability). A score below ~80 is generally a reliable prediction; above ~100 should be treated with caution.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# 1. Fork the repo and clone your fork
git clone https://github.com/<your-username>/OPENCV-CELEBRITY-FACE-RECOGNISATION.git

# 2. Create a feature branch
git checkout -b feature/your-improvement

# 3. Make your changes, then commit
git commit -m "feat: describe your change"

# 4. Push and open a Pull Request
git push origin feature/your-improvement
```

### Ideas for Improvement

- [ ] Add a `requirements.txt` for one-command dependency install
- [ ] Replace hard-coded absolute paths with `argparse` CLI arguments or relative paths
- [ ] Add **real-time webcam recognition** using `cv2.VideoCapture(0)`
- [ ] Expand the dataset with more celebrities or larger image sets
- [ ] Add a confidence **threshold** to reject unknown / unrecognised faces
- [ ] Build a simple GUI with Tkinter or a web interface with Streamlit
- [ ] Add a data-augmentation step to improve accuracy on varied lighting

---

## 📄 License

This project is released under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

| | |
|--|--|
| **Author** | Yeshwanth Dandu |
| **GitHub** | [@YeshwanthDandu180903](https://github.com/YeshwanthDandu180903) |
| **Repository** | [OPENCV-CELEBRITY-FACE-RECOGNISATION](https://github.com/YeshwanthDandu180903/OPENCV-CELEBRITY-FACE-RECOGNISATION) |
| **Issues** | [Open an issue](https://github.com/YeshwanthDandu180903/OPENCV-CELEBRITY-FACE-RECOGNISATION/issues) |

---

<div align="center">

Made with ❤️ using Python & OpenCV

⭐ **Star this repo if you found it helpful!**

</div>
