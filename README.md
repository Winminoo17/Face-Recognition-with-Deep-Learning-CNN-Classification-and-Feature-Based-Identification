# Face Recognition with Deep Learning: CNN Classification and Feature-Based Identification

Two related computer vision notebooks that build a small face recognition pipeline:

| Notebook | Task | Approach |
|----------|------|----------|
| `face_classification.ipynb` (Act 1) | **Face classification**: given a face image, predict *which person it is* from a fixed set of known people | Detect and crop faces with YuNet, then train a custom CNN |
| `face_identification.ipynb` (Act 2) | **Face identification (1:N matching)**: given a new photo, find the *closest match* in a gallery, or return `UNKNOWN` | Detect and crop faces with YuNet, extract embeddings with a pre-trained Xception model, then match by cosine similarity |

## How the two notebooks relate

Both notebooks share the same first stage, **face detection and cropping with YuNet**, and both solve the same bigger problem: *"who is this person?"* They solve it in different ways.

```
                    ┌─────────────────────────────────────────┐
  Raw photo ──────► │  YuNet face detector → crop face region │
                    └───────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              ▼                                            ▼
   Act 1: train a CNN classifier               Act 2: pre-trained Xception
   (closed set: fixed list of people)          → 2048-d feature vector
              │                                → cosine similarity vs. gallery
              ▼                                            ▼
     Predicted person ID + confidence          Best match, or UNKNOWN if below threshold
```

- **Act 1 (closed-set):** the model can only answer with one of the people it was trained on. Adding a new person means retraining.
- **Act 2 (open-set):** a new person is added by dropping one image into the gallery folder, with no retraining. A similarity threshold lets the system reject people who are not in the gallery.

## Notebook details

### Act 1: CNN face classifier (`act-8-1.ipynb`)

1. **Download the YuNet model** (`face_detection_yunet_2023mar.onnx`) from OpenCV Zoo.
2. **Crop faces** from the raw train and validation images, keeping only the face region plus a 15% margin. Removing the background stops the classifier from memorizing backgrounds instead of faces.
3. **Load the data** with `image_dataset_from_directory` at 128×128, batch size 16.
4. **Train a CNN** with:
   - Data augmentation: random zoom, contrast, and horizontal flip
   - Three Conv + MaxPooling blocks (32, 64, 128 filters)
   - Dense layers (256, 128) with dropout
   - Softmax output, Adam optimizer (learning rate 0.001)
   - Early stopping on validation loss (patience 5, up to 50 epochs, best weights restored)
5. **Plot** training vs. validation accuracy and loss.
6. **Predict** on any cropped face image (or a random image from a folder) and show the predicted ID with its confidence.

### Act 2: Gallery-based face identification (`cv-act-9-1_Week9_.ipynb`)

1. **Download the YuNet model** (same as Act 1).
2. **Load a trained Xception model** (`.keras`) and cut it at the `GlobalAveragePooling2D` layer, so the output is a **2048-dimensional feature vector** instead of class probabilities.
3. **Detect and crop** the face in each image, with a fallback to the full image if no face is found. Resize to 299×299.
4. **Build the gallery**: extract one feature vector per gallery image, labeled by filename.
5. **1:N matching**: extract features from the test image and compare with every gallery vector using **cosine similarity**. Print the ranking, the best match, and the margin over the second-best match.
6. **Threshold decision**: if the best similarity is below `THRESHOLD` (0.70 by default), the result is `UNKNOWN`.
7. **Visualize** the test face next to every gallery face with similarity percentages, and save the gallery features to `gallery_features.npy`.

## Tech stack

- Python 3
- TensorFlow / Keras
- OpenCV (`cv2.FaceDetectorYN` / YuNet)
- NumPy, scikit-learn, Matplotlib

## Setup and instructions

Both notebooks were written for **Kaggle Notebooks** and use `/kaggle/input/...` and `/kaggle/working/...` paths. The easiest way to run them is on Kaggle, with each dataset or model added through **Add Input**.

### Option A: Run on Kaggle (recommended)

1. Create a new Kaggle notebook and import the `.ipynb` file (**File → Import Notebook**).
2. Turn on a GPU: **Settings → Accelerator → GPU**.
3. Turn on **Internet** in the notebook settings (needed to download the YuNet model).
4. Add your inputs with **Add Input**:

   | Needed for | Input | Path used in the notebook |
   |------------|-------|---------------------------|
   | Act 1 | Face dataset with `gt_db_train/` and `gt_db_val/` folders, one sub-folder per person | `RAW_TRAIN_DIR`, `RAW_VAL_DIR` |
   | Act 2 | Trained Xception model (`xception_model.keras`) | `MODEL_PATH` |
   | Act 2 | Folder of gallery images, one image per person, named after the person | `GALLERY_DIR` |
   | Act 2 | One test image | `TEST_IMAGE` |

5. If your dataset has a different name or location, edit the path variables at the top of each notebook.
6. Run all cells from top to bottom.

### Option B: Run locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install tensorflow opencv-python numpy scikit-learn matplotlib jupyter

# 3. Download the YuNet face detector
wget -O face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# 4. Launch Jupyter
jupyter notebook
```

Then edit these in each notebook:
- Replace every `/kaggle/input/...` and `/kaggle/working/...` path with a local path.
- Replace the `!wget ...` line in the first cell with the download above, or remove it if the file already exists.

> `cv2.FaceDetectorYN` needs a recent OpenCV (4.5.4 or newer).

### Recommended order

1. Run **Act 1** first to understand the cropping step and the classification approach.
2. Run **Act 2** next to see how the same cropped faces can be turned into feature vectors for matching.

Act 2 loads a pre-trained Xception model rather than the CNN trained in Act 1, so the two notebooks can also be run independently.

## Expected folder structure

```
.
├── act-8-1.ipynb                # Act 1
├── cv-act-9-1_Week9_.ipynb      # Act 2
├── README.md
└── data/                        # not included in the repo
    ├── gt_db_train/             # Act 1
    │   ├── s01/  s02/  ...      # one folder per person
    ├── gt_db_val/
    │   ├── s01/  s02/  ...
    └── gallery/                 # Act 2
        ├── person_a.jpg
        └── person_b.jpg
```

## Key settings you can tune

| Setting | Notebook | Default | Effect |
|---------|----------|---------|--------|
| `MARGIN` | Act 1 | 0.15 | Padding around the detected face |
| `score_threshold` | Both | 0.6 (Act 1), 0.7 (Act 2) | Minimum face detection confidence |
| `BATCH_SIZE`, `LEARNING_RATE` | Act 1 | 16, 0.001 | Training behaviour |
| `THRESHOLD` | Act 2 | 0.70 | Minimum similarity to accept a match. Lower is more permissive, higher is stricter |

## Results

Add your own results after running the notebooks, for example:

- **Act 1:** final validation accuracy and the training/validation curves
- **Act 2:** example match output, similarity scores, and the threshold you chose

## Limitations and next steps

- The datasets are small, so results are illustrative and may not generalize to other people or conditions.
- Act 2 uses only the first detected face and takes the face as given, so group photos can give the wrong face.
- The similarity threshold should be tuned on your own data, with genuine pairs and impostor pairs.
- Possible improvements: a stronger face-embedding model, cross-validation, and a proper held-out test set.

