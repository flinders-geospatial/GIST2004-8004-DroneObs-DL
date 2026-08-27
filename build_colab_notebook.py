"""Build the student-facing Google Colab notebook.

The generated notebook is the file students use. Keeping the cells here makes
the JSON notebook reproducible and lets us syntax-check its Python code.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("drone_observation_yolo_colab.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    cell_source = text.strip()
    python_only = "\n".join(
        line for line in cell_source.splitlines() if not line.lstrip().startswith("%")
    )
    compile(python_only, "<notebook cell>", "exec")
    return nbf.v4.new_code_cell(cell_source)


cells = [
    md(
        r"""
# Drone observation: train, track and count with YOLO

In this practical you will:

1. watch a short drone video and inspect a labelled image dataset;
2. see what a pretrained YOLO model detects before any extra training;
3. fine-tune that model on drone imagery;
4. track vehicles through the drone video; and
5. count tracked vehicles that cross a line.

The footage is low-altitude oblique drone video: the drone hovers at roughly 50 to 60 m and looks across at an intersection, rather than straight down. The pilot dataset has three labels: `bicycle`, `motorcycle`, and `vehicle`. `vehicle` combines cars, vans, trucks, and buses. People are not a class in this dataset.
"""
    ),
    md(
        r"""
## Before running anything

1. Open this notebook in Google Colab and choose **File → Save a copy in Drive**.
2. Choose **Runtime → Change runtime type → GPU**. A T4 is sufficient when one is available.
3. Run the cells in order from the top: click a cell and press **Shift+Enter**, or click the play button at its left. A cell has finished when the spinner at its left becomes a number.

Some cells show a settings panel beside the code, with tick boxes and text fields. These are Colab form fields: they set values in that cell without you editing the code. After changing one, run the cell again.

Colab runtimes reset without warning. The notebook saves training and video outputs to your Drive.
"""
    ),
    md(
        r"""
## 1. Set up the runtime

Install the package versions used to test this practical.
"""
    ),
    code(
        r"""
%pip install -q "ultralytics==8.4.51" "supervision==0.27.0.post2"
"""
    ),
    code(
        r"""
from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
from zipfile import ZipFile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
import torch
import ultralytics
import yaml
from IPython.display import Image as NotebookImage
from IPython.display import Video, display
from matplotlib.patches import Rectangle
from ultralytics import YOLO

from google.colab import drive, files

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("Ultralytics:", ultralytics.__version__)
print("Supervision:", sv.__version__)
print("PyTorch:", torch.__version__)
print("Compute:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

if not torch.cuda.is_available():
    print("\nTraining is disabled until a GPU runtime is connected.")
    print("In Colab: Runtime → Change runtime type → GPU, then rerun this cell.")
"""
    ),
    md(
        r"""
### Core terms

| Term | What it means here |
|---|---|
| Detection | A box, class, and confidence for one object in one frame |
| Tracking | Giving the same moving object a persistent ID across frames |
| Line crossing | Counting a tracked ID when it moves across a chosen line |
| Validation | Testing the detector on labelled images it did not train on |
"""
    ),
    md(
        r"""
## 2. Fetch the course files

The course link is already filled in below. The four course files download once to this Colab runtime. Your own outputs are saved to your Google Drive, which Colab will ask permission to connect.
"""
    ),
    code(
        r"""
drive.mount("/content/drive")

COURSE_DATA_URL = "https://gist2004-droneobs-2026.s3.ap-southeast-2.amazonaws.com"  # @param {type:"string"}

OUTPUT_FOLDER = Path("/content/drive/MyDrive/GIST2004-8004/my-droneobs-results")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

ASSET_FOLDER = Path("/content/droneobs_course_assets")
ASSET_FOLDER.mkdir(parents=True, exist_ok=True)
DATASET_ZIP = ASSET_FOLDER / "droneobs-demo.zip"
BACKUP_MODEL = ASSET_FOLDER / "reference_model.pt"
BIGGER_MODEL = ASSET_FOLDER / "strong_vehicle_model.pt"
VIDEO_SOURCE = ASSET_FOLDER / "DJI_0022_teaching_clip.mp4"

assert COURSE_DATA_URL.startswith("http"), "Paste the course link first, then rerun this cell."
for destination in (DATASET_ZIP, BACKUP_MODEL, BIGGER_MODEL, VIDEO_SOURCE):
    if not destination.exists():
        print("Downloading", destination.name, "...")
        # Download under a temporary name so an interrupted download is retried.
        partial = destination.with_name(destination.name + ".part")
        urlretrieve(f"{COURSE_DATA_URL.rstrip('/')}/{destination.name}", partial)
        partial.rename(destination)

for path in (DATASET_ZIP, BACKUP_MODEL, BIGGER_MODEL, VIDEO_SOURCE):
    print(f"{path.stat().st_size / 1024**2:8.1f} MB  {path.name}")
print("Outputs will be saved to", OUTPUT_FOLDER)
"""
    ),
    md(
        r"""
### Watch the teaching clip

This is the video you will process later. Watch it once before any modelling and note what you would count: how many vehicles cross the intersection, and in which directions.
"""
    ),
    code(
        r"""
# Streams from the course link; the downloaded copy is used for processing.
display(Video(f"{COURSE_DATA_URL.rstrip('/')}/{VIDEO_SOURCE.name}", width=960))
"""
    ),
    md(
        r"""
## 3. Unpack and check the dataset

Each YOLO label row contains a class number followed by the box centre, width and height, expressed as fractions of image size.

| Split | Use |
|---|---|
| `train` | Adjust the model |
| `valid` | Check progress during training |
| `test` | Final held-out check |

Extract to Colab's local disk for faster training.
"""
    ),
    code(
        r"""
WORK_ROOT = Path("/content/droneobs_work")
EXTRACT_ROOT = Path("/content")
WORK_ROOT.mkdir(parents=True, exist_ok=True)

SOURCE_DATA_YAML = EXTRACT_ROOT / "droneobs-demo" / "data.yaml"
if not SOURCE_DATA_YAML.exists():
    print("Extracting the dataset once to Colab's local disk ...")
    with ZipFile(DATASET_ZIP) as archive:
        archive.extractall(EXTRACT_ROOT)
else:
    print("Dataset is already extracted in this runtime.")

DATASET_ROOT = SOURCE_DATA_YAML.parent.resolve()

# Make the path explicit even if the course folder is renamed later.
data_config = yaml.safe_load(SOURCE_DATA_YAML.read_text())
data_config["path"] = str(DATASET_ROOT)
LOCAL_DATA_YAML = WORK_ROOT / "data_colab.yaml"
LOCAL_DATA_YAML.write_text(yaml.safe_dump(data_config, sort_keys=False))

print("Dataset root:", DATASET_ROOT)
print(LOCAL_DATA_YAML.read_text())
"""
    ),
    code(
        r"""
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {int(k): v for k, v in data_config["names"].items()}

split_rows = []
split_images = {}
for split_name, yaml_key in (("train", "train"), ("valid", "val"), ("test", "test")):
    image_dir = DATASET_ROOT / data_config[yaml_key]
    label_dir = image_dir.parent / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    split_images[split_name] = images

    class_counts = Counter()
    empty_labels = 0
    for image_path in images:
        rows = (label_dir / f"{image_path.stem}.txt").read_text().splitlines()
        if not rows:
            empty_labels += 1
        for row in rows:
            class_counts[int(row.split()[0])] += 1

    summary = {
        "split": split_name,
        "images": len(images),
        "empty images": empty_labels,
        "boxes": sum(class_counts.values()),
    }
    summary.update({CLASS_NAMES[i]: class_counts[i] for i in sorted(CLASS_NAMES)})
    split_rows.append(summary)

dataset_summary = pd.DataFrame(split_rows).set_index("split")
display(dataset_summary)
print("\nEmpty label files are deliberate background examples, not missing annotations.")
print("Notice the class imbalance: vehicles are much more common than bicycles.")
"""
    ),
    md(
        r"""
### Look before training

Check for misplaced boxes, missing objects and inconsistent classes.
"""
    ),
    code(
        r"""
rng = np.random.default_rng()
sample_paths = rng.choice(split_images["train"], size=6, replace=False)  # rerun for a fresh six

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
colors = {0: "gold", 1: "darkorange", 2: "deepskyblue"}

for axis, image_path in zip(axes.flat, sample_paths):
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    for row in label_path.read_text().splitlines():
        class_id, cx, cy, box_w, box_h = map(float, row.split())
        x = (cx - box_w / 2) * width
        y = (cy - box_h / 2) * height
        rect = Rectangle(
            (x, y), box_w * width, box_h * height,
            fill=False, linewidth=1.2, edgecolor=colors[int(class_id)],
        )
        axis.add_patch(rect)
    axis.imshow(image)
    axis.set_title(image_path.name[:45], fontsize=8)
    axis.axis("off")

plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 4. Fine-tune a pretrained model

Training a detector from nothing needs millions of labelled images. Instead we start from `yolo26n.pt`, a small pretrained model from [Ultralytics](https://docs.ultralytics.com/models/). It has already been trained on COCO, a large collection of everyday photos covering 80 common object types, so it already knows a great deal about edges, textures, vehicles and people. Adjusting a pretrained model with a much smaller set of our own examples is called transfer learning.

First, try it exactly as downloaded. The address below points to a classic photo used to test detectors. Run the cell, then paste any other image address from the web into the field and run it again.
"""
    ),
    code(
        r"""
stock_model = YOLO("yolo26n.pt")  # downloads the pretrained COCO weights once

IMAGE_URL = "https://ultralytics.com/images/bus.jpg"  # @param {type:"string"}

stock_result = stock_model.predict(IMAGE_URL, imgsz=960, conf=0.25, device=DEVICE, verbose=False)[0]

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(stock_result.plot(line_width=2), cv2.COLOR_BGR2RGB))
plt.title(f"Pretrained COCO model, no fine-tuning: {len(stock_result.boxes)} detections")
plt.axis("off")
plt.show()

detected_classes = Counter(stock_model.names[int(c)] for c in stock_result.boxes.cls)
print("Detected:", dict(detected_classes))
"""
    ),
    md(
        r"""
On everyday photos the pretrained model already performs well, because COCO covers scenes like this. Our task is different: small objects seen obliquely from 50 to 60 m, and three classes (`bicycle`, `motorcycle`, `vehicle`) that COCO does not use. Fine-tuning keeps what the model knows about images in general and adjusts it to our labels and viewpoint.

### Training settings

| Setting | Meaning |
|---|---|
| `epochs` | Passes through the training images |
| `imgsz` | Working image size; larger costs more time and memory |
| `batch` | Images processed together; 12 is sized for the free T4 GPU |

The tick box beside the next cell is a form field that stops a stray **Run all** from starting a long training run: training only runs while it is ticked. During training the run keeps its best weights up to date after every epoch and copies them, with the diagnostics, to your Drive when it finishes. If your run is interrupted, a backup model trained in advance is supplied; the next section shows how to switch to it.
"""
    ),
    code(
        r"""
TRAIN_MODEL = False  # @param {type:"boolean"}
BASE_MODEL = "yolo26n.pt"

TRAIN_SETTINGS = {"epochs": 20, "imgsz": 960, "batch": 12}
RUN_NAME = "droneobs_yolo26n"
RUNS_ROOT = WORK_ROOT / "runs"
MY_MODEL = RUNS_ROOT / RUN_NAME / "weights" / "best.pt"

print(TRAIN_SETTINGS)
print("Training enabled:", TRAIN_MODEL)
if not TRAIN_MODEL:
    print("Set TRAIN_MODEL to True when you are ready to start the run.")
"""
    ),
    code(
        r"""
if TRAIN_MODEL:
    assert torch.cuda.is_available(), "Connect a GPU runtime before training."
    training_model = YOLO(BASE_MODEL)
    train_result = training_model.train(
        data=str(LOCAL_DATA_YAML),
        device=DEVICE,
        project=str(RUNS_ROOT),
        name=RUN_NAME,
        exist_ok=True,
        workers=2,
        seed=2004,
        patience=7,
        plots=True,
        **TRAIN_SETTINGS,
    )
    MY_MODEL = Path(train_result.save_dir) / "weights" / "best.pt"

    saved_run = OUTPUT_FOLDER / RUN_NAME
    saved_run.mkdir(parents=True, exist_ok=True)
    useful_files = [
        MY_MODEL,
        Path(train_result.save_dir) / "weights" / "last.pt",
        Path(train_result.save_dir) / "results.csv",
        Path(train_result.save_dir) / "results.png",
        Path(train_result.save_dir) / "confusion_matrix.png",
        Path(train_result.save_dir) / "confusion_matrix_normalized.png",
    ]
    for source in useful_files:
        if source.exists():
            shutil.copy2(source, saved_run / source.name)
    print("Saved model and diagnostics to", saved_run)
else:
    print("Training skipped. The remaining sections can use the supplied backup model.")
"""
    ),
    md(
        r"""
### Reading the training curves

The loss curves should fall and the mAP curves should rise as training progresses. If the validation loss starts rising while the training loss keeps falling, the model is memorising the training images and its results on new images will get worse.
"""
    ),
    code(
        r"""
results_plot = MY_MODEL.parents[1] / "results.png"
if results_plot.exists():
    display(NotebookImage(filename=str(results_plot)))
else:
    print("No new training plot yet. Run the training cell, or continue with the backup model.")
"""
    ),
    md(
        r"""
## 5. Try the detector on one held-out image

Three models appear from here on. The dropdown menus let you switch between them.

| Choice | What it is |
|---|---|
| `my model` | The YOLO26n you fine-tuned above |
| `backup model` | A small model like yours, trained on the same dataset in advance |
| `bigger model` | A larger model trained in advance on far more drone footage; it appears in the video section |

If your training run went wrong, or ran out of time, untick `TRAIN_MODEL` in the training cell, choose `backup model` below, and carry on. Nothing below depends on your own run finishing.

The test images were never used during training, so they are a fair check of what the model learned. Every detection carries a confidence score. Confidence is a filter score, not a true probability: a lower threshold gives more detections and more false detections. Try changing `conf` in the prediction cell to 0.15 and then 0.45 and compare what appears and disappears.
"""
    ),
    code(
        r"""
WEIGHTS_CHOICE = "my model"  # @param ["my model", "backup model"]

weights_path = MY_MODEL if WEIGHTS_CHOICE == "my model" else BACKUP_MODEL
assert weights_path.exists(), "Weights not found. If training did not finish, choose 'backup model'."
inference_model = YOLO(str(weights_path))
MODEL_CLASS_NAMES = {int(k): v for k, v in inference_model.names.items()}
print("Using:", weights_path)
print("Classes:", MODEL_CLASS_NAMES)
"""
    ),
    code(
        r"""
test_image = rng.choice(split_images["test"])  # rerun for a different image
prediction = inference_model.predict(
    source=str(test_image), imgsz=960, conf=0.25, device=DEVICE, verbose=False
)[0]
annotated = cv2.cvtColor(prediction.plot(line_width=1), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(14, 9))
plt.imshow(annotated)
plt.title(f"{len(prediction.boxes)} detections at confidence ≥ 0.25")
plt.axis("off")
plt.show()
"""
    ),
    md(
        r"""
VisDrone and the field footage differ in location, flight height, camera angle, shadows and compression. This is domain shift. Inspect field footage before trusting the counts.
"""
    ),
    md(
        r"""
## 6. Prepare the video and a counting line

Today you rehearse the full workflow on the supplied clip. Next week you will fly the drone; the class footage will be published as a few short clips at course web links. Choose `video web link`, paste a link, and the same workflow runs on it.

A fixed counting line is only valid while the camera holds still. If the drone pans or drifts, the road moves underneath the line and creates false crossings. Longer moving-camera work needs stabilisation or georeferencing.
"""
    ),
    code(
        r"""
VIDEO_SOURCE_MODE = "supplied clip"  # @param ["supplied clip", "video web link"]
NEW_VIDEO_URL = ""  # @param {type:"string"}

if VIDEO_SOURCE_MODE == "video web link":
    LOCAL_VIDEO = WORK_ROOT / (Path(urlparse(NEW_VIDEO_URL).path).name or "new_drone_video.mp4")
    print("Downloading the video ...")
    urlretrieve(NEW_VIDEO_URL, LOCAL_VIDEO)
else:
    LOCAL_VIDEO = WORK_ROOT / VIDEO_SOURCE.name
    if not LOCAL_VIDEO.exists():
        shutil.copy2(VIDEO_SOURCE, LOCAL_VIDEO)

cap = cv2.VideoCapture(str(LOCAL_VIDEO))
assert cap.isOpened(), f"Could not open {LOCAL_VIDEO}"
SOURCE_FPS = cap.get(cv2.CAP_PROP_FPS)
SOURCE_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
SOURCE_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SOURCE_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(
    f"{LOCAL_VIDEO.name}: {SOURCE_WIDTH}×{SOURCE_HEIGHT}, "
    f"{SOURCE_FPS:.2f} fps, {SOURCE_FRAMES / SOURCE_FPS:.1f} seconds, "
    f"{LOCAL_VIDEO.stat().st_size / 1024**2:.1f} MB"
)
"""
    ),
    md(
        r"""
### Choose the detector for the video

Start with your own model. The pilot dataset is small, so expect missed vehicles and dropped tracks. Then come back, choose `bigger model`, and rerun the cells from here: it was trained on far more footage and holds each vehicle from frame to frame.
"""
    ),
    code(
        r"""
VIDEO_MODEL_CHOICE = "my model"  # @param ["my model", "backup model", "bigger model"]

video_weights = {
    "my model": MY_MODEL,
    "backup model": BACKUP_MODEL,
    "bigger model": BIGGER_MODEL,
}[VIDEO_MODEL_CHOICE]
assert video_weights.exists(), "Weights not found. If training did not finish, choose 'backup model' or 'bigger model'."
inference_model = YOLO(str(video_weights))
MODEL_CLASS_NAMES = {int(k): v for k, v in inference_model.names.items()}
print("Video model:", video_weights.name)
print("Classes:", MODEL_CLASS_NAMES)
"""
    ),
    code(
        r"""
PROCESS_WIDTH = 1280  # every processed frame is resized to this width

def resize_to_width(frame, target_width):
    height, width = frame.shape[:2]
    if width == target_width:
        return frame
    scale = target_width / width
    return cv2.resize(
        frame, (target_width, round(height * scale)), interpolation=cv2.INTER_AREA
    )

cap = cv2.VideoCapture(str(LOCAL_VIDEO))
ok, first_frame = cap.read()
cap.release()
assert ok, "Could not read the first video frame"

zone_frame = resize_to_width(first_frame, PROCESS_WIDTH)
ZONE_HEIGHT, ZONE_WIDTH = zone_frame.shape[:2]
ZONE_IMAGE = WORK_ROOT / "zone_reference.jpg"
cv2.imwrite(str(ZONE_IMAGE), zone_frame)
print(f"Reference frame saved: {ZONE_WIDTH}×{ZONE_HEIGHT}")
"""
    ),
    md(
        r"""
### Your model and the bigger model on the same frame

The two results are stacked above each other so you can scan the same region in both. Model size and training data both change, so this is not a controlled comparison. Check missed objects, false detections, box placement and confidence.

For reference, mAP50 scores how often a model's boxes match the human labels with at least 50% overlap (0 is no matches, 1.0 is perfect). The backup model scored mAP50 0.449 on the pilot test set. The bigger model scored 0.852, but on its own test set with different labels, so the two numbers are not a head-to-head score.
"""
    ),
    code(
        r"""
RUN_MODEL_COMPARISON = True  # @param {type:"boolean"}

if RUN_MODEL_COMPARISON:
    small_model_path = MY_MODEL if MY_MODEL.exists() else BACKUP_MODEL
    comparison_specs = [
        ("my model" if small_model_path == MY_MODEL else "backup model", small_model_path, None),
        ("bigger model (vehicle only)", BIGGER_MODEL, [0]),
    ]
    comparison_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    for axis, (title, model_path, class_filter) in zip(axes, comparison_specs):
        model = YOLO(str(model_path))
        result = model.predict(
            zone_frame, imgsz=960, conf=0.25, classes=class_filter,
            device=DEVICE, verbose=False,
        )[0]
        axis.imshow(cv2.cvtColor(result.plot(line_width=1), cv2.COLOR_BGR2RGB))
        axis.set_title(f"{title}: {len(result.boxes)} detections")
        axis.axis("off")
        comparison_rows.append(
            {
                "model": title,
                "parameters (millions)": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1),
                "file size (MB)": round(model_path.stat().st_size / 1024**2, 1),
                "detections on this frame": len(result.boxes),
            }
        )
    plt.tight_layout()
    plt.show()
    display(pd.DataFrame(comparison_rows).set_index("model"))
else:
    print("Comparison skipped.")
"""
    ),
    md(
        r"""
### Draw the counting lines with Roboflow PolygonZone

PolygonZone picks coordinates on an image; it does not label training data.

1. Run the next cell. It downloads `zone_reference.jpg` and shows the same image for reference.
2. Open [Roboflow PolygonZone](https://polygonzone.roboflow.com/).
3. Upload `zone_reference.jpg` and choose the line tool.
4. Draw one or more lines across the roads you want to count, each roughly perpendicular to the traffic flow.
5. Copy the NumPy-format coordinates from the panel beside the image and paste them over the example in the `LINES` cell, keeping the `LINES =` part.

Each line has a direction. If a line's `in` and `out` come out reversed, swap its two endpoints.
"""
    ),
    code(
        r"""
print(f"Draw on this exact {ZONE_WIDTH}×{ZONE_HEIGHT} image.")
files.download(str(ZONE_IMAGE))
display(NotebookImage(filename=str(ZONE_IMAGE), width=960))
"""
    ),
    code(
        r"""
# Replace the example with the NumPy-format lines copied from PolygonZone.
LINES = [
    np.array([[610, 235], [610, 420]]),
]

preview = zone_frame.copy()
for line_number, line in enumerate(LINES, start=1):
    (x1, y1), (x2, y2) = np.asarray(line, dtype=int).tolist()
    for x, y in ((x1, y1), (x2, y2)):
        assert 0 <= x < ZONE_WIDTH and 0 <= y < ZONE_HEIGHT, (
            f"Line {line_number} point ({x}, {y}) is outside the {ZONE_WIDTH}×{ZONE_HEIGHT} image."
        )
    cv2.arrowedLine(preview, (x1, y1), (x2, y2), (0, 0, 255), 4, tipLength=0.06)
    cv2.putText(
        preview, str(line_number), (x1 + 10, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3,
    )

plt.figure(figsize=(14, 8))
plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
plt.title("Counting lines (each arrow points from start to end)")
plt.axis("off")
plt.show()
"""
    ),
    md(
        r"""
## 7. Detect, track and count

Three things happen to every frame. The detector finds the vehicles. The tracker (ByteTrack) matches each detection to the previous frames, so a moving vehicle keeps the same ID. When a tracked vehicle's bottom centre moves across one of your lines, that ID is counted once for that line, in whichever direction it crossed. The cells below save an annotated video and a CSV with one row per crossing.
"""
    ),
    code(
        r"""
CONFIDENCE = 0.25  # @param {type:"number"}
# MAX_SECONDS = 0 processes the whole video. Keep it short while rehearsing.
MAX_SECONDS = 30  # @param {type:"integer"}
# 1 processes every frame; a larger stride skips frames to speed up long videos.
FRAME_STRIDE = 1  # @param {type:"integer"}
COUNT_CLASSES = "vehicle only"  # @param ["vehicle only", "all model classes"]
VIDEO_IMGSZ = 960  # model input size, the same as training

if COUNT_CLASSES == "vehicle only":
    KEEP_CLASS_IDS = [
        class_id for class_id, name in MODEL_CLASS_NAMES.items() if name == "vehicle"
    ]
    assert KEEP_CLASS_IDS, f"This model has no class named 'vehicle': {MODEL_CLASS_NAMES}"
else:
    KEEP_CLASS_IDS = list(MODEL_CLASS_NAMES)

output_stem = f"{LOCAL_VIDEO.stem}_{VIDEO_MODEL_CHOICE}_tracked_counted".replace(" ", "_")
OUTPUT_AVI = WORK_ROOT / f"{output_stem}_raw.avi"
OUTPUT_MP4 = WORK_ROOT / f"{output_stem}.mp4"
EVENTS_CSV = WORK_ROOT / f"{output_stem}_crossings.csv"

max_raw_frames = SOURCE_FRAMES if MAX_SECONDS == 0 else min(
    SOURCE_FRAMES, round(MAX_SECONDS * SOURCE_FPS)
)
print(f"Will process {max_raw_frames} frames and count class ids {KEEP_CLASS_IDS}.")
"""
    ),
    code(
        r"""
assert torch.cuda.is_available(), "Connect a GPU runtime for the video section."

# The tracker, line counter and trace annotator keep state between frames,
# so this cell always starts them fresh.
tracker = sv.ByteTrack(frame_rate=max(1, round(SOURCE_FPS / FRAME_STRIDE)))
line_zones = [
    sv.LineZone(
        start=sv.Point(int(line[0][0]), int(line[0][1])),
        end=sv.Point(int(line[1][0]), int(line[1][1])),
        triggering_anchors=(sv.Position.BOTTOM_CENTER,),
    )
    for line in LINES
]
box_annotator = sv.BoxAnnotator(thickness=1)
label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1, text_padding=2)
trace_annotator = sv.TraceAnnotator(trace_length=30, thickness=2)
line_annotator = sv.LineZoneAnnotator(thickness=3, text_scale=0.6, text_thickness=2)

cap = cv2.VideoCapture(str(LOCAL_VIDEO))
writer = cv2.VideoWriter(
    str(OUTPUT_AVI),
    cv2.VideoWriter_fourcc(*"MJPG"),
    SOURCE_FPS / FRAME_STRIDE,
    (ZONE_WIDTH, ZONE_HEIGHT),
)
assert cap.isOpened() and writer.isOpened()

events = []
raw_frame_index = 0
processed_frames = 0

while raw_frame_index < max_raw_frames:
    ok, frame = cap.read()
    if not ok:
        break
    if raw_frame_index % FRAME_STRIDE != 0:
        raw_frame_index += 1
        continue

    frame = resize_to_width(frame, PROCESS_WIDTH)
    result = inference_model.predict(
        frame,
        imgsz=VIDEO_IMGSZ,
        conf=CONFIDENCE,
        classes=KEEP_CLASS_IDS,
        device=DEVICE,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)

    for line_number, line_zone in enumerate(line_zones, start=1):
        crossed_in, crossed_out = line_zone.trigger(detections)
        for detection_index in np.flatnonzero(crossed_in | crossed_out):
            class_id = int(detections.class_id[detection_index])
            events.append(
                {
                    "frame": raw_frame_index,
                    "seconds": raw_frame_index / SOURCE_FPS,
                    "line": line_number,
                    "track_id": int(detections.tracker_id[detection_index]),
                    "class_id": class_id,
                    "class_name": MODEL_CLASS_NAMES[class_id],
                    "direction": "in" if crossed_in[detection_index] else "out",
                }
            )

    labels = []
    if detections.tracker_id is not None:
        labels = [
            f"#{int(tracker_id)} {MODEL_CLASS_NAMES[int(class_id)]} {confidence:.2f}"
            for class_id, confidence, tracker_id in zip(
                detections.class_id, detections.confidence, detections.tracker_id
            )
        ]

    annotated = trace_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )
    for line_zone in line_zones:
        annotated = line_annotator.annotate(frame=annotated, line_counter=line_zone)
    writer.write(annotated)

    processed_frames += 1
    raw_frame_index += 1
    if processed_frames % 100 == 0:
        print(f"Processed {processed_frames} frames")

cap.release()
writer.release()

subprocess.run(
    [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(OUTPUT_AVI), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(OUTPUT_MP4),
    ],
    check=True,
)

event_columns = ["frame", "seconds", "line", "track_id", "class_id", "class_name", "direction"]
events_df = pd.DataFrame(events, columns=event_columns)
events_df.to_csv(EVENTS_CSV, index=False)

for path in (OUTPUT_MP4, EVENTS_CSV):
    shutil.copy2(path, OUTPUT_FOLDER / path.name)

print(f"Processed {processed_frames} frames; recorded {len(events_df)} crossings.")
print("Saved outputs to", OUTPUT_FOLDER)
"""
    ),
    code(
        r"""
if events_df.empty:
    print("No crossings were recorded. Check the line position, confidence, and detections.")
else:
    crossing_summary = (
        events_df.groupby(["line", "class_name", "direction"])
        .size()
        .rename("crossings")
        .to_frame()
    )
    display(crossing_summary)
    display(events_df.head(10))

plt.figure(figsize=(14, 8))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title("Final annotated frame (each line shows its cumulative in/out counts)")
plt.axis("off")
plt.show()

print(f"The annotated video is {OUTPUT_MP4.stat().st_size / 1024**2:.0f} MB.")
print("Play the copy in your Drive output folder, or use the download cell below.")
"""
    ),
    md(
        r"""
### Download the batch result

The files are already in your Drive output folder. Set the option below to download local copies.
"""
    ),
    code(
        r"""
DOWNLOAD_RESULTS = False  # @param {type:"boolean"}

if DOWNLOAD_RESULTS:
    files.download(str(OUTPUT_MP4))
    files.download(str(EVENTS_CSV))
else:
    print("Set DOWNLOAD_RESULTS to True to download the annotated MP4 and CSV.")
"""
    ),
    md(
        r"""
### Audit the count

Watch the annotated clip and manually count one direction. Then ask:

- Which vehicles were missed?
- Which false detections appeared?
- Did any track ID switch from one vehicle to another?
- Did a box jitter across the line more than once?
- What changes when confidence is 0.15 or 0.45?
"""
    ),
    md(
        r"""
## References

[Ultralytics training](https://docs.ultralytics.com/modes/train/) · [Ultralytics tracking](https://docs.ultralytics.com/modes/track/) · [Supervision line zones](https://supervision.roboflow.com/latest/detection/tools/line_zone/)
"""
    ),
]


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "accelerator": "GPU",
    "colab": {
        "name": "drone_observation_yolo_colab.ipynb",
        "provenance": [],
        "toc_visible": True,
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

nbf.validate(notebook)
nbf.write(notebook, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
