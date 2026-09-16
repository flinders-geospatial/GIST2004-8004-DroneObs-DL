"""Build the second student notebook: count vehicles in the class footage.

This is the short follow-up to build_colab_notebook.py. There is no training:
students load the model they saved last time (or a supplied one), fetch one
of the class clips, draw counting lines with PolygonZone, and run the same
detect, track and count loop. Edit this file and regenerate rather than
touching the notebook JSON.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUT = Path(__file__).with_name("drone_observation_field_footage_colab.ipynb")


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
# Drone observation: count vehicles in the class footage

In the first practical you fine-tuned a YOLO detector on drone imagery, tracked vehicles through a supplied clip, and counted the ones that crossed a line. Since then the class has flown two DJI Minis over two intersections near the campus. This notebook runs the same detect, track and count workflow on that footage, using the model you trained.

There is no training this time. You will:

1. load your saved model from Google Drive, or a supplied one;
2. fetch one of the class clips and watch it;
3. draw counting lines on a frame of it with PolygonZone; and
4. detect, track and count, then check the result against your own count.

The footage is low-altitude drone video: the drone hovers and looks down or across at an intersection. Your model was trained on VisDrone imagery from other cities and has not seen this footage.
"""
    ),
    md(
        r"""
## Before running anything

1. Open this notebook in Google Colab and choose **File → Save a copy in Drive**.
2. Choose **Runtime → Change runtime type → GPU**. A T4 is sufficient when one is available.
3. Run the cells in order from the top: click a cell and press **Shift+Enter**, or click the play button at its left. A cell has finished when the spinner at its left becomes a number.

Some cells show a settings panel beside the code, with dropdowns and text fields. These are Colab form fields: they set values in that cell without you editing the code. After changing one, run the cell again.

Colab runtimes reset without warning. The notebook saves the video outputs to your Drive.
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
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import supervision as sv
import torch
import ultralytics
from IPython.display import Image as NotebookImage
from IPython.display import Video, display
from ultralytics import YOLO

from google.colab import drive, files

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("Ultralytics:", ultralytics.__version__)
print("Supervision:", sv.__version__)
print("PyTorch:", torch.__version__)
print("Compute:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

if not torch.cuda.is_available():
    print("\nThe video section needs a GPU runtime.")
    print("In Colab: Runtime → Change runtime type → GPU, then rerun this cell.")
"""
    ),
    md(
        r"""
### Core terms

| Term | What it means here |
|---|---|
| Weights | The numbers a model learned during training, saved as a `.pt` file; loading them recreates the trained model without retraining |
| Detection | A box, class, and confidence for one object in one frame |
| Tracking | Giving the same moving object a persistent ID across frames |
| Line crossing | Counting a tracked ID when it moves across a chosen line |
| Domain shift | The gap between the images a model was trained on and the images it is used on |
"""
    ),
    md(
        r"""
## 2. Load your model

Last time your training run saved `best.pt` to `MyDrive/GIST2004-8004/my-droneobs-results/droneobs_yolo26n` in your Google Drive. That file is the result of the training, about 5 MB of weights. Loading it gives back the trained detector.

The cell below connects your Drive, looks for that file, and downloads the two supplied models from the course link.

| Choice | What it is |
|---|---|
| `my model` | The YOLO26n you fine-tuned last time, from your Drive |
| `backup model` | A small model like yours, trained on the same dataset in advance |
| `bigger model` | A larger model trained in advance on far more drone footage |

If your run did not finish last time, or you trained under a different Google account, choose `backup model` or `bigger model`.
"""
    ),
    code(
        r"""
drive.mount("/content/drive")

COURSE_DATA_URL = "https://gist2004-droneobs-2026.s3.ap-southeast-2.amazonaws.com"  # @param {type:"string"}

OUTPUT_FOLDER = Path("/content/drive/MyDrive/GIST2004-8004/my-droneobs-results")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
MY_MODEL = OUTPUT_FOLDER / "droneobs_yolo26n" / "best.pt"

ASSET_FOLDER = Path("/content/droneobs_course_assets")
ASSET_FOLDER.mkdir(parents=True, exist_ok=True)
WORK_ROOT = Path("/content/droneobs_work")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
BACKUP_MODEL = ASSET_FOLDER / "reference_model.pt"
BIGGER_MODEL = ASSET_FOLDER / "strong_vehicle_model.pt"

assert COURSE_DATA_URL.startswith("http"), "Paste the course link first, then rerun this cell."
for destination in (BACKUP_MODEL, BIGGER_MODEL):
    if not destination.exists():
        print("Downloading", destination.name, "...")
        # Download under a temporary name so an interrupted download is retried.
        partial = destination.with_name(destination.name + ".part")
        urlretrieve(f"{COURSE_DATA_URL.rstrip('/')}/{destination.name}", partial)
        partial.rename(destination)

MODEL_PATHS = {
    "my model": MY_MODEL,
    "backup model": BACKUP_MODEL,
    "bigger model": BIGGER_MODEL,
}
for name, path in MODEL_PATHS.items():
    status = f"{path.stat().st_size / 1024**2:5.1f} MB" if path.exists() else "not found"
    print(f"{name:13s} {status:9s} {path}")
print("Outputs will be saved to", OUTPUT_FOLDER)
"""
    ),
    code(
        r"""
MODEL_CHOICE = "my model"  # @param ["my model", "backup model", "bigger model"]

weights_path = MODEL_PATHS[MODEL_CHOICE]
assert weights_path.exists(), (
    f"'{MODEL_CHOICE}' was not found at {weights_path}. "
    "Choose 'backup model' or 'bigger model' and rerun this cell."
)
inference_model = YOLO(str(weights_path))
MODEL_CLASS_NAMES = {int(k): v for k, v in inference_model.names.items()}
print("Using", MODEL_CHOICE, "from", weights_path)
print("Classes:", MODEL_CLASS_NAMES)
"""
    ),
    md(
        r"""
The printed class list is the label set the model was trained with. This notebook counts `vehicle`. [VisDrone](https://docs.ultralytics.com/datasets/detect/visdrone/) was captured in Chinese cities at a range of heights and angles, so the location, flight height, camera angle, shadows, road markings and video compression all differ from this footage. That gap is domain shift, and it is why a model that scored well on its own test set can do worse here.
"""
    ),
    md(
        r"""
## 3. Fetch a class clip

The class footage is three short clips, each at the course link followed by the file name. Paste one over the value below and run the cell. The field opens with last practical's teaching clip, which is a quick way to check the notebook runs.

| File name | What it shows |
|---|---|
| `intersection_a_mini4.mp4` | Mini 4, 3 minutes, steep view of a signalised intersection with roadworks on one corner |
| `intersection_b_low_mini5.mp4` | Mini 5, 1 minute 45 seconds, low view across a large intersection beside a noise wall |
| `intersection_b_high_mini5.mp4` | Mini 5, 3 minutes, the same intersection from higher up, with smaller vehicles |

Watch the clip once and count one direction by hand. You will compare the model's count with yours at the end.
"""
    ),
    code(
        r"""
VIDEO_URL = "https://gist2004-droneobs-2026.s3.ap-southeast-2.amazonaws.com/DJI_0022_teaching_clip.mp4"  # @param {type:"string"}

assert VIDEO_URL.startswith("http"), "Paste a class clip link first, then rerun this cell."
LOCAL_VIDEO = WORK_ROOT / (Path(urlparse(VIDEO_URL).path).name or "class_clip.mp4")
if not LOCAL_VIDEO.exists():
    print("Downloading", LOCAL_VIDEO.name, "...")
    partial = LOCAL_VIDEO.with_name(LOCAL_VIDEO.name + ".part")
    urlretrieve(VIDEO_URL, partial)
    partial.rename(LOCAL_VIDEO)

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

# Streams from the web link; the downloaded copy is used for processing.
display(Video(VIDEO_URL, width=960))
"""
    ),
    md(
        r"""
Note the frame size and frame rate. The drone records at a higher resolution than this; the clips were reduced to 1280 pixels wide for class use, and the model then works at 960 pixels on the long side, the same size it was trained at. Vehicles on the far side of the intersection cover far fewer pixels than those on the near side, and the detector misses small ones more often. The [predict settings](https://docs.ultralytics.com/modes/predict/) page describes `imgsz` and `conf`.
"""
    ),
    md(
        r"""
## 4. Draw the counting lines

The next cell saves the first frame of the clip, resized to the width the model will process, and downloads it. Then, as last time:

1. Open [Roboflow PolygonZone](https://polygonzone.roboflow.com/).
2. Upload `zone_reference.jpg` and choose the line tool.
3. Draw one or more lines across the roads you want to count, each roughly perpendicular to the traffic flow.
4. Copy the NumPy-format coordinates from the panel beside the image and paste them over the example in the `LINES` cell, keeping the `LINES =` part.

Each line has a direction. If a line's `in` and `out` come out reversed, swap its two endpoints.

Placing a line:

- Put it across one approach to the intersection, clear of the middle where turning vehicles cross each other's paths.
- Make it long enough to cover every lane and no longer, so vehicles parked or queued beside the road are not counted.
- Prefer the near side of the frame, where vehicles are larger and tracks are steadier.
- A vehicle is counted when the bottom centre of its box crosses the line, because in an oblique view that point is close to where the vehicle meets the road.

A fixed line only works while the camera holds still. The drone hovered in place for each clip and the view holds steady, apart from a couple of bumps of turbulence, one of them caused by a bird. The `LINES` cell draws your lines on the first frame and on a frame near the end of the clip, so you can see whether the road moved under them. Studies that track vehicles for minutes, such as [pNEUMA](https://open-traffic.epfl.ch/) and [highD](https://levelxdata.com/highd-dataset/), georeference the video first.
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

def read_frame(frame_index):
    cap = cv2.VideoCapture(str(LOCAL_VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    assert ok, f"Could not read frame {frame_index} of {LOCAL_VIDEO.name}"
    return resize_to_width(frame, PROCESS_WIDTH)

zone_frame = read_frame(0)
# The reported frame count can be a few frames high, so stop a second short of the end.
late_frame = read_frame(max(0, SOURCE_FRAMES - round(SOURCE_FPS)))
ZONE_HEIGHT, ZONE_WIDTH = zone_frame.shape[:2]
ZONE_IMAGE = WORK_ROOT / "zone_reference.jpg"
cv2.imwrite(str(ZONE_IMAGE), zone_frame)

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

def draw_lines(frame):
    preview = frame.copy()
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
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 1, figsize=(14, 16))
for axis, (title, frame) in zip(axes, (("First frame", zone_frame), ("Near the last frame", late_frame))):
    axis.imshow(draw_lines(frame))
    axis.set_title(f"{title}: each arrow points from start to end")
    axis.axis("off")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
## 5. Detect, track and count

For each frame, the detector finds the vehicles, the tracker matches them to the previous frames so a moving vehicle keeps its ID, and the line counter records an ID when its bottom centre crosses one of your lines. The cells below save an annotated video and a CSV with one row per crossing.

The tracker is [ByteTrack](https://arxiv.org/abs/2110.06864). Most trackers throw away low-confidence detections; ByteTrack keeps them as a second pool for matching, so a vehicle whose confidence dips for a few frames, behind a tree or a pole, usually keeps its ID instead of starting a new one. The [Supervision trackers](https://supervision.roboflow.com/latest/trackers/) page shows its settings.

Leave `MAX_SECONDS` at 30 for a first pass to check the lines. Then set it to 0 and rerun the two cells below to process the whole clip.
"""
    ),
    code(
        r"""
CONFIDENCE = 0.5  # @param {type:"number"}
# MAX_SECONDS = 0 processes the whole video.
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

output_stem = f"{LOCAL_VIDEO.stem}_{MODEL_CHOICE}_tracked_counted".replace(" ", "_")
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
# Hot pink boxes and labels, cyan trails: both stand out on grey road and green verges.
BOX_COLOUR = sv.Color.from_hex("#FF1493")
TRAIL_COLOUR = sv.Color.from_hex("#00FFFF")
box_annotator = sv.BoxAnnotator(color=BOX_COLOUR, thickness=2)
label_annotator = sv.LabelAnnotator(
    color=BOX_COLOUR, text_color=sv.Color.WHITE, text_scale=0.4, text_thickness=1, text_padding=2
)
trace_annotator = sv.TraceAnnotator(color=TRAIL_COLOUR, trace_length=30, thickness=2)
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
Each CSV row is one crossing: frame, time, line, track ID and direction. The same ID crossing one line twice within a second or two is a box jittering across the line, and it counts twice. Two IDs for one vehicle means the tracker lost it and started again; if that happened on the line, the vehicle was counted twice or not at all.
"""
    ),
    md(
        r"""
### Download the result

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

Compare the model's count for one line and direction with your hand count. Matching totals can hide errors that cancel, so look at the video and the CSV as well:

- Which false detections appeared? In our runs the model boxed roadworks machinery, parked vehicles and a wheelie bin.
- Does `bigger model` give a different count? Choose it in section 2 and rerun from there.
- What changes when `CONFIDENCE` is 0.25 or 0.75? A lower threshold finds more vehicles and more false detections; the [performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/) guide explains how precision and recall trade off against each other.
"""
    ),
    md(
        r"""
## 6. Where to from here

- Fine-tune on this footage. Label a few dozen frames in [Roboflow Annotate](https://roboflow.com/annotate) and train as in the first practical. Test on frames from the other intersection, otherwise near-identical adjacent frames end up in both train and test.
- Swap the parts. The detector, tracker and line counter are separate. [Supervision](https://supervision.roboflow.com/latest/) has other trackers, anchor points and polygon zones that use the same detections.
"""
    ),
    md(
        r"""
## References

[Ultralytics predict](https://docs.ultralytics.com/modes/predict/) · [Ultralytics tracking](https://docs.ultralytics.com/modes/track/) · [Ultralytics performance metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/) · [Supervision line zones](https://supervision.roboflow.com/latest/detection/tools/line_zone/) · [ByteTrack paper](https://arxiv.org/abs/2110.06864) · [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) · [pNEUMA drone trajectories](https://open-traffic.epfl.ch/) · [highD drone trajectories](https://levelxdata.com/highd-dataset/) · [Roboflow PolygonZone](https://polygonzone.roboflow.com/)
"""
    ),
    md(
        r"""
## Photogrammetry scratch

Pix4D and the other photogrammetry packages start from the metadata inside each photo. A drone JPEG carries two blocks in its header. EXIF is written by the camera: lens, exposure, image size and the GPS position in degrees, minutes and seconds. XMP is written by the flight controller under a `drone-dji` namespace: decimal position, absolute and relative altitude, gimbal and aircraft angles, and on an RTK aircraft the fix quality and the calibrated camera model.

This section stands on its own: it needs no GPU and none of the cells above. Run the cell and choose one or two photos from the photogrammetry set when the file chooser appears. The XMP block is plain text, so the same fields turn up if you open the photo in a text editor and search for `drone-dji`.
"""
    ),
    code(
        r"""
import re
from pathlib import Path

from PIL import ExifTags
from PIL import Image as PilImage

from google.colab import files

PHOTO_FOLDER = Path("/content/photo_scratch")
PHOTO_FOLDER.mkdir(parents=True, exist_ok=True)

def read_exif(path):
    exif = PilImage.open(path).getexif()
    main = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    main.update({ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(ExifTags.IFD.Exif).items()})
    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif.get_ifd(ExifTags.IFD.GPSInfo).items()}
    return main, gps

def read_xmp(path):
    raw = Path(path).read_bytes()
    start, end = raw.find(b"<x:xmpmeta"), raw.find(b"</x:xmpmeta>")
    if start < 0 or end < 0:
        return {}
    xmp = raw[start:end].decode("utf-8", "replace")
    return dict(re.findall(r'drone-dji:(\w+)="([^"]*)"', xmp))

EXIF_KEYS = [
    "Make", "Model", "DateTimeOriginal", "ExifImageWidth", "ExifImageHeight",
    "FocalLength", "FocalLengthIn35mmFilm", "FNumber", "ExposureTime", "ISOSpeedRatings",
    "GPSLatitudeRef", "GPSLatitude", "GPSLongitudeRef", "GPSLongitude", "GPSAltitudeRef", "GPSAltitude",
]

uploaded = files.upload()
if not uploaded:
    print("No photos chosen. Run the cell again and pick one or two JPEGs.")
for name, data in uploaded.items():
    photo_path = PHOTO_FOLDER / Path(name).name
    photo_path.write_bytes(data)
    main, gps = read_exif(photo_path)
    print(f"\n=== {photo_path.name} ===")
    print("EXIF, written by the camera:")
    for key in EXIF_KEYS:
        value = main.get(key, gps.get(key))
        if value is not None:
            print(f"  {key:24s} {value}")
    print("XMP, written by the flight controller:")
    for key, value in read_xmp(photo_path).items():
        print(f"  {key:24s} {value}")
"""
    ),
]


notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "accelerator": "GPU",
    "colab": {
        "name": "drone_observation_field_footage_colab.ipynb",
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
