# Instructor setup for the drone-observation Colab

The student notebook is `drone_observation_yolo_colab.ipynb`. It assumes no
deep-learning background and is designed for live narration.

## Course assets

Host these four files at one public web location (the class S3 bucket) so that
`<base URL>/<file name>` resolves for each. The notebook's link field is
pre-filled with the current base URL; when the bucket changes, update
`COURSE_DATA_URL` in `build_colab_notebook.py` and regenerate. Make the bucket
or prefix public for the lab day and delete it afterwards.

| File | Approximate size | Purpose |
|---|---:|---|
| `droneobs-demo.zip` | 523 MB | 1,500 train, 300 validation, and 300 test images |
| `reference_model.pt` | 5.3 MB | fallback model, so later sections work without training |
| `strong_vehicle_model.pt` | 54.4 MB | multi-source YOLO11s comparison and batch-video model |
| `DJI_0022_teaching_clip.mp4` | 8 MB | 30-second, 1280×720 stable-camera practical clip |

Student outputs are saved to `MyDrive/GIST2004-8004/my-droneobs-results` in
each student's own Drive.

The archive contains 2,100 VisDrone images and 73,869 labelled objects. The
class distribution is strongly imbalanced:

| Class | Boxes |
|---|---:|
| bicycle | 3,059 |
| motorcycle | 9,424 |
| vehicle | 61,386 |

There are 38 valid empty-label images. They are background examples, not
missing annotations.

The supplied reference is YOLO11n trained for 50 epochs at 1024 px. Its saved
validation metrics are mAP50 0.449 and mAP50–95 0.245. It is the fallback when
a student cannot finish training. New runs start from `yolo26n.pt`.

The stronger comparison is YOLO11s trained at 960 px on the old multi-source
pipeline. Its saved held-out metrics are mAP50 0.852 and mAP50–95 0.535. Its
test set and two-class `vehicle`/`person` scheme differ from the pilot, so the
metric comparison is illustrative rather than a controlled benchmark. The
notebook uses only its `vehicle` output. The 327 MB YOLO11x accuracy-reference
checkpoint remains an instructor artefact rather than a required Colab asset.

## Distribution

Keep the notebook in the standalone public
`flinders-geospatial/GIST2004-8004-DroneObs-DL` GitHub repository and link to
its GitHub URL with a standard **Open in Colab** badge. The four larger assets
live in S3, not the repository. The notebook builds each file's URL from the
single public base URL. Do not put AWS credentials in the notebook.

## Recommended class sequence

1. Run setup and the fetch cell, watch the embedded teaching clip, then run
   extraction, the dataset audit, and the label visualisation.
2. Show the stock pretrained model's detections on the video frame, then
   explain transfer learning and the training settings.
3. Set `TRAIN_MODEL = True` and train on the available GPU.
4. Continue with the reference model if a student's runtime is interrupted.
5. Compare the pilot and stronger model on the same DJI frame.
6. Draw a line in PolygonZone and run the tracking/counting cells.
7. Download the annotated video and compare the CSV with a manual count.

For footage the students capture next week, upload it to object storage and
paste its URL into `NEW_VIDEO_URL`, or put it in Drive and select `file in
Drive`. Re-run the video cell before choosing the counting line.
`MAX_SECONDS = 0` processes the complete video; retain a short limit while
rehearsing the workflow.

The notebook trains from `/content`, then copies the model and plots to
Drive. Avoid training directly against thousands of files in mounted Drive.
Colab's free GPU type, limits, and availability are dynamic, so do not promise
a particular completion time.

## PolygonZone

PolygonZone is only a coordinate picker for the counting line; it does not
label training data. If a later iteration adds labelling of new frames, split
video-derived data by whole clip or time block: randomly placing nearly
identical adjacent frames into both train and test produces misleadingly good
validation results.

## Rehearsal checklist

- Test that the public base URL resolves for all four file names.
- Confirm the teaching clip embeds and plays after the fetch cell.
- Confirm the dataset audit reports 1,500/300/300 images.
- Visually inspect the six random label examples.
- Confirm the stock pretrained model cell renders detections on the video frame.
- For setup testing, set `epochs` to 1 in the training-settings cell.
- Confirm `best.pt`, `results.csv`, and plots are copied to Drive.
- Run still-image inference using both reference and newly trained weights.
- Confirm the pilot/stronger-model comparison renders on the same DJI frame.
- Download `zone_reference.jpg`, confirm PolygonZone reports coordinates for a
  1280×720 image, and preview the pasted line.
- Confirm the annotated MP4 plays inline and the model-named crossing CSV is
  copied to Drive and can be downloaded.
- Reverse the line endpoints if the displayed in/out convention is unhelpful.

Line counts assume a stationary or stabilised camera. A drone pan or translation
makes a road-fixed interpretation invalid; that limitation is intentionally
called out in the student notebook.

Before distributing the pilot archive, retain its VisDrone provenance and
confirm that the intended sharing arrangement is consistent with the source
dataset's terms.
