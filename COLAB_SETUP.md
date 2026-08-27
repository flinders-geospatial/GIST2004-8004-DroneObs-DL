# Instructor setup for the drone observation Colab

The student notebook is `drone_observation_yolo_colab.ipynb`. It is generated
by `build_colab_notebook.py`. Edit the builder and regenerate; do not edit the
notebook JSON.

## Course files

Host these four files somewhere public so `<base URL>/<file name>` works
for each one. The notebook link field is pre-filled with the current bucket
URL. When the bucket changes, update
`COURSE_DATA_URL` in the builder and regenerate. The bucket is public for the
lab day only; delete it afterwards.

| File | Approximate size | Purpose |
|---|---:|---|
| `droneobs-demo.zip` | 523 MB | 1,500 train, 300 validation, and 300 test images |
| `reference_model.pt` | 5.3 MB | backup model if a student's training run fails |
| `strong_vehicle_model.pt` | 54.4 MB | larger comparison and video model |
| `DJI_0022_teaching_clip.mp4` | 8 MB | 30-second, 1280×720 stable-camera clip |

Student outputs save to `MyDrive/GIST2004-8004/my-droneobs-results` in each
student's Drive.

## Dataset

The archive contains 2,100 VisDrone images and 73,869 labelled boxes:

| Class | Boxes |
|---|---:|
| bicycle | 3,059 |
| motorcycle | 9,424 |
| vehicle | 61,386 |

38 images have valid empty label files. These are background examples.

Before sharing the archive more widely, check it against the VisDrone terms
and keep the provenance note.

## Models

The notebook uses plain names: the student's run is `my model`,
`reference_model.pt` is `backup model`, and `strong_vehicle_model.pt` is
`bigger model`. Every dropdown defaults to `my model`.

The backup is YOLO11n, 50 epochs at 1024 px on this dataset: mAP50 0.449,
mAP50-95 0.245 on the pilot test set. Student runs start from `yolo26n.pt`.

The bigger model is YOLO11s at 960 px from the old multi-source pipeline:
mAP50 0.852 on its own test set, which used a two-class vehicle/person
scheme. The two mAP figures come from different test sets and label schemes,
so they are not comparable. The notebook uses only its `vehicle` class. The
327 MB YOLO11x checkpoint stays local.

## Training settings

`epochs` 20, `imgsz` 960, `batch` 12. Batch 8 peaked at about 7 GB on the
Colab T4. A local run of the notebook at batch 16 peaked at 13.7 GB reserved,
too close to the T4's 15 GB for class use, so the notebook uses 12. If a run
still hits an out-of-memory error, drop back to 8.

Training runs on `/content` and copies `best.pt` and the plots to Drive when
it finishes. Do not train against files mounted from Drive. Colab's free GPU
type and availability vary, so do not promise a completion time.

## Distribution

The notebook lives in the public
`flinders-geospatial/GIST2004-8004-DroneObs-DL` repository with an
**Open in Colab** badge in the README. The four large files stay in S3, out
of the repository. There are no AWS credentials in the notebook.

## Class sequence

1. Run setup and the fetch cell, watch the embedded clip, then run
   extraction, the dataset audit, and the label examples.
2. Run the stock pretrained model on the default test photo; students can
   paste any image address and rerun. Then cover transfer learning and the
   training settings.
3. Tick `TRAIN_MODEL` and train.
4. If a runtime is interrupted, the note at the top of section 5 covers the
   switch to the backup model.
5. Compare the student's model and the bigger model, stacked on the same
   frame with a parameter table.
6. Draw one or more lines in PolygonZone, paste its NumPy output over
   `LINES`, and run the tracking and counting cells. The CSV has a `line`
   column.
7. Download the annotated video and compare the CSV with a manual count.

Line counts assume a stationary or stabilised camera. If the camera pans or
drifts, the line no longer sits over the same stretch of road; the notebook
warns students about this.

## Next week's footage

Upload the 3 or 4 class clips to public object storage, same pattern as the
course bucket, and give out the links. Students choose `video web link` and
paste one into `NEW_VIDEO_URL`, then re-run the video cell before choosing
lines. `MAX_SECONDS = 0` processes the whole video; keep a short limit while
rehearsing.

## PolygonZone

PolygonZone picks the counting-line coordinates. It does not label training
data. If students label their own frames in a future version, split the data
by whole clip or time block: near-identical adjacent frames in both train
and test inflate the validation results.

## Rehearsal checklist

- The public base URL resolves for all four file names.
- The teaching clip streams and plays after the fetch cell. It streams from
  the bucket URL, so this player stops working once the bucket is deleted;
  processing uses the downloaded copy and is unaffected.
- The dataset audit reports 1,500/300/300 images.
- The six random label examples look right.
- The stock model cell renders detections on the default photo; paste a
  different image address to check the field.
- Set `epochs` to 1 for setup testing and watch the first epoch's VRAM at
  batch 12.
- `best.pt`, `results.csv`, and the plots copy to Drive.
- Still-image inference runs with both `my model` and `backup model`.
- The stacked comparison and its parameter table render.
- `zone_reference.jpg` downloads, PolygonZone's NumPy output pastes over
  `LINES`, and the numbered lines preview.
- The final annotated frame renders with cumulative counts, and the
  annotated MP4 and crossing CSV, with its `line` column, copy to Drive.
  Videos are never embedded in cell outputs; large embeds make the page
  unresponsive.
- Swap a line's endpoints if in and out come out the wrong way round.

## Version note

`supervision` 0.27 `LineZone.trigger` calls `np.cross` on 2-D vectors, which
NumPy 2 removed. Colab ships NumPy 1.x today, so it works. If Colab moves to
NumPy 2, bump `supervision` and re-test the counting loop.
