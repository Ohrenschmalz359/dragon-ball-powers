# MediaPipe models

Dragon Ball Powers tracks your hands and body with Google's
[MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide) models.
They run locally on your PC; no camera image ever leaves the machine.

| File | Size | Used for |
|------|------|----------|
| `hand_landmarker.task` | 7.5 MB | Hand Landmarker (float16, v1): palm detector + 21 landmarks per hand, up to two hands. Every gesture depends on it. |
| `pose_landmarker_lite.task` | 5.5 MB | Pose Landmarker *lite*: 33 body landmarks + person segmentation mask (aura). The default (`tracking.pose_model = "lite"`). |
| `pose_landmarker_full.task` | 9.0 MB | Pose Landmarker *full*: more accurate, slower. |
| `pose_landmarker_heavy.task` | 29.2 MB | Pose Landmarker *heavy*: most accurate, needs a fast CPU. |

The pose model is selected in the settings panel or in `config/settings.json`
(`tracking.pose_model`: `lite`, `full` or `heavy`).

## Where they come from

All files are downloaded from Google's official model storage, for example
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`.
[`models.json`](models.json) lists the exact URL, size in bytes and SHA256 checksum
of every file. A file is used only if its size and checksum match, so a damaged or
tampered download is detected and replaced.

Model documentation and model cards:

* Hand Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker>
* Pose Landmarker: <https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker>

## License

The models are published by Google as part of MediaPipe under the
**Apache License 2.0** (<https://www.apache.org/licenses/LICENSE-2.0>). See the model
cards linked from the pages above for intended use and limitations. The `.task`
files are not stored in the git repository (`.gitignore`); they are downloaded
during setup.

## Downloading them again

`setup.bat` downloads and verifies the models automatically (step 5/6). To check or
repair them by hand, open a terminal in the project folder (cmd or PowerShell):

| Command | Effect |
|---------|--------|
| `.venv\Scripts\python.exe scripts\download_models.py` | download missing or damaged files |
| `.venv\Scripts\python.exe scripts\download_models.py --check` | verify only (size + SHA256), never download |
| `.venv\Scripts\python.exe scripts\download_models.py --force` | download everything again |
| `.venv\Scripts\python.exe scripts\download_models.py --only pose_landmarker_heavy` | handle a single model |

The script needs only the Python standard library, so any Python 3.8+ can run it.
Downloads go to a temporary `.part` file and are moved into place only after the
size and SHA256 check passed. Behind a proxy, set it first in the same window:
`set HTTPS_PROXY=http://proxy.example.com:8080` (cmd) or
`$env:HTTPS_PROXY = "http://proxy.example.com:8080"` (PowerShell).

If you keep the models in another folder, set the environment variable
`DBP_MODELS_DIR` to it: the app and `download_models.py` both use it.
