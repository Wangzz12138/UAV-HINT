# UAV-HINT

## Dataset

**Coming soon.**

- [Dataset Release Agreement (PDF)](docs/UAV-HINT_Dataset_Release_Agreement.pdf)
- [Data Concerns and Removal Request Form (PDF)](docs/UAV-HINT_Data_Concerns_and_Removal_Requests.pdf)

## Data organization

```text
data/
|-- manifest.jsonl
`-- frames/
    |-- video001/
    |   |-- 000.jpg
    |   |-- 001.jpg
    |   `-- ...
    `-- video002/
        `-- ...
```

Each line of `manifest.jsonl` is a JSON record for one annotated person track. See [configs/manifest.example.jsonl](configs/manifest.example.jsonl) for complete examples.

| Field | Format |
| --- | --- |
| `video_id` | Video identifier. |
| `instance_id` | Person-track identifier within the video. |
| `split` | `train`, `val`, or `test`. |
| `frames` | Ordered frame-file paths, relative to the directory containing the manifest. |
| `bboxes` | Bounding boxes in the same order as `frames`, using original-image `[x1, y1, x2, y2]` coordinates. |
| `text` | Annotation fields: `scene`, `time`, `weather`, `age`, `gender`, and `state`. |
| `labels` | Labels for `action`, `short_term`, and `long_term`. |

## Environment setup

Use Python 3.10.

```bash
git clone https://github.com/Wangzz12138/UAV-HINT-A-Dataset-and-Baseline-for-Hierarchical-Human-Intention-Prediction-from-Aerial-Perspectives.git HINT-Net
cd HINT-Net
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
