# UAV-HINT: A Dataset and Baseline for Hierarchical Human Intention Prediction from Aerial Perspectives

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
git clone https://github.com/Wangzz12138/UAV-HINT.git
cd UAV-HINT
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

## Training

Run the following commands from the repository root after preparing `data/manifest.jsonl` and the frame files described above.

### Prepare pretrained weights

Download the ActionCLIP ViT-B/16 (8-frame, Transf) checkpoint from the [official ActionCLIP model zoo](https://github.com/sallymmx/ActionCLIP/blob/master/MODEL_ZOO.md) and save it as `pretrained/actionclip_vit_b16_8f.pt`. The checkpoint must contain both `model_state_dict` and `fusion_model_state_dict`.

Prepare T5-base and verify the pretrained files:

```bash
python tools/prepare_pretrained.py --t5-dir pretrained/t5-base --actionclip-checkpoint pretrained/actionclip_vit_b16_8f.pt
```

### Start training

```bash
python train.py --config configs/hintnet_336.json --manifest data/manifest.jsonl --t5-model pretrained/t5-base --local-files-only --output outputs/hintnet_train --device cuda
```

Training options are defined in [configs/hintnet_336.json](configs/hintnet_336.json). Logs and checkpoints are saved under `outputs/hintnet_train/`.
