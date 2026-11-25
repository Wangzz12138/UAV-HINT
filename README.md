# HINT-Net — 336 × 336

[中文说明](README_zh-CN.md)

HINT-Net predicts three labels for an observed person in an aerial video: **action (7 classes), short-term intention (8 classes), and long-term intention (4 classes)**. This source-code-only repository packages the **336 × 336 experimental variant** and its training and evaluation code. **No trained checkpoint or pretrained model weights are distributed here.** The paper's original image-size setting was 224 × 224; 336 is the evaluated higher-resolution variant.

The model uses a frozen ActionCLIP ViT-B/16 encoder for global frames, person crops, and contextual text, followed by hierarchical queries, a frozen T5-base decoder with progressive token unveiling, and three classification heads. This release contains the HINT-Net main model only.

**We also provide the Label Studio XML configuration used for annotation:** [annotation/label_studio_config.xml](annotation/label_studio_config.xml).

## What is included

```text
.
├── annotation/label_studio_config.xml # Label Studio annotation interface
├── checkpoints/README.md            # Instructions for locally generated checkpoints
├── configs/
│   ├── hintnet_336.json              # Raw-frame training, 336 × 336
│   ├── hintnet_336_cached.json       # Training on original/flip feature caches
│   ├── labels.json                  # Exact output-class order
│   ├── compatibility_81.json        # Experimental compatibility definition
│   ├── splits_standard.json         # Generated standard video split
│   └── manifest.example.jsonl
├── hintnet/                         # Model, data loader, loss, and metrics
├── tools/                           # Preparation, caching, export, and verification
├── tests/
├── train.py
├── evaluate.py                     # Evaluation or label-free prediction
└── THIRD_PARTY_NOTICES.md
```

The video dataset, feature caches, virtual environment, trained HINT-Net checkpoints, and pretrained encoders are not included. Train HINT-Net using the instructions below, or separately obtain a compatible trained checkpoint before evaluation. The optional export format stores **20 trainable tensors / 2,087,731 parameters**; the repository includes its loader and export tool, but no binary weights. Frozen ActionCLIP and T5 weights must also be supplied separately.

## Annotation configuration

The supplied [Label Studio XML configuration](annotation/label_studio_config.xml) defines video person-track bounding boxes, **7 action / 8 short-term intention / 4 long-term intention** choices, environmental scene/time/weather fields, and per-person age/gender/state attributes. Environment fields follow acquisition records and author-provided labels. This XML defines the annotation interface; the per-video annotation results must be supplied separately.

Its annotation instructions verify short-term intentions **within 30 seconds after the observation clip ends** and determine long-term intentions from the **full flight recording**. These rules establish human ground-truth labels. Model inference receives only the observed clip and its specified context. Setup instructions and an example task are in [annotation/README.md](annotation/README.md).

## Installation

Run commands from this repository's root. The code was validated with Python 3.10, PyTorch 2.2.2 with CUDA 12.1, and Transformers 4.44.2; the tested dependency versions are recorded in [requirements.lock.txt](requirements.lock.txt).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CPU or CUDA environment. GPU commands below use `--device cuda`; `--device cpu` is supported for functional testing, but full-size inference is substantially slower.

To install the recorded CUDA 12.1 lock file, provide the PyTorch wheel index; its `+cu121` versions are not available from the ordinary PyPI index alone:

```bash
python -m pip install -r requirements.lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

## Pretrained encoders

### ActionCLIP

Obtain the **ViT-B/16, 8-frame, Transf** checkpoint from the [ActionCLIP project](https://github.com/sallymmx/ActionCLIP), and place it at:

```text
pretrained/actionclip_vit_b16_8f.pt
```

The checkpoint must contain both `model_state_dict` and `fusion_model_state_dict`. A plain CLIP checkpoint does not contain the trained temporal fusion module and cannot replace it. The encoder used in the reported reference experiments has SHA256:

```text
0acfe0359eb73c02e93f0b1ee4761e7099747d15fe305aa833776f6681c8aa5b
```

The required inference source and BPE vocabulary are vendored under `hintnet/vendor/actionclip/`, with the upstream license retained. An adjacent ActionCLIP source checkout is not required.

### T5-base

Training and trained-adapter restoration use the genuine frozen `google-t5/t5-base` decoder at revision:

```text
a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
```

Transformers can load it on first use with network access. For offline use, prepare the corresponding local Hugging Face model directory at `pretrained/t5-base/`, containing its configuration and weights. Add these options to the evaluation command:

```bash
--t5-model pretrained/t5-base --local-files-only
```

The supplied helper downloads the pinned configuration and safetensors files and verifies their SHA256 values:

```bash
python tools/prepare_pretrained.py --t5-dir pretrained/t5-base
```

Once both encoders are present, verify the existing files without network access:

```bash
python tools/prepare_pretrained.py --t5-dir pretrained/t5-base \
  --verify-only --actionclip-checkpoint pretrained/actionclip_vit_b16_8f.pt
```

The helper verifies ActionCLIP but does not download it. For offline training, add the same `--t5-model pretrained/t5-base --local-files-only` options, or set `model.t5_model` and `model.local_files_only` in the training configuration.

Loading a separately prepared trained adapter checks the frozen decoder's tensor fingerprint. A missing or different base model is an error; it does not silently fall back to a random decoder.

## Input format

Each JSONL record describes **one person's observed clip**, with time-ordered frame paths, one original-image `xyxy` bounding box per frame, contextual text, and optional ground-truth labels. See [configs/manifest.example.jsonl](configs/manifest.example.jsonl). A complete eight-frame example is:

```json
{
  "video_id": "video001",
  "instance_id": "person001",
  "split": "test",
  "frames": ["frames/video001/000.jpg", "frames/video001/001.jpg", "frames/video001/002.jpg", "frames/video001/003.jpg", "frames/video001/004.jpg", "frames/video001/005.jpg", "frames/video001/006.jpg", "frames/video001/007.jpg"],
  "bboxes": [[100,150,220,450], [101,150,221,450], [102,150,222,450], [103,150,223,450], [104,150,224,450], [105,150,225,450], [106,150,226,450], [107,150,227,450]],
  "text": {"scene":"park", "time":"morning", "weather":"sunny", "age":"adult", "gender":"male", "state":"alone"},
  "labels": {"action":"Walking", "short_term":"Keeping", "long_term":"Daily routine"}
}
```

- Frame paths are relative to the manifest directory, unless `--root` overrides the frame root. For `data/manifest.jsonl` and `data/frames/...`, omit `--root`.
- Global frames and local person crops are each resized to 336 × 336. The standard experiment uses **ground-truth boxes**; it does not include an automatic person detector.
- Only the six context fields shown above enter dictionary-based text encoding. Label fields are not encoded as contextual text.
- Labels may be names or zero-based IDs in the exact order in `configs/labels.json`: action / short-term / long-term. Remove `labels` for prediction-only records.
- A `split` field (`train`, `val`, or `test`) is required for these commands. All records from one video must belong to one split; cross-split video leakage is rejected.
- Use observed frames only. The reported experiment used eight-frame observations prepared from the first half of each annotated track. Different observation windows constitute a different evaluation protocol.

The supplied standard split was **generated at video level with a 7:2:1 ratio**, yielding **249 / 71 / 33 labelled person tracks** in train / validation / test. It is not claimed to be an official published split. The dataset itself must be obtained separately.

## Evaluate a trained checkpoint

**First complete the cache-training instructions below** to create `outputs/hintnet_336_cached_seed42/last.pt`, or prepare a compatible trained checkpoint and replace the path. The checkpoint in this example is an output of training, not a file included in this repository. Prepare a labelled `data/manifest.jsonl` and the ActionCLIP encoder, then run:

```bash
python evaluate.py \
  --checkpoint outputs/hintnet_336_cached_seed42/last.pt \
  --manifest data/manifest.jsonl --input-mode video --split test \
  --actionclip-checkpoint pretrained/actionclip_vit_b16_8f.pt \
  --device cuda --batch-size 8 --num-workers 4 \
  --output outputs/seed42_metrics.json \
  --predictions outputs/seed42_predictions.jsonl \
  --per-label-output outputs/seed42_per_label.csv
```

A checkpoint trained with the supplied configuration retains its **336 input size**, including when training used cached tokens and evaluation uses raw frames. A full training checkpoint already contains the frozen T5 decoder; add the local T5 options above when loading a separately exported adapter offline.

For unlabelled inputs, use the same command with `--predict-only`, retain `--predictions`, and omit `--per-label-output`. Keep the appropriate `split` field in every input record. Predictions include all three class IDs, names, and softmax probability vectors.

## Train at 336 × 336

### Raw-frame training

```bash
python train.py --config configs/hintnet_336.json \
  --manifest data/manifest.jsonl \
  --compatibility configs/compatibility_81.json \
  --actionclip-checkpoint pretrained/actionclip_vit_b16_8f.pt \
  --output outputs/hintnet_336_seed42 --device cuda
```

The full configuration uses 50 epochs, batch size 8, AdamW, learning rate `5e-5`, weight decay `0.01`, 10% warmup followed by cosine decay, gradient clipping at 1, and LVP weight 5. The encoders remain frozen. Raw training applies synchronized horizontal flips to global frames and local crops.

Training saves `config.json`, `history.jsonl`, `last.pt`, and validation-selected `best.pt`. The reported primary results use **epoch-50 `last.pt`**, not a test-selected checkpoint. Resume a full training checkpoint with:

```bash
python train.py --resume outputs/hintnet_336_seed42/last.pt --device cuda
```

An exported adapter is for inference, not resuming an optimizer trajectory. Resume requires a full training checkpoint containing optimizer, scheduler, and RNG state.

### Exact original/flip cache training

Caching avoids recomputing the frozen encoder. For exact reproduction of the existing fixed observations, **every record must already contain exactly eight frames**. The extractor stores both original and synchronized pre-resize horizontal-flip features; validation and test always use the original features.

```bash
python tools/extract_augmented_features.py \
  --config configs/hintnet_336.json --manifest data/manifest.jsonl \
  --checkpoint pretrained/actionclip_vit_b16_8f.pt \
  --output-dir data/actionclip_cache_r336 --device cuda

python train.py --config configs/hintnet_336_cached.json \
  --manifest data/actionclip_cache_r336/manifest.jsonl \
  --compatibility configs/compatibility_81.json \
  --actionclip-checkpoint pretrained/actionclip_vit_b16_8f.pt \
  --output outputs/hintnet_336_cached_seed42 --device cuda
```

The cache extractor reads the image size from `--config`; its cache features are computed at 336, not resized from a 224 feature cache. Cache validation checks both variants against the raw preprocessing path. A single original-only cache does not reproduce random-flip training.

### Export a trained adapter

```bash
python tools/export_adapter.py \
  --checkpoint outputs/hintnet_336_cached_seed42/last.pt \
  --output checkpoints/my_hintnet_336.pt
```

Export removes the frozen decoder, optimizer, and RNG state, and records the base-weight fingerprint and training provenance. It refuses to overwrite an existing export. The historical seed42 [reference-checkpoint metadata](docs/reference_checkpoint_metadata.json) is retained only as an experiment provenance record; the corresponding binary weights are not distributed.

## Model details

| Component | Implementation in this release |
| --- | --- |
| Input encoder | Shared frozen ActionCLIP ViT-B/16 + trained Transf temporal fusion; global/local/text each contribute a normalized 512-D token |
| 336 resolution | Bicubic interpolation of the visual spatial positional embedding; CLS and pretrained parameters remain unchanged |
| HQR | Three groups of 16 queries; learned per-token scalar sigmoid gates, shared across examples rather than predicted from each input |
| Projection and prefix | 512 → 768 MLP, 16 learnable prefix tokens, 64 decoder tokens in total |
| PTU | Decoder layers 1–4 expose prefix + action keys; 5–8 add short-term keys; 9–12 add long-term keys; every query row, including prefix rows, receives the key mask |
| Classification | Mean pooling of the corresponding 16 query tokens after the final T5 decoder layer and final normalization, followed by one linear head per level; prefix tokens are excluded from pooling |
| Training objective | Sum of three cross-entropy losses plus LVP; frozen T5 parameters still transmit gradients to the trainable modules |

The decoder uses progressive key visibility rather than T5's default causal triangle. Reproduction choices such as scalar gates, pooled ActionCLIP tokens, and mean pooling are explicit in the code; they should not be mistaken for additional details supplied by the original paper.

## Measured 336 results

These are the **five independently trained seeds 42–46**, evaluated at epoch 50 on the same 33-person standard test set. Values are percentages, reported as mean ± **sample standard deviation** (`ddof=1`), not confidence intervals.

| Level | mAP (%) | Accuracy (%) | Macro-F1 (%) |
| --- | ---: | ---: | ---: |
| Action | 57.8537 ± 6.0117 | 36.9697 ± 1.3552 | 17.9913 ± 4.7457 |
| Short-term intention | 48.3417 ± 5.2918 | 38.7879 ± 4.9793 | 20.6504 ± 6.5142 |
| Long-term intention | 63.2461 ± 4.2412 | 53.3333 ± 4.5957 | 55.0128 ± 4.5093 |

Scons: **96.3636 ± 2.5353%**. Three-label exact-match accuracy: **7.2727 ± 2.7104%**.

For reference, the historical seed42 model achieved mAP **57.3868 / 52.0490 / 63.1299%** for action / short-term / long-term; this single-run result differs from the five-seed mean. These are measured reference results, not predictions available from an untrained model. **No trained weights or dataset files are distributed with this code**; reproduction requires training with the same data and preprocessing, or separately obtaining the corresponding trained weights.

### Metric and compatibility definitions

- AP uses continuous softmax-score rankings, with no fixed classification threshold. mAP averages one-vs-rest AP over all declared classes; a class without positive test examples contributes AP 0.
- Accuracy and Macro-F1 use each level's independent `argmax`. Macro-F1 includes every declared class and uses `zero_division=0`.
- Scons is the fraction of the three independent predictions whose triplet belongs to the compatibility table. It is not the fraction for which all three labels match ground truth.
- **The supplied 81-triplet table was statistically obtained from the full train + validation + test annotations.** It is an empirical compatibility definition, not an independently specified semantic ontology. The models in the reported reference experiments were trained with this table. This includes test-label relationship information and limits strict independent-generalization claims. For a training-only or manually specified graph protocol, build a suitable table and retrain; do not report the existing checkpoint as trained under that different protocol.

## Verification

Run the offline unit tests:

```bash
python -m unittest discover -s tests -v
```

For a small, fully synthetic CPU training/evaluation check that downloads no models:

```bash
python tools/make_demo.py --output outputs/demo/data --feature-dim 32
python train.py --config outputs/demo/data/smoke.json \
  --output outputs/demo/run --device cpu
python evaluate.py --checkpoint outputs/demo/run/last.pt --device cpu \
  --output outputs/demo/test_metrics.json \
  --predictions outputs/demo/test_predictions.jsonl
```

This smoke test uses artificial features and a small random frozen decoder. It verifies the software pipeline; its metrics are not UAV-HINT or 336-model benchmark results.

By default, the ActionCLIP adapter disables cuDNN only around visual encoding, matching the numerical-stability setting used for this experiment. The trained model never switches to random pretrained components when weights are missing.

## Third-party components

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the retained [ActionCLIP license](hintnet/vendor/actionclip/LICENSE). Pretrained encoder weights are obtained from their upstream projects and remain subject to their respective terms. No new license for the project's own code is implied by the vendored component's license.

Release checks: [verification record](docs/RELEASE_VALIDATION.md) · [file checksums](SHA256SUMS).
