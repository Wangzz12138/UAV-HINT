# Required frozen pretrained weights

The repository contains the trained HINT-Net adapter, not the large frozen
encoders. Place the following files here, or override their paths on the CLI.

```text
pretrained/
  actionclip_vit_b16_8f.pt
  t5-base/
    config.json
    model.safetensors
```

- ActionCLIP: obtain the **ViT-B/16, 8-frame, Kinetics-400** model from the
  [official model zoo](https://github.com/sallymmx/ActionCLIP/blob/master/MODEL_ZOO.md).
  The file must include `model_state_dict` and `fusion_model_state_dict`.
  Expected SHA256:
  `0acfe0359eb73c02e93f0b1ee4761e7099747d15fe305aa833776f6681c8aa5b`.
- T5-base: the loader can download `google-t5/t5-base` at pinned revision
  `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`. Alternatively prepare local files:

  ```bash
  python tools/prepare_pretrained.py --t5-dir pretrained/t5-base
  ```

Verify already downloaded files, without network access:

```bash
python tools/prepare_pretrained.py --verify-only \
  --t5-dir pretrained/t5-base \
  --actionclip-checkpoint pretrained/actionclip_vit_b16_8f.pt
```

No T5 tokenizer is required: HINT-Net passes learned continuous prefix/query
embeddings into the frozen decoder. Context text uses the vendored CLIP tokenizer.
