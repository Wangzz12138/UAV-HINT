# Third-party components

The HINT-Net project-level license has not been selected in this release folder.
No license for the original project is inferred from a dependency's license.

## ActionCLIP / CLIP

The inference implementation and BPE vocabulary in `hintnet/vendor/actionclip/`
derive from [sallymmx/ActionCLIP](https://github.com/sallymmx/ActionCLIP), which is
based on OpenAI CLIP. The supplied upstream MIT license and original source
headers are preserved. See [the vendored license](hintnet/vendor/actionclip/LICENSE).

This release modifies the visual position-embedding lookup to support bicubic
spatial interpolation at 336 pixels. It keeps the original CLS position and the
stored pretrained parameter shapes. The input wrapper uses the shared visual
encoder and trained temporal fusion branch for global and local observations.

ActionCLIP pretrained weights are not distributed in this repository. Obtain
the ViT-B/16, 8-frame Kinetics-400 checkpoint from the upstream model zoo and
follow its terms.

## T5 / Transformers

The frozen decoder is loaded through Hugging Face Transformers. The upstream
[google-t5/t5-base model card](https://huggingface.co/google-t5/t5-base) identifies
its license as Apache-2.0. The release pins the model revision to
`a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`. T5 pretrained weights are not bundled;
their upstream terms remain applicable.

Other dependencies retain their own licenses. No dataset videos, extracted
frames, annotation export, or full pretrained model is included.
