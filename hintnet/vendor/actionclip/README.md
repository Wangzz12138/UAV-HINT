# Vendored ActionCLIP inference components

Source: [sallymmx/ActionCLIP](https://github.com/sallymmx/ActionCLIP).
The supplied upstream MIT license and original source headers are retained.
This directory includes the visual/text model, BPE tokenizer and vocabulary,
and the trained `Transf` temporal-fusion implementation.

Global and person-crop clips share one frozen encoder and temporal head. Text
uses the CLIP EOT embedding. The wrapper normalizes the three output tokens.
An ActionCLIP checkpoint with both `model_state_dict` and
`fusion_model_state_dict` must be provided separately.

The visual implementation supports the 336-pixel experiment by bicubically
interpolating only the spatial position embeddings; CLS positions and stored
pretrained parameters remain unchanged. Native 224-pixel execution retains its
original numerical path.

The input adapter defaults to native visual convolutions (`disable_cudnn=True`)
to preserve the validated numerical setup. This changes the convolution backend,
not the model architecture. Enable cuDNN explicitly only when accepting a
different runtime/numerical setup.
