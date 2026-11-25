# Release verification

Verified on 2026-11-25 using an isolated copy of this repository, with the original
project removed from PYTHONPATH. The dataset and pretrained encoder files were
provided externally; neither is bundled in this repository.

- **54 offline unit tests passed**, including exact portable adapter restoration,
  missing/wrong tensor rejection, frozen-decoder identity checks, progressive
  attention, logical loss, AP/F1 definitions, data loading and resolution transfer.
- A synthetic CPU training/evaluation pipeline completed **2 epochs** and evaluated
  **56 artificial test examples**. These are software checks, not benchmark scores.
- Before checkpoint removal, the reference adapter restored every core state tensor of the seed42 epoch50
  snapshot: **178 tensors / 115,362,739 core parameters**, including the genuine
  frozen T5 decoder. The reference adapter stores only 20 trainable tensors; no adapter binary is distributed here.
- Using author-held trained weights, the standalone repository evaluated all **33 real standard-test instances**
  from raw frames with **336-pixel global/local inputs**. All 31 metric fields,
  all three argmax predictions, and all softmax scores were **identical** to the
  original trained model: maximum metric error **0**, maximum score error **0**.
- Exported metrics were independently recomputed from the per-instance scores.
- A Python wheel built successfully, including the vendored BPE vocabulary and
  third-party license. All local README links and internal imports were checked.
- Source, configuration and adapter metadata contain no original-machine paths.
  Dataset videos, caches, virtual environments and full pretrained weights are
  excluded.

The validation checkpoint was seed42 only; five-seed averages in the README describe separate
completed experiments. This is now a code-only distribution. Trained weights must
be obtained or generated separately. See [per-seed metrics](336_seed_results.csv), the
[machine-readable verification](release_validation.json) and the
[adapter metadata](reference_checkpoint_metadata.json).

Run `sha256sum -c SHA256SUMS` from the repository root to check file integrity.

## Code-only distribution update

The model-weight binary has been removed. The author-supplied XML is included at
[annotation/label_studio_config.xml](../annotation/label_studio_config.xml). XML
well-formedness, control references and the 7/8/4 vocabularies were checked. No
Label Studio browser session was executed. Historical model-verification results
above describe author-held reference weights, which are not distributed here.

See [upload instructions](UPLOAD_TO_GITHUB.md) for an existing GitHub repository.
