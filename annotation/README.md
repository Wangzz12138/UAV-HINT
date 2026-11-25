# UAV-HINT annotation interface

We provide the authors' Label Studio XML configuration used for person-track,
action and hierarchical-intention annotation:

- [label_studio_config.xml](label_studio_config.xml)

The XML is the annotation interface definition, not the per-video annotation
results. It was supplied directly by the authors for this release.

It includes video bounding boxes, 7 action choices, 8 short-term intention
choices, 4 long-term intention choices, scene/time/weather, and per-person
age/gender/state. Its instructions specify using acquisition records or
author-provided environmental labels rather than guessing those fields from
appearance. Person attributes and intention labels are assigned to the selected
person track.

Short-term intentions are verified within 30 seconds after the observation clip;
long-term intentions are determined from the full flight recording. These are
instructions for establishing ground-truth labels. HINT-Net receives only the
observed clip and its specified context at inference time.

Use the XML in the Label Studio project's labeling-interface configuration.
Imported task data must provide the `video` URL and `fps` referenced by `$video`
and `$fps`; for example:

```json
[{"data": {"video": "https://example.org/video001.mp4", "fps": 30}}]
```

Replace the example URL with a video accessible to your annotation deployment.
The XML has been checked for well-formedness, named-control references and
class-vocabulary agreement. A Label Studio browser session was not executed
as part of the code-release verification.
