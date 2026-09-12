# Pyannote Community-1 ONNX Model Conversion

This document describes the methodology used to convert the `pyannote/speaker-diarization-community-1` segmentation and embedding models from PyTorch to ONNX.

The goal is to make the models independently executable from Go using ONNX Runtime, while preserving the behavior of the original PyTorch models.

The conversion is intentionally performed at the **model level**, rather than attempting to reproduce the entire `Pipeline.from_pretrained()` machinery. The pyannote pipeline is orchestration: segmentation, embedding, and clustering are separate components that can be manually composed.

---

# 1. Overall Architecture

The Community-1 diarization pipeline can be thought of as:

```text
                    Raw audio
                       │
                       ▼
              ┌─────────────────┐
              │   Segmentation  │
              │     model       │
              └────────┬────────┘
                       │
                       ▼
              speaker activity
                   estimates
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        speaker 1 mask       speaker 2 mask
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    Embedding    │
              │     model       │
              └────────┬────────┘
                       │
                       ▼
              speaker embeddings
                       │
                       ▼
              ┌─────────────────┐
              │   Clustering /  │
              │   VBx + PLDA    │
              └────────┬────────┘
                       │
                       ▼
                  diarization
```

The ONNX conversion therefore produces two independently callable models:

```text
segmentation.onnx
embedding.onnx
```

Clustering/PLDA is a separate component and does not need to be part of either ONNX model.

---

# 2. Why Convert the Models Separately?

The segmentation and embedding models solve different problems.

## Segmentation

The segmentation model answers:

> "Who appears to be speaking at each point in time?"

Its output is a sequence of frame-level speaker activity scores.

For Community-1, the model produces:

```text
[batch, frames, classes]
```

For the example input used during conversion:

```text
input:
[1, 1, 160000]

output:
[1, 589, 7]
```

The exact interpretation of the 7 output classes should be obtained from the model's `Specifications` rather than inferred from the tensor shape.

---

## Embedding

The embedding model answers:

> "What speaker characteristics are present in this audio segment?"

It converts speech into a fixed-dimensional speaker representation.

The model used by Community-1 is a WeSpeaker ResNet34 architecture.

The relevant computation is:

```text
audio
  │
  ▼
fbank
  │
  ▼
WeSpeaker ResNet34
  │
  ▼
frame-level feature maps
  │
  ▼
weighted statistics pooling
  │
  ▼
speaker embedding
```

The resulting embedding is a fixed-size vector.

For this model the embedding dimension is:

```text
128
```

---

# 3. General Conversion Principle

The main principle is:

> Export the actual PyTorch computation needed at inference time, rather than exporting pyannote's entire pipeline.

The pyannote pipeline contains orchestration logic such as:

- audio chunking
- batching
- segmentation inference
- speaker mask manipulation
- embedding extraction
- clustering
- PLDA/VBx processing

Those concerns are better implemented in the Go application.

The ONNX models should therefore expose relatively simple tensor interfaces.

---

# 4. Segmentation Model

## 4.1 Load the Model

The segmentation model is loaded from the Community-1 repository:

```python
from pyannote.audio import Model

segmentation = Model.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    subfolder="segmentation",
    token=token,
    map_location="cpu",
    strict=False,
).eval()
```

Calling:

```python
.eval()
```

puts the model into inference/evaluation mode.

This is important because layers such as BatchNorm and Dropout can behave differently during training.

---

# 5. Segmentation Input

The segmentation model reports its own required duration and sample rate.

Do not hard-code these values if they can be obtained from the model.

```python
duration = segmentation.specifications.duration
sample_rate = segmentation.audio.sample_rate

num_samples = round(duration * sample_rate)
```

For Community-1:

```text
duration   = 10 seconds
sample rate = 16000 Hz
```

Therefore:

```text
10 × 16000 = 160000 samples
```

The example input is:

```python
example_waveform = torch.randn(
    1,
    1,
    160000,
    dtype=torch.float32,
)
```

The dimensions are:

```text
[B, C, samples]

[1, 1, 160000]
```

where:

- `B` = batch size
- `C` = audio channel count
- `samples` = audio samples

---

# 6. Why Use a Random Waveform?

The waveform passed to `torch.onnx.export()` is an **example input**.

It is not training data.

The exporter needs an actual tensor so that it can execute the PyTorch model and trace/build the ONNX computation graph.

Therefore:

```python
torch.randn(...)
```

is sufficient for export.

For example:

```python
example_waveform = torch.randn(
    1,
    1,
    num_samples,
    dtype=torch.float32,
)
```

For validation, however, real audio should eventually be used to compare PyTorch and ONNX outputs.

---

# 7. Segmentation Export

Initial exports should use a static shape.

```python
import torch
import onnx
from pyannote.audio import Model


segmentation = Model.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    subfolder="segmentation",
    token=token,
    map_location="cpu",
    strict=False,
).eval()


duration = segmentation.specifications.duration
sample_rate = segmentation.audio.sample_rate
num_samples = round(duration * sample_rate)

example_waveform = torch.randn(
    1,
    1,
    num_samples,
    dtype=torch.float32,
)


with torch.inference_mode():
    torch.onnx.export(
        segmentation,
        example_waveform,
        "segmentation.onnx",
        input_names=["waveform"],
        output_names=["powerset_scores"],
        opset_version=17,
        dynamo=False,
    )
```

---

# 8. Why `dynamo=False`?

Recent versions of PyTorch have a newer `torch.export`-based ONNX exporter.

The older exporter is deprecated, so PyTorch may display:

```text
DeprecationWarning:
You are using the legacy TorchScript-based ONNX export.
```

This is a warning, not a conversion failure.

For this project, `dynamo=False` is intentionally used during the initial conversion because:

1. the goal is to establish a working ONNX representation first;
2. the model contains operations such as LSTM and InstanceNorm;
3. the older exporter provides a straightforward way to debug compatibility;
4. ONNX Runtime compatibility is more important than using the newest exporter immediately.

Once the model is successfully converted and numerically validated, the newer exporter can be evaluated separately.

---

# 9. Why Not Use Dynamic Batch Immediately?

The segmentation model contains an LSTM.

The ONNX exporter may produce a warning similar to:

```text
Exporting a model to ONNX with a batch_size other than 1,
with a variable length with LSTM can cause an error when
running the ONNX model with a different batch size.
```

This is why the initial export intentionally does **not** specify:

```python
dynamic_axes=...
```

The first objective is:

```text
PyTorch
   ↓
ONNX
   ↓
ONNX Runtime
```

with a known shape.

Once that works, dynamic dimensions can be introduced deliberately and tested.

---

# 10. InstanceNorm Warning

The segmentation model contains `InstanceNorm1d`.

The exporter may report:

```text
ONNX export mode is set to TrainingMode.EVAL,
but operator 'instance_norm' is set to train=True.
```

Inspection of the actual model showed:

```text
InstanceNorm1d
training = False
track_running_stats = False
affine = True
```

This is important.

`track_running_stats=False` means the InstanceNorm layer uses statistics from the current input rather than maintaining running statistics.

Therefore the layer should **not** be modified merely to silence the exporter warning.

The correct approach is:

```python
segmentation.eval()
```

and leave the pretrained model architecture unchanged.

---

# 11. Segmentation Output

The model produced:

```text
input:
[1, 1, 160000]

output:
[1, 589, 7]
```

The output is called:

```text
powerset_scores
```

The model uses powerset-style speaker activity representation.

Do not assume that the 7 classes correspond directly to a particular speaker-count mapping.

The authoritative information should be obtained from:

```python
print(segmentation.specifications)
print(segmentation.specifications.classes)
print(segmentation.specifications.powerset)
print(segmentation.specifications.powerset_max_classes)
```

The tensor shape alone is not sufficient to determine the semantic class mapping.

---

# 12. Validate the Segmentation ONNX Model

First validate the ONNX graph:

```python
import onnx

model = onnx.load("segmentation.onnx")
onnx.checker.check_model(model)

print("ONNX model is valid.")
```

Then compare PyTorch and ONNX Runtime outputs.

```python
import numpy as np
import onnxruntime as ort
import torch


with torch.inference_mode():
    torch_output = segmentation(example_waveform)


session = ort.InferenceSession(
    "segmentation.onnx",
    providers=["CPUExecutionProvider"],
)

onnx_output = session.run(
    ["powerset_scores"],
    {
        "waveform": example_waveform.numpy()
    },
)[0]


print("PyTorch:", torch_output.shape)
print("ONNX:", onnx_output.shape)

print(
    "max abs difference:",
    np.max(
        np.abs(
            torch_output.numpy() - onnx_output
        )
    ),
)

print(
    "mean abs difference:",
    np.mean(
        np.abs(
            torch_output.numpy() - onnx_output
        )
    ),
)
```

The important checks are:

```text
same shape
        +
small numerical difference
```

Shape equality alone is not sufficient.

---

# 13. Embedding Model

The Community-1 embedding model is different from segmentation.

Its specifications showed:

```text
problem:       REPRESENTATION
resolution:    CHUNK
duration:      5.0 seconds
sample rate:   16000 Hz
```

Therefore the embedding model uses:

```text
5 seconds × 16000 Hz
=
80000 samples
```

The embedding model's duration must be used independently of the segmentation model.

Do not use:

```python
segmentation.specifications.duration
```

to construct embedding inputs.

Instead:

```python
duration = embedding.specifications.duration
sample_rate = embedding.audio.sample_rate

num_samples = round(
    duration * sample_rate
)
```

---

# 14. Embedding Waveform

Create the example waveform:

```python
example_waveform = torch.randn(
    1,
    1,
    num_samples,
    dtype=torch.float32,
)
```

For Community-1 this corresponds to:

```text
[1, 1, 80000]
```

---

# 15. Computing the Fbank

The WeSpeaker model does not directly consume raw waveform.

The pyannote embedding model computes filter-bank features:

```python
with torch.inference_mode():
    example_fbank = embedding.compute_fbank(
        example_waveform
    )
```

The resulting shape observed was:

```text
[1, 998, 80]
```

Therefore:

```text
B = 1
T = 998
F = 80
```

or:

```text
[batch, frames, features]
```

The embedding model's ResNet expects this layout.

---

# 16. WeSpeaker ResNet34

The embedding model contains:

```text
WeSpeakerResNet34
```

The relevant architecture is:

```text
fbank
  │
  ▼
permute:
[B, T, F]
      ↓
[B, F, T]
  │
  ▼
Conv2D
  │
  ▼
ResNet layer1
  │
  ▼
ResNet layer2
  │
  ▼
ResNet layer3
  │
  ▼
ResNet layer4
  │
  ▼
statistics pooling
  │
  ▼
Linear
  │
  ▼
embedding
```

The model contains:

```text
conv1
layer1
layer2
layer3
layer4
pool
seg_1
seg_bn_1
seg_2
```

The final embedding dimension is:

```text
128
```

---

# 17. Why the ResNet Output Has 125 Temporal Frames

The fbank contains:

```text
998 frames
```

but the ResNet performs temporal downsampling.

The observed ResNet output was:

```text
[1, 256, 10, 125]
```

This represents:

```text
[batch, channels, frequency, time]
```

Therefore the final temporal dimension is:

```text
125
```

The approximate temporal reduction is:

```text
998
 ↓ layer2 (stride 2)
499
 ↓ layer3 (stride 2)
250
 ↓ layer4 (stride 2)
125
```

Therefore the statistics pooling layer sees approximately:

```text
125 temporal positions
```

rather than the original 998 fbank frames.

---

# 18. Speaker Mask / Pooling Weights

The ResNet's `forward()` method explicitly documents:

```python
weights : (batch, frames) or (batch, speakers, frames)
```

The weights are passed to the statistics pooling layer:

```python
stats = self.pool(
    out,
    weights=weights
)
```

Therefore the weights are **temporal pooling weights**.

They determine which temporal positions contribute to the speaker statistics.

For a single speaker:

```text
[B, T]
```

For multiple speakers:

```text
[B, S, T]
```

where:

- `B` = batch
- `S` = number of speakers
- `T` = ResNet temporal frames

For the current example:

```text
T = 125
```

Therefore a single-speaker example mask is:

```python
example_mask = torch.ones(
    1,
    125,
    dtype=torch.float32,
)
```

not:

```python
torch.ones(1, 998)
```

and not necessarily:

```python
torch.ones(
    1,
    segmentation_output.shape[1]
)
```

The reason is that the pooling layer operates on the ResNet's temporal output, which is 125 frames.

---

# 19. Why the Mask Is Not 998 Frames

This distinction is critical.

The fbank has:

```text
998 frames
```

but the ResNet transforms them into:

```text
125 temporal positions
```

The pooling operation receives:

```text
frames = [B, 256, 10, 125]
```

and therefore the temporal weights must correspond to:

```text
125
```

temporal positions.

The pipeline therefore conceptually needs:

```text
segmentation timeline
       │
       ▼
speaker activity
       │
       ▼
resample / align to embedding temporal resolution
       │
       ▼
speaker mask [speakers, 125]
       │
       ▼
statistics pooling
```

This alignment should eventually be implemented explicitly in the Go pipeline.

---

# 20. Embedding ONNX Wrapper

The pyannote embedding model contains additional wrapper behavior.

For ONNX conversion, the relevant WeSpeaker ResNet can be exposed directly.

```python
import torch.nn as nn


class WeSpeakerONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.resnet = model.resnet

    def forward(self, fbank, speaker_mask):
        _, speaker_embedding = self.resnet(
            fbank,
            weights=speaker_mask,
        )

        return speaker_embedding
```

The purpose of this wrapper is to expose a clean ONNX interface:

```text
fbank
speaker_mask
   │
   ▼
WeSpeaker ResNet
   │
   ▼
speaker embedding
```

The first return value from the ResNet is intentionally ignored:

```python
_, speaker_embedding = ...
```

because the desired output is the speaker embedding.

---

# 21. Embedding Export

The initial static export is:

```python
import torch
import torch.nn as nn
import onnx


class WeSpeakerONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.resnet = model.resnet

    def forward(self, fbank, speaker_mask):
        _, speaker_embedding = self.resnet(
            fbank,
            weights=speaker_mask,
        )
        return speaker_embedding


# ---------------------------------------------------------
# 1. Use the EMBEDDING model's own specifications
# ---------------------------------------------------------

embedding.eval()

duration = embedding.specifications.duration
sample_rate = embedding.audio.sample_rate

num_samples = round(
    duration * sample_rate
)

example_waveform = torch.randn(
    1,
    1,
    num_samples,
    dtype=torch.float32,
)


# ---------------------------------------------------------
# 2. Compute fbank
# ---------------------------------------------------------

with torch.inference_mode():
    example_fbank = embedding.compute_fbank(
        example_waveform
    )

print(
    "fbank:",
    example_fbank.shape,
)

# Expected:
# [1, 998, 80]


# ---------------------------------------------------------
# 3. Determine ResNet temporal dimension
# ---------------------------------------------------------

with torch.inference_mode():
    example_frames = embedding.resnet.forward_frames(
        example_fbank
    )

print(
    "resnet frames:",
    example_frames.shape,
)

# Expected:
# [1, 256, 10, 125]


num_embedding_frames = (
    example_frames.shape[-1]
)


# ---------------------------------------------------------
# 4. Create temporal speaker mask
# ---------------------------------------------------------

example_mask = torch.ones(
    example_fbank.shape[0],
    num_embedding_frames,
    dtype=torch.float32,
)

print(
    "speaker mask:",
    example_mask.shape,
)

# Expected:
# [1, 125]


# ---------------------------------------------------------
# 5. Create ONNX wrapper
# ---------------------------------------------------------

embedding_onnx = WeSpeakerONNX(
    embedding
).eval()


# ---------------------------------------------------------
# 6. Export
# ---------------------------------------------------------

with torch.inference_mode():
    torch.onnx.export(
        embedding_onnx,
        (
            example_fbank,
            example_mask,
        ),
        "embedding.onnx",

        input_names=[
            "fbank",
            "speaker_mask",
        ],

        output_names=[
            "embedding",
        ],

        opset_version=17,

        # Use the legacy exporter initially
        # for easier debugging.
        dynamo=False,
    )


# ---------------------------------------------------------
# 7. Validate ONNX graph
# ---------------------------------------------------------

onnx_model = onnx.load(
    "embedding.onnx"
)

onnx.checker.check_model(
    onnx_model
)

print(
    "ONNX model is valid."
)
```

---

# 22. Expected Embedding ONNX Interface

The initial model should conceptually have:

```text
INPUT:

fbank
[1, 998, 80]

speaker_mask
[1, 125]


OUTPUT:

embedding
[1, 128]
```

The complete computation is:

```text
fbank [1, 998, 80]
        │
        ▼
   ResNet34
        │
        ▼
[1, 256, 10, 125]
        │
        │ speaker_mask [1,125]
        ▼
Statistics Pooling
        │
        ▼
     Linear
        │
        ▼
embedding [1,128]
```

---

# 23. Why Start With a Single-Speaker Mask?

The model documentation indicates that the pooling layer can accept:

```text
[B, T]
```

or:

```text
[B, S, T]
```

However, the initial ONNX conversion should use:

```text
[B, T]
```

because it is simpler to validate.

The first goal is:

```text
one audio chunk
+
one speaker mask
        ↓
one embedding
```

Once this is validated, multi-speaker batching can be tested.

For example:

```text
speaker_mask:

[1, 3, 125]
```

could represent three speaker masks.

Conceptually:

```text
              time
          0  1  2  3 ... 124

speaker 0  1  1  1  0 ...   0
speaker 1  0  0  1  1 ...   0
speaker 2  0  0  0  0 ...   1
```

This could potentially produce:

```text
[1, 3, 128]
```

speaker embeddings.

However, the exact behavior should be verified against the pooling implementation before making this the production ONNX interface.

---

# 24. Important: Segmentation and Embedding Have Different Time Resolutions

One of the most important implementation details is that the segmentation and embedding models do not necessarily operate on the same temporal grid.

For example:

```text
Segmentation:

audio
  ↓
589 frames
```

while:

```text
Embedding:

fbank
  ↓
998 frames
  ↓
ResNet
  ↓
125 temporal positions
```

Therefore we should not directly assume:

```text
segmentation frame i == embedding frame i
```

Instead, the production pipeline needs an explicit temporal alignment step.

Conceptually:

```text
segmentation scores
        │
        ▼
speaker activity
        │
        ▼
convert to embedding temporal grid
        │
        ▼
125-frame speaker mask
        │
        ▼
embedding model
```

This alignment should be treated as pipeline logic, not hidden inside the ONNX embedding model.

---

# 25. Why Keep Fbank Computation Outside ONNX?

The current embedding ONNX model accepts:

```text
fbank
```

rather than raw waveform.

This is intentional.

The pyannote `compute_fbank()` implementation performs feature extraction before the WeSpeaker ResNet.

Keeping this outside the ONNX model has several advantages:

1. The ONNX graph is simpler.
2. The ResNet conversion is easier to validate.
3. Feature extraction can be implemented explicitly in Go.
4. The same fbank implementation can be reused.
5. Debugging is easier because feature extraction and neural inference are separate stages.

The eventual Go pipeline becomes:

```text
audio
  │
  ▼
fbank extraction
  │
  ▼
embedding.onnx
```

rather than requiring ONNX to reproduce the entire audio preprocessing stack.

However, the Go fbank implementation must match pyannote's preprocessing exactly if embeddings are expected to be numerically equivalent.

---

# 26. Inference Mode vs Eval Mode

Both are used intentionally:

```python
model.eval()
```

and:

```python
with torch.inference_mode():
```

They solve different problems.

## `eval()`

Changes model behavior for inference.

For example:

- Dropout becomes disabled.
- BatchNorm uses its inference behavior.

## `inference_mode()`

Disables autograd-related tracking.

This reduces unnecessary computation and memory overhead when running inference/export examples.

Therefore:

```python
embedding.eval()

with torch.inference_mode():
    ...
```

is appropriate.

---

# 27. ONNX Validation Strategy

Every converted model should go through three levels of validation.

## Level 1 — ONNX Graph Validation

```python
onnx.checker.check_model(
    onnx_model
)
```

This verifies that the ONNX graph is structurally valid.

---

## Level 2 — Shape Validation

Verify:

```text
PyTorch input shape
=
ONNX input shape
```

and:

```text
PyTorch output shape
=
ONNX output shape
```

For segmentation:

```text
[1, 1, 160000]
        ↓
[1, 589, 7]
```

For embedding:

```text
fbank:
[1, 998, 80]

mask:
[1, 125]

        ↓

embedding:
[1, 128]
```

---

## Level 3 — Numerical Validation

Run the exact same input through:

```text
PyTorch
```

and:

```text
ONNX Runtime
```

and compare:

```python
np.max(
    np.abs(
        pytorch_output
        -
        onnx_output
    )
)
```

and:

```python
np.mean(
    np.abs(
        pytorch_output
        -
        onnx_output
    )
)
```

This is the most important validation step.

A model that loads successfully in ONNX Runtime can still have incorrect numerical behavior.

---

# 28. Recommended Development Sequence

The safest conversion process is:

```text
1. Load PyTorch model
        ↓
2. Inspect model specifications
        ↓
3. Determine input shape
        ↓
4. Run PyTorch inference
        ↓
5. Inspect intermediate tensor shapes
        ↓
6. Create minimal ONNX wrapper if necessary
        ↓
7. Export with static dimensions
        ↓
8. Validate ONNX graph
        ↓
9. Run ONNX Runtime
        ↓
10. Compare PyTorch vs ONNX
        ↓
11. Test real audio
        ↓
12. Only then introduce dynamic dimensions
        ↓
13. Only then optimize/batch
```

Avoid starting with dynamic batching and multiple speakers because it makes exporter and runtime problems harder to isolate.

---

# 29. Production Go Architecture

Once both models have been validated, the Go implementation can be structured approximately as:

```text
Audio
 │
 ├──────────────────────────────┐
 │                              │
 ▼                              ▼
Segmentation ONNX          Fbank extraction
 │                              │
 ▼                              ▼
speaker activity           embedding fbank
 │                              │
 ▼                              │
temporal alignment              │
 │                              │
 ▼                              ▼
speaker masks ────────────> Embedding ONNX
                                │
                                ▼
                         speaker embeddings
                                │
                                ▼
                           PLDA / VBx
                                │
                                ▼
                           diarization
```

This separation is useful because each stage can be tested independently.

---

# 30. Current Known Tensor Shapes

The shapes established during conversion are:

## Segmentation

```text
sample rate:
16000 Hz

duration:
10 seconds

waveform:
[1, 1, 160000]

output:
[1, 589, 7]
```

## Embedding

```text
sample rate:
16000 Hz

duration:
5 seconds

waveform:
[1, 1, 80000]

fbank:
[1, 998, 80]

ResNet output:
[1, 256, 10, 125]

speaker mask:
[1, 125]

embedding:
[1, 128]
```

---

# 31. Important Design Decisions

## Decision 1 — Use each model's own duration

Correct:

```python
segmentation.specifications.duration
```

for segmentation.

Correct:

```python
embedding.specifications.duration
```

for embedding.

Do not assume both models use the same chunk duration.

---

## Decision 2 — Keep segmentation and embedding separate

The models solve different problems and operate at different temporal resolutions.

Do not combine them into one ONNX graph initially.

---

## Decision 3 — Export the WeSpeaker ResNet rather than the entire pyannote embedding wrapper

The useful computation is:

```text
fbank
+
speaker weights
        ↓
WeSpeaker ResNet
        ↓
embedding
```

This produces a clean interface for the Go implementation.

---

## Decision 4 — Speaker mask follows ResNet temporal resolution

The fbank has:

```text
998 frames
```

but the pooling layer operates over:

```text
125 temporal positions
```

Therefore:

```text
speaker_mask = [B, 125]
```

for the initial single-speaker interface.

---

## Decision 5 — Start with static ONNX shapes

Static shapes make initial debugging substantially easier.

Dynamic batch support should be added only after numerical equivalence is established.

---

## Decision 6 — Do not modify pretrained normalization layers just to eliminate exporter warnings

The segmentation model's InstanceNorm configuration is part of the pretrained model behavior.

Warnings from the exporter should not automatically lead to architectural changes.

---

# 32. What Still Needs to Be Verified

The following should be verified before considering the conversion complete:

### Segmentation

- [ ] ONNX graph passes `onnx.checker`.
- [ ] ONNX Runtime loads the model.
- [ ] PyTorch and ONNX outputs numerically agree.
- [ ] Exact powerset class mapping is obtained from `segmentation.specifications`.
- [ ] Real audio produces sensible segmentation scores.
- [ ] Dynamic batch support is tested separately if required.

### Embedding

- [ ] Exact statistics pooling implementation is inspected.
- [ ] `[B,T]` mask behavior is verified.
- [ ] `[B,S,T]` multi-speaker mask behavior is verified.
- [ ] ONNX graph passes `onnx.checker`.
- [ ] ONNX Runtime loads the model.
- [ ] PyTorch and ONNX embeddings numerically agree.
- [ ] Fbank implementation is reproduced exactly in Go.
- [ ] Segmentation-to-embedding temporal alignment is implemented and tested.
- [ ] Real speech produces stable embeddings.

---

# 33. Final Target

The desired end state is not:

```text
Python pipeline → ONNX
```

but rather:

```text
                    Go application

                         Audio
                           │
                           ▼
                  ┌─────────────────┐
                  │ Audio preprocessing│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ segmentation.onnx│
                  └────────┬────────┘
                           │
                           ▼
                  speaker activity
                           │
                           ▼
                    temporal alignment
                           │
                           ▼
                    speaker masks
                           │
                           │
             ┌─────────────┘
             │
             ▼
       Fbank extraction
             │
             ▼
      ┌─────────────────┐
      │  embedding.onnx │
      └────────┬────────┘
               │
               ▼
       speaker embeddings
               │
               ▼
          PLDA / VBx
               │
               ▼
          diarization
```

The ONNX models should remain relatively small, well-defined inference components, while the Go application owns the orchestration and temporal alignment logic.