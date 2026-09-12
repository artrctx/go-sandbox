# pyannote Community-1 Diarization Research

Research target: [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)

This document explains what the Community-1 artifacts do, how the diarization pipeline combines them, and how to export the neural components to ONNX for a runtime that does not depend on the `pyannote.audio` Python package.

## Executive Summary

Community-1 is not one end-to-end neural network. It combines two neural models with conventional signal processing and statistical clustering:

```text
16 kHz mono audio
  -> local speaker segmentation
  -> masked speaker embeddings
  -> AHC initialization
  -> x-vector transform
  -> PLDA transform
  -> VBx clustering
  -> constrained local-to-global assignment
  -> timestamp reconstruction
```

Each component answers a different question:

| Component | Question |
| --- | --- |
| Segmentation | When is speech occurring, and how many speakers are active? |
| WeSpeaker embedding | What does each locally detected voice sound like? |
| x-vector transform | How should embeddings be centered, normalized, and projected? |
| PLDA | Which embedding differences represent speaker identity rather than ordinary variation? |
| AHC and VBx | Which local embeddings belong to the same global person? |
| Reconstruction | Which global speaker is active at each final timestamp? |

No speech-separation model is required for the Community-1 diarization pipeline.

## Repository Artifacts

At model revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`, the repository contains these runtime assets:

| Artifact | Role |
| --- | --- |
| `segmentation/pytorch_model.bin` | Local speaker segmentation, speech activity, overlap detection, and speaker-count evidence |
| `embedding/pytorch_model.bin` | pyannote-packaged WeSpeaker speaker encoder |
| `plda/xvec_transform.npz` | Embedding centering, normalization, and LDA projection |
| `plda/plda.npz` | PLDA mean, transform, and between-speaker covariance |
| `config.yaml` | Pipeline selection and tuned parameters |

The `.bin` files are PyTorch Lightning/pyannote checkpoints, not plain TorchScript modules. They include the state dictionary plus information such as:

- Architecture module and class
- Constructor hyperparameters
- Model specifications
- Training chunk duration
- Frame resolution and receptive field
- Powerset configuration
- Compatible `pyannote.audio` version

The checkpoint metadata is the authoritative source for exact architecture and dimensions.

## Why Segmentation and Embeddings Are Both Necessary

### Segmentation provides timing

The segmentation model analyzes short, overlapping windows and predicts local speaker activity at frame resolution:

```text
Time       Local speaker 0   Local speaker 1
0.0-1.2s       active           inactive
1.2-1.5s       active           active
1.5-2.0s       inactive         active
```

This identifies:

- Speech and silence
- Speaker changes
- Overlapping speech
- The instantaneous number of active speakers

Its speaker slots are local to each window. Local speaker 0 in one window is not guaranteed to represent the same person as local speaker 0 in another window.

### Embeddings provide identity

The embedding model maps a voice to a numerical vector. Recordings from the same person should have nearby vectors even when they occur in different windows.

```text
Alice, window 1 -> [ 0.13, -0.42, 0.71, ...]
Alice, window 4 -> [ 0.15, -0.39, 0.69, ...]
Bob, window 2   -> [-0.51,  0.22, 0.08, ...]
```

The segmentation mask tells the embedding model which frames in a window belong to a particular local speaker. Clustering then connects local speaker slots into recording-wide identities.

Embeddings alone do not provide precise boundaries or reliable overlap detection. Segmentation alone cannot preserve identity across a long recording. Diarization needs both.

## Segmentation Model

Unlike the third-party WeSpeaker embedding architecture, segmentation is based on pyannote's own powerset segmentation approach. The expected architecture is PyanNet, with the exact class recorded in the checkpoint:

```text
raw 16 kHz waveform
  -> SincNet learnable audio filters
  -> bidirectional LSTM layers
  -> feed-forward layers
  -> powerset classifier
```

Confirm the class rather than assuming it:

```python
checkpoint["pyannote.audio"]["architecture"]
```

### Powerset output

The model predicts one mutually exclusive powerset class per frame. A typical three-local-speaker, maximum-two-overlap configuration has these classes:

```text
{}
{speaker 0}
{speaker 1}
{speaker 2}
{speaker 0, speaker 1}
{speaker 0, speaker 2}
{speaker 1, speaker 2}
```

The pipeline performs hard decoding:

```python
powerset_class = argmax(log_probabilities, axis=-1)
local_activity = powerset_mapping[powerset_class]
```

This is not independent sigmoid thresholding and is not soft marginal decoding.

## Embedding Model

Community-1's embedding component is a pyannote wrapper around a trained WeSpeaker ResNet speaker encoder. WeSpeaker is the neural architecture, not an additional model that must be downloaded.

```text
waveform
  -> Kaldi-compatible 80-bin filterbank
  -> WeSpeaker ResNet
  -> mask-weighted statistics pooling
  -> speaker embedding
```

The expected preprocessing includes:

- Mono 16 kHz input
- Waveform multiplication by `32768`
- 80 mel filterbank bins
- 25 ms frame length
- 10 ms frame shift
- Hamming window
- Zero dither
- No energy coefficient
- Mean subtraction across frames

Read the actual values from checkpoint hyperparameters.

For every `(window, local speaker)` pair, pyannote runs the complete window with that local speaker's activity mask:

```python
embedding = embedding_model(window_audio, weights=local_speaker_mask)
```

The mask is used by weighted statistics pooling. The implementation interpolates it to the internal ResNet frame rate with nearest-neighbor interpolation.

## Complete Inference Sequence

### 1. Prepare audio

1. Decode the recording.
2. Average multiple channels to mono.
3. Resample to exactly 16 kHz.
4. Retain the expected floating-point waveform scale.
5. Zero-pad the final incomplete segmentation window.

### 2. Run sliding-window segmentation

Use the training duration stored in the checkpoint:

```text
window duration = checkpoint specification duration
window step     = duration * 0.1
```

The source default therefore uses 90% overlap. Do not assume a fixed 5- or 10-second duration without reading the checkpoint.

The logical output is:

```text
[num_windows, num_segmentation_frames, num_local_speakers]
```

### 3. Estimate instantaneous speaker count

Sum local activity in each window and overlap-average those counts on the global timeline:

```python
local_count = local_activity.sum(axis=-1)
global_count = round(overlap_average(local_count))
```

The result can represent silence, one speaker, or overlapping speakers.

### 4. Extract masked speaker embeddings

Produce one embedding for each local speaker slot in each segmentation window. Community-1 includes overlapping frames in the embedding mask by default.

### 5. Filter clustering embeddings

An embedding participates in clustering initialization only when:

- It contains no NaN values.
- Its local speaker has enough single-speaker activity.
- Clean activity covers at least 20% of the window frames.

The pipeline can still assign excluded embeddings later after global centroids have been discovered.

### 6. Initialize with AHC

L2-normalize usable embeddings and run centroid-linkage agglomerative hierarchical clustering with Euclidean distance. The Community-1 source default threshold is `0.6`.

AHC gives VBx an initial speaker assignment. It is not the final clustering result.

### 7. Apply the x-vector transform

`xvec_transform.npz` contains:

```text
mean1
mean2
lda
```

The current implementation computes the equivalent of:

```python
x = embeddings - mean1
x = sqrt(input_dimension) * l2_normalize(x)
x = x @ lda
x = x - mean2
x = sqrt(lda_dimension) * l2_normalize(x)
```

Matrix orientation must be validated against the NPZ shapes.

### 8. Apply the PLDA transform

`plda.npz` contains:

```text
mu
tr
psi
```

pyannote reconstructs within- and between-class covariance matrices, solves a generalized eigenproblem, reverses the resulting order, and retains the first 128 dimensions:

```python
W = inverse(tr.T @ tr)
B = inverse((tr.T / psi) @ tr)

eigenvalues, eigenvectors = scipy.linalg.eigh(B, W)
plda_psi = eigenvalues[::-1]
plda_tr = eigenvectors.T[::-1]

plda_features = (xvec_features - mu) @ plda_tr.T
plda_features = plda_features[:, :128]
phi = plda_psi[:128]
```

These arrays were trained for this embedding model. Replacing the embedding model while retaining the PLDA parameters is not valid without retraining the backend.

### 9. Run VBx

VBx starts from smoothed AHC assignments and refines speaker responsibilities in PLDA space. Source defaults are:

```text
AHC threshold = 0.6
Fa            = 0.07
Fb            = 0.8
max iterations = 20
```

The downloaded `config.yaml` should remain the final source of truth for a pinned model revision.

Although the cited VBx paper describes Bayesian HMM clustering, the current pyannote implementation explicitly dropped HMM support and uses a GMM-style variational update. Implement the pyannote source behavior when exact parity is required.

Speakers whose learned prior is at most `1e-7` are removed.

### 10. Build global centroids

Use VBx posterior responsibilities to calculate speaker centroids in the original WeSpeaker embedding space, not PLDA space:

```python
weights = responsibilities[:, speaker_priors > 1e-7]
centroids = weights.T @ original_embeddings
centroids = centroids / weights.sum(axis=0)[:, None]
```

### 11. Assign every local speaker slot

Compare all original local embeddings with the global centroids using cosine distance. Apply Hungarian constrained assignment independently in each window so two local speakers in the same window cannot map to the same global speaker.

When a requested speaker count forces a K-means fallback, constrained assignment is disabled.

### 12. Reconstruct diarization

Map local activity tracks to global speaker dimensions and overlap-add the windows. At each frame, activate the top speakers according to the estimated instantaneous count:

```python
active_speakers[t] = top_k(
    global_activations[t],
    k=instantaneous_speaker_count[t],
)
```

Merge adjacent active frames into timestamped turns.

For exclusive diarization, reuse the same activations and clusters but cap the count at one:

```python
exclusive_count[t] = min(regular_count[t], 1)
```

## Recommended Deployment Components

```text
segmentation.onnx
embedding.onnx
xvec_transform.npz
plda.npz
pipeline.json
```

ONNX should handle neural inference. Conventional numerical code should handle filterbank extraction, transforms, clustering, assignment, and reconstruction.

`pipeline.json` should record at least:

- Model repository and revision
- Sample rate
- Segmentation window duration and step
- Segmentation receptive-field timing
- Powerset mapping
- Filterbank configuration
- Minimum clean activity ratio
- AHC threshold and linkage method
- PLDA dimension
- `Fa`, `Fb`, and VBx iterations
- Speaker-prior pruning threshold
- Mask interpolation mode
- Reconstruction parameters

## ONNX Conversion Strategy

Use `pyannote.audio` only in a trusted conversion environment. Deployment can then use ONNX Runtime without pyannote.

### Load directly from Hugging Face

Accept the gated model conditions first and provide `HF_TOKEN`:

```python
import os

from pyannote.audio import Model

repository = "pyannote/speaker-diarization-community-1"
token = os.environ["HF_TOKEN"]

segmentation = Model.from_pretrained(
    repository,
    subfolder="segmentation",
    token=token,
    map_location="cpu",
    strict=False,
).eval()

embedding = Model.from_pretrained(
    repository,
    subfolder="embedding",
    token=token,
    map_location="cpu",
    strict=False,
).eval()
```

This automatically downloads each `pytorch_model.bin`, reconstructs the architecture recorded in the checkpoint, and loads its weights.

### Export segmentation

```python
import torch

duration = segmentation.specifications.duration
sample_rate = segmentation.audio.sample_rate
num_samples = round(duration * sample_rate)
example_waveform = torch.randn(1, 1, num_samples)

torch.onnx.export(
    segmentation,
    example_waveform,
    "segmentation.onnx",
    input_names=["waveform"],
    output_names=["powerset_scores"],
    opset_version=17,
    dynamo=False,
    dynamic_axes={
        "waveform": {0: "batch"},
        "powerset_scores": {0: "batch"},
    },
)
```

This exports raw powerset scores. Powerset decoding can be implemented outside ONNX or included in an export wrapper using `ArgMax` and a constant mapping tensor.

Keep the sample dimension fixed for parity. The production pipeline already operates on fixed windows and pads the last one.

### Export the embedding network

The neural WeSpeaker ResNet is ONNX-compatible. The built-in `torchaudio.compliance.kaldi.fbank` frontend may not export reliably, so the recommended ONNX boundary accepts filterbanks rather than raw waveform:

```python
import torch
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


duration = segmentation.specifications.duration
num_samples = round(duration * embedding.audio.sample_rate)
example_waveform = torch.randn(1, 1, num_samples)

with torch.inference_mode():
    example_fbank = embedding.compute_fbank(example_waveform)
    segmentation_output = segmentation(example_waveform)

example_mask = torch.ones(1, segmentation_output.shape[1])
embedding_onnx = WeSpeakerONNX(embedding).eval()

torch.onnx.export(
    embedding_onnx,
    (example_fbank, example_mask),
    "embedding.onnx",
    input_names=["fbank", "speaker_mask"],
    output_names=["embedding"],
    opset_version=17,
    dynamo=False,
    dynamic_axes={
        "fbank": {0: "batch"},
        "speaker_mask": {0: "batch"},
        "embedding": {0: "batch"},
    },
)
```

The deployment runtime must reproduce the exact Kaldi-compatible filterbank before invoking `embedding.onnx`.

A raw-waveform embedding ONNX is possible if the filterbank frontend is rewritten entirely with exportable ONNX operations. It is harder to maintain numerical parity, so separating preprocessing from the neural graph is safer.

## Extracting Parameters From ONNX

ONNX initializers contain neural weights and constant tensors:

```python
import onnx
from onnx import numpy_helper

model = onnx.load("segmentation.onnx")

parameters = {
    initializer.name: numpy_helper.to_array(initializer)
    for initializer in model.graph.initializer
}

for name, value in parameters.items():
    print(name, value.shape, value.dtype)
```

ONNX naturally preserves:

- Convolution, recurrent, linear, and batch-normalization parameters
- Constant tensors included in the graph
- Operators and their attributes
- Tensor types and dimensions
- Custom metadata explicitly added during export

It does not automatically preserve pipeline semantics such as:

- Sample rate
- Sliding-window duration and step
- Receptive-field timestamp mapping
- Audio resampling and downmix behavior
- Filterbank settings when preprocessing is external
- AHC threshold
- Minimum clean activity ratio
- `Fa`, `Fb`, and VBx iteration count
- Speaker-count bounds
- Reconstruction rules

The NPZ parameters also do not appear in either neural ONNX graph unless they are deliberately wrapped into that graph. Keeping them as separate assets is simpler.

## Validation Plan

Establish parity component by component rather than comparing only final speaker turns. Save golden outputs from `pyannote.audio` for a short recording:

```text
segmentation.npy
speaker_count.npy
embeddings.npy
usable_embedding_indices.npy
ahc_labels.npy
plda_features.npy
vbx_responsibilities.npy
vbx_priors.npy
centroids.npy
local_to_global.npy
regular.rttm
exclusive.rttm
```

Validate in this order:

1. Segmentation powerset scores
2. Decoded local activity
3. Speaker count and timestamp mapping
4. Filterbank values
5. Masked speaker embeddings
6. x-vector transformed embeddings
7. PLDA features and `phi`
8. AHC initialization
9. VBx responsibilities and priors
10. Global centroids
11. Local-to-global assignments
12. Final regular and exclusive RTTM

## Main Parity Risks

- Treating local segmentation slots as global identities
- Running an unnecessary VAD or speech-separation model
- Using soft powerset marginals instead of hard argmax decoding
- Cropping speech instead of using mask-weighted embedding pooling
- Producing filterbanks that differ from Kaldi behavior
- Forgetting waveform multiplication by `32768`
- Getting LDA or PLDA matrix orientation wrong
- Using PLDA-space vectors for final centroid assignment
- Omitting one-to-one Hungarian assignment inside each window
- Ignoring instantaneous speaker count during reconstruction
- Implementing the original HMM VBx instead of pyannote's current update
- Guessing window duration or frame timing instead of reading checkpoint metadata
- Loading only the state dictionary without reconstructing the saved architecture

## Source References

- [Community-1 model repository](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [Speaker diarization pipeline](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/speaker_diarization.py)
- [Clustering and VBx integration](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/clustering.py)
- [PLDA loader and transform](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/core/plda.py)
- [VBx implementation](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/utils/vbx.py)
- [WeSpeaker wrapper and preprocessing](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/models/embedding/wespeaker/__init__.py)
- [WeSpeaker ResNet](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/models/embedding/wespeaker/resnet.py)
- [PyanNet segmentation architecture](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/models/segmentation/PyanNet.py)
- [Powerset conversion](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/utils/powerset.py)
- [Diarization reconstruction](https://github.com/pyannote/pyannote-audio/blob/develop/src/pyannote/audio/pipelines/utils/diarization.py)

## Licensing Note

The Community-1 model repository is published under CC-BY-4.0. The pyannote.audio source is MIT-licensed, while incorporated WeSpeaker code carries its own Apache-2.0 notices. Review and retain the applicable attribution and license notices when redistributing converted model artifacts or copied implementation code.
