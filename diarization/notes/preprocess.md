Yes. This is the **actual FBANK pipeline**, and we can simplify it substantially for your use case because you're doing inference with this Sortformer model.

I'll walk through it in execution order and point out **what matters for your Go implementation**.

---

## 1. Calculate the output frame length

```python
seq_len_time = seq_len
seq_len_unfixed = self.get_seq_len(seq_len)

seq_len = torch.where(
    seq_len == 0,
    torch.zeros_like(seq_len_unfixed),
    seq_len_unfixed
)
```

Input:

```text
x       = [B, samples]
seq_len = [B]
```

For example:

```text
x.shape = [1, 16000]
seq_len = [16000]
```

`get_seq_len()` determines how many **STFT frames** are valid after preprocessing.

With your configuration:

```text
sample rate = 16000
window      = 400 samples (25 ms)
hop         = 160 samples (10 ms)
```

So approximately:

```text
16000 samples → 100 feature frames
```

`seq_len` is therefore tracking **time frames**, not audio samples, after this point.

The `seq_len == 0` special case is just for streaming edge cases.

**Go relevance:** you need this calculation because the returned `seq_len` is passed along to the Sortformer.

---

# 2. Optional STFT padding

```python
if self.stft_pad_amount is not None:
    x = torch.nn.functional.pad(
        x.unsqueeze(1),
        (self.stft_pad_amount, self.stft_pad_amount),
        "constant"
    ).squeeze(1)
```

This adds zeros to both ends of the waveform.

But for your current Sortformer configuration, you should check whether:

```python
self.stft_pad_amount
```

is actually set.

From the configuration we've inspected, `exact_pad=False`, so **this isn't the main padding mechanism you need to worry about**.

---

# 3. Dither

```python
if self.training and self.dither > 0:
    x += self.dither * torch.randn_like(x)
```

This adds tiny random noise.

Your model has:

```text
dither = 1e-5
```

But you're doing:

```python
model.eval()
```

Therefore:

```text
self.training == False
```

so this **doesn't happen during inference**.

### For your Go implementation:

**Ignore dither.**

---

# 4. Pre-emphasis

```python
timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)

x = torch.cat(
    (
        x[:, 0].unsqueeze(1),
        x[:, 1:] - self.preemph * x[:, :-1]
    ),
    dim=1
)

x = x.masked_fill(~timemask, 0.0)
```

This is the first actual transformation of your waveform.

Your model has:

```text
preemph = 0.97
```

Mathematically:

```text
y[0] = x[0]

y[n] = x[n] - 0.97 * x[n-1]
```

So:

```text
raw:
x0 x1 x2 x3 x4 ...

        ↓

pre-emphasis:
x0
x1 - 0.97*x0
x2 - 0.97*x1
x3 - 0.97*x2
...
```

The `timemask` portion makes sure samples beyond the valid `seq_len_time` are zero.

### For Go:

This is straightforward and **you need it**.

---

# 5. STFT

```python
with torch.amp.autocast(x.device.type, enabled=False):
    x = self.stft(x)
```

This calls NeMo's:

```python
FilterbankFeatures.stft()
```

which ultimately uses:

```python
torch.stft(...)
```

Your configuration is:

```text
n_fft      = 512
window     = 400
hop        = 160
window     = Hann
center     = True
```

Conceptually:

```text
waveform
   ↓
STFT
   ↓
complex spectrum
```

The shape becomes approximately:

```text
[B, 257, frames]
```

because:

```text
n_fft / 2 + 1
= 512 / 2 + 1
= 257
```

So:

```text
[1, 16000]
      ↓
STFT
      ↓
[1, 257, ~100] complex
```

---

# 6. Complex STFT → magnitude

```python
x = torch.view_as_real(x)

x = torch.sqrt(
    x.pow(2).sum(-1) + guard
)
```

STFT gives you complex numbers:

```text
a + bi
```

This converts each one to its magnitude:

```text
sqrt(a² + b²)
```

So:

```text
complex spectrum
      ↓
magnitude spectrum
```

Shape goes back to:

```text
[B, 257, frames]
```

---

# 7. Noise-band augmentation

```python
if self.training and self.nb_augmentation_prob > 0.0:
    ...
```

Again, you're in:

```python
model.eval()
```

so this does **nothing**.

Ignore it for your implementation.

---

# 8. Convert magnitude → power spectrum

```python
if self.mag_power != 1.0:
    x = x.pow(self.mag_power)
```

Your configuration:

```text
mag_power = 2.0
```

Therefore:

```text
magnitude²
```

So now:

```text
[1, 257, frames]
```

contains the **power spectrum**.

---

# 9. Apply Mel filterbank

This is the important FBANK step:

```python
x = torch.matmul(self.fb.to(x.dtype), x)
```

You have:

```text
self.fb.shape = [128, 257]
```

and:

```text
x.shape = [B, 257, frames]
```

Matrix multiplication:

```text
[128, 257]
      ×
[257, frames]

      ↓

[128, frames]
```

So:

```text
power spectrum
[1, 257, frames]

      ↓ Mel filterbank

[1, 128, frames]
```

This is where the **257 FFT frequency bins become 128 Mel bins**.

Your `self.fb` is the actual Mel filterbank matrix.

---

# 10. Take logarithm

```python
if self.log:
    x = torch.log(
        x + self.log_zero_guard_value_fn(x)
    )
```

Your configuration says:

```text
log = True
log_zero_guard_type = "add"
log_zero_guard_value = 5.960464477539063e-08
```

So effectively:

```text
log(mel_energy + 5.96e-8)
```

Now you have **log-Mel features**.

This is what people often casually call "FBANK."

---

# 11. Frame splicing

```python
if self.frame_splicing > 1:
    x = splice_frames(...)
```

Your configuration:

```text
frame_splicing = 1
```

Therefore:

**nothing happens.**

---

# 12. Normalization

```python
if self.normalize:
    x, _, _ = normalize_batch(...)
```

Your configuration:

```text
normalize = "NA"
```

So:

**nothing happens.**

This is important: you don't need to reproduce some hidden mean/std normalization.

---

# 13. Mask invalid frames

```python
max_len = x.size(-1)

mask = torch.arange(max_len, device=x.device)
mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)

x = x.masked_fill(
    mask.unsqueeze(1).type(torch.bool),
    self.pad_value
)
```

This handles batches where audio sequences have different lengths.

For example:

```text
audio A → 100 valid frames
audio B → 70 valid frames
```

The feature tensor might be:

```text
[B, 128, 100]
```

and B's frames 70–99 get replaced with:

```text
pad_value = 0
```

For a single full-length audio input, this isn't particularly interesting.

---

# 14. Pad number of frames to a multiple of 16

```python
pad_to = self.pad_to

if pad_to > 0:
    pad_amt = x.size(-1) % pad_to

    if pad_amt != 0:
        x = torch.nn.functional.pad(
            x,
            (0, pad_to - pad_amt),
            value=self.pad_value
        )
```

Your configuration has:

```text
pad_to = 16
```

So if you have:

```text
100 frames
```

it becomes:

```text
112 frames
```

because:

```text
112 % 16 = 0
```

This is **tensor padding for efficiency**. It doesn't mean those extra frames contain actual audio.

---

# 15. Return

Finally:

```python
return x, seq_len
```

So the result is:

```text
x
[B, 128, padded_frames]

seq_len
[B]
```

For example:

```text
input:

[1, 16000]

       ↓

STFT

[1, 257, 100]

       ↓

Mel

[1, 128, 100]

       ↓

pad_to=16

[1, 128, 112]

       ↓

return
```

with:

```text
seq_len = [100]
```

Notice the important distinction:

```text
x.shape[-1] = 112
seq_len     = 100
```

**112 is the allocated/padded feature length. 100 is the actual valid feature length.**

---

## So the entire thing for your model

You can mentally reduce this huge function to:

```text
[B, samples] raw audio
       │
       ▼
pre-emphasis (0.97)
       │
       ▼
STFT
  n_fft=512
  window=400
  hop=160
  Hann
       │
       ▼
magnitude
       │
       ▼
power²
       │
       ▼
128-bin Mel filterbank
       │
       ▼
log()
       │
       ▼
padding to multiple of 16
       │
       ▼
[B, 128, frames]
```

And **that `[B, 128, frames]` tensor is the output you care about before feeding the Sortformer pipeline.**

One correction to keep in mind from our earlier discussion: **the exact Sortformer ONNX boundary may transpose this to `[B, frames, 128]`**, depending on the exported graph/version. The FBANK function itself returns `[B, 128, frames]`.
