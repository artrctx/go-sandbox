package fbank

import (
	"fmt"
	"math"

	"github.com/artrctx/gossiper/internal/util/cond"
	"github.com/artrctx/gossiper/internal/util/stft"
)

// mel filter bank impl by nvidia
//https://docs.nvidia.com/nemo/speech/nightly/asr/api.html
//https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html
//https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html#preprocessor-configuration
// https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/preprocessing/features.py

/*
SortformerEncLabelModel
        │
        ▼
model.preprocessor
        │
        ▼
AudioToMelSpectrogramPreprocessor
        │
        ▼
self.featurizer
        │
        ▼
FilterbankFeatures
        │
        ▼
FilterbankFeatures.forward()


[FilterbankFeatures]
sample_rate               = 16000
win_length                = 400
hop_length                = 160
n_fft                     = 512
window                    = Tensor(shape=(400,))
normalize                 = NA
preemph                   = 0.97
nfilt                     = 128
log                       = True
log_zero_guard_type       = add
log_zero_guard_value      = 5.960464477539063e-08
dither                    = 1e-05
pad_to                    = 16
frame_splicing            = 1
exact_pad                 = False
pad_value                 = 0
mag_power                 = 2.0
nb_augmentation_prob      = 0.0
*/

type MelFeatures struct{}

type Source struct {
	data   []float32
	length int
}

type Config struct {
	sampleRate        int
	logZeroGuardValue float32
	windowLen         int
	hopLen            int
	nFFT              int
	exactPad          bool
	frameSplicing     int
	nfilt             int
	preempt           float32
	padTo             int
	// default should be 1e-5
	guard float32
}

func (c *Config) StftPadAmount() *int {
	if !c.exactPad {
		return nil
	}
	val := (c.nFFT - c.hopLen) / 2
	return new(val)
}

// this is specifically for "nvidia/diar_streaming_sortformer_4spk-v2.1"
// https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/parts/preprocessing/features.py
func Preprocess(srcs []Source, cfg Config) ([]MelFeatures, error) {
	return nil, nil
}

func process(s *Source, cfg *Config) (*MelFeatures, error) {
	// ---- Prep ----
	_ = s.length
	stftPadAmt := cfg.StftPadAmount()
	seqLenUnfixed := (s.length + cond.Ternary(stftPadAmt != nil, *stftPadAmt*2, (cfg.nFFT/2)*2) - cfg.nFFT) / cfg.hopLen
	s.length = cond.Ternary(s.length == 0, 0, seqLenUnfixed)

	if stftPadAmt != nil {
		s.data = pad(s.data, *stftPadAmt, *stftPadAmt)
	}

	// ---- Preemphasis ----
	new, err := preemphasis(s.data, s.length, cfg.preempt)
	if err != nil {
		return nil, err
	}
	s.data = new

	// ---- STFT ----
	_ = stft.New(cfg.windowLen, cfg.hopLen, cfg.nFFT, stftPadAmt != nil).Process(s.data)

	return nil, nil
}

func pad(data []float32, leftPad, rightPad int) []float32 {
	newSlice := make([]float32, len(data)+leftPad+rightPad)
	copy(newSlice[leftPad:], data)
	return newSlice
}

func preemphasis(data []float32, seqLen int, preempt float32) ([]float32, error) {
	if len(data) < seqLen {
		return nil, fmt.Errorf("source data is shorter then seq len expected at least %d got %d", seqLen, len(data))
	}

	if seqLen == 0 {
		clear(data)
		return data, nil
	}

	prev := data[0]
	for i := 1; i < seqLen; i++ {
		curr := data[i]
		data[i] = data[i] - preempt*prev
		prev = curr

	}

	clear(data[seqLen:])

	return data, nil
}

func complexToFloat(cs []complex64, guard float32) []float32 {
	out := make([]float32, len(cs))
	for idx, c := range cs {
		r, i := real(c), imag(c)
		out[idx] = float32(math.Sqrt(float64(r*r + i*i + guard)))
	}
	return out
}
