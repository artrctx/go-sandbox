package stft

// Package stft provides support for Short-Time Fourier Transform (STFT)
// analysis.
import (
	"math"

	"github.com/artrctx/gossiper/internal/util/stft/fft"
)

// CreateHanning returns a symmetric Hann window.
func CreateHanning(length int) []float32 {
	window := make([]float32, length)

	arg := 2.0 * math.Pi / float64(length-1)
	for i := range window {
		window[i] = 0.5 - 0.5*float32(math.Cos(arg*float64(i)))
	}

	return window
}

// Windowing multiplies input by window.
func Windowing(input, window []float32) []float32 {
	result := make([]float32, len(input))

	for i := range input {
		result[i] = input[i] * window[i]
	}

	return result
}

// STFT represents Short-Time Fourier Transform analysis.
type STFT struct {
	FrameShift int
	FrameLen   int
	NFFT       int
	Center     bool

	// Window is the original win_length-sized window.
	Window []float32

	// PaddedWindow is the n_fft-sized window used for the FFT.
	PaddedWindow []float32
}

// New returns a new STFT instance.
func New(frameShift, frameLen, nFFT int, center bool) *STFT {
	window := CreateHanning(frameLen)

	// torch.stft pads a window shorter than n_fft symmetrically.
	windowPadding := (nFFT - frameLen) / 2
	paddedWindow := padXY(window, windowPadding)

	return &STFT{
		FrameShift:   frameShift,
		FrameLen:     frameLen,
		NFFT:         nFFT,
		Center:       center,
		Window:       window,
		PaddedWindow: paddedWindow,
	}
}

// NumFrames returns the number of n_fft-sized frames.
func (s *STFT) NumFrames(input []float32) int {
	if len(input) < s.NFFT {
		return 0
	}

	return (len(input)-s.NFFT)/s.FrameShift + 1
}

// DivideFrames returns overlapping n_fft-sized frames.
func (s *STFT) DivideFrames(input []float32) [][]float32 {
	numFrames := s.NumFrames(input)

	frames := make([][]float32, numFrames)

	for i := 0; i < numFrames; i++ {
		frames[i] = s.FrameAt(input, i)
	}

	return frames
}

// FrameAt returns an n_fft-sized frame.
//
// The returned slice references the original input.
func (s *STFT) FrameAt(input []float32, index int) []float32 {
	start := index * s.FrameShift
	end := start + s.NFFT
	return input[start:end]
}

// Process returns the complex spectrogram.
func (s *STFT) Process(input []float32) [][]complex64 {
	if s.Center {
		input = padXY(input, s.NFFT/2)
	}

	numFrames := s.NumFrames(input)
	spectrogram := make([][]complex64, numFrames)

	for i := 0; i < numFrames; i++ {
		frame := s.FrameAt(input, i)
		windowed := Windowing(frame, s.PaddedWindow)
		spectrogram[i] = fft.FFTReal(windowed)
	}

	return spectrogram
}

// padXY zero-pads input symmetrically on both sides.
func padXY(input []float32, padAmt int) []float32 {
	if padAmt <= 0 {
		return input
	}

	out := make([]float32, len(input)+padAmt*2)
	copy(out[padAmt:], input)

	return out
}
