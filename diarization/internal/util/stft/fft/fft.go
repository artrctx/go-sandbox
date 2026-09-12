package fft

// FFTReal returns the forward FFT of the real-valued slice.
func FFTReal(x []float32) []complex64 {
	return FFT(ToComplex(x))
}

// IFFTReal returns the inverse FFT of the real-valued slice.
func IFFTReal(x []float32) []complex64 {
	return IFFT(ToComplex(x))
}

// IFFT returns the inverse FFT of the complex-valued slice.
func IFFT(x []complex64) []complex64 {
	lx := len(x)
	r := make([]complex64, lx)

	// Reverse inputs, which is calculated with modulo N, hence x[0] as an outlier
	r[0] = x[0]
	for i := 1; i < lx; i++ {
		r[i] = x[lx-i]
	}

	r = FFT(r)

	N := complex(float32(lx), 0)
	for n := range r {
		r[n] /= N
	}
	return r
}

// Convolve returns the convolution of x ∗ y.
func Convolve(x, y []complex64) []complex64 {
	if len(x) != len(y) {
		panic("arrays not of equal size")
	}

	fft_x := FFT(x)
	fft_y := FFT(y)

	r := make([]complex64, len(x))
	for i := 0; i < len(r); i++ {
		r[i] = fft_x[i] * fft_y[i]
	}

	return IFFT(r)
}

// FFT returns the forward FFT of the complex-valued slice.
func FFT(x []complex64) []complex64 {
	lx := len(x)

	// todo: non-hack handling length <= 1 cases
	if lx <= 1 {
		r := make([]complex64, lx)
		copy(r, x)
		return r
	}

	if IsPowerOf2(lx) {
		return radix2FFT(x)
	}

	return bluesteinFFT(x)
}

var (
	worker_pool_size = 0
)
