package fft

import (
	"math"
	"sync"
)

var (
	bluesteinLock       sync.RWMutex
	bluesteinFactors    = map[int][]complex64{}
	bluesteinInvFactors = map[int][]complex64{}
)

func getBluesteinFactors(input_len int) ([]complex64, []complex64) {
	bluesteinLock.RLock()

	if hasBluesteinFactors(input_len) {
		defer bluesteinLock.RUnlock()
		return bluesteinFactors[input_len], bluesteinInvFactors[input_len]
	}

	bluesteinLock.RUnlock()
	bluesteinLock.Lock()
	defer bluesteinLock.Unlock()

	if !hasBluesteinFactors(input_len) {
		bluesteinFactors[input_len] = make([]complex64, input_len)
		bluesteinInvFactors[input_len] = make([]complex64, input_len)

		var sin, cos float32
		for i := 0; i < input_len; i++ {
			if i == 0 {
				sin, cos = 0, 1
			} else {
				sin64, cos64 := math.Sincos(float64(math.Pi / float32(input_len) * float32(i*i)))
				sin, cos = float32(sin64), float32(cos64)
			}
			bluesteinFactors[input_len][i] = complex(cos, sin)
			bluesteinInvFactors[input_len][i] = complex(cos, -sin)
		}
	}

	return bluesteinFactors[input_len], bluesteinInvFactors[input_len]
}

func hasBluesteinFactors(idx int) bool {
	return bluesteinFactors[idx] != nil
}

// bluesteinFFT returns the FFT calculated using the Bluestein algorithm.
func bluesteinFFT(x []complex64) []complex64 {
	lx := len(x)
	a := ZeroPad(x, NextPowerOf2(lx*2-1))
	la := len(a)
	factors, invFactors := getBluesteinFactors(lx)

	for n, v := range x {
		a[n] = v * invFactors[n]
	}

	b := make([]complex64, la)
	for i := 0; i < lx; i++ {
		b[i] = factors[i]

		if i != 0 {
			b[la-i] = factors[i]
		}
	}

	r := Convolve(a, b)

	for i := 0; i < lx; i++ {
		r[i] *= invFactors[i]
	}

	return r[:lx]
}
