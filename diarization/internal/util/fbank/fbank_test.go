package fbank

import (
	"math/rand/v2"
	"slices"
	"testing"
)

func TestPadSlice(t *testing.T) {
	type testcase struct {
		slice    []float32
		leftPad  int
		rightPad int
	}
	createTestCase := func() testcase {
		slice := make([]float32, rand.IntN(10)+2)
		for idx := range len(slice) {
			slice[idx] = rand.Float32()
		}

		return testcase{slice: slice, leftPad: rand.IntN(100), rightPad: rand.IntN(100)}
	}

	tests := []testcase{
		createTestCase(),
		createTestCase(),
		createTestCase(),
		createTestCase(),
		createTestCase(),
	}

	for _, tc := range tests {
		pd := pad(tc.slice, tc.leftPad, tc.rightPad)

		lpd, corePd, rpd := pd[:tc.leftPad], pd[tc.leftPad:len(tc.slice)+tc.leftPad], pd[len(tc.slice)+tc.leftPad:]

		if len(lpd) != tc.leftPad {
			t.Errorf("left padded amount expected = %d got = %d", tc.leftPad, len(lpd))
		}

		if len(rpd) != tc.rightPad {
			t.Errorf("right padded amount expected = %d got = %d", tc.rightPad, len(rpd))
		}

		if !slices.Equal(corePd, tc.slice) {
			t.Errorf("padded slice does not contain valid slice")
		}
	}
}

func TestPreemphasis(t *testing.T) {
	tCases := []struct {
		input   []float32
		seqLen  int
		preempt float32
		output  []float32
	}{
		{input: []float32{}, seqLen: 0, output: []float32{}},
		{input: []float32{1}, seqLen: 1, output: []float32{1}},
		{input: []float32{1, 2, 3}, seqLen: 3, preempt: 0.97, output: []float32{1, 1.03, 1.06}},
		{input: []float32{1, 2, 3, 4, 5}, seqLen: 3, preempt: 0.97, output: []float32{1, 1.03, 1.06, 0, 0}},
		{input: []float32{10, 20, 30, 40}, seqLen: 2, preempt: 0.5, output: []float32{10, 15, 0, 0}},
		{input: []float32{1, 2, 3}, seqLen: 3, preempt: 0, output: []float32{1, 2, 3}},
		{input: []float32{1, 2, 3}, seqLen: 3, preempt: 1, output: []float32{1, 1, 1}},
	}

	for _, tt := range tCases {
		o, err := preemphasis(tt.input, tt.seqLen, tt.preempt)
		if err != nil {
			t.Fatalf("expected result to be successful but got err=%v", err)
		}
		if !slices.Equal(o, tt.output) {
			t.Errorf("expected slices to be equal expected=%v got=%v", tt.output, o)
		}
	}

	// -- errored test case
	_, err := preemphasis([]float32{1, 2, 3}, 4, 0.3)
	if err == nil {
		t.Error("expected error but received nil")
	}
}
