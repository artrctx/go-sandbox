package sha1

import (
	"encoding/binary"
	"math/bits"
	"slices"
)

// franctional part of the sqrt of the first five primes
const (
	H0 uint32 = 0x67452301
	H1 uint32 = 0xEFCDAB89
	H2 uint32 = 0x98BADCFE
	H3 uint32 = 0x10325476
	H4 uint32 = 0xC3D2E1F0

	ROUND1 uint32 = 0x5A827999
	ROUND2 uint32 = 0x6ED9EBA1
	ROUND3 uint32 = 0x8F1BBCDC
	ROUND4 uint32 = 0xCA62C1D6
)

// will return 20 byte hash
func Hash(key []byte) []byte {
	h0, h1, h2, h3, h4 := H0, H1, H2, H3, H4

	m := padMessage(key)

	// 512 bit chunks 8 * 64
	for chunk := range slices.Chunk(m, 64) {
		a, b, c, d, e := h0, h1, h2, h3, h4
		sch := buildPreliminarySchedule(chunk)

		for i, v := range sch {
			var f, k uint32
			switch {
			case i < 20:
				// previous impl used (b & c) | ((^b) & d)
				f, k = (b&c)^((^b)&d), ROUND1
			case i < 40:
				f, k = b^c^d, ROUND2
			case i < 60:
				f, k = (b&c)^(b&d)^(c&d), ROUND3
			default:
				f, k = b^c^d, ROUND4
			}

			// wrapping additions
			tmp := bits.RotateLeft32(a, 5) + f + e + k + v

			e = d
			d = c
			c = bits.RotateLeft32(b, 30)
			b = a
			a = tmp
		}

		h0, h1, h2, h3, h4 = h0+a, h1+b, h2+c, h3+d, h4+e
	}

	hash := make([]byte, 0, 20)

	// h0
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, h0)
	hash = append(hash, buf...)

	// h1
	buf = make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, h1)
	hash = append(hash, buf...)

	// h2
	buf = make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, h2)
	hash = append(hash, buf[:4]...)

	// h3
	buf = make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, h3)
	hash = append(hash, buf...)

	// h4
	buf = make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, h4)
	hash = append(hash, buf...)

	return hash
}

func padMessage(k []byte) []byte {
	bitLen := len(k) * 8

	// 0x80 = 128 = 1000000 (binary)
	k = append(k, 0x80)

	// 64 bits remaining
	for (len(k)*8)%512 != 448 {
		k = append(k, 0)
	}

	// put length of key at the end
	k = binary.BigEndian.AppendUint64(k, uint64(bitLen))

	return k
}

// expects 512 bits / 64 bytes
func buildPreliminarySchedule(chunk []byte) [80]uint32 {
	schedule := [80]uint32{}

	idx := 0
	// 16 count
	for blk := range slices.Chunk(chunk, 4) {
		schedule[idx] = binary.LittleEndian.Uint32(blk)
		idx++
	}

	for i := 16; i < 80; i++ {
		schedule[i] = schedule[i-3] ^ schedule[i-8] ^ schedule[i-14] ^ schedule[i-16]
		schedule[i] = bits.RotateLeft32(schedule[i], 1)
	}

	return schedule
}
