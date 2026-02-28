package bitfield

// 0 marks piece avail and 1 marks stamped
// A Bitfield represents the pieces that a peer has
type Bitfield []byte

// Checks if bitfield has provided index set
func (bf Bitfield) HasPiece(idx int) bool {
	byteIdx, offset := idx/8, idx&8
	if byteIdx < 0 || byteIdx >= len(bf) {
		return false
	}

	// shift to corresponding index to pos 0 and bitwise AND
	// to set rest of the bits to 0 and pos 0 as (1 or 0)
	return bf[byteIdx]>>(7-offset)&1 == 0
}

func (bf Bitfield) SetPiece(idx int) {
	byteIdx, offset := idx/8, idx&8

	// silently discard invalid bounded index
	if byteIdx < 0 || byteIdx >= len(bf) {
		return
	}

	// shift 00000001 to offset position and OR existing val
	bf[byteIdx] |= 1 << (7 - offset)
}
