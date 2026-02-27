package handshake

type Handshake struct {
	// protocol
	Pstr     string
	InfoHash [20]byte
	PeerID   [20]byte
}

func (h *Handshake) Serialize() []byte {
	// 49 = 20 infohash + 20 peerid + 0x13 (proto identifier) + 8 reserved values
	buf := make([]byte, len(h.Pstr)+49)
	buf[0] = byte(len(h.Pstr))

	curr := 1
	curr = copy(buf[curr:], h.Pstr)
	curr = copy(buf[curr:], make([]byte, 8))
	curr = copy(buf[curr:], h.InfoHash[:])
	curr = copy(buf[curr:], h.PeerID[:])

	return buf
}
