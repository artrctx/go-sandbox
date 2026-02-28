package handshake

import (
	"fmt"
	"io"
)

type Handshake struct {
	// The protocol identifier, called the pstr which is always BitTorrent protocol
	Pstr     string
	InfoHash [20]byte
	PeerID   [20]byte
}

func New(infoHash, peerID [20]byte) Handshake {
	return Handshake{
		// standard pstr
		Pstr:     "BitTorrent protocol",
		InfoHash: infoHash,
		PeerID:   peerID,
	}
}

func (h *Handshake) Serialize() []byte {
	// 49 = 20 infohash + 20 peerid + 0x13 (proto identifier) + 8 reserved values
	buf := make([]byte, len(h.Pstr)+49)
	buf[0] = byte(len(h.Pstr))

	curr := 1
	curr = copy(buf[curr:], h.Pstr)
	// update if you want to extend
	// https://www.bittorrent.org/beps/bep_0010.html
	curr = copy(buf[curr:], make([]byte, 8))
	curr = copy(buf[curr:], h.InfoHash[:])
	curr = copy(buf[curr:], h.PeerID[:])

	return buf
}

func Open(r io.Reader) (*Handshake, error) {
	lengthBuf := make([]byte, 1)
	if _, err := io.ReadFull(r, lengthBuf); err != nil {
		return nil, err
	}

	pstrLen := int(lengthBuf[0])
	if pstrLen == 0 {
		return nil, fmt.Errorf("pstrlen cannot be 0")
	}

	handshakeBuf := make([]byte, pstrLen+48)
	if _, err := io.ReadFull(r, handshakeBuf); err != nil {
		return nil, err
	}

	var infoHash, peerID [20]byte
	copy(infoHash[:], handshakeBuf[pstrLen+8:pstrLen+8+20])
	copy(peerID[:], handshakeBuf[pstrLen+8+20:])

	hs := New(infoHash, peerID)
	return &hs, nil
}
