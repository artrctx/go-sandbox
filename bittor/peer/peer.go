package peer

import (
	"encoding/binary"
	"fmt"
	"net"
	"strconv"
)

const PeerBinSize = 6

type Peer struct {
	IP   net.IP
	Port uint16
}

func Unmarshal(peersBin []byte) ([]Peer, error) {
	if len(peersBin)%PeerBinSize != 0 {
		return []Peer{}, fmt.Errorf("invalid peer binary length expected: mod %v, got: %v", PeerBinSize, len(peersBin))
	}

	peerCount := len(peersBin) / PeerBinSize
	peers := make([]Peer, peerCount)
	for i := range peerCount {
		offset := i * PeerBinSize
		peers[i].IP = net.IP(peersBin[offset : offset+4])
		peers[i].Port = binary.BigEndian.Uint16(peersBin[offset+4 : offset+PeerBinSize])
	}
	return peers, nil
}

func (p Peer) String() string {
	return net.JoinHostPort(p.IP.String(), strconv.Itoa(int(p.Port)))
}
