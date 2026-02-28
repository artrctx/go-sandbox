package client

import (
	"bittor/bitfield"
	"bittor/handshake"
	"bittor/message"
	"bittor/peer"
	"bytes"
	"fmt"
	"net"
	"time"
)

// TCP connection with peer
type Client struct {
	Conn     net.Conn
	Choked   bool
	BitField bitfield.Bitfield
	peer     peer.Peer
	infoHash [20]byte
	peerID   [20]byte
}

func completeHandshake(conn net.Conn, infoHash, peerID [20]byte) (*handshake.Handshake, error) {
	conn.SetDeadline(time.Now().Add(3 * time.Second))
	defer conn.SetDeadline(time.Time{}) // reset deadline

	req := handshake.New(infoHash, peerID)
	if _, err := conn.Write(req.Serialize()); err != nil {
		return nil, err
	}

	res, err := handshake.Read(conn)
	if err != nil {
		return nil, err
	}
	if !bytes.Equal(res.InfoHash[:], infoHash[:]) {
		return nil, fmt.Errorf("res infohash mismatch expected: %x got: %x", infoHash, res.InfoHash)
	}

	return res, nil
}

func recvBitField(conn net.Conn) (bitfield.Bitfield, error) {
	conn.SetDeadline(time.Now().Add(5 * time.Second))
	defer conn.SetDeadline(time.Time{}) // reset deadline

	msg, err := message.Read(conn)
	if err != nil {
		return nil, err
	}
	if msg.ID != message.MsgBitfield {
		return nil, fmt.Errorf("expected bitfield message but received %d", msg.ID)
	}
	return msg.Payload, nil
}

// New connects with a peer, completes a handshake, and receives a handshake
func New(peer peer.Peer, infoHash, peerID [20]byte) (*Client, error) {
	// Timeout set to 3 seconds
	conn, err := net.DialTimeout("tcp", peer.String(), 3*time.Second)
	if err != nil {
		return nil, err
	}

	if _, err := completeHandshake(conn, infoHash, peerID); err != nil {
		return nil, err
	}

	bf, err := recvBitField(conn)
	if err != nil {
		conn.Close()
		return nil, err
	}

	return &Client{
		Conn:     conn,
		Choked:   true,
		BitField: bf,
		peer:     peer,
		infoHash: infoHash,
		peerID:   peerID,
	}, nil
}

// read and consume message from conn
func (c *Client) Read() (*message.Message, error) {
	return message.Read(c.Conn)
}

// func (c *Client) SendRequest(idx, begin, length int) error {

// }
