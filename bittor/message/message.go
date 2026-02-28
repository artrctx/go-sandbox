package message

import (
	"encoding/binary"
	"io"
)

type MessageID uint8

// https://www.bittorrent.org/beps/bep_0004.html
// Core Protocol:
// 0x00   choke
// 0x01   unchoke
// 0x02   interested
// 0x03   not interested
// 0x04   have
// 0x05   bitfield
// 0x06   request
// 0x07   piece
// 0x08   cancel

const (
	// MsgChoke chockes the receiver
	MsgChoke MessageID = 0
	// MsgUnchoke unchokes the receiver
	MsgUnchoke MessageID = 1
	// MsgInterested express interest in receiving data
	MsgInterested MessageID = 2
	// MsgNotInterested expresses disinterest in receiving data
	MsgNotInterested MessageID = 3
	// MsgHave alerts the receiver that the sender has doanloded a piece
	MsgHave MessageID = 4
	// MsgBitfield encodes which pieces that the sender has downloaded
	MsgBitfield MessageID = 5
	// MsgRequest request a block of data from the receiver
	MsgRequest MessageID = 6
	// MsgPiece delivers a block of a data to fulfill a request
	MsgPiece MessageID = 7
	// MsgCancel cancels a request
	MsgCancel MessageID = 8
)

type Message struct {
	ID      MessageID
	Payload []byte
}

// Serialize serializes a message into a buffer of the form
// <length prefix><message ID><payload>
// Interprets `nil` as a keep-alive message
func (m *Message) Serialize() []byte {
	if m == nil {
		return []byte{}
	}
	// message id (1) + payload len
	len := uint32(len(m.Payload) + 1)
	// 4 == length prefix
	buf := make([]byte, 4+len)
	binary.BigEndian.AppendUint32(buf[:4], len)
	buf[4] = byte(m.ID)
	copy(buf[5:], m.Payload)
	return buf
}

func Read(r io.Reader) (*Message, error) {
	// length prefix buffer
	lenBuf := make([]byte, 4)
	if _, err := io.ReadFull(r, lenBuf); err != nil {
		return nil, err
	}

	length := binary.BigEndian.Uint32(lenBuf)
	// keep-alive message
	if length == 0 {
		return nil, nil
	}
	msgBuf := make([]byte, length)
	if _, err := io.ReadFull(r, msgBuf); err != nil {
		return nil, err
	}
	return &Message{
		ID:      MessageID(msgBuf[0]),
		Payload: msgBuf[1:],
	}, nil

}
