package message

import (
	"encoding/binary"
	"fmt"
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

// Creates Request Msg
func FormatRequest(idx, begin, length int) Message {
	// 4 byte idx + 4 byte begin + 4 byte length
	payload := make([]byte, 12)
	binary.BigEndian.AppendUint32(payload[:4], uint32(idx))
	binary.BigEndian.AppendUint32(payload[4:8], uint32(begin))
	binary.BigEndian.AppendUint32(payload[8:], uint32(length))
	return Message{MsgRequest, payload}
}

// Creates Have Msg
func FormatHave(idx int) Message {
	payload := make([]byte, 4)
	binary.BigEndian.AppendUint32(payload, uint32(idx))
	return Message{MsgHave, payload}
}

// parse Piece message and copy payload to buf
func ParsePiece(idx int, buf []byte, msg *Message) (int, error) {
	if msg.ID != MsgPiece {
		return 0, fmt.Errorf("expected Piece ID (%d) but got ID %v", MsgPiece, msg.ID)
	}

	if len(msg.Payload) < 8 {
		return 0, fmt.Errorf("payload too short. %d < 8", len(msg.Payload))
	}

	parsedIdx := int(binary.BigEndian.Uint32(msg.Payload[:4]))
	if parsedIdx != idx {
		return 0, fmt.Errorf("expected index %d, got %d", idx, parsedIdx)
	}
	begin := int(binary.BigEndian.Uint32(msg.Payload[4:8]))
	if begin >= len(buf) {
		return 0, fmt.Errorf("begin offset too high. %d >= %d", begin, len(buf))
	}
	data := msg.Payload[8:]
	if (len(data) + begin) > len(buf) {
		return 0, fmt.Errorf("Data too long [%d] for offset %d with length %d", len(data), begin, len(buf))
	}
	copy(buf[begin:], msg.Payload[8:])
	return len(data), nil
}

// parses Have message and return index
func ParseHave(msg *Message) (int, error) {
	if msg.ID != MsgHave {
		return 0, fmt.Errorf("expected Have ID (%d) but got ID %v", MsgHave, msg.ID)
	}
	if len(msg.Payload) != 4 {
		return 0, fmt.Errorf("xpected payload length 4 but got length %d", len(msg.Payload))
	}
	return int(binary.BigEndian.Uint32(msg.Payload)), nil
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

func (m *Message) name() string {
	if m == nil {
		return "KeepAlive"
	}
	switch m.ID {
	case MsgChoke:
		return "Choke"
	case MsgUnchoke:
		return "Unchoke"
	case MsgInterested:
		return "Interested"
	case MsgNotInterested:
		return "NotInterested"
	case MsgHave:
		return "Have"
	case MsgBitfield:
		return "Bitfield"
	case MsgRequest:
		return "Request"
	case MsgPiece:
		return "Piece"
	case MsgCancel:
		return "Cancel"
	default:
		return fmt.Sprintf("Unknown%d", m.ID)
	}
}

func (m *Message) String() string {
	if m == nil {
		return m.name()
	}
	return fmt.Sprintf("%s [%d]", m.name(), len(m.Payload))
}
