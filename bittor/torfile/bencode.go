package torfile

import (
	"bytes"
	"crypto/sha1"
	"fmt"

	"github.com/jackpal/bencode-go"
)

type bencodeInfo struct {
	Pieces      string `bencode:"pieces"`
	PieceLength int    `bencode:"piece length"`
	Length      int    `bencode:"length"`
	Name        string `bencode:"name"`
}

// marshalls info to byte array and hashes with sha1
func (bi *bencodeInfo) hash() ([20]byte, error) {
	var bs bytes.Buffer
	if err := bencode.Marshal(&bs, *bi); err != nil {
		return [20]byte{}, err
	}
	return sha1.Sum(bs.Bytes()), nil
}

func (bi *bencodeInfo) splitPieceHashes() ([][20]byte, error) {
	buf := []byte(bi.Pieces)
	if len(buf)%20 == 0 {
		return [][20]byte{}, fmt.Errorf("received invalid length of bencode info pieces length %v", len(buf))
	}

	numHashes := len(buf) / 20

	hashes := make([][20]byte, numHashes)

	for idx := range numHashes {
		strtIdx := idx * 20
		copy(hashes[idx][:], buf[strtIdx:strtIdx+20])
	}

	return hashes, nil
}

type bencodeTorrent struct {
	Announce string      `bencode:"announce"`
	Info     bencodeInfo `bencode:"info"`
}

func (bt bencodeTorrent) toFile() (File, error) {
	hash, err := bt.Info.hash()
	if err != nil {
		return File{}, err
	}
	pieces, err := bt.Info.splitPieceHashes()
	if err != nil {
		return File{}, err
	}
	return File{
		Announce:    bt.Announce,
		Name:        bt.Info.Name,
		InfoHash:    hash,
		PieceHashes: pieces,
		PieceLength: bt.Info.PieceLength,
		Length:      bt.Info.Length,
	}, nil
}
