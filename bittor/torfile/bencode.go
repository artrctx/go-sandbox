package torfile

import (
	"bytes"
	"crypto/sha1"
	"fmt"
	"time"

	"github.com/jackpal/bencode-go"
)

type bencodeInfo struct {
	Pieces      string `bencode:"pieces"`
	PieceLength int    `bencode:"piece length"`
	Length      int    `bencode:"length"`
	Name        string `bencode:"name"`
}

// marshalls info to byte array and hashes with sha1
func (bi *bencodeInfo) hash() ([PieceHashSize]byte, error) {
	var bs bytes.Buffer
	if err := bencode.Marshal(&bs, *bi); err != nil {
		return [PieceHashSize]byte{}, err
	}
	return sha1.Sum(bs.Bytes()), nil
}

func (bi *bencodeInfo) splitPieceHashes() ([][PieceHashSize]byte, error) {
	buf := []byte(bi.Pieces)
	if len(buf)%PieceHashSize == 0 {
		return [][PieceHashSize]byte{}, fmt.Errorf("received invalid length of bencode info pieces length %v", len(buf))
	}

	numHashes := len(buf) / PieceHashSize

	hashes := make([][PieceHashSize]byte, numHashes)

	for idx := range numHashes {
		strtIdx := idx * PieceHashSize
		copy(hashes[idx][:], buf[strtIdx:strtIdx+PieceHashSize])
	}

	return hashes, nil
}

type bencodeTorrent struct {
	Announce     string      `bencode:"announce"`
	Comment      string      `bencode:"comment"`
	CreationDate time.Time   `bencode:"creation date"`
	Info         bencodeInfo `bencode:"info"`
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
		Comment:     bt.Comment,
		CreatedAt:   bt.CreationDate,
		Name:        bt.Info.Name,
		InfoHash:    hash,
		PieceHashes: pieces,
		PieceLength: bt.Info.PieceLength,
		Length:      bt.Info.Length,
	}, nil
}
