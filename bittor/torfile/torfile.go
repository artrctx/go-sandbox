package torfile

import (
	"crypto/rand"
	"os"
	"time"

	"github.com/jackpal/bencode-go"
)

const (
	PieceHashSize        = 20
	Port          uint16 = 6881
)

type File struct {
	Announce    string
	Comment     string
	CreatedAt   time.Time
	InfoHash    [PieceHashSize]byte
	PieceHashes [][PieceHashSize]byte
	PieceLength int
	Length      int
	Name        string
}

func Read(path string) (File, error) {
	file, err := os.Open(path)
	if err != nil {
		return File{}, err
	}
	defer file.Close()

	var bt bencodeTorrent
	if err := bencode.Unmarshal(file, &bt); err != nil {
		return File{}, err
	}

	return bt.toFile()
}

func (f *File) Download(path string) error {
	var peerID [PieceHashSize]byte
	// This never returns error
	rand.Read(peerID[:])

	return nil
}
