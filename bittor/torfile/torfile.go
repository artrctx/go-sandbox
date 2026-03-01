package torfile

import (
	"bittor/p2p"
	"crypto/rand"
	"os"

	"github.com/jackpal/bencode-go"
)

// Port to listen on
const Port uint16 = 6881

type File struct {
	Announce    string
	InfoHash    [20]byte
	PieceHashes [][20]byte
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

	bt := bencodeTorrent{}
	if err := bencode.Unmarshal(file, &bt); err != nil {
		return File{}, err
	}

	return bt.toFile()
}

func (f *File) Download(path string) error {
	var peerID [20]byte
	// This never returns error
	rand.Read(peerID[:])

	//? PORT NEEDS TO BE DYNAMIC
	peers, err := f.requestPeers(peerID, Port)
	if err != nil {
		return err
	}

	tor := p2p.Torrent{
		Peers:       peers,
		PeerID:      peerID,
		InfoHash:    f.InfoHash,
		PieceHashes: f.PieceHashes,
		PieceLength: f.PieceLength,
		Length:      f.Length,
		Name:        f.Name,
	}
	buf, err := tor.Download()
	if err != nil {
		return err
	}

	outFile, err := os.Create(path)
	if err != nil {
		return err
	}
	defer outFile.Close()

	if _, err := outFile.Write(buf); err != nil {
		return err
	}
	return nil
}
