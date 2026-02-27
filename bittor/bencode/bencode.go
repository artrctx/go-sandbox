package bencode

import (
	"io"
	"time"

	"github.com/jackpal/bencode-go"
)

/*
d
  8:announce
    41:http://bttracker.debian.org:6969/announce
  7:comment
    35:"Debian CD from cdimage.debian.org"
  13:creation date
    i1573903810e
  4:info
    d
      6:length
        i351272960e
      4:name
        31:debian-10.2.0-amd64-netinst.iso
      12:piece length
        i262144e
      6:pieces
        26800:�����PS�^�� (binary blob of the hashes of each piece)
    e
e
*/

type Info struct {
	Pieces      string `bencode:"pieces"`
	PieceLength int    `bencode:"piece length"`
	Length      int    `bencode:"length"`
	Name        string `bencode:"name"`
}

type Torrent struct {
	Announce     string    `bencode:"announce"`
	Comment      string    `bencode:"comment"`
	CreationDate time.Time `bencode:"creation date"`
	Info         Info      `bencode:"info"`
}

func Open(r io.Reader) (*Torrent, error) {
	var tor Torrent
	if err := bencode.Unmarshal(r, &tor); err != nil {
		return nil, err
	}
	return &tor, nil
}
