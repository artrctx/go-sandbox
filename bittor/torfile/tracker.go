package torfile

import (
	"bittor/peer"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/jackpal/bencode-go"
)

type bencodeTrackerResp struct {
	Interval int    `bencode:"interval"`
	Peers    string `bencode:"peers"`
}

/*
peer_id: A 20 byte name to identify ourselves to trackers and peers.
We’ll just generate 20 random bytes for this.
Real BitTorrent clients have IDs like -TR2940-k8hj0wgej6ch which identify the client software and version—in this case,
TR2940 stands for Transmission client 2.94.
*/
func (f *File) buildTrackerURL(peerID [20]byte, port uint16) (string, error) {
	base, err := url.Parse(f.Announce)
	if err != nil {
		return "", err
	}
	params := url.Values{
		"info_hash":  []string{string(f.InfoHash[:])},
		"peer_id":    []string{string(peerID[:])},
		"port":       []string{strconv.Itoa(int(port))},
		"uploaded":   []string{"0"},
		"downloaded": []string{"0"},
		"compact":    []string{"1"},
		"left":       []string{strconv.Itoa(f.Length)},
	}

	base.RawQuery = params.Encode()
	return base.String(), nil
}

func (f *File) requestPeers(peerID [20]byte, port uint16) ([]peer.Peer, error) {
	url, err := f.buildTrackerURL(peerID, port)
	if err != nil {
		return nil, err
	}

	c := &http.Client{Timeout: 15 * time.Second}
	resp, err := c.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var btResp bencodeTrackerResp
	if err := bencode.Unmarshal(resp.Body, &btResp); err != nil {
		return nil, err
	}

	return peer.Unmarshal([]byte(btResp.Peers))
}
