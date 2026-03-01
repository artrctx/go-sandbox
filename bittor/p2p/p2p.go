package p2p

import (
	"bittor/client"
	"bittor/message"
	"bittor/peer"
	"bytes"
	"crypto/sha1"
	"fmt"
	"log"
	"runtime"
	"time"
)

const (
	// largest number of bytes a request can ask for (16KB)
	MaxBlockSize = 16384
	// num of unfulfilled req a client can hasve in its pipeline
	// can be fine tuned to increate download speed
	//TODO: THIS CAN BE DYNAMICALLY SET
	MaxBackLog = 5
)

type Torrent struct {
	Peers       []peer.Peer
	PeerID      [20]byte
	InfoHash    [20]byte
	PieceHashes [][20]byte
	PieceLength int
	Length      int
	Name        string
}

type pieceWork struct {
	index  int
	hash   [20]byte
	length int
}

type pieceResult struct {
	index int
	buf   []byte
}

type pieceProgress struct {
	index      int
	client     *client.Client
	buf        []byte
	downloaded int
	requested  int
	backlog    int
}

// reads message from client and updates state
func (state *pieceProgress) readMessage() error {
	msg, err := state.client.Read()
	if err != nil {
		return err
	}
	// keep-alive
	if msg == nil {
		return nil
	}

	switch msg.ID {
	case message.MsgUnchoke:
		state.client.Choked = false
	case message.MsgChoke:
		state.client.Choked = true
	case message.MsgHave:
		idx, err := message.ParseHave(msg)
		if err != nil {
			return err
		}
		state.client.Bitfield.HasPiece(idx)
	case message.MsgPiece:
		n, err := message.ParsePiece(state.index, state.buf, msg)
		if err != nil {
			return err
		}
		state.downloaded += n
		state.backlog--
	}
	return nil
}

func attemptDownloadPiece(c *client.Client, pw *pieceWork) ([]byte, error) {
	state := pieceProgress{
		index:  pw.index,
		client: c,
		buf:    make([]byte, pw.length),
	}

	// setting a deadline helps get unresponsive peer unstuck
	// 30 seconds should be more then enough to download 262kb piece
	//TODO: THIS CAN BE DYNAMICALLY SET
	c.Conn.SetDeadline(time.Now().Add(30 * time.Second))
	defer c.Conn.SetDeadline(time.Time{})

	for state.downloaded < pw.length {
		if !state.client.Choked {
			// grouping for performance imrpovement
			// batching request
			for state.backlog < MaxBackLog && state.requested < pw.length {
				blockSize := MaxBlockSize
				if pw.length-state.requested < blockSize {
					blockSize = pw.length - state.requested
				}

				if err := c.SendRequest(pw.index, state.requested, blockSize); err != nil {
					return nil, err
				}

				state.backlog++
				state.requested += blockSize
			}
		}
	}

	if err := state.readMessage(); err != nil {
		return nil, err
	}

	return state.buf, nil
}

// checking byt comparing hash in the .torrent
func checkIntegrity(pw *pieceWork, buf []byte) error {
	hash := sha1.Sum(buf)
	if !bytes.Equal(hash[:], pw.hash[:]) {
		return fmt.Errorf("index %d failed integrity check", pw.index)
	}
	return nil
}

func (t *Torrent) startDownloadWorker(peer peer.Peer, workQueue chan *pieceWork, results chan *pieceResult) {
	c, err := client.New(peer, t.PeerID, t.InfoHash)
	if err != nil {
		log.Printf("could not handshake with %s. error: %v. disconnecting\n", peer.IP, err)
		return
	}
	defer c.Conn.Close()
	log.Printf("completed handshake with peer %s", peer.IP)

	c.SendUnchoke()
	c.SendInterested()

	for pw := range workQueue {
		// if doesn't have piece put the work back on the queue to retry
		if !c.Bitfield.HasPiece(pw.index) {
			workQueue <- pw
			return
		}

		buf, err := attemptDownloadPiece(c, pw)
		// when failed put the work back on the queue to retry
		if err != nil {
			log.Println("failed downloading", err)
			workQueue <- pw
			return
		}

		if err = checkIntegrity(pw, buf); err != nil {
			log.Printf("piece %d failed integrity check\n", pw.index)
			workQueue <- pw
			continue
		}

		c.SendHave()
		results <- &pieceResult{pw.index, buf}
	}
}

func (t *Torrent) calculateBoundsForPiece(idx int) (begin, end int) {
	begin = idx * t.PieceLength
	end = begin + t.PieceLength
	if end > t.Length {
		end = t.Length
	}
	return begin, end
}

func (t *Torrent) calculatePieceSize(idx int) int {
	begin, end := t.calculateBoundsForPiece(idx)
	return end - begin
}

func (t *Torrent) Download() ([]byte, error) {
	log.Printf("starting download for %s", t.Name)

	// initialize work queue according to amount of piece hashes
	totalPieces := len(t.PieceHashes)
	workQueue, result := make(chan *pieceWork, totalPieces), make(chan *pieceResult)
	for idx, hash := range t.PieceHashes {
		workQueue <- &pieceWork{idx, hash, t.calculatePieceSize(idx)}
	}

	// start workers
	for _, peer := range t.Peers {
		go t.startDownloadWorker(peer, workQueue, result)
	}

	// collect results into a buffer until full
	// instead of memory might be able to just use file system
	buf := make([]byte, t.Length)
	donePieces := 0
	for donePieces < totalPieces {
		res := <-result
		begin, end := t.calculateBoundsForPiece(res.index)
		copy(buf[begin:end], res.buf)
		donePieces++

		percent := (float64(donePieces) / float64(totalPieces)) * 100
		numWorkers := runtime.NumGoroutine() - 1 // subtract 1 for main thread count
		log.Printf("(%0.2f%%) downloaded piece #%d from #%d peers", percent, res.index, numWorkers)
	}
	close(workQueue)

	return buf, nil
}
