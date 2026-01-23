package main

import (
	"encoding/hex"
	"fmt"
	"sha1/sha1"
)

func main() {
	fmt.Printf("hash: %v", hex.EncodeToString(sha1.Hash([]byte("THISHASH"))))
}
