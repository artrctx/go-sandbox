package main

import (
	"encoding/binary"
	"fmt"
)

func main() {
	val := make([]byte, 8)
	binary.BigEndian.PutUint64(val, 0x4f52)
	fmt.Println(val)
	fmt.Printf("-- %d %d\n", 0x4f, 0x52)
}
