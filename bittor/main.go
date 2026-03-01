// bittorrent implementaion
// https://blog.jse.li/posts/torrent/
// maybe base for future project idea: `give me that`
package main

import (
	"bittor/torfile"
	"log"
	"os"
)

func main() {
	inPath, outPath := os.Args[1], os.Args[2]
	log.Println("in path:", inPath, "out path:", outPath)

	tf, err := torfile.Read(inPath)
	if err != nil {
		log.Fatal(err)
	}

	if err = tf.Download(outPath); err != nil {
		log.Fatal(err)
	}
}
