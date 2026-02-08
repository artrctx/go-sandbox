package main

import (
	"fmt"
	"log"
	"os"

	"github.com/twpayne/go-kmz"
)

// Name | Address | Business Type | Description (if any) | Website
// https://apps.ilsos.gov/businessentitysearch/businessentitysearch
func main() {
	kmz := kmz.NewKMZ()
	file, err := os.ReadFile("./scrapper/test.kmz")
	if err != nil {
		log.Panicf("failed toread test file: %v", err)
	}

	upedKmz := kmz.AddFile("test", file)

	fmt.Println(upedKmz)
}
