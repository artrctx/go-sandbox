package main

import (
	"bytes"
	"log"
	"os"
	"scrapper/pdl"

	"github.com/twpayne/go-kmz"
)

// Name | Address | Business Type | Description (if any) | Website
// https://apps.ilsos.gov/businessentitysearch/businessentitysearch
// https://www.google.com/maps/d/u/0/viewer?mid=18EFqHj_e5KFy_iknwP8aGpMp7JQGDyc&ll=41.96780775781806%2C-87.71260880628354&z=16
// https://developers.google.com/kml/documentation/kmlreference
func main() {
	kmz := kmz.NewKMZ(
		pdl.GetFromList()...,
	)

	w := &bytes.Buffer{}
	if err := kmz.WriteIndent(w, "", "\t"); err != nil {
		log.Fatal(err)
	}

	os.WriteFile("./scraper/test_gen.kmz", w.Bytes(), 0644)

	pdl.GetFromList()
}
