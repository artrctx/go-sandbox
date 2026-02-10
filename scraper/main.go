package main

import (
	"bytes"
	"log"
	"os"
	"scrapper/pdl"

	"github.com/twpayne/go-kml"
	"github.com/twpayne/go-kmz"
)

// Name | Address | Business Type | Description (if any) | Website
// https://apps.ilsos.gov/businessentitysearch/businessentitysearch
// https://www.google.com/maps/d/u/0/viewer?mid=18EFqHj_e5KFy_iknwP8aGpMp7JQGDyc&ll=41.96780775781806%2C-87.71260880628354&z=16
// https://developers.google.com/kml/documentation/kmlreference
func main() {
	kmz := kmz.NewKMZ(
		kml.Folder(
			kml.Name("this is folder"),
			kml.Placemark(
				kml.Name("Simple placemark"),
				kml.Description("Attached to the ground. Intelligently places itself at the height of the underlying terrain."),
				kml.Point(
					kml.Coordinates(kml.Coordinate{Lon: -122.0822035425683, Lat: 37.42228990140251}),
				),
			),
		),
		kml.Folder(
			kml.Name("this is folder2"),
			kml.Placemark(
				kml.Name("Simple placemark"),
				kml.Description("Attached to the ground. Intelligently places itself at the height of the underlying terrain."),
				kml.Point(
					kml.Coordinates(kml.Coordinate{Lon: -122.0822035425683, Lat: 37.42228990140251}),
				),
			),
		),
	)

	w := &bytes.Buffer{}
	if err := kmz.WriteIndent(w, "", "\t"); err != nil {
		log.Fatal(err)
	}

	os.WriteFile("test_gen.kmz", w.Bytes(), 0644)

	pdl.GetFromList()
}
