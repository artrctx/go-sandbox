// People Data Lab
package pdl

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"scrapper/geocode"

	"github.com/twpayne/go-kml"
)

type Loc struct {
	Street1    string `json:"street_address"`
	Street2    string `json:"address_line_2"`
	PostalCode string `json:"postal_code"`
	State      string `json:"locality"`
	Country    string `json:"country"`
}

type Place struct {
	Name        string  `json:"name"`
	DisplayName string  `json:"display_name"`
	Founded     int     `json:"founded"`
	Industry    string  `json:"industry"`
	IndustryV2  string  `json:"industry_v2"`
	Location    Loc     `json:"location"`
	Website     string  `json:"website"`
	Summary     string  `json:"summary"`
	Headline    string  `json:"headline"`
	Lat         float64 `json:"lat"`
	Lng         float64 `json:"lng"`
}

func (p *Place) toCompundElement() kml.Element {
	return kml.Placemark(
		kml.Name(p.Name),
		kml.Description(p.Summary),
		kml.Point(
			kml.Coordinates(kml.Coordinate{Lon: p.Lng, Lat: p.Lat}),
		),
	)
}

type LocList struct {
	Data []Place `json:"data"`
}

func readFromFile(dir string) LocList {
	lst, err := os.ReadFile(dir)
	if err != nil {
		log.Fatalf("Failed to read list; %v", err)
	}
	var locList LocList
	if err := json.Unmarshal(lst, &locList); err != nil {
		log.Fatalf("failed to unmarshall with err: %v", err)
	}
	return locList
}

func readFromCached(dir string) LocList {
	cachedDir := "./scraper/pdl/cahced.json"
	_, err := os.Stat(cachedDir)
	if err == nil {
		return readFromFile(cachedDir)
	}

	readLoc := readFromFile(dir)
	for idx, p := range readLoc.Data {
		loc := geocode.GetLocFromAddr(fmt.Sprintf("%s,%s,%s %s", p.Location.Street1, p.Location.State, p.Location.PostalCode, p.Location.Country))

		readLoc.Data[idx].Lat = loc.Lat
		readLoc.Data[idx].Lng = loc.Lng
	}

	cJson, err := json.Marshal(readLoc)
	if err != nil {
		panic(err)
	}

	if err := os.WriteFile(cachedDir, cJson, 0644); err != nil {
		panic(err)
	}

	return readLoc
}

func GetFromList() []kml.Element {
	locList := readFromCached("./scraper/pdl/list.json")

	kmlMap := map[string][]kml.Element{}
	for _, p := range locList.Data {
		ce, ok := kmlMap[p.Industry]
		if !ok {
			kmlMap[p.Industry] = []kml.Element{p.toCompundElement()}
		}
		kmlMap[p.Industry] = append(ce, p.toCompundElement())
	}

	els := []kml.Element{}
	for k, v := range kmlMap {
		compEls := make([]kml.Element, len(v)+1)
		compEls = append(compEls, kml.Name(k))

		for _, e := range v {
			compEls = append(compEls, e)
		}

		els = append(els, kml.Folder(v...))
	}

	return els
}
