// People Data Lab
package pdl

import (
	"encoding/json"
	"log"
	"os"
)

type Loc struct {
	Street1    string `json:"street_address"`
	Street2    string `json:"address_line_2"`
	PostalCode string `json:"postal_code"`
	State      string `json:"locality"`
	Country    string `json:"country"`
}

type Place struct {
	Name        string `json:"name"`
	DisplayName string `json:"display_name"`
	Founded     int    `json:"founded"`
	Industry    string `json:"industry"`
	IndustryV2  string `json:"industry_v2"`
	Location    Loc    `json:"location"`
	Website     string `json:"website"`
	Summary     string `json:"summary"`
	Headline    string `json:"headline"`
}

type LocList struct {
	Data []Place `json:"data"`
}

// func GetFromList() []*kml.CompoundElement {
func GetFromList() {
	lst, err := os.ReadFile("./scraper/pdl/list.json")
	if err != nil {
		log.Fatalf("Failed to read list; %v", err)
	}
	var locList LocList
	if err := json.Unmarshal(lst, &locList); err != nil {
		log.Fatalf("failed to unmarshall with err: %v", err)
	}

	// var kmlMap map[string][]*kml.CompoundElement
	// for _, p := range locList.Data {
	// 	//
	// }

	// els := []*kml.CompoundElement{}
	// for k, v := range els {
	// 	//
	// }
}
