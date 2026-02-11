package geocode

import (
	"context"
	"fmt"
	"log"
	"os"

	_ "github.com/joho/godotenv/autoload"
	"googlemaps.github.io/maps"
)

type Loc struct {
	Lat float64
	Lng float64
}

var client *maps.Client

func GetLocFromAddr(addr string) Loc {
	if client == nil {
		fmt.Println(os.Getenv("GOOGLE_API_KEY"))
		c, err := maps.NewClient(maps.WithAPIKey(os.Getenv("GOOGLE_API_KEY")))
		if err != nil {
			log.Panicf("failed to create maps client %v", err)
		}
		client = c
	}

	req := &maps.GeocodingRequest{
		Address: addr,
	}

	result, err := client.Geocode(context.Background(), req)

	if err != nil {
		log.Panicf("failed to get geo code with err: %v", err)
	}

	if len(result) == 0 {
		return Loc{41.881832, -87.623177}
	}
	return Loc{result[0].Geometry.Location.Lat, result[0].Geometry.Location.Lng}
}
