package main

import (
	"context"
	"fmt"

	"github.com/artrctx/gossiper/internal/diarization"
	"github.com/joho/godotenv"
)

func main() {
	fmt.Println("Initializing Gossiper...")
	err := godotenv.Load()
	if err != nil {
		fmt.Printf("Error loading .env file: %v\n", err)
	}

	ctx := context.Background()
	diar, err := diarization.New(ctx)
	if err != nil {
		panic(err)
	}
	defer diar.Close()

	fmt.Println("Concluded")
}
