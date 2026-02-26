package main

import (
	"log"
	"net/http"
)

func main() {
	brkr := NewBroker()

	startMessaging(&brkr)

	http.Handle("/", http.FileServer(http.Dir("./sse/static")))
	http.HandleFunc("/events", brkr.registerEventRouteFunc)

	log.Println("listening on http://localhost:8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Failed to serve with err: %v", err)
	}
}
