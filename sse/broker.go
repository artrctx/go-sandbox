package main

import (
	"fmt"
	"net/http"
	"strings"
	"sync"
)

type EventType = string

const (
	ERROR_EVENT EventType = "error"
	MSG_EVENT   EventType = "message"
)

type Event struct {
	ID    string    `json:"id"`
	Event EventType `json:"event"`
	Data  string    `json:"data"`
	Retry int       `json:"retry"`
}

func (e Event) format() string {
	var s strings.Builder
	if e.ID != "" {
		s.WriteString(fmt.Sprintf("id: %s\n", e.ID))
	}

	if e.Event != "" {
		s.WriteString(fmt.Sprintf("event: %s\n", e.Event))
	}

	if e.Retry > 0 {
		s.WriteString(fmt.Sprintf("retry: %d\n", e.Retry))
	}

	s.WriteString(fmt.Sprintf("data: %s\n\n", e.Data))

	return s.String()
}

type Broker struct {
	clients map[chan Event]struct{}
	mu      sync.RWMutex
}

func NewBroker() Broker {
	return Broker{
		clients: make(map[chan Event]struct{}),
	}
}

func (b *Broker) Sub() chan Event {
	ch := make(chan Event)
	b.mu.Lock()
	b.clients[ch] = struct{}{}
	b.mu.Unlock()
	return ch
}

func (b *Broker) UnSub(ch chan Event) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.clients, ch)
}

func (b *Broker) Publish(evt Event) {
	b.mu.Lock()
	defer b.mu.Unlock()
	for ch := range b.clients {
		// Non blocking channel send
		select {
		case ch <- evt:
		default:
		}
	}
}

func (b *Broker) registerEventRouteFunc(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streamining is not supported", http.StatusInternalServerError)
		return
	}

	ch := b.Sub()
	defer b.UnSub(ch)

	for {
		select {
		case evt := <-ch:
			fmt.Fprint(w, evt.format())
			flusher.Flush()
		case <-r.Context().Done():
			return
		}
	}
}
