package main

import (
	"fmt"
	"math/rand"
	"time"
)

var msgs = []string{
	"It is the time you have wasted for your rose that makes your rose so important",
	"You become responsible, forever, for what you have tamed",
	"If you love a flower that lives on a star, it is sweet to look at the sky at night",
	"To me, you will be unique in all the world. To you, I shall be unique in all the world",
}

func startMessaging(b *Broker) {
	var count uint = 0
	// 2 second timer
	ticker := time.NewTicker(2 * time.Second)
	go func() {
		for range ticker.C {
			count++
			msgToSend := msgs[rand.Intn(len(msgs))]
			b.Publish(Event{
				ID:    fmt.Sprintf("%d", count),
				Event: MSG_EVENT,
				Data:  msgToSend,
			})
		}
	}()
}
