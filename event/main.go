package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os/signal"
	"sync"
	"syscall"
	"time"
	"uuid"

	"github.com/twmb/franz-go/pkg/kgo"
)

const TOPIC_NAME string = "test-topic"

func gracefulShutdown(srv *http.Server, done chan<- struct{}) {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	<-ctx.Done()

	slog.InfoContext(ctx, "shutting down. press Ctrl+C again to force quit")
	stop()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		slog.ErrorContext(ctx, "Server forced to shutdown with error", slog.Any("error", err))
	}

	slog.Info("Server exiting")
	done <- struct{}{}
}

func serverGen(client *kgo.Client) *http.Server {
	mux := http.NewServeMux()

	mux.HandleFunc("/produce", func(w http.ResponseWriter, r *http.Request) {
		newID := uuid.New().String()
		var wg sync.WaitGroup
		wg.Add(1)
		record := &kgo.Record{Topic: TOPIC_NAME, Value: fmt.Appendf(nil, "new event with id: %s", newID)}
		var produceErr error
		client.Produce(r.Context(), record, func(_ *kgo.Record, err error) {
			defer wg.Done()
			if err == nil {
				return
			}
			produceErr = err
		})
		wg.Wait()

		w.Header().Add("Content-Type", "application/json")
		res := map[string]string{"eventID": newID}
		if produceErr == nil {
			w.WriteHeader(http.StatusOK)
			res["message"] = "succesful"
		} else {
			w.WriteHeader(http.StatusInternalServerError)
			res["message"] = fmt.Sprintf("error: %v", produceErr)
		}

		json.NewEncoder(w).Encode(res)
	})

	return &http.Server{
		Addr:         ":8000",
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  15 * time.Second,
	}
}

func eventConsumer(client *kgo.Client, done chan<- struct{}) {
	ctx := context.Background()
	for {
		fetches := client.PollFetches(ctx)
		if errs := fetches.Errors(); len(errs) > 0 {
			for idx, err := range errs {
				fmt.Printf("Consumer error %d: %v", idx, err)
			}
			break
		}

		iter := fetches.RecordIter()
		for !iter.Done() {
			record := iter.Next()
			fmt.Printf("Record Received: %v\n", string(record.Value))
		}
	}
	done <- struct{}{}
}

func main() {
	seeds := []string{"localhost:9092"}
	cl, err := kgo.NewClient(
		kgo.SeedBrokers(seeds...),
		kgo.ConsumeTopics(TOPIC_NAME),
	)

	if err != nil {
		panic(err)
	}
	defer cl.Close()

	done := make(chan struct{})
	srv := serverGen(cl)

	go eventConsumer(cl, done)
	go gracefulShutdown(srv, done)

	fmt.Printf("Serving on %s\n", srv.Addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		fmt.Printf("failed shutting down server on %s. %v\n", srv.Addr, err)
	} else {
		fmt.Println("Shutdown server on 8000")
	}

	<-done
	fmt.Println("all down shutdown")
}
