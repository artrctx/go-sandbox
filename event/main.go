package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os/signal"
	"syscall"
	"time"
)

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

func serverGen() *http.Server {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /route-1", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		json.NewEncoder(w).Encode(map[string]string{"value": "This is route 1"})
	})

	mux.HandleFunc("POST /route-2", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{"value": "This is route 2"})
	})

	return &http.Server{
		Addr:         ":8080",
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  15 * time.Second,
	}
}

func main() {
	done := make(chan struct{})
	srv := serverGen()

	go gracefulShutdown(srv, done)

	fmt.Printf("Serving on %s\n", srv.Addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		fmt.Printf("failed shutting down server on %s. %v\n", srv.Addr, err)
	} else {
		fmt.Println("Shutdown server on 8080")
	}

	<-done
	fmt.Println("all down shutdown")
}
