// https://kasvith.me/posts/lets-create-a-simple-lb-go/
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	Attempt int = iota
	Retry
)

var serverPool ServerPool

type Backend struct {
	url          *url.URL
	alive        bool
	mu           *sync.RWMutex
	reverseProxy *httputil.ReverseProxy
}

func (b *Backend) IsAlive() bool {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.alive
}

func (b *Backend) SetAlive(alive bool) {
	b.mu.Lock()
	b.alive = alive
	b.mu.Unlock()
}

type ServerPool struct {
	backends []*Backend
	// active index
	active uint32
}

func (s *ServerPool) AddBackend(be *Backend) {
	s.backends = append(s.backends, be)
}

func (s *ServerPool) NextIndex() int {
	return int(atomic.AddUint32(&s.active, 1) % uint32(len(s.backends)))
}

func (s *ServerPool) MarkBackendStatus(backendUrl *url.URL, alive bool) {
	for _, be := range s.backends {
		if be.url.String() != backendUrl.String() {
			continue
		}
		be.SetAlive(alive)
	}
}

func (s *ServerPool) GetNextPeer() *Backend {
	next, beLen := s.NextIndex(), len(s.backends)
	l := next + beLen

	for i := next; i < l; i++ {
		idx := i % beLen

		if s.backends[idx].IsAlive() {
			if i != next {
				atomic.StoreUint32(&s.active, uint32(idx))
			}
			return s.backends[idx]
		}
	}
	return nil
}

func (s *ServerPool) HealthCheck() {
	for _, b := range s.backends {
		status := "up"
		alive := isBackendAlive(b.url)
		b.SetAlive(alive)
		if !alive {
			status = "down"
		}
		log.Printf("%s [%s]\n", b.url, status)
	}
}

func isBackendAlive(url *url.URL) bool {
	conn, err := net.DialTimeout("tcp", url.String(), 2*time.Second)
	if err != nil {
		log.Printf("site unreachable to %s, err: %v", url.String(), err)
		return false
	}
	defer conn.Close()
	return true
}

func getAttemptFromRequest(r *http.Request) int {
	if retry, ok := r.Context().Value(Retry).(int); ok {
		return retry
	}
	return 1
}

func getRetryFromRequest(r *http.Request) int {
	if attempt, ok := r.Context().Value(Attempt).(int); ok {
		return attempt
	}
	return 0
}

func loadbalance(w http.ResponseWriter, r *http.Request) {
	attempt := getAttemptFromRequest(r)
	if attempt > 3 {
		log.Printf("%s(%s) Max attempt reached, terminated\n", r.RemoteAddr, r.URL.Path)
		http.Error(w, "Maximum attempt reached", http.StatusServiceUnavailable)
		return
	}

	peer := serverPool.GetNextPeer()
	if peer == nil {
		http.Error(w, "Service not available", http.StatusServiceUnavailable)
		return
	}

	peer.reverseProxy.ServeHTTP(w, r)
}
func healthCheck() {
	t := time.NewTicker(2 * time.Minute)
	defer t.Stop()
	for range t.C {
		log.Println("Starting health check")
		serverPool.HealthCheck()
		log.Println("Completed health check")
	}
}

func main() {
	var serverList string
	var port int
	flag.StringVar(&serverList, "backends", "", "Load balancer backends. Use comma to seperate")
	flag.IntVar(&port, "port", 3030, "Ports to serve ")
	flag.Parse()

	urls := strings.Split(serverList, ",")
	for _, urlStr := range urls {
		serverUrl, err := url.Parse(urlStr)
		if err != nil {
			log.Fatal(err)
		}

		proxy := httputil.NewSingleHostReverseProxy(serverUrl)
		proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("[%s] errored: %v", r.Host, err)
			retries := getRetryFromRequest(r)
			if retries < 3 {
				time.Sleep(10 * time.Millisecond)
				ctx := context.WithValue(r.Context(), Retry, retries+1)
				proxy.ServeHTTP(w, r.WithContext(ctx))
				return
			}

			serverPool.MarkBackendStatus(serverUrl, false)

			attempt := getAttemptFromRequest(r)
			log.Printf("%s(%s) Attempt retry %d\n", r.RemoteAddr, r.URL.Path, attempt)
			ctx := context.WithValue(r.Context(), Attempt, attempt+1)
			loadbalance(w, r.WithContext(ctx))
		}

		serverPool.AddBackend(&Backend{
			url:          serverUrl,
			alive:        true,
			reverseProxy: proxy,
		})
		log.Printf("Configure server: %s\n", serverUrl)
	}

	server := http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: http.HandlerFunc(loadbalance),
	}

	go healthCheck()

	log.Printf("Load balancer started at: %d", port)
	if err := server.ListenAndServe(); err != nil {
		log.Fatalln(err)
	}
}
