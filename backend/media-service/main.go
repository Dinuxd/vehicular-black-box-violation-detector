package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

type EventMedia struct {
	EvidenceID string `json:"evidence_id"`
	Type       string `json:"type"`
	SizeBytes  int64  `json:"size_bytes"`
}

type PrepareUploadsRequest struct {
	EventID string       `json:"event_id"`
	Media   []EventMedia `json:"media"`
}

type PrepareUploadsResponse struct {
	EventID    string            `json:"event_id"`
	UploadURLs map[string]string `json:"upload_urls"`
}

type MediaVerifyRequest struct {
	EventID   string   `json:"event_id"`
	Evidences []string `json:"evidences"`
}

type MediaVerifyResponse struct {
	EventID string `json:"event_id"`
	Status  string `json:"status"` // OK or ERROR
}

type UploadResponse struct {
	EventID    string `json:"event_id"`
	EvidenceID string `json:"evidence_id"`
	Status     string `json:"status"`
	Bytes      int64  `json:"bytes"`
	Path       string `json:"path"`
}

type PreparedMedia struct {
	Type         string
	ExpectedSize int64
}

type UploadedMedia struct {
	Path       string
	ActualSize int64
	UploadedAt time.Time
}

var (
	mu            sync.RWMutex
	prepared      = map[string]map[string]PreparedMedia{}
	uploaded      = map[string]map[string]UploadedMedia{}
	storageDir    string
	publicBaseURL string
)

func main() {
	port := getenv("MEDIA_PORT", "8081")
	storageDir = getenv("MEDIA_STORAGE_DIR", "uploads")
	publicBaseURL = strings.TrimRight(getenv("MEDIA_PUBLIC_BASE_URL", "http://localhost:"+port), "/")

	if err := os.MkdirAll(storageDir, 0755); err != nil {
		log.Fatalf("failed to create storage dir: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/media/prepare-uploads", prepareUploadsHandler)
	mux.HandleFunc("/media/verify-and-register", verifyAndRegisterHandler)
	mux.HandleFunc("/media/upload/", uploadHandler)

	log.Printf("Media service listening on :%s", port)
	if err := http.ListenAndServe(":"+port, corsMiddleware(mux)); err != nil {
		log.Fatal(err)
	}
}

func getenv(key, fallback string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	return v
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET,POST,PUT,OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("OK"))
}

// POST /media/prepare-uploads
func prepareUploadsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req PrepareUploadsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}

	req.EventID = strings.TrimSpace(req.EventID)
	if req.EventID == "" {
		http.Error(w, "event_id is required", http.StatusBadRequest)
		return
	}

	uploadURLs := make(map[string]string, len(req.Media))

	mu.Lock()
	defer mu.Unlock()

	if prepared[req.EventID] == nil {
		prepared[req.EventID] = map[string]PreparedMedia{}
	}
	if uploaded[req.EventID] == nil {
		uploaded[req.EventID] = map[string]UploadedMedia{}
	}

	for _, m := range req.Media {
		evidenceID := strings.TrimSpace(m.EvidenceID)
		if evidenceID == "" {
			http.Error(w, "media evidence_id is required", http.StatusBadRequest)
			return
		}

		prepared[req.EventID][evidenceID] = PreparedMedia{
			Type:         m.Type,
			ExpectedSize: m.SizeBytes,
		}

		uploadURLs[evidenceID] = publicBaseURL +
			"/media/upload/" +
			url.PathEscape(req.EventID) +
			"/" +
			url.PathEscape(evidenceID)
	}

	writeJSON(w, http.StatusOK, PrepareUploadsResponse{
		EventID:    req.EventID,
		UploadURLs: uploadURLs,
	})
}

// PUT /media/upload/{eventID}/{evidenceID}
func uploadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut && r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	rest := strings.TrimPrefix(r.URL.Path, "/media/upload/")
	parts := strings.SplitN(rest, "/", 2)
	if len(parts) != 2 {
		http.NotFound(w, r)
		return
	}

	eventID, err1 := url.PathUnescape(parts[0])
	evidenceID, err2 := url.PathUnescape(parts[1])
	if err1 != nil || err2 != nil || eventID == "" || evidenceID == "" {
		http.Error(w, "bad upload path", http.StatusBadRequest)
		return
	}

	mu.RLock()
	_, eventPrepared := prepared[eventID]
	_, evidencePrepared := prepared[eventID][evidenceID]
	mu.RUnlock()

	if !eventPrepared || !evidencePrepared {
		http.Error(w, "upload not prepared for this event/evidence", http.StatusNotFound)
		return
	}

	eventDir := filepath.Join(storageDir, safeName(eventID))
	if err := os.MkdirAll(eventDir, 0755); err != nil {
		http.Error(w, "failed to create event dir", http.StatusInternalServerError)
		return
	}

	filePath := filepath.Join(eventDir, safeName(evidenceID))
	f, err := os.Create(filePath)
	if err != nil {
		http.Error(w, "failed to create file", http.StatusInternalServerError)
		return
	}
	defer f.Close()

	n, err := io.Copy(f, r.Body)
	if err != nil {
		http.Error(w, "failed to store upload", http.StatusInternalServerError)
		return
	}

	mu.Lock()
	if uploaded[eventID] == nil {
		uploaded[eventID] = map[string]UploadedMedia{}
	}
	uploaded[eventID][evidenceID] = UploadedMedia{
		Path:       filePath,
		ActualSize: n,
		UploadedAt: time.Now(),
	}
	mu.Unlock()

	writeJSON(w, http.StatusOK, UploadResponse{
		EventID:    eventID,
		EvidenceID: evidenceID,
		Status:     "UPLOADED",
		Bytes:      n,
		Path:       filePath,
	})
}

// POST /media/verify-and-register
func verifyAndRegisterHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req MediaVerifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}

	req.EventID = strings.TrimSpace(req.EventID)
	if req.EventID == "" {
		http.Error(w, "event_id is required", http.StatusBadRequest)
		return
	}

	mu.RLock()
	defer mu.RUnlock()

	preparedForEvent, okPrepared := prepared[req.EventID]
	uploadedForEvent, okUploaded := uploaded[req.EventID]
	if !okPrepared || !okUploaded {
		writeJSON(w, http.StatusOK, MediaVerifyResponse{
			EventID: req.EventID,
			Status:  "ERROR",
		})
		return
	}

	for _, evidenceID := range req.Evidences {
		evidenceID = strings.TrimSpace(evidenceID)
		if evidenceID == "" {
			writeJSON(w, http.StatusOK, MediaVerifyResponse{
				EventID: req.EventID,
				Status:  "ERROR",
			})
			return
		}

		if _, ok := preparedForEvent[evidenceID]; !ok {
			writeJSON(w, http.StatusOK, MediaVerifyResponse{
				EventID: req.EventID,
				Status:  "ERROR",
			})
			return
		}

		if _, ok := uploadedForEvent[evidenceID]; !ok {
			writeJSON(w, http.StatusOK, MediaVerifyResponse{
				EventID: req.EventID,
				Status:  "ERROR",
			})
			return
		}
	}

	writeJSON(w, http.StatusOK, MediaVerifyResponse{
		EventID: req.EventID,
		Status:  "OK",
	})
}

func safeName(s string) string {
	replacer := strings.NewReplacer(
		"/", "_",
		"\\", "_",
		":", "_",
		"*", "_",
		"?", "_",
		"\"", "_",
		"<", "_",
		">", "_",
		"|", "_",
	)
	return replacer.Replace(s)
}