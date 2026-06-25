package dto

import "time"

type EventMedia struct {
	EvidenceID string `json:"evidence_id"`
	Type       string `json:"type"`
	SizeBytes  int64  `json:"size_bytes"`
}

type EventGPSRequest struct {
	Latitude   *float64 `json:"latitude"`
	Longitude  *float64 `json:"longitude"`
	CapturedAt string   `json:"captured_at"`
	AccuracyM  *float64 `json:"accuracy_m,omitempty"`
}

type EventGPSResponse struct {
	Latitude   float64   `json:"latitude"`
	Longitude  float64   `json:"longitude"`
	CapturedAt time.Time `json:"captured_at"`
	AccuracyM  *float64  `json:"accuracy_m,omitempty"`
}

type CreateEventRequest struct {
	EventID   string           `json:"event_id"`
	DeviceID  string           `json:"device_id"`
	Timestamp string           `json:"ts"`
	EventType string           `json:"event_type"`
	Severity  string           `json:"severity"`
	GPS       *EventGPSRequest `json:"gps,omitempty"`
	Media     []EventMedia     `json:"media"` // accepted for backward compatibility; ignored
}

type CreateEventResponse struct {
	EventID    string            `json:"event_id"`
	Status     string            `json:"status"`
	UploadURLs map[string]string `json:"upload_urls"`
}

type FinalizeEventResponse struct {
	EventID string `json:"event_id"`
	Status  string `json:"status"`
}

type EventRow struct {
	Seq       int               `json:"seq"`
	EventID   string            `json:"event_id"`
	TS        time.Time         `json:"ts"`
	EventType string            `json:"event_type"`
	Severity  string            `json:"severity"`
	GPS       *EventGPSResponse `json:"gps,omitempty"`
}

type EventsResponse struct {
	DeviceID  string     `json:"device_id"`
	EventType string     `json:"event_type"`
	Limit     int        `json:"limit"`
	Offset    int        `json:"offset"`
	Total     int        `json:"total"`
	Rows      []EventRow `json:"rows"`
}
