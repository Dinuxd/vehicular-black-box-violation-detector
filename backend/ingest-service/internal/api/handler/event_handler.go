package handler

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"ingest-service/internal/api/dto"
	"ingest-service/internal/common"
	"ingest-service/internal/service"
)

type EventHandler struct {
	eventService *service.EventService
}

func NewEventHandler(eventService *service.EventService) *EventHandler {
	return &EventHandler{eventService: eventService}
}

func (h *EventHandler) HandleEvents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request dto.CreateEventRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}

	event, err := h.eventService.CreateEvent(r.Context(), service.CreateEventInput{
		EventID:   request.EventID,
		DeviceID:  request.DeviceID,
		Timestamp: request.Timestamp,
		EventType: request.EventType,
		Severity:  request.Severity,
		GPS:       mapCreateEventGPS(request.GPS),
	})
	if err != nil {
		if errors.Is(err, service.ErrInvalidTimestamp) {
			http.Error(w, "bad timestamp", http.StatusBadRequest)
			return
		}
		if errors.Is(err, service.ErrInvalidGPS) || errors.Is(err, service.ErrInvalidGPSCapturedAt) {
			http.Error(w, "bad gps", http.StatusBadRequest)
			return
		}

		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}

	common.WriteJSON(w, http.StatusCreated, dto.CreateEventResponse{
		EventID:    event.ID,
		Status:     event.Status,
		UploadURLs: map[string]string{},
	})
}

func (h *EventHandler) HandleEventActions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.NotFound(w, r)
		return
	}

	eventID, ok := extractFinalizeEventID(r.URL.Path)
	if !ok {
		http.NotFound(w, r)
		return
	}

	if err := h.eventService.FinalizeEvent(r.Context(), eventID); err != nil {
		if errors.Is(err, service.ErrEventNotFound) {
			http.Error(w, "event not found", http.StatusNotFound)
			return
		}

		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}

	common.WriteJSON(w, http.StatusOK, dto.FinalizeEventResponse{
		EventID: eventID,
		Status:  "FINALIZED",
	})
}

func extractFinalizeEventID(path string) (string, bool) {
	if !strings.HasPrefix(path, "/events/") || !strings.HasSuffix(path, "/finalize") {
		return "", false
	}

	eventID := strings.TrimSuffix(strings.TrimPrefix(path, "/events/"), "/finalize")
	eventID = strings.TrimSuffix(eventID, "/")
	if eventID == "" {
		return "", false
	}

	return eventID, true
}

func mapCreateEventGPS(input *dto.EventGPSRequest) *service.CreateEventGPSInput {
	if input == nil {
		return nil
	}

	return &service.CreateEventGPSInput{
		Latitude:   input.Latitude,
		Longitude:  input.Longitude,
		CapturedAt: input.CapturedAt,
		AccuracyM:  input.AccuracyM,
	}
}
