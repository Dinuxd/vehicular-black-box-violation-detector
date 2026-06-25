package handler

import (
	"net/http"
	"strings"

	"ingest-service/internal/api/dto"
	"ingest-service/internal/common"
	"ingest-service/internal/domain"
	"ingest-service/internal/service"
)

type DeviceHandler struct {
	deviceService *service.DeviceService
}

func NewDeviceHandler(deviceService *service.DeviceService) *DeviceHandler {
	return &DeviceHandler{deviceService: deviceService}
}

func (h *DeviceHandler) HandleDevices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if r.URL.Path != "/devices" {
		http.NotFound(w, r)
		return
	}

	devices, err := h.deviceService.ListDevices(r.Context())
	if err != nil {
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}

	common.WriteJSON(w, http.StatusOK, dto.DevicesResponse{Devices: devices})
}

func (h *DeviceHandler) HandleDeviceSubroutes(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	deviceID, action, ok := parseDeviceSubroute(r.URL.Path)
	if !ok {
		http.NotFound(w, r)
		return
	}

	switch action {
	case "violations":
		h.handleViolations(w, r, deviceID)
	case "events":
		h.handleEvents(w, r, deviceID)
	case "score":
		h.handleScore(w, r, deviceID)
	default:
		http.NotFound(w, r)
	}
}

func (h *DeviceHandler) handleViolations(w http.ResponseWriter, r *http.Request, deviceID string) {
	violations, err := h.deviceService.ListViolations(r.Context(), deviceID)
	if err != nil {
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}

	common.WriteJSON(w, http.StatusOK, dto.ViolationsResponse{
		DeviceID:   deviceID,
		Violations: violations,
	})
}

func (h *DeviceHandler) handleEvents(w http.ResponseWriter, r *http.Request, deviceID string) {
	limit, offset := common.ParseLimitOffset(r.URL.Query(), 50, 500)

	page, err := h.deviceService.ListEvents(r.Context(), service.ListEventsInput{
		DeviceID:  deviceID,
		EventType: r.URL.Query().Get("event_type"),
		Limit:     limit,
		Offset:    offset,
	})
	if err != nil {
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}

	rows := make([]dto.EventRow, 0, len(page.Rows))
	for _, row := range page.Rows {
		rows = append(rows, dto.EventRow{
			Seq:       row.Seq,
			EventID:   row.EventID,
			TS:        row.Timestamp,
			EventType: row.EventType,
			Severity:  row.Severity,
			GPS:       mapEventGPSResponse(row.GPS),
		})
	}

	common.WriteJSON(w, http.StatusOK, dto.EventsResponse{
		DeviceID:  page.DeviceID,
		EventType: page.EventType,
		Limit:     page.Limit,
		Offset:    page.Offset,
		Total:     page.Total,
		Rows:      rows,
	})
}

func (h *DeviceHandler) handleScore(w http.ResponseWriter, r *http.Request, deviceID string) {
	score, err := h.deviceService.GetScore(r.Context(), deviceID)
	if err != nil {
		http.Error(w, "db error", http.StatusInternalServerError)
		return
	}

	common.WriteJSON(w, http.StatusOK, dto.DeviceScoreResponse{
		DeviceID:           score.DeviceID,
		Score:              score.Score,
		RiskBand:           score.RiskBand,
		TotalViolations:    score.TotalViolations,
		LastViolationAt:    score.LastViolationAt,
		ScorePolicyVersion: score.ScorePolicyVersion,
		HalfLifeDays:       score.HalfLifeDays,
		UpdatedAt:          score.UpdatedAt,
	})
}

func parseDeviceSubroute(path string) (deviceID string, action string, ok bool) {
	rest := strings.TrimPrefix(path, "/devices/")
	parts := strings.Split(rest, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", "", false
	}

	return parts[0], parts[1], true
}

func mapEventGPSResponse(gps *domain.EventGPS) *dto.EventGPSResponse {
	if gps == nil {
		return nil
	}

	return &dto.EventGPSResponse{
		Latitude:   gps.Latitude,
		Longitude:  gps.Longitude,
		CapturedAt: gps.CapturedAt,
		AccuracyM:  gps.AccuracyM,
	}
}
