package router

import (
	"net/http"

	"ingest-service/internal/api/handler"
	"ingest-service/internal/api/middleware"
)

func New(healthHandler *handler.HealthHandler, eventHandler *handler.EventHandler, deviceHandler *handler.DeviceHandler) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", healthHandler.Handle)
	mux.HandleFunc("/events", eventHandler.HandleEvents)
	mux.HandleFunc("/events/", eventHandler.HandleEventActions)
	mux.HandleFunc("/devices", deviceHandler.HandleDevices)
	mux.HandleFunc("/devices/", deviceHandler.HandleDeviceSubroutes)

	return middleware.CORS(mux)
}
