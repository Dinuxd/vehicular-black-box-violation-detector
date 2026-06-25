package service

import (
	"context"

	"ingest-service/internal/domain"
	"ingest-service/internal/repository"
)

type ListEventsInput struct {
	DeviceID  string
	EventType string
	Limit     int
	Offset    int
}

type DeviceService struct {
	repo repository.EventRepository
}

func NewDeviceService(repo repository.EventRepository) *DeviceService {
	return &DeviceService{repo: repo}
}

func (s *DeviceService) ListDevices(ctx context.Context) ([]string, error) {
	return s.repo.ListDevices(ctx)
}

func (s *DeviceService) ListViolations(ctx context.Context, deviceID string) ([]string, error) {
	return s.repo.ListViolationsByDevice(ctx, deviceID)
}

func (s *DeviceService) ListEvents(ctx context.Context, input ListEventsInput) (domain.EventPage, error) {
	return s.repo.ListEventsByDevice(ctx, domain.EventQuery{
		DeviceID:  input.DeviceID,
		EventType: input.EventType,
		Limit:     input.Limit,
		Offset:    input.Offset,
	})
}

func (s *DeviceService) GetScore(ctx context.Context, deviceID string) (domain.DeviceScore, error) {
	return s.repo.GetDeviceScore(ctx, deviceID)
}
