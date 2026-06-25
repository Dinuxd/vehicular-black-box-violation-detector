package main

import (
	"context"
	"flag"
	"log"
	"net/http"

	"ingest-service/internal/api/handler"
	"ingest-service/internal/api/router"
	"ingest-service/internal/config"
	"ingest-service/internal/platform/db"
	"ingest-service/internal/repository/postgres"
	"ingest-service/internal/service"
)

func main() {
	rebuildScores := flag.Bool("rebuild-scores", false, "rebuild device_scores from the events ledger and exit")
	flag.Parse()

	ctx := context.Background()
	cfg := config.Load()

	pool, err := db.NewPool(ctx, cfg.DBURL)
	if err != nil {
		log.Fatalf("failed to connect to ingest db: %v", err)
	}
	defer pool.Close()

	eventRepo := postgres.NewEventRepository(pool, cfg.ScoreHalfLifeDays)

	if *rebuildScores {
		if err := eventRepo.RebuildScores(ctx); err != nil {
			log.Fatalf("failed to rebuild scores: %v", err)
		}
		log.Printf("device scores rebuilt with policy half-life %.2f days", cfg.ScoreHalfLifeDays)
		return
	}

	eventService := service.NewEventService(eventRepo)
	deviceService := service.NewDeviceService(eventRepo)

	healthHandler := handler.NewHealthHandler()
	eventHandler := handler.NewEventHandler(eventService)
	deviceHandler := handler.NewDeviceHandler(deviceService)

	appRouter := router.New(healthHandler, eventHandler, deviceHandler)

	log.Printf("Ingest service listening on :%s", cfg.Port)
	if err := http.ListenAndServe(":"+cfg.Port, appRouter); err != nil {
		log.Fatal(err)
	}
}
