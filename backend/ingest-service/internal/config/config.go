package config

import (
	"os"
	"strconv"
	"strings"

	"ingest-service/internal/scoring"
)

type Config struct {
	DBURL             string
	Port              string
	ScoreHalfLifeDays float64
}

func Load() Config {
	return Config{
		DBURL:             getenv("DB_URL", ""),
		Port:              getenv("INGEST_PORT", "8080"),
		ScoreHalfLifeDays: getenvFloat("SCORE_HALF_LIFE_DAYS", scoring.DefaultHalfLifeDays),
	}
}

func getenv(key, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}

	return value
}

func getenvFloat(key string, fallback float64) float64 {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}

	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || parsed <= 0 {
		return fallback
	}

	return parsed
}
