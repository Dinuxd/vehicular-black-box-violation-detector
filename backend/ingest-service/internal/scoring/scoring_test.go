package scoring

import (
	"math"
	"testing"
)

func TestPointsKnownAliasesAndUnknowns(t *testing.T) {
	tests := []struct {
		eventType string
		want      int
	}{
		{"DROWSINESS", 25},
		{"drowsiness-detected", 25},
		{"solid line crossing", 25},
		{"HORN_ABUSE", 2},
		{"GPS_LTE_TEST", 0},
		{"HARSH_BRAKING", 0},
		{"AGGRESSIVE_DRIVING", 0},
		{"SOMETHING_NEW", 0},
	}

	for _, tt := range tests {
		if got := Points(tt.eventType); got != tt.want {
			t.Fatalf("Points(%q) = %d, want %d", tt.eventType, got, tt.want)
		}
	}
}

func TestDecayHalvesAfterHalfLife(t *testing.T) {
	got := Decay(100, 60*86400, 60)
	if math.Abs(float64(got-50)) > 1 {
		t.Fatalf("Decay(100, half-life) = %d, want about 50", got)
	}
}

func TestRiskBand(t *testing.T) {
	tests := []struct {
		score int
		want  string
	}{
		{0, "LOW"},
		{9, "LOW"},
		{10, "MODERATE"},
		{24, "MODERATE"},
		{25, "HIGH"},
		{49, "HIGH"},
		{50, "CRITICAL"},
	}

	for _, tt := range tests {
		if got := RiskBand(tt.score); got != tt.want {
			t.Fatalf("RiskBand(%d) = %q, want %q", tt.score, got, tt.want)
		}
	}
}
