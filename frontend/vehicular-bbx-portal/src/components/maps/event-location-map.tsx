import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

type EventLocationMapProps = {
  latitude: number;
  longitude: number;
  accuracyM?: number | null;
  title: string;
};

const DEFAULT_TILE_URL =
  import.meta.env.VITE_MAP_TILE_URL ?? "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";

const DEFAULT_TILE_ATTRIBUTION =
  import.meta.env.VITE_MAP_ATTRIBUTION ??
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

export function EventLocationMap({
  latitude,
  longitude,
  accuracyM,
  title,
}: EventLocationMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!containerRef.current) return undefined;

    const map = L.map(containerRef.current, {
      attributionControl: true,
      scrollWheelZoom: false,
      zoomControl: false,
    }).setView([latitude, longitude], 16);

    L.control.zoom({ position: "bottomright" }).addTo(map);

    L.tileLayer(DEFAULT_TILE_URL, {
      attribution: DEFAULT_TILE_ATTRIBUTION,
      maxZoom: 19,
    }).addTo(map);

    const point = L.latLng(latitude, longitude);

    if (accuracyM && accuracyM > 0) {
      L.circle(point, {
        color: "#f97316",
        fillColor: "#fdba74",
        fillOpacity: 0.22,
        radius: accuracyM,
        weight: 1,
      }).addTo(map);
    }

    const marker = L.circleMarker(point, {
      color: "#7f1d1d",
      fillColor: "#dc2626",
      fillOpacity: 1,
      radius: 8,
      weight: 3,
    }).addTo(map);

    marker.bindPopup(title).openPopup();

    requestAnimationFrame(() => {
      map.invalidateSize();
    });

    return () => {
      map.remove();
    };
  }, [accuracyM, latitude, longitude, title]);

  return (
    <div className="overflow-hidden rounded-xl border bg-muted/20 shadow-sm">
      <div ref={containerRef} className="h-[320px] w-full md:h-[420px]" />
    </div>
  );
}
