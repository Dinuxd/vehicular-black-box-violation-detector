import { Navigate, Route, Routes } from "react-router";
import { VehicularShell } from "@/layout/vehicular-shell";
import { BlankHome } from "@/pages/blank-home";
import { DevicesPage } from "@/pages/devices/devices-page";
import { DeviceEventsPage } from "@/pages/devices/device-events-page";
import { DeviceEventDetailPage } from "@/pages/devices/device-event-detail-page";

export default function App() {
  return (
    <Routes>
      <Route element={<VehicularShell />}>
        <Route index element={<BlankHome />} />
        <Route path="devices" element={<DevicesPage />} />
        <Route path="devices/:deviceId" element={<DeviceEventsPage />} />
        <Route
          path="devices/:deviceId/events/:eventId"
          element={<DeviceEventDetailPage />}
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
