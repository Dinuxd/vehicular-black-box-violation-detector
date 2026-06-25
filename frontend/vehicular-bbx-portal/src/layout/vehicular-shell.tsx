import { Link, Outlet, useLocation, useParams } from "react-router";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
  Separator,
} from "@/lib/platform-ui-common";
import { Car, Monitor, ChevronRight } from "lucide-react";

function usePageTitle() {
  const location = useLocation();
  const params = useParams();

  if (location.pathname === "/devices") return "Devices";
  if (location.pathname.includes("/events/")) return `Event: ${params.eventId ?? ""}`;
  if (location.pathname.startsWith("/devices/")) return `Device: ${params.deviceId ?? ""}`;
  return ""; // base route -> blank
}

export function VehicularShell() {
  const title = usePageTitle();

  return (
    <SidebarProvider>
      <Sidebar className="border-r border-slate-800 bg-slate-900 text-slate-100">
        <SidebarHeader className="border-b border-slate-800 px-4 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-indigo-500 text-white shadow-sm">
              <Car className="h-5 w-5" />
            </div>
            <div className="leading-tight">
              <div className="font-semibold text-white">Vehicular BBX</div>
              <div className="text-xs text-slate-400">Portal</div>
            </div>
          </div>
        </SidebarHeader>

        <SidebarContent>
          <SidebarGroup>
            <SidebarGroupLabel className="text-slate-300">Navigation</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                <SidebarMenuItem>
                  <SidebarMenuButton asChild>
                    <Link
                      to="/devices"
                      className="text-slate-200 hover:bg-slate-800 hover:text-white"
                    >
                      <Monitor />
                      <span>Devices</span>
                      <ChevronRight className="ml-auto opacity-50" />
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        </SidebarContent>
      </Sidebar>

      <SidebarInset>
        <header className="flex h-14 items-center gap-3 border-b px-4">
          <SidebarTrigger />
          <Separator orientation="vertical" className="h-6" />
          <div className="font-semibold">{title}</div>
        </header>

        <main className="min-h-[calc(100vh-3.5rem)] bg-background">
          <Outlet />
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
