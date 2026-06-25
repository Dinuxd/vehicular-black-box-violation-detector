import * as React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";

import "@/app/globals.css";

import App from "@/app/app";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
