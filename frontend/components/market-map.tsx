"use client";

import "maplibre-gl/dist/maplibre-gl.css";
import maplibregl from "maplibre-gl";
import { useEffect, useRef } from "react";

export function MarketMap() {
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!container.current) return;
    const map = new maplibregl.Map({
      container: container.current,
      style: "https://demotiles.maplibre.org/style.json",
      center: [-87.68, 41.84],
      zoom: 9.3,
      attributionControl: false,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }));
    return () => map.remove();
  }, []);
  return <div ref={container} className="market-map" aria-label="Interactive Chicago market map" />;
}
