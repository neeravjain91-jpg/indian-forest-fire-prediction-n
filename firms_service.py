from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from shapely.geometry import Point, shape
from shapely.prepared import prep

logger = logging.getLogger(__name__)

# Indian Standard Time (IST) offset is UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))

# Official NASA FIRMS Area API endpoint
FIRMS_AREA_API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# Bounding box for India region to query NASA FIRMS (West, South, East, North)
INDIA_QUERY_BBOX = "68,6,98,38"

# Maximum in-memory cache entries
MAX_CACHE_SIZE = 50

# Supported VIIRS Near-Real-Time (NRT) satellite sources
VALID_SOURCES = {
    "VIIRS_SNPP_NRT": "Suomi-NPP VIIRS",
    "VIIRS_NOAA20_NRT": "NOAA-20 VIIRS",
    "VIIRS_NOAA21_NRT": "NOAA-21 VIIRS",
}

BASE_DIR = Path(__file__).resolve().parent
BOUNDARY_PATH = BASE_DIR / "data" / "processed" / "india_boundary.geojson"


class FIRMSService:
    """Service to fetch, filter, and analyze real-time NASA FIRMS active fire data for India."""

    def __init__(self, boundary_path: Path = BOUNDARY_PATH, cache_ttl_seconds: int = 900):
        self.boundary_path = boundary_path
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._india_polygon = None
        self._prepared_india = None
        self._init_boundary()

    def _init_boundary(self) -> None:
        """Load and prepare India sovereign boundary polygon for strict spatial filtering."""
        if not self.boundary_path.exists():
            logger.error(f"India sovereign boundary file missing at: {self.boundary_path}")
            return
        try:
            with open(self.boundary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("type") == "FeatureCollection" and data.get("features"):
                geom = shape(data["features"][0]["geometry"])
            elif "geometry" in data:
                geom = shape(data["geometry"])
            else:
                geom = shape(data)

            if not geom.is_valid:
                logger.warning("Loaded boundary polygon is invalid, attempting repair with buffer(0)")
                geom = geom.buffer(0)

            self._india_polygon = geom
            self._prepared_india = prep(geom)
            logger.info("India boundary polygon loaded and indexed with covers semantics.")
        except Exception as exc:
            logger.error(f"Failed to load India boundary polygon: {exc}")
            self._india_polygon = None
            self._prepared_india = None

    def get_api_key(self) -> Optional[str]:
        """Retrieve FIRMS MAP_KEY securely from environment or .env file (never exposed to client)."""
        key = os.environ.get("FIRMS_MAP_KEY", "").strip()
        if key:
            return key
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("FIRMS_MAP_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
            except Exception:
                pass
        return None

    def is_inside_india(self, lon: float, lat: float) -> bool:
        """Check if coordinate strictly lies inside or on the sovereign India boundary.
        
        Uses Shapely 'covers' semantics so boundary points are intentionally included.
        Raises RuntimeError if the boundary polygon is unavailable.
        """
        if self._prepared_india is None:
            raise RuntimeError(
                "India sovereign boundary polygon is not loaded. Spatial filtering cannot proceed."
            )
        point = Point(lon, lat)
        return bool(self._prepared_india.covers(point))

    def categorize_severity(self, frp: float) -> str:
        """Classify Fire Radiative Power (MW) into standard operational severity categories."""
        if frp >= 50.0:
            return "severe"
        if frp >= 20.0:
            return "high"
        if frp >= 5.0:
            return "moderate"
        return "low"

    def fetch_live_fires(
        self,
        source: str = "ALL",
        day_range: int = 1,
        min_frp: float = 0.0,
        min_confidence: str = "all",
    ) -> Dict[str, Any]:
        """Fetch active fire detections for India from NASA FIRMS Area API.
        
        Strictly distinguishes between REAL FIRMS DATA (is_demo=False) and
        DEMO DATA (is_demo=True).
        """
        day_range = max(1, min(5, int(day_range)))
        cache_key = f"{source}_{day_range}_{min_frp}_{min_confidence}"

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached:
                cached_time, cached_data = cached
                if time.time() - cached_time < self.cache_ttl:
                    return cached_data

        api_key = self.get_api_key()
        if not api_key:
            data = self._generate_fallback_data(
                reason="DEMO MODE: Simulated near-real-time active fire observations across India. Set the server environment variable FIRMS_MAP_KEY to stream live observations from NASA FIRMS.",
                source=source,
                day_range=day_range,
                min_frp=min_frp,
            )
            return data

        try:
            sources_to_query = (
                list(VALID_SOURCES.keys()) if source.upper() in ("ALL", "ALL_VIIRS") else [source]
            )
            raw_records: List[Dict[str, Any]] = []

            for src in sources_to_query:
                url = f"{FIRMS_AREA_API_BASE}/{api_key}/{src}/{INDIA_QUERY_BBOX}/{day_range}"
                resp = requests.get(url, timeout=20)

                if resp.status_code in (400, 401, 403) or "invalid map_key" in resp.text.lower():
                    logger.warning(f"NASA FIRMS API rejected MAP_KEY for {src}: {resp.text[:200]}")
                    return self._generate_fallback_data(
                        reason="DEMO MODE: NASA FIRMS API rejected the server MAP_KEY. Verify the key at https://firms.modaps.eosdis.nasa.gov/api/map_key. Displaying calibrated demo observations.",
                        source=source,
                        day_range=day_range,
                        min_frp=min_frp,
                    )

                if resp.status_code != 200:
                    logger.warning(f"FIRMS API returned status {resp.status_code} for {src}: {resp.text[:200]}")
                    continue

                text = resp.text.strip()
                if not text or "latitude" not in text.lower():
                    if "invalid" in text.lower() or "not authorized" in text.lower():
                        return self._generate_fallback_data(
                            reason="DEMO MODE: NASA FIRMS key authorization failed. Displaying calibrated demo observations.",
                            source=source,
                            day_range=day_range,
                            min_frp=min_frp,
                        )
                    continue

                reader = csv.DictReader(io.StringIO(text))
                for row in reader:
                    row["source_satellite"] = VALID_SOURCES.get(src, src)
                    raw_records.append(row)

            processed = self._process_records(raw_records, min_frp=min_frp, min_confidence=min_confidence)
            result = {
                "status": "live",
                "is_demo": False,
                "mode_label": "LIVE SATELLITE (NASA FIRMS VIIRS)",
                "message": "LIVE OBSERVATIONS: Streaming near-real-time observations from NASA FIRMS VIIRS Area API strictly filtered to sovereign India.",
                "source": source,
                "day_range": day_range,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "summary": processed["summary"],
                "fires": processed["fires"],
            }

            with self._lock:
                if len(self._cache) >= MAX_CACHE_SIZE:
                    oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][0])
                    del self._cache[oldest_key]
                self._cache[cache_key] = (time.time(), result)

            return result

        except Exception as exc:
            logger.error(f"Error querying NASA FIRMS API: {exc}")
            return self._generate_fallback_data(
                reason=f"DEMO MODE: NASA FIRMS API connection error ({exc}). Displaying calibrated demo observations.",
                source=source,
                day_range=day_range,
                min_frp=min_frp,
            )

    def _process_records(
        self,
        records: List[Dict[str, Any]],
        min_frp: float = 0.0,
        min_confidence: str = "all",
    ) -> Dict[str, Any]:
        """Filter points strictly to India sovereign boundary and compute analytics."""
        filtered_fires: List[Dict[str, Any]] = []
        conf_filter = min_confidence.lower().strip()

        for r in records:
            try:
                lat = float(r.get("latitude", 0))
                lon = float(r.get("longitude", 0))
            except (ValueError, TypeError):
                continue

            # Strict sovereign India boundary check using polygon covers semantics
            if not self.is_inside_india(lon, lat):
                continue

            try:
                frp = float(r.get("frp", 0.0))
            except (ValueError, TypeError):
                frp = 0.0

            if frp < min_frp:
                continue

            raw_conf = str(r.get("confidence", "nominal")).strip().lower()
            if conf_filter == "nominal_or_high" and raw_conf in ("l", "low"):
                continue
            if conf_filter == "high" and raw_conf not in ("h", "high"):
                continue

            conf_label = "High" if raw_conf in ("h", "high") else ("Low" if raw_conf in ("l", "low") else "Nominal")
            acq_date = str(r.get("acq_date", ""))
            acq_time = str(r.get("acq_time", "")).zfill(4)
            formatted_time_utc = f"{acq_time[:2]}:{acq_time[2:]} UTC" if len(acq_time) == 4 else acq_time

            # Compute IST time (UTC+5:30)
            try:
                dt_utc = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
                dt_ist = dt_utc.astimezone(IST)
                acq_time_ist = dt_ist.strftime("%H:%M IST")
                acq_date_ist = dt_ist.strftime("%Y-%m-%d")
                acq_datetime_ist = dt_ist.isoformat()
            except Exception:
                acq_time_ist = formatted_time_utc
                acq_date_ist = acq_date
                acq_datetime_ist = None

            satellite = r.get("source_satellite") or r.get("satellite", "VIIRS")
            severity = self.categorize_severity(frp)

            try:
                bright_ti4 = float(r.get("bright_ti4", 0.0))
            except (ValueError, TypeError):
                bright_ti4 = None

            try:
                bright_ti5 = float(r.get("bright_ti5", 0.0))
            except (ValueError, TypeError):
                bright_ti5 = None

            filtered_fires.append({
                "id": f"{lat:.4f}_{lon:.4f}_{acq_date}_{acq_time}_{satellite[:4]}",
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "grid_lat": round(round(lat / 0.1) * 0.1, 1),
                "grid_lon": round(round(lon / 0.1) * 0.1, 1),
                "frp": round(frp, 2),
                "severity": severity,
                "confidence": conf_label,
                "satellite": satellite,
                "acq_date": acq_date,
                "acq_time": formatted_time_utc,
                "acq_time_ist": acq_time_ist,
                "acq_date_ist": acq_date_ist,
                "acq_datetime_ist": acq_datetime_ist,
                "raw_time": acq_time,
                "bright_ti4": round(bright_ti4, 1) if bright_ti4 else None,
                "bright_ti5": round(bright_ti5, 1) if bright_ti5 else None,
                "daynight": "Day" if str(r.get("daynight", "")).upper() == "D" else "Night",
            })

        filtered_fires.sort(key=lambda x: x["frp"], reverse=True)
        total = len(filtered_fires)
        severe_count = sum(1 for f in filtered_fires if f["severity"] == "severe")
        high_count = sum(1 for f in filtered_fires if f["severity"] == "high")
        max_frp = max((f["frp"] for f in filtered_fires), default=0.0)
        mean_frp = round(sum(f["frp"] for f in filtered_fires) / total, 2) if total > 0 else 0.0

        return {
            "fires": filtered_fires,
            "summary": {
                "total_fires": total,
                "severe_count": severe_count,
                "high_count": high_count,
                "max_frp": max_frp,
                "mean_frp": mean_frp,
            },
        }

    def _generate_fallback_data(
        self,
        reason: str,
        source: str = "ALL",
        day_range: int = 1,
        min_frp: float = 0.0,
    ) -> Dict[str, Any]:
        """Realistic sample of active forest fire detections across India for demonstration when API is unconfigured."""
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        sample_points = [
            # Similipal / Mayurbhanj, Odisha (Dry Deciduous / Sal Forest)
            {"lat": 21.8542, "lon": 86.3218, "frp": 88.5, "conf": "High", "sat": "Suomi-NPP VIIRS", "time": "0830", "ti4": 358.4, "dn": "Day"},
            {"lat": 21.7820, "lon": 86.2940, "frp": 44.2, "conf": "High", "sat": "NOAA-20 VIIRS", "time": "0915", "ti4": 341.2, "dn": "Day"},
            {"lat": 21.9100, "lon": 86.4150, "frp": 16.8, "conf": "Nominal", "sat": "NOAA-21 VIIRS", "time": "1000", "ti4": 326.5, "dn": "Day"},
            # Bandipur - Mudumalai - Nagarhole corridor (Western Ghats / Karnataka-Tamil Nadu)
            {"lat": 11.6640, "lon": 76.6280, "frp": 62.1, "conf": "High", "sat": "Suomi-NPP VIIRS", "time": "0825", "ti4": 352.0, "dn": "Day"},
            {"lat": 11.7210, "lon": 76.5410, "frp": 27.4, "conf": "Nominal", "sat": "NOAA-20 VIIRS", "time": "0910", "ti4": 334.8, "dn": "Day"},
            {"lat": 11.5830, "lon": 76.7190, "frp": 9.3, "conf": "Nominal", "sat": "NOAA-21 VIIRS", "time": "2045", "ti4": 312.4, "dn": "Night"},
            # Satpura - Kanha Tiger Reserve corridor, Madhya Pradesh
            {"lat": 22.3420, "lon": 78.4510, "frp": 54.7, "conf": "High", "sat": "Suomi-NPP VIIRS", "time": "0835", "ti4": 349.5, "dn": "Day"},
            {"lat": 22.4100, "lon": 78.5200, "frp": 31.9, "conf": "High", "sat": "NOAA-20 VIIRS", "time": "0920", "ti4": 338.1, "dn": "Day"},
            {"lat": 22.2850, "lon": 78.3880, "frp": 12.4, "conf": "Nominal", "sat": "NOAA-21 VIIRS", "time": "1005", "ti4": 322.0, "dn": "Day"},
            {"lat": 22.1500, "lon": 80.6100, "frp": 38.6, "conf": "High", "sat": "Suomi-NPP VIIRS", "time": "0832", "ti4": 343.7, "dn": "Day"},
            # Corbett National Park / Shivalik foothills, Uttarakhand
            {"lat": 29.5300, "lon": 78.7740, "frp": 49.3, "conf": "High", "sat": "NOAA-20 VIIRS", "time": "0925", "ti4": 346.0, "dn": "Day"},
            {"lat": 29.6100, "lon": 78.8920, "frp": 22.8, "conf": "Nominal", "sat": "Suomi-NPP VIIRS", "time": "0840", "ti4": 331.2, "dn": "Day"},
            {"lat": 29.4800, "lon": 78.6900, "frp": 7.1, "conf": "Low", "sat": "NOAA-21 VIIRS", "time": "2105", "ti4": 308.9, "dn": "Night"},
            # Melghat / Vidarbha forests, Maharashtra
            {"lat": 21.4120, "lon": 77.1980, "frp": 33.5, "conf": "High", "sat": "NOAA-20 VIIRS", "time": "0918", "ti4": 339.4, "dn": "Day"},
            {"lat": 21.3400, "lon": 77.2650, "frp": 14.2, "conf": "Nominal", "sat": "Suomi-NPP VIIRS", "time": "0838", "ti4": 325.6, "dn": "Day"},
            # Dampa / Mizoram Bamboo & Jhum forest belt
            {"lat": 23.6820, "lon": 92.4110, "frp": 58.2, "conf": "High", "sat": "Suomi-NPP VIIRS", "time": "0745", "ti4": 351.8, "dn": "Day"},
            {"lat": 23.5900, "lon": 92.5200, "frp": 26.0, "conf": "Nominal", "sat": "NOAA-20 VIIRS", "time": "0845", "ti4": 333.5, "dn": "Day"},
            {"lat": 23.7500, "lon": 92.3400, "frp": 11.5, "conf": "Nominal", "sat": "NOAA-21 VIIRS", "time": "0930", "ti4": 320.1, "dn": "Day"},
            # Saranda / Jharkhand Sal Forests
            {"lat": 22.2100, "lon": 85.3200, "frp": 41.0, "conf": "High", "sat": "Suomi-NPP VIIRS", "time": "0828", "ti4": 344.0, "dn": "Day"},
            {"lat": 22.1400, "lon": 85.2500, "frp": 18.3, "conf": "Nominal", "sat": "NOAA-20 VIIRS", "time": "0912", "ti4": 328.7, "dn": "Day"},
            # Bastar / Kanger Valley, Chhattisgarh
            {"lat": 18.8900, "lon": 81.9800, "frp": 36.4, "conf": "High", "sat": "NOAA-21 VIIRS", "time": "0958", "ti4": 340.2, "dn": "Day"},
            {"lat": 18.9600, "lon": 82.0500, "frp": 15.6, "conf": "Nominal", "sat": "Suomi-NPP VIIRS", "time": "0830", "ti4": 324.9, "dn": "Day"},
            # Shendurney / Agasthyamalai, Kerala
            {"lat": 8.8400, "lon": 77.1500, "frp": 24.1, "conf": "Nominal", "sat": "NOAA-20 VIIRS", "time": "0905", "ti4": 332.0, "dn": "Day"},
        ]

        fires: List[Dict[str, Any]] = []
        for p in sample_points:
            frp = p["frp"]
            if frp < min_frp:
                continue

            time_str = p["time"]
            formatted_utc = f"{time_str[:2]}:{time_str[2:]} UTC"
            try:
                dt_utc = datetime.strptime(f"{today_str} {time_str}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
                dt_ist = dt_utc.astimezone(IST)
                acq_time_ist = dt_ist.strftime("%H:%M IST")
                acq_date_ist = dt_ist.strftime("%Y-%m-%d")
                acq_datetime_ist = dt_ist.isoformat()
            except Exception:
                acq_time_ist = formatted_utc
                acq_date_ist = today_str
                acq_datetime_ist = None

            fires.append({
                "id": f"{p['lat']:.4f}_{p['lon']:.4f}_{today_str}_{time_str}_{p['sat'][:4]}",
                "lat": round(p["lat"], 4),
                "lon": round(p["lon"], 4),
                "grid_lat": round(round(p["lat"] / 0.1) * 0.1, 1),
                "grid_lon": round(round(p["lon"] / 0.1) * 0.1, 1),
                "frp": round(frp, 2),
                "severity": self.categorize_severity(frp),
                "confidence": p["conf"],
                "satellite": p["sat"],
                "acq_date": today_str,
                "acq_time": formatted_utc,
                "acq_time_ist": acq_time_ist,
                "acq_date_ist": acq_date_ist,
                "acq_datetime_ist": acq_datetime_ist,
                "raw_time": time_str,
                "bright_ti4": p["ti4"],
                "bright_ti5": round(p["ti4"] - 35.0, 1),
                "daynight": p["dn"],
            })

        fires.sort(key=lambda x: x["frp"], reverse=True)
        total = len(fires)
        severe_count = sum(1 for f in fires if f["severity"] == "severe")
        high_count = sum(1 for f in fires if f["severity"] == "high")
        max_frp = max((f["frp"] for f in fires), default=0.0)
        mean_frp = round(sum(f["frp"] for f in fires) / total, 2) if total > 0 else 0.0

        return {
            "status": "demo",
            "is_demo": True,
            "mode_label": "DEMO MODE (Simulated Surveillance)",
            "message": reason,
            "source": source,
            "day_range": day_range,
            "timestamp_utc": now.isoformat(),
            "summary": {
                "total_fires": total,
                "severe_count": severe_count,
                "high_count": high_count,
                "max_frp": max_frp,
                "mean_frp": mean_frp,
            },
            "fires": fires,
        }
