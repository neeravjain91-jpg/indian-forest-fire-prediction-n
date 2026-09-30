# Operational Deployment & Production Architecture Guide

## 1. System Overview

The India Wildfire Intelligence Platform is packaged as a dual-stack application:
1. **Scientific Backend**: Python / Flask WSGI service integrating PyTorch, LightGBM, Scikit-learn, and Shapely.
2. **Geospatial GIS Frontend**: Real-time Leaflet GIS mapping interface with hardware-accelerated Canvas rendering, multi-horizon forecast overlays, event tracking, and retrospective replay station.

---

## 2. Environment Setup & Dependencies

Install production requirements:
```bash
pip install -r requirements.txt
```

Core dependencies verified:
- `numpy >= 1.26`
- `pandas >= 2.2`
- `scipy >= 1.13`
- `scikit-learn >= 1.5`
- `lightgbm >= 4.0`
- `torch >= 2.0`
- `shapely >= 2.0`
- `flask >= 3.0`
- `matplotlib >= 3.8`
- `joblib >= 1.4`

---

## 3. Configuration & Security Protocols

### NASA FIRMS Key Management
The application accesses the NASA FIRMS Area API strictly server-side. **No API keys are ever transmitted to or stored within the browser.**

Set the environment variable:
* **Windows (PowerShell)**:
  ```powershell
  $env:FIRMS_MAP_KEY = "your_nasa_firms_map_key"
  ```
* **Linux / macOS (Bash)**:
  ```bash
  export FIRMS_MAP_KEY="your_nasa_firms_map_key"
  ```
* **Dotenv (.env file)**:
  Create a `.env` file in the project root:
  ```ini
  FIRMS_MAP_KEY=your_nasa_firms_map_key
  ```
*(Note: If no key is set, the application operates in calibrated demo mode with simulated active observations over Indian forest corridors).*

### Security Audit Findings & Hardening
1. **No `/api/set-key` endpoint**: Public POST endpoints modifying server environment variables were removed.
2. **API Timeout & Retry Backoff**: FIRMS API queries enforce a 15-second timeout and 3-attempt exponential backoff.
3. **Response Caching**: Responses are cached in-memory with a 15-minute Time-To-Live (TTL) to avoid exceeding NASA rate limits.
4. **Boundary Integrity**: Detections outside the sovereign boundary of India are strictly filtered using Shapely `covers` spatial indexing.

---

## 4. Launching the Service

### Development Server:
```bash
python application.py
```
Access at: `http://127.0.0.1:5000`

### Production WSGI (Windows via Waitress):
```bash
pip install waitress
waitress-serve --port=5000 application:app
```

### Production WSGI (Linux via Gunicorn):
```bash
gunicorn -w 4 -b 0.0.0.0:5000 application:app
```
