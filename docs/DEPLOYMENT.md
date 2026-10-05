# Deployment & Operational Guide

## 1. Local Deployment

### 1.1 Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### 1.2 Installation & Startup
```bash
# Clone the repository
git clone https://github.com/neeravjain91-jpg/indian-forest-fire-prediction-n.git
cd indian-forest-fire-prediction-n

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Provide NASA FIRMS MAP Key
export FIRMS_MAP_KEY="your_nasa_firms_key_here"  # On Windows: set FIRMS_MAP_KEY=your_nasa_firms_key_here

# Launch Flask application
python application.py
```
The application will start locally on `http://127.0.0.1:5000`.

---

## 2. Cloud Serverless Deployment (Vercel)

The application is structured for instant serverless deployment on Vercel:

1. **Vercel Entrypoint**: Configured in `pyproject.toml`:
   ```toml
   [tool.vercel]
   entrypoint = "application:app"
   ```
2. **Minimal Dependency Footprint**:
   By strictly eliminating bulky deep-learning libraries (`torch`) and complex compiled packages, the production dependency footprint stays well within Vercel's serverless package limits (<250 MB compressed).
3. **Ignored Artifacts**:
   `.vercelignore` excludes large CSV datasets and raw archives from the serverless deployment artifact, deploying only the precomputed 1.15 MB model checkpoint (`results/final_model/final_hgb_model.joblib`), metrics, and UI templates.

---

## 3. Configuration & Security Protocols

### 3.1 NASA FIRMS API Key Management
- The application reads `FIRMS_MAP_KEY` exclusively from server-side environment variables.
- **Graceful Fallback (DEMO Mode)**: If `FIRMS_MAP_KEY` is not provided, the application automatically enters **DEMO Mode**, serving realistic historical active fire samples across India without failing or prompting the user.
- **Zero Key Leakage**: The `/api/firms-status` endpoint exposes only operational state flags (`LIVE` vs. `DEMO`) and whether the boundary polygon is initialized. Raw key values are never returned to client browsers.

### 3.2 Geospatial Boundary Enforcement
All incoming satellite detections from NASA FIRMS are passed through a spatial filter (`shapely.prepared.prep(polygon)`) using the official Survey of India GeoJSON boundary (`data/processed/india_boundary.geojson`). Any coordinate outside sovereign Indian territory is filtered prior to UI transmission.
