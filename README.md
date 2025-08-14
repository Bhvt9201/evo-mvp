# Evo Genesis — MVP (Demo)

## Quick start (local)
1. Make executable: `chmod +x setup_and_run.sh`
2. Run: `./setup_and_run.sh`
3. Open: `http://localhost:8000` (demo) and `http://localhost:8000/admin` (admin)

## Deploy (Render / Railway / similar)
- Create a repo, push this code.
- For Render: create a Web Service, connect GitHub, build command: `pip install -r requirements.txt`, start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
- Set PORT env var if required.

## What it does
- Captures lightweight telemetry via `static/telemetry.js`
- Runs a tiny intent classifier (synthetic trained) to infer `purchase/research/browse`
- Uses Thompson-sampling bandit to choose CTA variants
- Exposes an admin UI that shows decisions and bandit stats
- Accepts conversion outcomes via `/api/outcome` for bandit learning

## Next steps
- Replace in-memory stores with Redis/Postgres for scale
- Swap synthetic classifier with trained model on real pilot data
- Add persistent user profiles & federated learning
