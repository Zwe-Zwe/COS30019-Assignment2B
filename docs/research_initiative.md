# Research Initiative (Visualisation via Flask UI)

Use the existing Flask map to provide “Google Maps–style” evidence of how incident severity affects routing. Capture screenshots for a few scenarios and include them in the report’s Research section.

## How to capture evidence
1. Start the app: `FLASK_APP=ics_app.py flask run` (or `python3 ics_app.py`), then open http://127.0.0.1:5000.
2. For each scenario below:
   - Set Origin/Destination.
   - Pick the affected Way ID.
   - Choose Manual severity (or upload an image that matches the severity).
   - Set `k` to 3–5 routes.
   - Click “Compute Routes.”
   - Take a screenshot showing the map (incident highlight + top-k routes) and the route table with times.

## Suggested scenarios
- **Moderate slowdown on Way 2003 (Masjid → Plaza)**  
  Start: `1` (Masjid) → Goal: `3` (Plaza), Way: `2003`, Severity: `02-moderate`, k=3.
- **Severe accident on Way 2014 (Padang → Art Museum)**  
  Start: `2` (Padang) → Goal: `12` (Art Museum), Way: `2014`, Severity: `03-severe`, k=3.
- **Moderate on Way 2026 (Wisma Hopoh connector)**  
  Start: `14` (Wisma Hopoh) → Goal: `9` (Museum Admin), Way: `2026`, Severity: `02-moderate`, k=3.

## What to note in the report
- The incident highlight on the affected way and the severity you chose.
- Route diversity and cost changes (e.g., severe incidents force detours; moderate may keep main arterials).
- Any observation about robustness (are there still 3–5 viable routes? how much extra time?).

These screenshots + notes satisfy the optional research/visualisation component without extra tooling.
