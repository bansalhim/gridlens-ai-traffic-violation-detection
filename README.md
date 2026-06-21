# gridlens-ai-traffic-violation-detection

# GridLens AI — Flipkart Gridlock Hackathon Theme 3 Prototype

## Problem Statement
Automated Photo Identification and Classification for Traffic Violations Using Computer Vision.

## Prototype Summary
GridLens AI processes a traffic image, detects vehicles and road users, identifies possible violations,
generates annotated evidence, extracts number plate text when OCR is enabled, and produces a downloadable
evidence report.

## Current MVP Features
- Streamlit web app
- YOLO-based vehicle/person detection
- Triple-riding heuristic
- Possible illegal parking/obstruction candidate
- Optional EasyOCR for plate/text detection
- Annotated evidence output
- Violation report CSV
- Simple analytics dashboard

## Recommended Demo Scope
For Round 2, demonstrate:
1. Helmet non-compliance workflow
2. Triple riding
3. Number plate OCR
4. Evidence packet generation
5. Dashboard analytics

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The default model is `yolov8n.pt`. It may download automatically on first run.

## Important Hackathon Note
For Theme 3, the solution can be submitted as a concept note, prototype proposal, or solution framework.
This MVP uses public/sample traffic images only for demonstration. In real deployment, it can connect to
official BTP CCTV/photo evidence feeds.

## Future Scope
- Train custom helmet/no-helmet model
- Train number plate detector for Indian plates
- Multi-frame illegal parking confirmation
- Stop-line and red-light region-of-interest calibration
- Wrong-side driving using lane direction and object tracking
- Human review workflow for enforcement officers
