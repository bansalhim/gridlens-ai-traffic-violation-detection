
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from datetime import datetime

st.set_page_config(page_title="GridLens AI", page_icon="🚦", layout="wide")

COCO_VEHICLES = {"car", "motorcycle", "bus", "truck", "bicycle"}
ROAD_USERS = {"person"}
VIOLATION_CLASSES = {
    "no_helmet", "without_helmet", "helmet_missing", "faceWithNoHelmet",
    "triple_riding", "illegal_parking", "stop_line_violation", "red_light_violation"
}

@st.cache_resource
def load_yolo(model_name_or_path: str):
    from ultralytics import YOLO
    return YOLO(model_name_or_path)

@st.cache_resource
def load_ocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)

def preprocess_image(rgb):
    """Basic enhancement for low-light/shadow images."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def expand_box(box, img_w, img_h, scale_x=1.35, scale_y=1.8):
    x1, y1, x2, y2 = box
    cx, cy = box_center(box)
    w = (x2 - x1) * scale_x
    h = (y2 - y1) * scale_y
    nx1 = max(0, cx - w / 2)
    ny1 = max(0, cy - h / 2)
    nx2 = min(img_w, cx + w / 2)
    ny2 = min(img_h, cy + h / 2)
    return [nx1, ny1, nx2, ny2]

def point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_box(img, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.rectangle(img, (x1, max(0, y1 - 25)), (x1 + max(160, 8 * len(label)), y1), color, -1)
    cv2.putText(img, label, (x1 + 4, max(15, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

def evidence_record(violation, confidence, evidence, vehicle_id="-", plate_text="-", status="Review Required"):
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "violation_type": violation,
        "confidence": round(float(confidence), 3),
        "vehicle_id": vehicle_id,
        "number_plate": plate_text,
        "evidence": evidence,
        "status": status,
    }

def run_detection(rgb, model_name, conf_threshold):
    model = load_yolo(model_name)
    results = model.predict(rgb, conf=conf_threshold, verbose=False)
    result = results[0]

    detections = []
    names = result.names

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "label": label,
                "confidence": conf,
                "box": xyxy,
            })
    return detections

def run_ocr(rgb):
    try:
        reader = load_ocr()
        results = reader.readtext(rgb)
        texts = []
        for bbox, text, conf in results:
            cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
            if len(cleaned) >= 5:
                texts.append((cleaned, conf))
        texts = sorted(texts, key=lambda x: x[1], reverse=True)
        return texts[:5]
    except Exception as e:
        return [("OCR_NOT_AVAILABLE", 0.0)]

def detect_violations(rgb, detections, demo_helmet=False, demo_stopline=False, demo_redlight=False):
    h, w = rgb.shape[:2]
    annotated = rgb.copy()
    records = []

    persons = [d for d in detections if d["label"] in ROAD_USERS]
    vehicles = [d for d in detections if d["label"] in COCO_VEHICLES]
    motorcycles = [d for d in detections if d["label"] == "motorcycle"]

    # Draw all detections first
    for d in detections:
        label = f'{d["label"]} {d["confidence"]:.2f}'
        color = (0, 200, 0)
        if d["label"] == "person":
            color = (255, 180, 0)
        elif d["label"] in COCO_VEHICLES:
            color = (0, 200, 255)
        draw_box(annotated, d["box"], label, color=color, thickness=2)

    # Triple riding heuristic: 3+ persons around one motorcycle.
    for idx, moto in enumerate(motorcycles, start=1):
        zone = expand_box(moto["box"], w, h, scale_x=2.2, scale_y=2.2)
        nearby_persons = [
            p for p in persons if point_inside_box(box_center(p["box"]), zone)
        ]
        if len(nearby_persons) >= 3:
            draw_box(annotated, zone, "TRIPLE RIDING - REVIEW", color=(255, 0, 0), thickness=3)
            records.append(evidence_record(
                "Triple Riding",
                min(0.95, 0.60 + 0.08 * len(nearby_persons)),
                f"{len(nearby_persons)} persons detected around motorcycle #{idx}",
                vehicle_id=f"motorcycle_{idx}",
            ))

    # Illegal parking candidate heuristic for single image.
    # In real deployment, this should be confirmed using multi-frame tracking/stationary duration.
    for idx, v in enumerate(vehicles, start=1):
        x1, y1, x2, y2 = v["box"]
        vehicle_area = (x2 - x1) * (y2 - y1)
        img_area = w * h
        is_large_foreground_vehicle = vehicle_area / img_area > 0.08
        is_lower_road_zone = y2 > h * 0.65
        if is_large_foreground_vehicle and is_lower_road_zone and v["label"] in {"car", "truck", "bus"}:
            records.append(evidence_record(
                "Possible Illegal Parking / Obstruction",
                0.55,
                "Vehicle detected in lower road zone; needs time-based confirmation",
                vehicle_id=f'{v["label"]}_{idx}',
            ))

    # Demo toggles let you show the full evidence workflow even before a custom violation model is trained.
    if demo_helmet:
        records.append(evidence_record(
            "Helmet Non-Compliance",
            0.82,
            "Demo mode: no-helmet class would be detected by custom helmet model",
            vehicle_id="two_wheeler_demo",
        ))
        cv2.putText(annotated, "DEMO: HELMET VIOLATION EVIDENCE", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

    if demo_stopline:
        cv2.line(annotated, (int(w * 0.1), int(h * 0.72)), (int(w * 0.9), int(h * 0.72)), (255, 0, 0), 4)
        records.append(evidence_record(
            "Stop-Line Violation",
            0.78,
            "Demo mode: vehicle beyond virtual stop-line ROI",
            vehicle_id="vehicle_demo",
        ))

    if demo_redlight:
        records.append(evidence_record(
            "Red-Light Violation",
            0.76,
            "Demo mode: red signal state + vehicle crossing ROI",
            vehicle_id="vehicle_demo",
        ))

    return annotated, records

st.title("🚦 GridLens AI")
st.subheader("Automated Traffic Violation Detection and Evidence Generation Prototype")

with st.sidebar:
    st.header("Prototype Settings")
    model_name = st.text_input("YOLO model", value="yolov8n.pt", help="Use yolov8n.pt for quick demo or custom helmet/violation model path.")
    conf_threshold = st.slider("Detection confidence", 0.10, 0.90, 0.35, 0.05)
    use_preprocess = st.checkbox("Enhance image before detection", value=True)
    enable_ocr = st.checkbox("Enable number plate OCR", value=False)
    st.divider()
    st.caption("Demo switches for full evidence workflow before custom training:")
    demo_helmet = st.checkbox("Simulate helmet violation", value=False)
    demo_stopline = st.checkbox("Simulate stop-line violation", value=False)
    demo_redlight = st.checkbox("Simulate red-light violation", value=False)

uploaded = st.file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a traffic image to run the prototype. For first demo, use an image with bikes/cars/people on road.")
    st.markdown("""
    ### MVP features
    - Vehicle and road-user detection using YOLO
    - Triple-riding heuristic using rider count near motorcycle
    - Possible illegal parking/obstruction candidate
    - Optional OCR for number plates
    - Annotated evidence image and downloadable violation report
    """)
else:
    image = Image.open(uploaded).convert("RGB")
    rgb = np.array(image)
    processed = preprocess_image(rgb) if use_preprocess else rgb

    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb, caption="Original Image", use_container_width=True)

    try:
        detections = run_detection(processed, model_name, conf_threshold)
        annotated, records = detect_violations(
            processed, detections,
            demo_helmet=demo_helmet,
            demo_stopline=demo_stopline,
            demo_redlight=demo_redlight,
        )

        plate_candidates = []
        if enable_ocr:
            plate_candidates = run_ocr(processed)
            best_plate = plate_candidates[0][0] if plate_candidates else "-"
            for r in records:
                if r["number_plate"] == "-":
                    r["number_plate"] = best_plate

        with c2:
            st.image(annotated, caption="Annotated Evidence Output", use_container_width=True)

        st.divider()
        st.subheader("Detection Summary")
        det_df = pd.DataFrame(detections)
        if len(det_df) > 0:
            st.dataframe(det_df[["label", "confidence"]], use_container_width=True)
        else:
            st.warning("No objects detected. Try lowering confidence or using a clearer image.")

        st.subheader("Violation Evidence Report")
        report_df = pd.DataFrame(records)
        if len(report_df) > 0:
            st.dataframe(report_df, use_container_width=True)
            st.bar_chart(report_df["violation_type"].value_counts())

            csv = report_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download evidence report CSV",
                data=csv,
                file_name="gridlens_evidence_report.csv",
                mime="text/csv",
            )
        else:
            st.success("No high-confidence violation detected. Marked clean / no actionable violation in this frame.")

        if enable_ocr:
            st.subheader("OCR Plate/Text Candidates")
            st.write(plate_candidates)

        st.divider()
        st.subheader("Deployment Note")
        st.write("""
        This prototype demonstrates the evidence-generation pipeline. In real deployment, illegal parking,
        stop-line, and red-light violations should be confirmed using camera calibration, region-of-interest
        zones, signal state integration, and multi-frame tracking.
        """)

    except Exception as e:
        st.error("Model could not run. Check requirements installation and internet/model path.")
        st.exception(e)
