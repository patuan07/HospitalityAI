import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"

try:
    API_URL = st.secrets.get("API_URL", DEFAULT_API_URL)
except Exception:
    API_URL = DEFAULT_API_URL

st.set_page_config(page_title="Hospitality AI Product Demo", layout="wide")

st.title("Hospitality AI — Product Demo")
st.caption("Upload a bed image. The template runs Stage 1→4 and visualizes outputs.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

# Define columns for the main layout
colA, colB = st.columns([1, 1])

with colB:
    run_stage5 = st.checkbox("Run Stage 5 (Robustness)", value=False)

if uploaded is not None:
    # 1. Input Section (Top-Left)
    with colA:
        st.subheader("Input Image")
        st.image(uploaded, use_container_width=True)

    if st.button("Analyze", type="primary"):
        files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        with st.spinner("Running deep learning pipeline..."):
            try:
                r = requests.post(
                    f"{API_URL}/analyze",
                    params={"run_stage5": str(run_stage5).lower()},
                    files=files,
                    timeout=180,
                )
                
                if r.status_code != 200:
                    st.error(f"Backend error: {r.status_code}\n{r.text}")
                else:
                    data = r.json()

                    # 2. Results Section (Top-Right)
                    with colB:
                        # --- Stage 1: Binary Made/Unmade ---
                        st.subheader("Stage 1 — Binary Classification")
                        s1_col1, s1_col2 = st.columns(2)
                        with s1_col1:
                            st.metric("P(made)", f"{data['stage1']['prob_made']:.3f}")
                        with s1_col2:
                            top_label = data['stage1'].get('debug', {}).get('top_label', 'Unknown')
                            if data["stage1"]["pred_made"]:
                                st.success(f"Result: {top_label.upper()}")
                            else:
                                st.error(f"Result: {top_label.upper()}")

                        # --- Stage 2: Multi-label Summary (Progress Bars) ---
                        st.subheader("Stage 2 — Defect Severity")
                        defects = data.get("defects", {}) 
                        if defects:
                            for label, prob in defects.items():
                                display_name = label.replace('_', ' ').title()
                                st.progress(float(max(0.0, min(1.0, prob))), text=f"{display_name}: {prob:.2f}")
                        else:
                            st.info("No defects detected.")

                        # --- Stage 3: Spatial Localizations (Table) ---
                        st.subheader("Stage 3 — Spatial Localizations")
                        localizations = data.get("localizations", [])
                        if localizations:
                            ui_table = []
                            for det in localizations:
                                if isinstance(det, dict):
                                    label = det.get("label", "Unknown")
                                    conf = det.get("confidence", 0.0)
                                    pos_info = det.get("method", "yolo")
                                else:
                                    label = getattr(det, "label", "Unknown")
                                    conf = getattr(det, "confidence", 0.0)
                                    pos_info = getattr(det, "method", "yolo")

                                ui_table.append({
                                    "Defect Type": label,
                                    "Confidence": f"{conf:.2f}",
                                    "Position": pos_info
                                })
                            st.table(ui_table)
                        
                        # --- Stage 4: Geometry Alignment ---
                        st.subheader("Stage 4 — Geometry Alignment")
                        s4_col1, s4_col2 = st.columns(2)
                        with s4_col1:
                            st.metric("Score", f"{data['alignment_score']:.3f}")
                        with s4_col2:
                            if data["alignment_pass"]:
                                st.success("ALIGNMENT PASS")
                            else:
                                st.error("ALIGNMENT FAIL")

                    # 3. Visual Artifacts Section (Bottom-Left - Now below the input)
                    with colA:
                        st.divider()
                        st.subheader("Visual Analysis Results")
                        art = data.get("artifacts", {})
                        if art:
                            # Prioritize the YOLO detection image for full-width display
                            yolo_url = art.get("YOLO Detections")
                            if yolo_url:
                                st.write("**YOLO Detection Overlay**")
                                st.image(yolo_url, use_container_width=True, caption="Stage 2 & 3 Output")
                            
                            # Display other artifacts (like Stage 4) in smaller columns below
                            other_arts = {k: v for k, v in art.items() if k != "YOLO Detections"}
                            if other_arts:
                                st.write("**Other Geometric Artifacts**")
                                sub_cols = st.columns(len(other_arts))
                                for i, (k, url) in enumerate(other_arts.items()):
                                    with sub_cols[i]:
                                        st.image(url, use_container_width=True, caption=k)
                        else:
                            st.info("Waiting for visual artifacts...")

                    # --- Stage 5: Robustness (Bottom-Right) ---
                    if data.get("stage5"):
                        with colB:
                            st.divider()
                            st.subheader("Stage 5 — System Robustness")
                            st.metric("Robustness Score", f"{data['stage5']['robustness_score']:.3f}")

            except requests.exceptions.RequestException as e:
                st.error(f"API Connection Error: {e}")