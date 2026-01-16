import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"

try:
    API_URL = st.secrets.get("API_URL", DEFAULT_API_URL)
except Exception:
    API_URL = DEFAULT_API_URL

st.set_page_config(page_title="Hospitality AI Product Demo", layout="wide")

st.title("Hospitality AI — Product Demo")
st.caption("Upload a bed image. The template runs Stage 1→4 (and optional Stage 5) and visualizes outputs.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

colA, colB = st.columns([1, 1])

with colB:
    run_stage5 = st.checkbox("Run Stage 5 (Robustness)", value=False)

if uploaded is not None:
    with colA:
        st.subheader("Input")
        st.image(uploaded, use_container_width=True)

    if st.button("Analyze", type="primary"):
        files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        with st.spinner("Running analysis..."):
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

            with colB:
                st.subheader("Stage 1 — Binary")
                st.metric("P(made)", f"{data['stage1']['prob_made']:.3f}")
                st.success("MADE") if data["stage1"]["pred_made"] else st.error("NOT MADE")

                st.subheader("Stage 2 — Multi-label")
                st.json(data.get("defects", {}), expanded=False)

                st.subheader("Stage 4 — Geometry alignment")
                st.metric("Alignment score", f"{data['alignment_score']:.3f}")
                st.success("PASS") if data["alignment_pass"] else st.error("FAIL")

                if data.get("stage5"):
                    st.subheader("Stage 5 — Robustness")
                    st.metric("Robustness score", f"{data['stage5']['robustness_score']:.3f}")

            st.divider()
            st.subheader("Artifacts")
            art = data.get("artifacts", {})
            if not art:
                st.info("No artifacts returned.")
            else:
                for k, url in art.items():
                    st.write(k)
                    st.image(url, use_container_width=True)

            st.divider()
            st.subheader("Stage 3 — Localizations")
            locs = data.get("localizations", [])
            if not locs:
                st.info("No localizations (try an image with higher defect probability).")
            for loc in locs:
                st.write(f"**{loc['label']}** — conf={loc['confidence']:.2f} ({loc['method']})")
                c1, c2 = st.columns(2)
                if loc.get("heatmap_path"):
                    with c1:
                        st.caption("Heatmap")
                        st.image(loc["heatmap_path"], use_container_width=True)
                if loc.get("overlay_path"):
                    with c2:
                        st.caption("Overlay")
                        st.image(loc["overlay_path"], use_container_width=True)
