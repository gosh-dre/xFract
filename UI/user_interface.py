import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path

# =========================
# 1) CONFIG & THEME
# =========================
st.set_page_config(layout="wide", page_title="xFract AI | Neural Review", page_icon="🧬")

st.markdown(
    """
    <style>
    .stApp { background: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important; border-right: 1px solid #30363d;
    }
    .report-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #30363d;
        margin-bottom: 15px;
    }
    .outcome-box {
        background: linear-gradient(90deg, #1f6feb 0%, #8957e5 100%);
        padding: 2px; border-radius: 10px; margin-top: 20px;
    }
    .outcome-inner {
        background: #0d1117; padding: 15px; border-radius: 8px; text-align: center;
    }
    h3 { color: #58a6ff !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ====================================
# 2) CSV DATA UTILITIES
# ====================================
# Load config file
config_path = "config/config.yaml"
config = data_ingestion.load_config(config_path)

output_folder = config["output"]["output_path"]
CSV_path = os.path.join(output_folder, "final_results.csv")

REQUIRED_COLUMNS = [
    "id", "patient", "report", "evidence", "rationale",
    "outcome", "verification", "status", "feedback"
]

PLACEHOLDER_VALUES = {
    "report_findings": "No findings available.",
    "rationale": "AI rationale not generated.",
    "Confidence_score": "N/A",
    "outcome": "Pending outcome",
    "status": "Pending",
    "feedback": ""
}

def load_data(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        st.error(f"CSV not found at: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)

    # Map narrative → report if needed
    if "report" not in df.columns:
        if "narrative" in df.columns:
            df["report"] = df["narrative"]
        else:
            df["report"] = ""

    # Ensure patient column
    if "patient" not in df.columns:
        df["patient"] = [f"Patient {i+1}" for i in range(len(df))]

    # Ensure id column and make it numeric/unique-ish for buttons
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    else:
        # Coerce to int if possible; fill missing with new seq
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        if df["id"].isna().any():
            # Assign ids to NaNs without disturbing existing ids
            base = df["id"].dropna()
            start = int(base.max() + 1) if not base.empty else 1
            new_ids = []
            counter = start
            for is_na in df["id"].isna():
                if is_na:
                    new_ids.append(counter)
                    counter += 1
                else:
                    new_ids.append(None)
            df.loc[df["id"].isna(), "id"] = [i for i in new_ids if i is not None]
        df["id"] = df["id"].astype(int)

    # Ensure placeholders for missing columns
    for col, default in PLACEHOLDER_VALUES.items():
        if col not in df.columns:
            df[col] = default

    # Keep only expected cols + any extra columns for safety
    return df

def save_data(df: pd.DataFrame, csv_path: str):
    df.to_csv(csv_path, index=False)

# Load once per run (no cache to keep writes simple with st.rerun)
df = load_data(CSV_PATH)

# ====================================
# 3) SESSION STATE
# ====================================
if "selected_id" not in st.session_state:
    st.session_state.selected_id = None
if "mode" not in st.session_state:
    st.session_state.mode = None  # "review" or "view"

# ==================
# 4) SIDEBAR
# ==================
st.sidebar.markdown(
    "<h1 style='text-align: center; color:#58a6ff;'>xFract <span style='color:#8957e5'>AI</span></h1>",
    unsafe_allow_html=True
)

# Build a lightweight view for buttons
all_reports = df[["id", "patient", "status"]].copy()

pending = all_reports[all_reports["status"] == "Pending"]
archived = all_reports[all_reports["status"] != "Pending"]

st.sidebar.subheader("📥 PENDING")
# Create deterministic ordering for consistent button order
for _, row in pending.sort_values(by=["patient", "id"]).iterrows():
    if st.sidebar.button(f"📄 {row['patient']}", key=f"btn_p_{row['id']}", use_container_width=True):
        st.session_state.selected_id = int(row["id"])
        st.session_state.mode = "review"
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🗄️ ARCHIVE")
for _, row in archived.sort_values(by=["patient", "id"]).iterrows():
    icon = "✅" if row["status"] == "Approved" else "❌"
    if st.sidebar.button(f"{icon} {row['patient']}", key=f"btn_a_{row['id']}", use_container_width=True):
        st.session_state.selected_id = int(row["id"])
        st.session_state.mode = "view"
        st.rerun()

# =====================
# 5) MAIN CONTENT
# =====================
if st.session_state.selected_id is not None and st.session_state.selected_id in df["id"].values:
    res = df[df["id"] == st.session_state.selected_id].iloc[0].to_dict()

    # Convert decimal confidence score to a whole-number percentage
    raw_score = res.get("Confidence_score", None)

    try:
        if raw_score is not None:
            pct = round(float(raw_score) * 100)
            confidence_display = f"{pct}%"
        else:
            confidence_display = "N/A"
    except:
        confidence_display = "N/A"

    # Header
    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.markdown(
            f"## {res['patient']} <small style='color:gray'>(ID: {res['id']})</small>",
            unsafe_allow_html=True
        )
    with col_s:
        s_color = "#f1e05a" if res["status"] == "Pending" else "#3fb950" if res["status"] == "Approved" else "#f85149"
        st.markdown(
            f"<div style='text-align:right'>"
            f"<span style='background:{s_color}; color:black; padding:5px 15px; "
            f"border-radius:20px; font-weight:bold;'>{res['status']}</span></div>",
            unsafe_allow_html=True
        )

    # Grid (cards)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="report-card"><h3>📄 Original Report</h3>{res.get("report", "")}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="report-card"><h3>🧠 AI Rationale</h3>{res.get("rationale", "")}</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div class="report-card"><h3>🔍 Evidence</h3>{res.get("report_findings", "")}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="report-card" style="text-align:center;">'
            f'<h3>✅ Fact Check</h3>'
            f'<h2 style="color:#3fb950">{confidence_display}</h2>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div class="outcome-box"><div class="outcome-inner">'
        f'<h1 style="margin:0;">{str(res.get("outcome","")).upper()}</h1>'
        f'</div></div>',
        unsafe_allow_html=True
    )

    st.divider()

    # --- ACTION LOGIC ---
    if st.session_state.mode == "review":
        st.subheader("🖋️ Clinician Decision")

        # Use unique key and prefill with any existing feedback
        clinician_notes = st.text_area(
            "Observations:",
            value=str(res.get("feedback", "")),
            key=f"ta_{res['id']}"
        )

        b1, b2, _ = st.columns([1, 1, 2])

        # APPROVE
        if b1.button("✅ APPROVE", key=f"app_{res['id']}", use_container_width=True):
            # Update in-memory df
            df.loc[df["id"] == res["id"], ["status", "feedback"]] = ["Approved", clinician_notes]
            # Persist to CSV
            save_data(df, CSV_PATH)
            # Switch to view mode after decision
            st.session_state.mode = "view"
            st.success("Case approved and saved.")
            st.rerun()

        # REJECT
        if b2.button("❌ REJECT", key=f"rej_{res['id']}", use_container_width=True):
            df.loc[df["id"] == res["id"], ["status", "feedback"]] = ["Rejected", clinician_notes]
            save_data(df, CSV_PATH)
            st.session_state.mode = "view"
            st.warning("Case rejected and saved.")
            st.rerun()

    else:
        # VIEWER MODE
        st.subheader("👁️ Archive Viewer")
        past_fb = res.get("feedback", "")
        st.info(f"**Past Feedback:** {past_fb if past_fb else 'None provided.'}")

        if st.button("♻️ Re-open Case", key=f"reopen_{res['id']}"):
            df.loc[df["id"] == res["id"], "status"] = "Pending"
            save_data(df, CSV_PATH)
            st.session_state.mode = "review"
            st.rerun()

else:
    st.info("Select a patient record from the sidebar to begin analysis.")