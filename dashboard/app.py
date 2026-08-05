"""Phase 7 Streamlit dashboard: a read-only viewer over the OTel trace file,
the live Qdrant corpus, and the deletion-receipt audit trail.

Does not submit queries itself -- generate traffic by hitting the running
`/query` endpoint (`uvicorn src.api.main:app`) or an `eval/run_eval.py` stage,
then use the sidebar "Refresh" button (Streamlit reruns the whole script on
any interaction, which is enough for a single-user local demo).

Run from the repo root: `streamlit run dashboard/app.py` -- the trace file
and `receipts/` default paths are resolved relative to the current working
directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` only adds this file's own directory (dashboard/) to
# sys.path, not the repo root -- unlike `pytest` (which resolves `import
# src...`/`import dashboard...` via `pythonpath = ["."]` in pyproject.toml),
# Streamlit has no equivalent config hook. Without this, `import src.*` and
# `import dashboard.*` below fail with `ModuleNotFoundError: No module named
# 'src'` regardless of which directory `streamlit run` is invoked from. This
# must run before any `src`/`dashboard` import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from src.observability.config import DEFAULT_CONFIG as OBSERVABILITY_CONFIG  # noqa: E402
from src.unlearning.config import DEFAULT_CONFIG as UNLEARNING_CONFIG  # noqa: E402
from src.vectorstore.schema import TrustTier  # noqa: E402

from dashboard.data import (  # noqa: E402
    detection_rate_timeseries,
    live_trust_tier_distribution,
    load_trace_events,
    spans_to_dataframe,
    verified_receipts,
)

# Fixed-order categorical palette (dataviz skill's reference instance,
# references/palette.md): slots 1/2/3 (blue/orange/aqua) are the only
# slot-count that validates all-pairs CVD separation in both light and dark
# modes. Reused across two separate 3-category charts below (trust tiers,
# gate decisions) -- color identity is per-chart, reuse across unrelated
# charts is fine.
_CATEGORICAL_3 = ["#2a78d6", "#eb6834", "#1baf7a"]
_SEQUENTIAL_BLUE = "#2a78d6"

st.set_page_config(page_title="VeritasRAG Observability", layout="wide")
st.title("VeritasRAG — Live Observability Dashboard")
st.caption(
    "Read-only viewer. Generate traffic by hitting the running `/query` endpoint "
    "or an eval script, then refresh."
)

if st.sidebar.button("Refresh"):
    st.rerun()

events = load_trace_events(OBSERVABILITY_CONFIG.trace_file_path)
spans_df = spans_to_dataframe(events)

st.header("Live Query Traces")
root_spans = spans_df[spans_df["name"] == "veritasrag.query"].sort_values(
    "start_time", ascending=False
)
if root_spans.empty:
    st.info("No query traces recorded yet. Trace file: " + OBSERVABILITY_CONFIG.trace_file_path)
else:
    table = root_spans.copy()
    table["question"] = table["attributes"].apply(lambda a: a.get("veritasrag.query.question", ""))
    st.dataframe(
        table[["start_time", "question", "duration_ms", "trace_id"]],
        hide_index=True,
    )

    for _, root in root_spans.head(20).iterrows():
        question = root["attributes"].get("veritasrag.query.question", "(no question recorded)")
        with st.expander(f"{root['start_time']} — {question}"):
            children = spans_df[spans_df["trace_id"] == root["trace_id"]].sort_values("start_time")
            st.dataframe(
                children[["name", "start_time", "duration_ms", "parent_span_id"]],
                hide_index=True,
            )

st.header("Detection Rate Over Time")
rate_df = detection_rate_timeseries(spans_df)
if rate_df.empty:
    st.info("No retrieval spans recorded yet.")
else:
    chart_df = rate_df.set_index("start_time")
    st.line_chart(chart_df[["detection_rate"]], color=[_SEQUENTIAL_BLUE])
    st.caption("Gate-decision breakdown (per query): allow / downweight / drop")
    st.line_chart(chart_df[["allow_count", "downweight_count", "drop_count"]], color=_CATEGORICAL_3)

st.header("Trust-Tier Distribution (Live Corpus Snapshot)")
try:
    tier_counts = live_trust_tier_distribution()
except Exception as exc:  # noqa: BLE001 -- surface any Qdrant connectivity issue as a page warning
    st.warning(f"Could not reach Qdrant for a live trust-tier snapshot: {exc}")
else:
    ordered_tiers = [
        TrustTier.VERIFIED.value,
        TrustTier.UNVERIFIED.value,
        TrustTier.QUARANTINED.value,
    ]
    wide = pd.DataFrame([{tier: tier_counts.get(tier, 0) for tier in ordered_tiers}])
    st.bar_chart(wide, color=_CATEGORICAL_3)

st.header("Deletion Audit Log")
receipts_df = verified_receipts(UNLEARNING_CONFIG.receipt_dir)
if receipts_df.empty:
    st.info("No deletion receipts found.")
else:
    st.dataframe(receipts_df, hide_index=True)
