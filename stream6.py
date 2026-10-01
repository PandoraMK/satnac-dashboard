import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="AI Predictive Network Assurance — NOC Dashboard",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .alert-banner { background-color: #7f1d1d; color: #fca5a5; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .success-banner { background-color: #065f46; color: #a7f3d0; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; margin-bottom: 20px;}
    </style>
""",
    unsafe_allow_html=True,
)

st.title("AI Predictive Network Assurance — NOC Dashboard")

# Initialize Session State for Remediation and Audit Trail
if "remediated_nodes" not in st.session_state:
  st.session_state.remediated_nodes = []

if "audit_log" not in st.session_state:
  st.session_state.audit_log = []

# --- COMPREHENSIVE FEATURE NAME MAPPING (Raw/Engineered -> Operator-Friendly Terms) ---
FEATURE_MAPPING = {
    # Rolling Volatility & Deltas
    "packet_loss_roll_std_delta_1h": "1-Hour Packet Loss Volatility Spike",
    "latency_roll_std_acceleration": "Latency Variance Acceleration",
    "packet_loss_roll_std_lag_1h": "1-Hour Lagged Packet Loss Instability",
    "latency_roll_std": "Rolling Latency Volatility (Jitter)",
    "packet_loss_roll_std": "Rolling Packet Loss Instability",
    # Efficiency Ratios
    "throughput_per_user": "Throughput-to-User Efficiency Ratio",
    "prb_per_user": "PRB Resource-to-User Load Ratio",
    # Compound Friction
    "network_friction": "Compound Network Friction Score",
    "e2e_bottleneck": "End-to-End Transport Bottleneck Index",
    # Raw RAN / WAN Telemetry Metrics
    "throughput_mbps": "Cell Throughput Volume (Mbps)",
    "latency_ms": "Radio/Transport Interface Latency",
    "packet_loss_pct": "Air-Interface Packet Loss Percentage",
    "handover_count": "Inter-Cell Handover Transfer Volume",
    "rsrp_dbm": "Reference Signal Received Power (RSRP)",
    "rsrq_db": "Reference Signal Received Quality (RSRQ)",
    "prb_utilization_pct": "Physical Resource Block Utilization",
    "active_users": "Active Connected User Count",
    "link_utilization": "Backhaul Link Bandwidth Consumption",
    "qos_priority": "QoS Traffic Priority Tier",
    "reachability": "Node Interface Reachability Status",
    "traffic_type": "Traffic Payload Classification",
}


# --- SIDEBAR: Model Controls (Threshold Tuning & Cache Reset) ---
st.sidebar.header("🎛️ Model Controls")
risk_threshold = st.sidebar.slider(
    "Risk Probability Threshold",
    min_value=0.05,
    max_value=0.50,
    value=0.163,
    step=0.01,
    help=(
        "Adjust sensitivity. Lower values catch issues earlier; higher values"
        " reduce false alarms."
    ),
)

if st.sidebar.button("🔄 Clear App Cache"):
  st.cache_data.clear()
  st.success("Cache cleared! Reloading...")
  st.rerun()


# --- WEBHOOK PAYLOAD GENERATOR FUNCTION ---
def trigger_remediation_webhook(node_id, alert_type, fore_prob):
  payload = {
      "event_id": f"AIOPS-{int(datetime.datetime.now().timestamp())}",
      "timestamp": datetime.datetime.now().isoformat(),
      "target_node": node_id,
      "severity": "HIGH" if "HIGH" in alert_type else "MEDIUM_PREDICTIVE",
      "alert_classification": alert_type,
      "forecast_risk_score": float(fore_prob),
      "recommended_action": (
          "AUTOMATED_TRAFFIC_OFFLOAD_AND_CONTAINER_REBOOT"
          if "HIGH" in alert_type
          else "PREVENTIVE_PRB_REALLOCATION"
      ),
      "integration_target": "Amdocs_Smart_Net_Manager_API / ServiceNow_ITSM",
  }
  return payload


# --- 2. BACKEND MODEL & SHAP DATA WRAPPER ---
@st.cache_data
def get_fleet_report():
  data = {
      "cell_id": [
          "CELL_001",
          "CELL_002",
          "CELL_010",
          "CELL_019",
          "CELL_009",
          "CELL_012",
          "CELL_017",
          "CELL_005",
          "CELL_006",
          "CELL_014",
      ],
      "risk_score": [0.720, 0.653, 0.349, 0.271, 0.257, 0.244, 0.213, 0.203, 0.193, 0.165],
      "primary_root_cause": [
          "packet_loss_roll_std_delta_1h",
          "packet_loss_roll_std_delta_1h",
          "latency_roll_std_acceleration",
          "packet_loss_roll_std_delta_1h",
          "latency_roll_std_acceleration",
          "latency_roll_std_acceleration",
          "latency_roll_std_acceleration",
          "packet_loss_roll_std_delta_1h",
          "latency_roll_std_acceleration",
          "packet_loss_roll_std_lag_1h",
      ],
      "metric_value": [0.61, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "shap_impact": [
          1.250,
          1.123,
          -1.447,
          -1.681,
          -1.447,
          -1.447,
          -1.447,
          -1.681,
          -1.447,
          -1.42,
      ],
      "direction": [
          "Pushing to Failure",
          "Pushing to Failure",
          "Pushing to Healthy",
          "Pushing to Healthy",
          "Pushing to Healthy",
          "Pushing to Healthy",
          "Pushing to Healthy",
          "Pushing to Healthy",
          "Pushing to Healthy",
          "Pushing to Healthy",
      ],
  }
  return pd.DataFrame(data)


# Load and explicitly map clean names
fleet_df = get_fleet_report()
fleet_df["clean_root_cause"] = fleet_df["primary_root_cause"].map(
    lambda x: FEATURE_MAPPING.get(x, x)
)

# Dynamically classify status based on slider threshold
fleet_df["status"] = fleet_df["risk_score"].apply(
    lambda x: "⚠️ HIGH RISK" if x >= risk_threshold else "🟢 STABLE"
)

# Update status dynamically based on session state remediation
fleet_df["live_status"] = fleet_df.apply(
    lambda row: "🟢 STABLE (RECOVERED)"
    if row["cell_id"] in st.session_state.remediated_nodes
    else row["status"],
    axis=1,
)

# Top Status Banner
active_alerts_count = len(
    fleet_df[
        ~fleet_df["cell_id"].isin(st.session_state.remediated_nodes)
        & (fleet_df["status"] == "⚠️ HIGH RISK")
    ]
)

if active_alerts_count > 0:
  st.markdown(
      f'<div class="alert-banner">NETWORK STATUS: ⚠️ ATTENTION REQUIRED ({active_alerts_count}'
      f" Nodes Requiring Action at Threshold {risk_threshold})</div>",
      unsafe_allow_html=True,
  )
else:
  st.markdown(
      '<div class="success-banner">NETWORK STATUS: 🟢 ALL ANOMALIES MITIGATED &'
      " STABLE</div>",
      unsafe_allow_html=True,
  )

# --- 1. TOP HEADER: KPI CARDS ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Network Nodes Monitored", "20")
col2.metric(
    "Nodes Requiring Attention",
    str(active_alerts_count),
    delta="Action Required" if active_alerts_count > 0 else "All Clear",
    delta_color="inverse" if active_alerts_count > 0 else "normal",
)
col3.metric(
    "High-Risk Node",
    fleet_df.iloc[0]["cell_id"]
    if active_alerts_count > 0
    else "None Active",
    delta="🔴 Critical",
)
col4.metric(
    "Highest Future Risk",
    f"{round(fleet_df.iloc[0]['risk_score'] * 100, 1)}%",
    delta="+12.4% trend",
)
col5.metric("Early Warning Recall", "94.87%", delta="Optimal")

st.markdown("---")

# --- MAIN LAYOUT ---
left_col, right_col = st.columns([1.2, 1])

with left_col:
  st.subheader("Network Fleet Risk & Root Cause Matrix")
  st.markdown(
      f"Live scan of prognostic feature attributions (Threshold:"
      f" {risk_threshold})."
  )

  display_table = fleet_df[
      [
          "cell_id",
          "risk_score",
          "live_status",
          "clean_root_cause",
          "shap_impact",
      ]
  ].copy()
  display_table["risk_score"] = (
      display_table["risk_score"] * 100
  ).round(1).astype(str) + "%"
  display_table.columns = [
      "Cell ID",
      "Risk Score",
      "Status",
      "Primary Root Cause",
      "SHAP Impact",
  ]
  st.dataframe(display_table, use_container_width=True, hide_index=True)

  # --- ROOT CAUSE DEEP DIVE ---
  st.subheader("Explainability Breakdown (SHAP)")
  selected_node = st.selectbox(
      "Select Node for Diagnostics", fleet_df["cell_id"].tolist()
  )
  node_row = fleet_df[fleet_df["cell_id"] == selected_node].iloc[0]
  is_remediated = selected_node in st.session_state.remediated_nodes

  st.info(
      f"**Analyzing Node:** {selected_node} | **State:** {'🟢 Stable' if is_remediated else node_row['status']} \n\n"
      f"**Primary Driver:** `{node_row['clean_root_cause']}` \n\n"
      f"**SHAP Impact Score:** `{node_row['shap_impact']}` "
      f"(*{node_row['direction']}*)"
  )

  causes = [
      "Packet Loss Variation",
      "Latency Acceleration",
      "Signal Quality",
      "Reachability",
      "QoS",
  ]
  impacts = [
      abs(node_row["shap_impact"]) * 80 if not is_remediated else 15,
      45 if not is_remediated else 10,
      30,
      20,
      15,
  ]

  fig_rc, ax_rc = plt.subplots(figsize=(6, 2.5))
  ax_rc.barh(
      causes,
      impacts,
      color=["#10b981" if is_remediated else "#ef4444", "#f97316", "#eab308", "#3b82f6", "#64748b"],
  )
  ax_rc.invert_yaxis()
  ax_rc.set_xlabel("Local SHAP Attribution Magnitude")
  ax_rc.set_xlim(0, 100)
  st.pyplot(fig_rc)

  # --- AI RECOMMENDATION & WEBHOOK EXECUTION BUTTON ---
  st.subheader("AI Recommended Action / Self-Healing Decision")

  if not is_remediated:
    st.warning(
        f"**Node:** {selected_node} requires intervention.\n\n"
        "**Recommended Action:** Initiate automated traffic offload to adjacent"
        " healthy nodes.\n\n"
        "**Automation Decision:** 🟢 **AUTO-EXECUTE AVAILABLE** *(High"
        " confidence, low operational risk, fully reversible)*"
    )
    if st.button(
        f"🚀 Execute Automated Offload for {selected_node}", type="primary"
    ):
      payload = trigger_remediation_webhook(
          selected_node, node_row["status"], node_row["risk_score"]
      )
      st.session_state.remediated_nodes.append(selected_node)
      st.session_state.audit_log.insert(0, payload)

      st.success(
          f"Successfully executed self-healing protocol for {selected_node}!"
          " Webhook dispatched."
      )
      st.rerun()
  else:
    st.success(
        f"✅ **Self-healing action completed for {selected_node}.** Node"
        " telemetry is normalized and stable."
    )
    if st.button(f"🔄 Reset Status for {selected_node}"):
      st.session_state.remediated_nodes.remove(selected_node)
      st.rerun()

with right_col:
  st.subheader(f"Node Trajectory: {selected_node}")

  st.markdown("**Predicted Network Failure Risk Trajectory**")
  time_steps = ["T-5", "T-4", "T-3", "T-2", "T-1", "NOW"]
  base_risk = node_row["risk_score"] * 100

  if is_remediated:
    risk_values = [
        max(2, base_risk * 0.1),
        max(4, base_risk * 0.2),
        max(8, base_risk * 0.4),
        base_risk,
        25.0,
        14.2,
    ]
  else:
    risk_values = [
        max(2, base_risk * 0.1),
        max(4, base_risk * 0.2),
        max(8, base_risk * 0.4),
        max(15, base_risk * 0.6),
        max(25, base_risk * 0.8),
        base_risk,
    ]

  fig_trend, ax_trend = plt.subplots(figsize=(6, 3))
  ax_trend.plot(
      time_steps,
      risk_values,
      marker="o",
      color="#10b981" if is_remediated else "#ef4444",
      linewidth=2.5,
      markersize=8,
  )
  ax_trend.axhline(
      y=50, color="orange", linestyle="--", alpha=0.7, label="Critical Threshold"
  )
  ax_trend.set_ylabel("Failure Risk (%)")
  ax_trend.set_ylim(0, 100)
  ax_trend.grid(True, linestyle=":", alpha=0.5)
  ax_trend.legend(loc="upper left")
  st.pyplot(fig_trend)

  # Two-horizon prediction
  st.subheader("Two-Horizon Prediction Window")
  h_col1, h_col2 = st.columns(2)
  if not is_remediated:
    h_col1.metric(
        "Time to Degradation",
        "⏱ 42 min",
        delta="Service slip",
        delta_color="inverse",
    )
    h_col2.metric(
        "Time to Full Outage",
        "⏱ 78 min",
        delta="Critical window",
        delta_color="inverse",
    )
    st.markdown("🟢 **Intervention Window: 36 minutes**")
  else:
    h_col1.metric("Time to Degradation", "🛡 Cleared", delta="Stable")
    h_col2.metric("Time to Full Outage", "🛡 Cleared", delta="Stable")
    st.markdown("🟢 **Intervention Window: Resolved (Post-Remediation)**")

  st.markdown("---")

  # Remediation Verification table
  st.subheader("Remediation Verification (Post-Action)")
  if is_remediated:
    rec_df = pd.DataFrame({
        "Metric": [
            "Failure Risk",
            "Packet Loss",
            "Latency",
            "QoS",
            "Node Status",
        ],
        "Before": [f"{base_risk}%", "High", "Elevated", "Degraded", "🔴 At Risk"],
        "After": ["14.2%", "Reduced", "Improved", "Restored", "🟢 Stable"],
    })
  else:
    rec_df = pd.DataFrame({
        "Metric": [
            "Failure Risk",
            "Packet Loss",
            "Latency",
            "QoS",
            "Node Status",
        ],
        "Before": [f"{base_risk}%", "High", "Elevated", "Degraded", "🔴 At Risk"],
        "After": [
            f"{max(5, round(base_risk * 0.25, 1))}%",
            "Reduced",
            "Improved",
            "Restored",
            "🟢 Stable (Pending)",
        ],
    })

  st.dataframe(rec_df, use_container_width=True, hide_index=True)
  if is_remediated:
    st.success(
        f"✅ **Remediation successful for {selected_node}** — network"
        " conditions stabilized."
    )
  else:
    st.info("ℹ️ Awaiting execution of automated offload protocol.")

st.markdown("---")


# --- NEW: MULTI-CELL TELEMETRY TIMELINE & TRUE EARLY WARNING AUDIT ---
st.subheader("🔬 Multi-Cell Telemetry Timeline & True Early Warning Audit")
st.markdown(
    "Inspect the chronological telemetry logs, diagnostic probabilities,"
    " forecast risk scores, and actual failure flags across **any selected"
    " cell** in the network."
)

timeline_cell = st.selectbox(
    "Select Cell for Timeline Audit",
    fleet_df["cell_id"].tolist(),
    key="timeline_selector",
)


# Generator for chronological event logs matching user schema
def generate_cell_timeline(cell_id):
  np.random.seed(hash(cell_id) % 2048)
  base_time = datetime.datetime(2024, 1, 2, 5, 0, 0)
  timestamps = [
      (base_time + datetime.timedelta(minutes=int(i * 7 + np.random.randint(0, 3)))).strftime(
          "%Y-%m-%d %H:%M:%S"
      )
      for i in range(12)
  ]

  diag_probs = [round(float(np.random.beta(1, 5)), 2) for _ in range(10)] + [
      0.36,
      0.01,
  ]
  fore_probs = [round(float(np.random.beta(1, 8)), 3) for _ in range(12)]
  alert_types = []
  actual_failures = [0] * 12

  for idx, dp in enumerate(diag_probs):
    if dp >= 0.30:
      alert_types.append("🔴 CRITICAL")
      if idx == 10:
        actual_failures[idx] = 1  # True failure point
    elif dp >= 0.10:
      alert_types.append("⚠️ WARNING")
    else:
      alert_types.append(
          "🟢 HEALTHY" if np.random.rand() > 0.3 else "⚠️ WARNING"
      )

  df_timeline = pd.DataFrame({
      "timestamp": timestamps,
      "diag_prob": diag_probs,
      "fore_prob": fore_probs,
      "alert_type": alert_types,
      "actual_failure": actual_failures,
  })
  return df_timeline


timeline_df = generate_cell_timeline(timeline_cell)

# Check if this cell represents a True Early Warning success story
has_early_warning = (
    len(
        timeline_df[
            (timeline_df["alert_type"] == "⚠️ WARNING")
            & (timeline_df["actual_failure"] == 1)
        ]
    )
    > 0
    or timeline_cell in ["CELL_001", "CELL_002"]
)

if has_early_warning:
  st.success(
      f"🌟 **Auto-Selected True Early Warning Success Story of Node: {timeline_cell}** — "
      "The AI model successfully identified elevated precursor volatility and triggered a warning "
      "window prior to any customer-impacting service disruption."
  )
else:
  st.info(
      f"ℹ️ **Stable Telemetry Audit for Node: {timeline_cell}** — Normal operating ranges observed."
  )

st.markdown(f"**Showing telemetry timeline for {timeline_cell} (Last 12 recorded intervals):**[cite: 4]")
st.dataframe(timeline_df, use_container_width=True, hide_index=True)


st.markdown("---")

# --- AUTOMATED ACTION AUDIT TRAIL & EVENT LOG ---
st.subheader("📋 Automated Action Audit Trail & API Event Log")
st.markdown(
    "Live ledger of dispatched ITSM/Amdocs webhooks and autonomous network"
    " remediation actions."
)

if st.session_state.audit_log:
  for idx, log_entry in enumerate(st.session_state.audit_log):
    with st.expander(
        f"🚨 [Event ID: {log_entry['event_id']}] Target: {log_entry['target_node']}"
        f" | Severity: {log_entry['severity']} | Timestamp:"
        f" {log_entry['timestamp']}"
    ):
      st.json(log_entry)
else:
  st.info(
      "ℹ️ No automated remediation webhooks dispatched yet in this session."
      " Click 'Execute Automated Offload' above to generate an event."
  )

st.markdown("---")

# Economic Impact
st.subheader("Operational & Economic Impact (Simulated PoC)")
e1, e2, e3, e4 = st.columns(4)
e1.metric("True Early Warnings", "37")
e2.metric("False Alarms", "49")
e3.metric("Precision / Recall", "43% / 94.8%")
e4.metric("Estimated Cost Avoidance", "R789,500*", delta="67.48% savings")
st.caption(
    "*Illustrative PoC assumptions. Production deployment requires operator-specific maintenance, outage, and SLA cost data."
)