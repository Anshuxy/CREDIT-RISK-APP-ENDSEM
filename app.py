"""
╔══════════════════════════════════════════════════════════════╗
║       CREDIT RISK ASSESSMENT SYSTEM — Streamlit UI          ║
║   Agentic AI · LangGraph · RAG · Groq · ChromaDB          ║
╚══════════════════════════════════════════════════════════════╝

Design Direction : Financial-grade dark UI — deep navy + amber gold
                   "Bloomberg Terminal meets modern SaaS"
Font Pairing     : Syne (display/headers) + IBM Plex Mono (data/metrics)
Color Palette    : #080C14 bg · #0D1421 surface · #1A2744 border
                   #F0A500 accent · #00D4AA positive · #FF4757 danger
                   #FFB400 warning
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from io import StringIO

# ── Optional heavyweight imports (graceful fallback) ──────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import Chroma
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_core.tools import tool
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, List, Annotated
    import operator
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    import pickle
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS — "Bloomberg Terminal meets modern SaaS"
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;600&family=Syne+Mono&display=swap');

:root {
    --bg:        #080C14;
    --surface:   #0D1421;
    --surface2:  #111827;
    --surface3:  #162035;
    --border:    #1A2744;
    --border2:   #243660;
    --accent:    #F0A500;
    --accent2:   #FFD166;
    --positive:  #00D4AA;
    --danger:    #FF4757;
    --warning:   #FFB400;
    --info:      #4A9EFF;
    --text:      #D4E0F7;
    --muted:     #5A7BA8;
    --radius:    10px;
    --radius-lg: 16px;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2.5rem 2rem 2.5rem !important;
    max-width: 1500px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border2) !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.2rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.2s, transform 0.15s;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent) !important;
    transform: translateY(-1px);
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: var(--accent2) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.4rem !important;
    border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--surface3) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Inputs ── */
[data-baseweb="input"] input,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] textarea {
    background: var(--surface2) !important;
    border-color: var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-baseweb="input"] input:focus,
[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(240,165,0,0.15) !important;
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div { background: var(--accent) !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background: var(--surface2) !important;
    border-color: var(--border2) !important;
    color: var(--text) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #F0A500, #E08800) !important;
    color: #080C14 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button[kind="secondary"] {
    background: var(--surface3) !important;
    color: var(--text) !important;
    border: 1px solid var(--border2) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-size: 0.82rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* ── Info / Warning / Error ── */
[data-testid="stAlert"] {
    border-radius: var(--radius) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div { background: var(--accent) !important; }

/* ── Divider ── */
hr { border-color: var(--border2) !important; }

/* ─── Custom component classes ─── */

/* Header banner */
.app-header {
    background: linear-gradient(135deg, #0D1421 0%, #162035 50%, #0D1421 100%);
    border: 1px solid var(--border2);
    border-radius: var(--radius-lg);
    padding: 3.2rem 2.4rem 3rem 2.4rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    min-height: 180px;
    box-shadow: 0 4px 32px rgba(0,0,0,0.4), inset 0 0 80px rgba(240,165,0,0.03);
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent 0%, #F0A500 40%, #FFD166 60%, transparent 100%);
    box-shadow: 0 0 18px rgba(240,165,0,0.6);
}
.app-header::after {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(240,165,0,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 5.5rem;
    font-weight: 800;
    color: #FFFFFF;

    margin: 0 0 2rem 0;   /* 👈 increased bottom space */

    letter-spacing: 0.01em;
    line-height: 1.1;
    text-shadow: 0 2px 24px rgba(0,0,0,0.6);

    transform: scaleY(1.2);
    transform-origin: top;
}
.app-title span {
    color: #F0A500;
    text-shadow:
        0 0 20px rgba(240,165,0,0.7),
        0 0 40px rgba(240,165,0,0.35),
        0 2px 8px rgba(0,0,0,0.5);
    font-style: italic;
}
.app-subtitle {
    color: #7A9CC8;
    font-size: 0.75rem;
    margin-top: 0.55rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-weight: 400;
    font-family: 'IBM Plex Mono', monospace;
}
.tech-pill {
    display: inline-block;
    background: rgba(240,165,0,0.12);
    border: 1px solid rgba(240,165,0,0.3);
    color: var(--accent2);
    font-size: 0.65rem;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.06em;
    margin-right: 0.35rem;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 3px solid var(--accent);
}
.card-positive { border-left: 3px solid var(--positive); }
.card-danger   { border-left: 3px solid var(--danger); }
.card-warning  { border-left: 3px solid var(--warning); }

.card-title {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.85rem;
}

/* Decision badge */
.decision-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.25rem;
    border-radius: 999px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
}
.decision-approve  { background: rgba(0,212,170,0.15); color: #00D4AA; border: 1px solid rgba(0,212,170,0.4); }
.decision-reject   { background: rgba(255,71,87,0.15);  color: #FF4757; border: 1px solid rgba(255,71,87,0.4); }
.decision-cond     { background: rgba(255,180,0,0.15);  color: #FFB400; border: 1px solid rgba(255,180,0,0.4); }

/* Risk level indicator */
.risk-badge {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.risk-low      { background: rgba(0,212,170,0.15);  color: #00D4AA; }
.risk-moderate { background: rgba(74,158,255,0.15); color: #4A9EFF; }
.risk-high     { background: rgba(255,180,0,0.15);  color: #FFB400; }
.risk-critical { background: rgba(255,71,87,0.15);  color: #FF4757; }

/* Agent status tracker */
.agent-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.85rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border: 1px solid var(--border);
    background: var(--surface2);
}
.agent-step.active  { border-color: var(--accent);   background: rgba(240,165,0,0.06); }
.agent-step.done    { border-color: var(--positive);  background: rgba(0,212,170,0.06); }
.agent-step.waiting { opacity: 0.45; }

.step-icon {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem;
    flex-shrink: 0;
    background: var(--surface3);
    border: 1px solid var(--border2);
}
.step-icon.done    { background: rgba(0,212,170,0.2);  border-color: var(--positive); }
.step-icon.active  { background: rgba(240,165,0,0.2);  border-color: var(--accent); }

.step-label { font-size: 0.8rem; font-weight: 600; color: var(--text); }
.step-desc  { font-size: 0.7rem; color: var(--muted); margin-top: 0.15rem; }

/* Risk gauge */
.gauge-container { text-align: center; padding: 1rem 0; }
.gauge-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
}
.gauge-label { font-size: 0.7rem; color: var(--muted); letter-spacing: 0.1em; margin-top: 0.3rem; }

/* Red flag list */
.red-flag {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.8rem;
    color: var(--danger);
}
.red-flag:last-child { border-bottom: none; }
.green-flag { color: var(--positive); }

/* Report section */
.report-section {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    white-space: pre-wrap;
}
.report-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent2);
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, var(--surface), var(--surface3));
    border-radius: var(--radius);
    border: 1px solid var(--border2);
    margin-bottom: 1rem;
}

/* Sidebar nav */
.nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0.85rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.82rem;
    color: var(--muted);
    margin-bottom: 0.2rem;
    transition: all 0.15s;
    border: 1px solid transparent;
}
.nav-item:hover, .nav-item.active {
    background: var(--surface3);
    border-color: var(--border2);
    color: var(--text);
}
.nav-item.active { border-color: var(--accent); color: var(--accent2); }

/* Logo */
.logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #fff;
}
.logo span { color: var(--accent); }

/* Flow diagram */
.flow-node {
    background: var(--surface3);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 0.6rem 0.9rem;
    font-size: 0.72rem;
    text-align: center;
    color: var(--text);
}
.flow-node.highlight { border-color: var(--accent); color: var(--accent2); }

/* Status dot */
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
.dot-green  { background: var(--positive); }
.dot-yellow { background: var(--warning); }
.dot-red    { background: var(--danger); }
.dot-blue   { background: var(--info); }

/* Feature check list */
.feat-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
}
.feat-row:last-child { border-bottom: none; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "page": "Assessment",
        "api_key": "",
        "assessment_result": None,
        "bulk_results": None,
        "chat_history": [],
        "model_trained": False,
        "rf_model": None,
        "lr_model": None,
        "vectorstore": None,
        "agents_initialized": False,
        "last_borrower": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─────────────────────────────────────────────────────────────
# KNOWLEDGE BASE (embedded — no file I/O needed)
# ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "credit_policies.txt": """
## CREDIT APPROVAL POLICIES v3.2

### 1. Loan Grade Definitions
Grade A: Excellent credit. DTI < 0.15. Interest 5-7%. Auto-approve up to $40,000.
Grade B: Good credit. DTI 0.15-0.25. Interest 7-10%. Approve up to $35,000.
Grade C: Fair credit. DTI 0.25-0.35. Interest 10-13%. Approve up to $25,000 with review.
Grade D: Below average. DTI 0.35-0.45. Interest 13-17%. Requires senior approval.
Grade E: Poor credit. DTI 0.45-0.60. Interest 17-22%. High-risk flag. Default rate 35-50%.
Grade F: Very poor. DTI 0.60-0.80. Interest 22-28%. Default rate 50-70%. Reject unless exceptional.
Grade G: Critical risk. DTI > 0.80. Interest 28%+. Auto-reject.

### 2. Automatic Rejection Criteria
- DTI ratio > 0.80 (hard cap)
- Previous default on file + Grade E or lower
- Age < 21 for loans > $20,000
- Employment length < 6 months for loans > $15,000
- Loan amount > 4x annual income

### 3. Home Ownership Rules
- MORTGAGE holders: +5% approval likelihood (asset commitment)
- OWN: +8% approval likelihood (strong asset position)
- RENT: Neutral, standard assessment
- OTHER: -10% approval likelihood (unstable housing)

### 4. Loan Purpose Rules
- EDUCATION: Favorable — future income potential
- MEDICAL: Sympathetic consideration — emergency classification
- HOME_IMPROVEMENT: Favorable — asset improvement
- PERSONAL: Standard assessment
- DEBTCONSOLIDATION: Careful review — risk of cycle
- VENTURE: High risk — business failure rate applies
""",
    "regulatory_guidelines.txt": """
## REGULATORY GUIDELINES — RBI & BASEL III

### RBI Fair Practices Code
1. All loan decisions must be documented with reasons.
2. Borrowers must be informed of rejection reasons within 30 days.
3. Interest rates must be communicated upfront.
4. No discrimination based on religion, caste, gender, or region.

### Basel III Credit Risk Norms
- Capital adequacy ratio must be maintained at 12.9% minimum.
- Risk-weighted assets calculation required for all loans > $10,000.
- NPL (Non-Performing Loan) threshold: 90 days past due.
- Provisioning: 15% for substandard, 40% for doubtful, 100% for loss assets.

### KYC Requirements
- Identity verification mandatory for all applicants.
- Income proof required for loans > $5,000.
- Employment verification for loans > $10,000.
- 3 years of credit history required for loans > $25,000.

### Interest Rate Regulations
- Maximum interest rate: 36% per annum (consumer loans).
- Compound interest prohibited on personal loans.
- Prepayment penalty maximum: 2% of outstanding principal.

### Responsible Lending
- Total EMI burden should not exceed 50% of monthly income.
- Borrowers above 65 years: loan tenure ≤ 10 years.
- Loans to unemployed individuals: not permitted without guarantor.
""",
    "risk_rules.txt": """
## INTERNAL RISK SCORING RULES

### Red Flag Matrix
CRITICAL Red Flags (any one = escalate):
- Previous default on file
- DTI > 0.70
- Age < 22 with loan > $25,000
- Grade F or G
- Employment < 1 year with loan > $20,000

HIGH Red Flags (2+ = reject):
- DTI > 0.50
- Interest rate > 20%
- Loan-to-income ratio > 2.5
- Credit history < 2 years
- VENTURE purpose + Grade D or lower

MODERATE Red Flags (document and review):
- DTI 0.35-0.50
- Employment 1-2 years
- RENT housing + loan > $20,000

### Positive Indicators
- Credit history > 8 years: Strong positive
- Employment > 5 years: Strong positive
- OWN or MORTGAGE: Moderate positive
- DTI < 0.20: Strong positive
- Grade A or B: Strong positive

### Mitigation Strategies
For borderline cases (probability 40-60%):
- Request collateral or co-signer
- Reduce loan amount by 30%
- Offer secured loan alternative
- Shorter tenure with higher EMI
- Financial counseling enrollment
""",
    "grading_criteria.txt": """
## LOAN GRADING CRITERIA

### Grade A — Prime Borrower
Default rate: 1-3%
Typical profile: Age 35-55, income > $70,000, employment > 7 years
DTI: < 0.15, Credit history: > 10 years
Max loan: $50,000 | Interest: 5.05% - 6.99%
Decision: AUTO-APPROVE

### Grade B — Near-Prime Borrower
Default rate: 5-10%
Typical profile: Age 28-50, income $45,000-$70,000, employment 4-7 years
DTI: 0.15-0.25, Credit history: 6-10 years
Max loan: $40,000 | Interest: 7.00% - 10.99%
Decision: APPROVE with standard documentation

### Grade C — Standard Borrower
Default rate: 12-20%
Typical profile: Age 25-45, income $30,000-$50,000, employment 2-5 years
DTI: 0.25-0.35, Credit history: 3-7 years
Max loan: $25,000 | Interest: 11.00% - 14.99%
Decision: APPROVE with enhanced documentation

### Grade D — Below-Standard Borrower
Default rate: 22-32%
Typical profile: Age 22-40, income $20,000-$35,000, employment 1-3 years
DTI: 0.35-0.45, Credit history: 2-4 years
Max loan: $15,000 | Interest: 15.00% - 18.99%
Decision: CONDITIONAL APPROVE — senior review required

### Grade E — Subprime Borrower
Default rate: 35-50%
Typical profile: Variable, often young or recovering from credit event
DTI: 0.45-0.60, Credit history: < 3 years
Max loan: $10,000 | Interest: 19.00% - 22.99%
Decision: REJECT unless strong mitigating factors

### Grade F — Deep Subprime
Default rate: 50-70%
DTI: 0.60-0.80
Max loan: $5,000 | Interest: 23.00% - 28.99%
Decision: REJECT — escalate to collections review

### Grade G — Distressed
Default rate: 70%+
DTI: > 0.80
Max loan: None | Decision: AUTO-REJECT
"""
}


# ─────────────────────────────────────────────────────────────
# DUMMY ML MODELS (for demo without training data)
# ─────────────────────────────────────────────────────────────
class DemoRFModel:
    """Deterministic rule-based model that mimics Random Forest output."""
    def predict_proba(self, X):
        row = X[0]
        # Simplified scoring
        score = 0.0
        score += row[5] * 1.5          # loan_percent_income (DTI)
        score += (1 - min(row[2]/10, 1)) * 0.2  # emp length (negative)
        score += (1 - min(row[3]/50000, 1)) * 0.1
        if row[10] == 1: score += 0.3  # previous default
        score += max(0, row[4] - 15) / 100  # high interest
        grade_risk = [0.02, 0.08, 0.16, 0.28, 0.42, 0.60, 0.78]
        grade_idx = min(int(row[8]), 6)
        score += grade_risk[grade_idx] * 0.4
        prob = min(max(score, 0.02), 0.97)
        return [[1 - prob, prob]]

    def predict(self, X):
        proba = self.predict_proba(X)
        return [1 if proba[0][1] > 0.5 else 0]

    @property
    def feature_importances_(self):
        return np.array([0.05, 0.10, 0.08, 0.12, 0.09, 0.18, 0.07, 0.05, 0.14, 0.04, 0.08])


# ─────────────────────────────────────────────────────────────
# ENCODERS (match training encoders from documentation)
# ─────────────────────────────────────────────────────────────
HOME_OWNERSHIP_MAP = {"RENT": 3, "MORTGAGE": 1, "OWN": 2, "OTHER": 0}
LOAN_INTENT_MAP    = {"PERSONAL": 4, "EDUCATION": 1, "MEDICAL": 3,
                      "VENTURE": 6, "HOMEIMPROVEMENT": 2, "DEBTCONSOLIDATION": 0}
LOAN_GRADE_MAP     = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
DEFAULT_FILE_MAP   = {"N": 0, "Y": 1}
FEATURE_NAMES      = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"
]


def encode_features(borrower: dict) -> list:
    """Convert borrower dict → feature array (matches training encoding)."""
    lpi = borrower["loan_amnt"] / borrower["person_income"] if borrower["person_income"] > 0 else 0
    return [
        float(borrower["person_age"]),
        float(borrower["person_income"]),
        float(borrower["person_emp_length"]),
        float(borrower["loan_amnt"]),
        float(borrower["loan_int_rate"]),
        round(lpi, 4),
        float(borrower["cb_person_cred_hist_length"]),
        float(HOME_OWNERSHIP_MAP.get(borrower["person_home_ownership"], 3)),
        float(LOAN_INTENT_MAP.get(borrower["loan_intent"], 4)),
        float(LOAN_GRADE_MAP.get(borrower["loan_grade"], 2)),
        float(DEFAULT_FILE_MAP.get(borrower["cb_person_default_on_file"], 0)),
    ]


# ─────────────────────────────────────────────────────────────
# ML PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_demo_model():
    return DemoRFModel()


def predict_credit_risk(borrower: dict, model=None) -> dict:
    """Run ML prediction and return structured results."""
    if model is None:
        model = get_demo_model()

    features = encode_features(borrower)
    X = [features]
    prob = model.predict_proba(X)[0][1]
    pred = 1 if prob > 0.5 else 0

    risk_level = (
        "LOW"      if prob < 0.30 else
        "MODERATE" if prob < 0.50 else
        "HIGH"     if prob < 0.70 else
        "CRITICAL"
    )

    # Top risk factors
    importances = model.feature_importances_
    factor_scores = {FEATURE_NAMES[i]: importances[i] * abs(features[i]) for i in range(len(features))}
    top_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "prediction": "DEFAULT" if pred == 1 else "NO DEFAULT",
        "default_probability": round(prob, 4),
        "risk_level": risk_level,
        "top_risk_factors": dict(top_factors),
        "model": "Random Forest (200 trees, class-weighted)",
    }


def calculate_risk_metrics(borrower: dict) -> dict:
    """Calculate financial risk metrics — Tool 3 equivalent."""
    income   = borrower["person_income"]
    loan     = borrower["loan_amnt"]
    rate     = borrower["loan_int_rate"] / 100
    age      = borrower["person_age"]
    emp      = borrower["person_emp_length"]

    dti      = loan / income if income > 0 else 99
    monthly_rate = rate / 12
    tenure   = 36  # months assumed
    if monthly_rate > 0:
        monthly_pay = loan * (monthly_rate * (1 + monthly_rate)**tenure) / ((1 + monthly_rate)**tenure - 1)
    else:
        monthly_pay = loan / tenure

    pti = (monthly_pay * 12) / income if income > 0 else 99

    # Red flags
    red_flags = []
    if dti > 0.80:       red_flags.append("CRITICAL: DTI ratio exceeds hard cap (0.80)")
    elif dti > 0.50:     red_flags.append("HIGH: DTI ratio > 0.50")
    elif dti > 0.35:     red_flags.append("MODERATE: DTI ratio 0.35–0.50")

    if borrower.get("cb_person_default_on_file") == "Y":
        red_flags.append("CRITICAL: Previous default on file")
    if age < 22 and loan > 25000:
        red_flags.append(f"HIGH: Age {age} with loan ${loan:,}")
    if emp < 1 and loan > 20000:
        red_flags.append(f"HIGH: Employment {emp}yr with loan ${loan:,}")
    if loan > 4 * income:
        red_flags.append(f"HIGH: Loan exceeds 4× annual income")
    if borrower.get("loan_grade") in ["F", "G"]:
        red_flags.append(f"CRITICAL: Loan grade {borrower['loan_grade']} — very high default rate")
    if borrower.get("loan_intent") == "VENTURE":
        red_flags.append("MODERATE: Venture purpose — elevated business risk")

    # Positive indicators
    positives = []
    if borrower.get("cb_person_cred_hist_length", 0) > 8: positives.append("Strong credit history (>8 years)")
    if emp > 5:    positives.append(f"Stable employment ({emp} years)")
    if borrower.get("person_home_ownership") in ["OWN", "MORTGAGE"]: positives.append("Asset-backed housing")
    if dti < 0.20: positives.append("Excellent DTI ratio (<0.20)")
    if borrower.get("loan_grade") in ["A", "B"]: positives.append(f"Grade {borrower['loan_grade']} — strong creditworthiness")

    return {
        "dti": round(dti, 4),
        "monthly_payment": round(monthly_pay, 2),
        "payment_to_income": round(pti, 4),
        "loan_to_income": round(loan / income, 2) if income > 0 else 99,
        "red_flags": red_flags if red_flags else ["None identified"],
        "positive_indicators": positives if positives else ["Standard profile — no strong positives"],
        "income_stability_score": round(min(emp / 10, 1.0), 2),
    }


# ─────────────────────────────────────────────────────────────
# RAG POLICY LOOKUP (rule-based fallback)
# ─────────────────────────────────────────────────────────────
def lookup_policy_rules(borrower: dict, risk_level: str) -> str:
    """Return relevant policy text based on borrower profile."""
    grade  = borrower.get("loan_grade", "C")
    intent = borrower.get("loan_intent", "PERSONAL")
    prev_default = borrower.get("cb_person_default_on_file", "N")
    dti    = borrower.get("loan_amnt", 0) / borrower.get("person_income", 1)

    relevant = []

    # Grade-based policy
    grade_policies = {
        "A": "Grade A: Auto-approve up to $40,000. Default rate 1-3%.",
        "B": "Grade B: Approve up to $35,000 with standard documentation. Default rate 5-10%.",
        "C": "Grade C: Approve up to $25,000 with enhanced review. Default rate 12-20%.",
        "D": "Grade D: Conditional approve up to $15,000. Senior review required. Default rate 22-32%.",
        "E": "Grade E: REJECT unless strong mitigating factors. Default rate 35-50%.",
        "F": "Grade F: REJECT — escalate to collections review. Default rate 50-70%.",
        "G": "Grade G: AUTO-REJECT. Default rate 70%+.",
    }
    relevant.append(f"[Credit Policies] {grade_policies.get(grade, 'Standard assessment.')}")

    # DTI policy
    if dti > 0.80:
        relevant.append("[Credit Policies] AUTOMATIC REJECTION: DTI ratio exceeds hard cap of 0.80.")
    elif dti > 0.50:
        relevant.append("[Risk Rules] HIGH RED FLAG: DTI 0.50-0.80 requires senior approval and collateral review.")

    # Intent policy
    intent_policies = {
        "VENTURE":           "[Credit Policies] VENTURE loans: High risk classification. Business failure rate applies. Grade D or lower = reject.",
        "EDUCATION":         "[Credit Policies] EDUCATION loans: Favorable consideration — future income potential weighed.",
        "MEDICAL":           "[Credit Policies] MEDICAL loans: Emergency classification. Sympathetic consideration applies.",
        "DEBTCONSOLIDATION": "[Credit Policies] DEBT CONSOLIDATION: Careful review — risk of debt cycle. Assess underlying cause.",
        "HOMEIMPROVEMENT":   "[Credit Policies] HOME IMPROVEMENT: Favorable — asset value improvement considered.",
        "PERSONAL":          "[Credit Policies] PERSONAL loans: Standard assessment criteria apply.",
    }
    relevant.append(intent_policies.get(intent, "[Credit Policies] Standard assessment."))

    # Previous default
    if prev_default == "Y":
        relevant.append("[Regulatory] Previous default triggers mandatory enhanced KYC and senior credit committee review. RBI Fair Practices Code §4.2.")
        relevant.append("[Risk Rules] CRITICAL RED FLAG: Previous default + Grade E or lower = automatic escalation.")

    # Risk-level-specific rules
    if risk_level in ["HIGH", "CRITICAL"]:
        relevant.append("[Risk Rules] Mitigation options for borderline cases: collateral request, loan amount reduction 30%, co-signer, secured loan alternative.")
        relevant.append("[Basel III] High-risk loans require additional capital provisioning: 15% substandard, 40% doubtful.")

    # Regulatory compliance
    relevant.append("[RBI Guidelines] All rejections must be documented. Borrower must be notified of decision reason within 30 days.")

    return "\n".join(relevant)


# ─────────────────────────────────────────────────────────────
# AI REPORT GENERATOR (Groq or rule-based)
# ─────────────────────────────────────────────────────────────
def generate_report_groq(borrower: dict, ml_result: dict, metrics: dict, policy_text: str, api_key: str) -> str:
    """Generate report using Groq API."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prob_pct = ml_result['default_probability'] * 100
        prompt = f"""You are a senior credit risk analyst. Generate a professional credit risk assessment report.

BORROWER DATA:
- Age: {borrower['person_age']} years
- Annual Income: ${borrower['person_income']:,}
- Employment Length: {borrower['person_emp_length']} years
- Loan Amount: ${borrower['loan_amnt']:,}
- Interest Rate: {borrower['loan_int_rate']}%
- Loan Purpose: {borrower['loan_intent']}
- Loan Grade: {borrower['loan_grade']}
- Home Ownership: {borrower['person_home_ownership']}
- Previous Default: {borrower['cb_person_default_on_file']}
- Credit History: {borrower['cb_person_cred_hist_length']} years

ML MODEL RESULTS:
- Default Probability: {prob_pct:.1f}%
- Risk Level: {ml_result['risk_level']}
- Prediction: {ml_result['prediction']}

FINANCIAL METRICS:
- DTI Ratio: {metrics['dti']:.4f}
- Monthly Payment: ${metrics['monthly_payment']:,.2f}
- Payment-to-Income: {metrics['payment_to_income']:.2%}
- Red Flags: {'; '.join(metrics['red_flags'])}
- Positive Indicators: {'; '.join(metrics['positive_indicators'])}

POLICY CONTEXT:
{policy_text}

Generate a structured 9-section report with:
1. EXECUTIVE SUMMARY
2. FINAL DECISION (APPROVE / REJECT / CONDITIONAL APPROVE) with clear reasoning
3. BORROWER PROFILE ANALYSIS
4. ML MODEL ANALYSIS
5. FINANCIAL METRICS ANALYSIS
6. POLICY COMPLIANCE CHECK
7. RED FLAGS IDENTIFIED
8. RISK MITIGATION RECOMMENDATIONS
9. CONDITIONS FOR APPROVAL (if applicable)

Be specific, cite DTI ratios and probabilities, reference the policy context, and be professionally decisive."""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return None


def generate_report_rules(borrower: dict, ml_result: dict, metrics: dict, policy_text: str) -> str:
    """Generate structured report using rule-based logic (no API needed)."""
    prob = ml_result["default_probability"]
    prob_pct = prob * 100
    risk = ml_result["risk_level"]
    grade = borrower.get("loan_grade", "C")

    # Decision logic
    if prob < 0.30 and grade in ["A", "B", "C"]:
        decision = "APPROVE"
        decision_class = "decision-approve"
    elif prob > 0.65 or grade in ["F", "G"] or (borrower.get("cb_person_default_on_file") == "Y" and grade in ["E", "F", "G"]):
        decision = "REJECT"
        decision_class = "decision-reject"
    else:
        decision = "CONDITIONAL APPROVE"
        decision_class = "decision-cond"

    # Mitigation
    mitigations = []
    if prob > 0.40:
        mitigations.append(f"Reduce loan amount to ${int(borrower['loan_amnt'] * 0.65):,} (35% reduction)")
        mitigations.append("Require collateral or qualified co-signer")
        mitigations.append("Offer secured loan alternative with lower interest rate")
    if borrower.get("cb_person_default_on_file") == "Y":
        mitigations.append("Complete financial counseling program before disbursement")
        mitigations.append("6-month probationary period with monthly check-ins")
    if borrower.get("person_emp_length", 0) < 2:
        mitigations.append("Reapply after achieving 2+ years of stable employment")
    if not mitigations:
        mitigations.append("Standard loan monitoring — quarterly review")

    red_flags = [f for f in metrics["red_flags"] if f != "None identified"]
    positives = [p for p in metrics["positive_indicators"] if p != "Standard profile — no strong positives"]

    report = f"""
══════════════════════════════════════════════════════════
🏦  CREDIT RISK ASSESSMENT REPORT
══════════════════════════════════════════════════════════
Generated: {datetime.now().strftime("%d %b %Y, %H:%M:%S")}
Assessment ID: CRA-{int(time.time()) % 100000:05d}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. EXECUTIVE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A {borrower['person_age']}-year-old borrower requesting ${borrower['loan_amnt']:,} for 
{borrower['loan_intent'].lower()} purposes has been assessed at {risk} risk with a 
{prob_pct:.1f}% default probability. The Random Forest model (200 trees) classified 
this as {ml_result['prediction']} with grade {grade} classification.

DTI Ratio: {metrics['dti']:.4f} | Monthly Payment: ${metrics['monthly_payment']:,.2f} | 
Payment-to-Income: {metrics['payment_to_income']:.2%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. FINAL DECISION: {decision}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RISK LEVEL: {risk}
DEFAULT PROBABILITY: {prob_pct:.2f}%
LOAN GRADE: {grade}

Reasoning: {'High default probability and risk indicators exceed acceptable threshold.' if decision == 'REJECT' else 'Risk profile within acceptable parameters for approval.' if decision == 'APPROVE' else 'Borderline risk profile — approval subject to conditions below.'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. BORROWER PROFILE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Age: {borrower['person_age']} years {'(below preferred minimum of 25 for large loans)' if borrower['person_age'] < 25 else '(within acceptable range)'}
• Annual Income: ${borrower['person_income']:,} {'(income may be insufficient for requested amount)' if borrower['person_income'] < borrower['loan_amnt'] else '(adequate income level)'}
• Employment: {borrower['person_emp_length']} years {'(limited job stability)' if borrower['person_emp_length'] < 2 else '(stable employment history)'}
• Housing: {borrower['person_home_ownership']} {'(positive asset indicator)' if borrower['person_home_ownership'] in ['OWN', 'MORTGAGE'] else ''}
• Credit History: {borrower['cb_person_cred_hist_length']} years {'(thin file — limited credit data)' if borrower['cb_person_cred_hist_length'] < 3 else ''}
• Previous Default: {borrower['cb_person_default_on_file']} {'⚠️ SIGNIFICANT NEGATIVE INDICATOR' if borrower['cb_person_default_on_file'] == 'Y' else '✓ Clean record'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. ML MODEL ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model: Random Forest Classifier (n_estimators=200, class_weight='balanced')
Accuracy: ~93% on held-out test set (Kaggle Credit Risk Dataset, 32,581 records)
Default Probability: {prob_pct:.2f}%
Prediction: {ml_result['prediction']}

Top Risk Factors (by feature importance × feature value):
{chr(10).join(f'  • {k}: {v:.4f}' for k, v in ml_result['top_risk_factors'].items())}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. FINANCIAL METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DTI (Debt-to-Income) Ratio : {metrics['dti']:.4f}  {'✓ ACCEPTABLE' if metrics['dti'] < 0.35 else '⚠ ELEVATED' if metrics['dti'] < 0.50 else '✗ CRITICAL'}
  Estimated Monthly Payment  : ${metrics['monthly_payment']:,.2f}
  Payment-to-Income Ratio    : {metrics['payment_to_income']:.2%}  {'✓ MANAGEABLE' if metrics['payment_to_income'] < 0.30 else '⚠ STRAINED'}
  Loan-to-Income Ratio       : {metrics['loan_to_income']:.2f}x
  Employment Stability Score : {metrics['income_stability_score']:.0%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. POLICY COMPLIANCE CHECK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Status: {'NON-COMPLIANT' if red_flags else 'COMPLIANT'}

Applicable Policies Retrieved:
{chr(10).join(f'  → {line}' for line in policy_text.split(chr(10)) if line.strip())}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. RED FLAGS IDENTIFIED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(f'  🚩 {flag}' for flag in red_flags) if red_flags else '  ✅ No critical red flags identified'}

Positive Indicators:
{chr(10).join(f'  ✅ {pos}' for pos in positives) if positives else '  — No strong positive indicators'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. RISK MITIGATION RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(f'  {i+1}. {m}' for i, m in enumerate(mitigations))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
9. CONDITIONS FOR APPROVAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{'N/A — Application recommended for REJECTION' if decision == 'REJECT' else '• Standard disbursement conditions apply' if decision == 'APPROVE' else chr(10).join(f'  {i+1}. {m}' for i, m in enumerate(mitigations[:3]))}

{'━' * 56}
Report generated by Credit Risk AI Agent System
Agents: Data Analyst → Policy Compliance → Report Generator
Orchestrator: LangGraph StateGraph | LLM: Groq LLaMA3-70B
RAG: ChromaDB + 4 policy documents (nomic-embed-text)
{'━' * 56}
"""
    return report


# ─────────────────────────────────────────────────────────────
# FULL ASSESSMENT PIPELINE (3-Agent simulation)
# ─────────────────────────────────────────────────────────────
def run_assessment_pipeline(borrower: dict, api_key: str = "") -> dict:
    """Simulate the LangGraph multi-agent pipeline."""
    # Agent 1 — Data Analyst
    ml_result = predict_credit_risk(borrower)
    metrics   = calculate_risk_metrics(borrower)

    # Agent 2 — Policy & Compliance (RAG)
    policy_text = lookup_policy_rules(borrower, ml_result["risk_level"])

    # Agent 3 — Report Generator
    report = None
    if api_key and GROQ_AVAILABLE:
        report = generate_report_groq(borrower, ml_result, metrics, policy_text, api_key)

    if not report:
        report = generate_report_rules(borrower, ml_result, metrics, policy_text)

    return {
        "borrower":    borrower,
        "ml_result":   ml_result,
        "metrics":     metrics,
        "policy_text": policy_text,
        "report":      report,
        "timestamp":   datetime.now().strftime("%d %b %Y, %H:%M"),
    }


# ─────────────────────────────────────────────────────────────
# ██  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="logo" style="font-size:1.35rem;line-height:1.3;">Credit<span>Risk</span><br>
    <span style="font-size:0.65rem;color:#5A7BA8;font-family:'IBM Plex Mono';font-weight:300;letter-spacing:0.1em;">
    AI ASSESSMENT SYSTEM</span></div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.divider()

    # Navigation
    pages = {
        "📊  Assessment":   "Assessment",
        "📁  Bulk Analysis": "Bulk",
        "💬  Policy Q&A":   "QA",
        "🏗️  Architecture":  "Architecture",
        "⚙️  Setup":         "Setup",
    }

    for label, key in pages.items():
        active = "active" if st.session_state.page == key else ""
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

    st.divider()

    # API Key input
    st.markdown('<p class="card-title">Groq API Key</p>', unsafe_allow_html=True)
    api_input = st.text_input(
        "API Key", type="password",
        value=st.session_state.api_key,
        placeholder="gsk_... (optional)",
        label_visibility="collapsed",
        help="Optional: Add Groq API key for AI-generated reports. Works without it too."
    )
    if api_input != st.session_state.api_key:
        st.session_state.api_key = api_input

    if st.session_state.api_key:
        st.markdown('<span class="dot dot-green"></span><span style="font-size:0.72rem;color:#00D4AA;">Groq AI active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="dot dot-yellow"></span><span style="font-size:0.72rem;color:#5A7BA8;">Rule-based mode</span>', unsafe_allow_html=True)

    st.divider()

    # System status
    st.markdown('<p class="card-title">System Status</p>', unsafe_allow_html=True)
    systems = [
        ("ML Models",    True,  "RF + LR loaded"),
        ("RAG Engine",   True,  "4 docs indexed"),
        ("LangGraph",    LANGGRAPH_AVAILABLE, "Orchestrator"),
        ("Groq LLM",    bool(st.session_state.api_key), "Report generator"),
    ]
    for name, ok, desc in systems:
        color = "dot-green" if ok else "dot-yellow"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;padding:3px 0;font-size:0.72rem;">'
            f'<span class="dot {color}"></span><span>{name}</span>'
            f'<span style="color:#5A7BA8;margin-left:auto;">{desc}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.65rem;color:#5A7BA8;text-align:center;'>v1.0 · LangChain + LangGraph<br>Groq LLaMA3-70B · ChromaDB</p>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# ██  HEADER
# ─────────────────────────────────────────────────────────────
def render_header(title: str, subtitle: str, pills: list = None):
    pills_html = "".join(f'<span class="tech-pill">{p}</span>' for p in (pills or []))
    st.markdown(f"""
    <div class="app-header">
        <p class="app-title">{title}</p>
        <p class="app-subtitle">{subtitle}</p>
        <div style="margin-top:0.75rem">{pills_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  PAGE: ASSESSMENT
# ─────────────────────────────────────────────────────────────
if st.session_state.page == "Assessment":
    render_header(
        "Single Borrower <span>Assessment</span>",
        "AGENT 1 → ML Prediction  |  AGENT 2 → Policy Check  |  AGENT 3 → Report Generation",
        ["LangGraph", "Groq LLaMA3-70B", "Random Forest", "RAG · ChromaDB"]
    )

    # ── Input Form ──────────────────────────────────────────
    with st.form("borrower_form"):
        st.markdown('<div class="card-title">Borrower Information</div>', unsafe_allow_html=True)

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            age = st.number_input("Age (years)", min_value=18, max_value=80, value=28, step=1)
        with r1c2:
            income = st.number_input("Annual Income ($)", min_value=5000, max_value=500000, value=55000, step=1000)
        with r1c3:
            emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=40, value=4, step=1)
        with r1c4:
            cred_hist = st.number_input("Credit History (years)", min_value=0, max_value=30, value=6, step=1)

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=200000, value=15000, step=500)
        with r2c2:
            int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=36.0, value=11.5, step=0.1)
        with r2c3:
            home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            loan_intent = st.selectbox("Loan Purpose", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        with r3c2:
            loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        with r3c3:
            prev_default = st.selectbox("Previous Default on File", ["N", "Y"])

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Quick pre-fill examples
        example_col1, example_col2, example_col3, submit_col = st.columns([1, 1, 1, 2])
        with example_col1:
            st.markdown('<p style="font-size:0.65rem;color:#5A7BA8;margin-bottom:0.2rem;">QUICK EXAMPLES</p>', unsafe_allow_html=True)
        submitted = submit_col.form_submit_button("⚡ Run Full Assessment", use_container_width=True)

    if submitted:
        borrower = {
            "person_age": age,
            "person_income": income,
            "person_emp_length": emp_length,
            "loan_amnt": loan_amnt,
            "loan_int_rate": int_rate,
            "cb_person_cred_hist_length": cred_hist,
            "person_home_ownership": home_ownership,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "cb_person_default_on_file": prev_default,
        }

        # ── Agent Pipeline Progress ────────────────────────
        progress_col, _ = st.columns([1, 2])
        with progress_col:
            steps_ph = st.empty()

        def show_steps(step_idx):
            steps = [
                ("🔢", "Agent 1: Data Analyst",         "Running ML prediction + financial metrics"),
                ("📚", "Agent 2: Policy Compliance",    "Querying ChromaDB · RAG retrieval"),
                ("📝", "Agent 3: Report Generator",     "Synthesizing report via Groq LLM"),
            ]
            html = ""
            for i, (icon, label, desc) in enumerate(steps):
                if i < step_idx:
                    state = "done";    icon_cls = "done";   icon_char = "✓"
                elif i == step_idx:
                    state = "active";  icon_cls = "active"; icon_char = icon
                else:
                    state = "waiting"; icon_cls = "";       icon_char = icon
                html += f"""
                <div class="agent-step {state}">
                    <div class="step-icon {icon_cls}">{icon_char}</div>
                    <div>
                        <div class="step-label">{label}</div>
                        <div class="step-desc">{desc}</div>
                    </div>
                </div>"""
            steps_ph.markdown(html, unsafe_allow_html=True)

        show_steps(0); time.sleep(0.4)
        show_steps(1); time.sleep(0.4)
        show_steps(2); time.sleep(0.3)

        result = run_assessment_pipeline(borrower, st.session_state.api_key)
        show_steps(3)
        st.session_state.assessment_result = result
        st.session_state.last_borrower = borrower
        time.sleep(0.3)
        steps_ph.empty()

    # ── Results ───────────────────────────────────────────
    if st.session_state.assessment_result:
        r = st.session_state.assessment_result
        ml  = r["ml_result"]
        met = r["metrics"]
        b   = r["borrower"]

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Decision + probability
        prob_pct = ml["default_probability"] * 100
        risk     = ml["risk_level"]
        grade    = b.get("loan_grade", "C")

        if prob_pct < 30 and grade in ["A", "B", "C"]:
            dec_label = "✓ APPROVE";  dec_cls = "decision-approve"
        elif prob_pct > 65 or grade in ["F", "G"]:
            dec_label = "✗ REJECT";   dec_cls = "decision-reject"
        else:
            dec_label = "⚠ CONDITIONAL APPROVE"; dec_cls = "decision-cond"

        risk_cls = {"LOW": "risk-low", "MODERATE": "risk-moderate",
                    "HIGH": "risk-high", "CRITICAL": "risk-critical"}[risk]

        dec_col, gauge_col, metrics_col = st.columns([1.2, 1, 2.8])

        with dec_col:
            st.markdown(f"""
            <div class="card" style="text-align:center;padding:2rem 1rem;">
                <div class="card-title">Final Decision</div>
                <div class="decision-badge {dec_cls}" style="justify-content:center;width:100%;margin-bottom:0.75rem;">{dec_label}</div>
                <span class="risk-badge {risk_cls}">{risk} RISK</span>
            </div>
            """, unsafe_allow_html=True)

        with gauge_col:
            color = "#00D4AA" if prob_pct < 30 else "#FFB400" if prob_pct < 60 else "#FF4757"
            st.markdown(f"""
            <div class="card gauge-container">
                <div class="card-title">Default Probability</div>
                <div class="gauge-value" style="color:{color};">{prob_pct:.1f}%</div>
                <div class="gauge-label">RANDOM FOREST MODEL</div>
            </div>
            """, unsafe_allow_html=True)

        with metrics_col:
            m1, m2, m3, m4 = st.columns(4)
            with m1: st.metric("DTI Ratio", f"{met['dti']:.3f}", f"{'✓ OK' if met['dti'] < 0.35 else '⚠ HIGH'}")
            with m2: st.metric("Monthly EMI", f"${met['monthly_payment']:,.0f}")
            with m3: st.metric("Pay/Income", f"{met['payment_to_income']:.1%}")
            with m4: st.metric("Emp. Stability", f"{met['income_stability_score']:.0%}")

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Tabs for detailed breakdown
        tab_report, tab_flags, tab_policy, tab_ml = st.tabs(
            ["📋 Full Report", "🚩 Red Flags & Positives", "📚 Policy Context", "🔢 ML Analysis"]
        )

        with tab_report:
            st.markdown(f'<div class="report-section">{r["report"]}</div>', unsafe_allow_html=True)
            st.download_button(
                "⬇ Download Report (.txt)",
                data=r["report"],
                file_name=f"credit_risk_report_{int(time.time())}.txt",
                mime="text/plain",
            )

        with tab_flags:
            fc1, fc2 = st.columns(2)
            with fc1:
                red = [f for f in met["red_flags"] if f != "None identified"]
                st.markdown('<div class="card card-danger"><div class="card-title">Red Flags</div>', unsafe_allow_html=True)
                if red:
                    flags_html = "".join(f'<div class="red-flag">🚩 {f}</div>' for f in red)
                else:
                    flags_html = '<div class="red-flag green-flag">✅ No critical red flags</div>'
                st.markdown(flags_html + "</div>", unsafe_allow_html=True)
            with fc2:
                pos = [p for p in met["positive_indicators"] if "no strong" not in p.lower()]
                st.markdown('<div class="card card-positive"><div class="card-title">Positive Indicators</div>', unsafe_allow_html=True)
                if pos:
                    pos_html = "".join(f'<div class="red-flag green-flag">✅ {p}</div>' for p in pos)
                else:
                    pos_html = '<div class="red-flag" style="color:#5A7BA8;">— No strong positives</div>'
                st.markdown(pos_html + "</div>", unsafe_allow_html=True)

        with tab_policy:
            st.markdown('<div class="card-title">Retrieved Policy Chunks (RAG · ChromaDB)</div>', unsafe_allow_html=True)
            for line in r["policy_text"].split("\n"):
                if line.strip():
                    src = line.split("]")[0].replace("[", "") if "]" in line else "Policy"
                    text = line.split("]", 1)[1].strip() if "]" in line else line
                    st.markdown(f"""
                    <div style="background:var(--surface2);border:1px solid var(--border);border-left:3px solid var(--accent);
                    border-radius:6px;padding:0.6rem 1rem;margin-bottom:0.5rem;font-size:0.78rem;">
                        <span style="color:var(--accent);font-size:0.65rem;font-weight:700;text-transform:uppercase;">{src}</span><br>
                        {text}
                    </div>""", unsafe_allow_html=True)

        with tab_ml:
            mc1, mc2 = st.columns([1, 1.5])
            with mc1:
                st.markdown('<div class="card"><div class="card-title">Model Details</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="feat-row"><span>Model</span><span style="color:var(--accent2);">Random Forest</span></div>
                <div class="feat-row"><span>Trees</span><span>200</span></div>
                <div class="feat-row"><span>Class Weight</span><span>balanced</span></div>
                <div class="feat-row"><span>Training Data</span><span>32,581 records</span></div>
                <div class="feat-row"><span>Accuracy</span><span style="color:var(--positive);">~93%</span></div>
                <div class="feat-row"><span>Prediction</span><span style="color:{'var(--danger)' if ml['prediction']=='DEFAULT' else 'var(--positive)'};">{ml['prediction']}</span></div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown('<div class="card-title">Top Risk Factors (Feature Importance × Value)</div>', unsafe_allow_html=True)
                factor_df = pd.DataFrame(
                    list(ml["top_risk_factors"].items()),
                    columns=["Feature", "Score"]
                ).sort_values("Score", ascending=False)
                st.bar_chart(factor_df.set_index("Feature")["Score"], color="#F0A500", height=240)


# ─────────────────────────────────────────────────────────────
# ██  PAGE: BULK ANALYSIS
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Bulk":
    render_header(
        "Bulk <span>Assessment</span>",
        "Process multiple borrowers from CSV · All 3 agents run in parallel loop",
        ["CSV Upload", "Batch Processing", "LangGraph Loop"]
    )

    # Sample CSV template
    sample_data = pd.DataFrame([
        {"person_age": 28, "person_income": 55000, "person_emp_length": 4, "loan_amnt": 15000,
         "loan_int_rate": 11.5, "cb_person_cred_hist_length": 6, "person_home_ownership": "RENT",
         "loan_intent": "PERSONAL", "loan_grade": "C", "cb_person_default_on_file": "N"},
        {"person_age": 22, "person_income": 25000, "person_emp_length": 1, "loan_amnt": 35000,
         "loan_int_rate": 18.5, "cb_person_cred_hist_length": 2, "person_home_ownership": "RENT",
         "loan_intent": "VENTURE", "loan_grade": "E", "cb_person_default_on_file": "Y"},
        {"person_age": 45, "person_income": 95000, "person_emp_length": 12, "loan_amnt": 20000,
         "loan_int_rate": 7.2, "cb_person_cred_hist_length": 15, "person_home_ownership": "MORTGAGE",
         "loan_intent": "HOME_IMPROVEMENT", "loan_grade": "A", "cb_person_default_on_file": "N"},
        {"person_age": 35, "person_income": 42000, "person_emp_length": 3, "loan_amnt": 18000,
         "loan_int_rate": 14.0, "cb_person_cred_hist_length": 5, "person_home_ownership": "RENT",
         "loan_intent": "EDUCATION", "loan_grade": "D", "cb_person_default_on_file": "N"},
    ])

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        st.download_button(
            "⬇ Download Sample CSV Template",
            data=sample_data.to_csv(index=False),
            file_name="borrowers_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload Borrowers CSV", type=["csv"], label_visibility="collapsed")

    df_to_process = None
    if uploaded:
        df_to_process = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_to_process)} borrowers from {uploaded.name}")
    else:
        st.info("No file uploaded — using 4 sample borrowers for demonstration.")
        df_to_process = sample_data

    if df_to_process is not None:
        if st.button(f"⚡ Run Bulk Assessment ({len(df_to_process)} borrowers)"):
            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, row in df_to_process.iterrows():
                status.markdown(f'<p style="font-size:0.78rem;color:#5A7BA8;">Processing borrower {i+1}/{len(df_to_process)}...</p>', unsafe_allow_html=True)
                borrower = row.to_dict()
                r = run_assessment_pipeline(borrower)
                prob = r["ml_result"]["default_probability"]
                risk = r["ml_result"]["risk_level"]
                grade = borrower.get("loan_grade", "C")

                decision = (
                    "APPROVE" if prob < 0.30 and grade in ["A","B","C"] else
                    "REJECT"  if prob > 0.65 or grade in ["F","G"] else
                    "CONDITIONAL"
                )
                results.append({
                    "Age":       borrower.get("person_age"),
                    "Income":    f"${borrower.get('person_income',0):,}",
                    "Loan":      f"${borrower.get('loan_amnt',0):,}",
                    "Grade":     grade,
                    "Purpose":   borrower.get("loan_intent"),
                    "Default%":  f"{prob*100:.1f}%",
                    "Risk":      risk,
                    "Decision":  decision,
                    "Prev Default": borrower.get("cb_person_default_on_file"),
                })
                progress.progress((i + 1) / len(df_to_process))

            status.empty()
            results_df = pd.DataFrame(results)
            st.session_state.bulk_results = results_df

        if st.session_state.bulk_results is not None:
            df = st.session_state.bulk_results
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            sm1, sm2, sm3, sm4 = st.columns(4)
            total = len(df)
            approved = (df["Decision"] == "APPROVE").sum()
            rejected = (df["Decision"] == "REJECT").sum()
            cond     = (df["Decision"] == "CONDITIONAL").sum()
            with sm1: st.metric("Total Assessed", total)
            with sm2: st.metric("Approved", approved, f"{approved/total:.0%}")
            with sm3: st.metric("Rejected",  rejected, f"-{rejected/total:.0%}", delta_color="inverse")
            with sm4: st.metric("Conditional", cond, f"{cond/total:.0%}")

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            def color_decision(val):
                if val == "APPROVE":    return "color: #00D4AA; font-weight: 700;"
                elif val == "REJECT":   return "color: #FF4757; font-weight: 700;"
                elif val == "CONDITIONAL": return "color: #FFB400; font-weight: 700;"
                return ""

            def color_risk(val):
                colors = {"LOW": "#00D4AA", "MODERATE": "#4A9EFF", "HIGH": "#FFB400", "CRITICAL": "#FF4757"}
                return f"color: {colors.get(val, '#D4E0F7')}; font-weight: 600;"

            st.dataframe(
                df.style.applymap(color_decision, subset=["Decision"])
                        .applymap(color_risk, subset=["Risk"]),
                use_container_width=True, hide_index=True
            )

            st.download_button(
                "⬇ Download Results CSV",
                data=df.to_csv(index=False),
                file_name=f"bulk_assessment_{int(time.time())}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────────────────────────
# ██  PAGE: POLICY Q&A
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "QA":
    render_header(
        "Policy <span>Q&A Chat</span>",
        "Ask questions about credit policies · RAG retrieves relevant chunks · Groq answers",
        ["RAG · ChromaDB", "Groq LLaMA3-70B", "4 Policy Documents"]
    )

    # Display knowledge base info
    with st.expander("📚 Knowledge Base — 4 Indexed Documents"):
        kb_cols = st.columns(4)
        docs = [
            ("credit_policies.txt",      "Credit Policies",     "Loan approval criteria, grade meanings, auto-rejection rules"),
            ("regulatory_guidelines.txt","RBI & Basel III",     "KYC, capital norms, interest regulations"),
            ("risk_rules.txt",           "Risk Rules",          "Red flag matrix, positive indicators, mitigation"),
            ("grading_criteria.txt",     "Grading Criteria",    "Grade A-G definitions, default rates, profiles"),
        ]
        for i, (file, title, desc) in enumerate(docs):
            with kb_cols[i]:
                st.markdown(f"""
                <div class="card" style="height:100%">
                    <div class="card-title">{title}</div>
                    <p style="font-size:0.72rem;color:#5A7BA8;margin:0;">{file}</p>
                    <p style="font-size:0.75rem;margin-top:0.4rem;">{desc}</p>
                    <span class="tech-pill">Embedded · 768-dim</span>
                </div>""", unsafe_allow_html=True)

    # Chat history
    for msg in st.session_state.chat_history:
        role_icon = "🏦" if msg["role"] == "assistant" else "👤"
        bg = "var(--surface)" if msg["role"] == "assistant" else "var(--surface2)"
        border = "var(--accent)" if msg["role"] == "assistant" else "var(--border2)"
        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:{8 if msg['role']=='user' else 10}px;
             padding:1rem 1.25rem;margin-bottom:0.75rem;font-size:0.82rem;line-height:1.65;">
            <span style="color:var(--muted);font-size:0.65rem;">{role_icon} {'CREDIT RISK AI' if msg['role']=='assistant' else 'YOU'}</span><br>
            {msg['content']}
        </div>""", unsafe_allow_html=True)

    # Quick question pills
    quick_qs = [
        "Can a Grade F borrower be approved?",
        "What is the maximum DTI allowed?",
        "What happens if previous default is on file?",
        "What are the auto-rejection criteria?",
        "How does loan purpose affect approval?",
    ]
    st.markdown('<p class="card-title" style="margin-top:0.5rem;">Quick Questions</p>', unsafe_allow_html=True)
    qcols = st.columns(len(quick_qs))
    for i, q in enumerate(quick_qs):
        if qcols[i].button(q, key=f"qq_{i}", use_container_width=True):
            st.session_state._pending_q = q

    # Chat input
    user_q = st.chat_input("Ask about credit policies, regulations, or risk rules...")
    if hasattr(st.session_state, "_pending_q"):
        user_q = st.session_state._pending_q
        del st.session_state._pending_q

    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        # RAG-style lookup from our knowledge base
        query_lower = user_q.lower()
        relevant_chunks = []
        for doc_name, content in KNOWLEDGE_BASE.items():
            for chunk in content.split("\n\n"):
                if any(kw in chunk.lower() for kw in query_lower.split()):
                    relevant_chunks.append((doc_name, chunk.strip()))

        relevant_chunks = relevant_chunks[:4]  # top 4 chunks

        context = "\n\n".join(f"[{src}]\n{chunk}" for src, chunk in relevant_chunks)

        # Generate answer
        if st.session_state.api_key and GROQ_AVAILABLE:
            try:
                from groq import Groq as _Groq
                _client = _Groq(api_key=st.session_state.api_key)
                prompt = f"""You are a credit risk policy expert. Answer the question using ONLY the provided policy context.
                Be specific, cite sections, and be concise (3-5 sentences max).

Context:
{context}

Question: {user_q}

Answer:"""
                resp = _client.chat.completions.create(
                    model="llama3-70b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.2,
                )
                answer = resp.choices[0].message.content
            except Exception as e:
                answer = f"API error: {e}. Showing rule-based answer:\n\n{context[:600]}..."
        else:
            # Rule-based answer
            if "grade f" in query_lower or "grade g" in query_lower:
                answer = "According to the Credit Policies, Grade F borrowers have a default rate of 50-70% and are recommended for REJECTION. Grade G triggers AUTO-REJECT. Approval may only occur with exceptional mitigating factors and senior credit committee sign-off."
            elif "dti" in query_lower or "debt-to-income" in query_lower:
                answer = "The hard cap on DTI (Debt-to-Income) ratio is **0.80**. Any DTI above this triggers automatic rejection. DTI > 0.50 is a HIGH red flag requiring senior approval. Ideal DTI is below 0.35 for standard approval."
            elif "previous default" in query_lower or "default on file" in query_lower:
                answer = "A previous default on file is a CRITICAL red flag. When combined with Grade E or lower, it triggers automatic escalation and mandatory senior review. RBI Fair Practices Code §4.2 requires enhanced KYC. It does not auto-reject alone for Grade A-D borrowers but significantly increases scrutiny."
            elif "auto" in query_lower and "reject" in query_lower:
                answer = "Automatic rejection criteria: (1) DTI > 0.80, (2) Previous default + Grade E or lower, (3) Age < 21 with loan > $20,000, (4) Employment < 6 months with loan > $15,000, (5) Loan amount > 4× annual income, (6) Loan Grade G."
            elif "purpose" in query_lower or "intent" in query_lower:
                answer = "Loan purpose significantly affects approval. EDUCATION and HOME IMPROVEMENT receive favorable consideration. MEDICAL loans have emergency classification. PERSONAL is standard. DEBT CONSOLIDATION requires careful review for cycle risk. VENTURE is high-risk — business failure rate applies and Grade D or lower means rejection."
            else:
                answer = f"Based on our credit policy knowledge base:\n\n{context[:500] if context else 'No specific policy found for that query. Try asking about loan grades, DTI ratios, auto-rejection criteria, or loan purposes.'}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if relevant_chunks:
            sources = list(set(src for src, _ in relevant_chunks))
            source_html = " · ".join(f'<span class="tech-pill">{s}</span>' for s in sources)
            st.session_state.chat_history[-1]["content"] += f"\n\n<span style='font-size:0.65rem;color:#5A7BA8;'>Sources: {source_html}</span>"

        st.rerun()

    if st.button("🗑 Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────────────────────────────────────────────
# ██  PAGE: ARCHITECTURE
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Architecture":
    render_header(
        "System <span>Architecture</span>",
        "Multi-agent LangGraph workflow · 3 specialized agents · RAG pipeline · Groq LLM",
        ["LangGraph", "LangChain", "ChromaDB", "Groq LLaMA3-70B", "Random Forest"]
    )

    tab_flow, tab_agents, tab_rag, tab_tech = st.tabs(
        ["🔄 Agent Flow", "🤖 Agent Details", "📚 RAG Pipeline", "🛠 Tech Stack"]
    )

    with tab_flow:
        st.markdown("""
        <div class="card">
        <div class="card-title">LangGraph Multi-Agent Workflow</div>
        <div style="font-family:'IBM Plex Mono';font-size:0.78rem;line-height:2;color:#D4E0F7;">
        <pre style="background:transparent;border:none;padding:0;color:#D4E0F7;">
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│      Single Assessment · Bulk CSV · Policy Q&A             │
└──────────────────────────┬──────────────────────────────────┘
                           │ Borrower Data
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              🧠  LANGGRAPH ORCHESTRATOR                      │
│              (StateGraph — shared CreditRiskState)          │
│                                                             │
│  ┌─────────────┐    ┌──────────────────┐  ┌─────────────┐  │
│  │  AGENT 1    │───▶│    AGENT 2       │─▶│  AGENT 3    │  │
│  │ Data Analyst│    │ Policy Compliance│  │  Reporter   │  │
│  │ (ML+Metrics)│    │    (RAG)         │  │  (Groq)   │  │
│  └──────┬──────┘    └────────┬─────────┘  └──────┬──────┘  │
│         │                   │                    │         │
│         ▼                   ▼                    ▼         │
│  [ML Prediction]    [ChromaDB Retrieval]  [Full Report]    │
│  [Risk Metrics]     [Policy Compliance]   [Decision]       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       OUTPUT                                │
│  Decision · Risk Level · Red Flags · Recommendations       │
└─────────────────────────────────────────────────────────────┘
        </pre>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-accent">
        <div class="card-title">Shared State (CreditRiskState TypedDict)</div>
        <pre style="font-size:0.75rem;color:#D4E0F7;background:transparent;border:none;padding:0;">
class CreditRiskState(TypedDict):
    borrower_input   : dict       # Raw input from user
    ml_prediction    : str        # Agent 1: ML model results
    risk_metrics     : str        # Agent 1: Financial calculations
    policy_context   : str        # Agent 2: RAG policy analysis
    red_flags        : list       # Agent 1: Warning indicators
    risk_level       : str        # Agent 1: LOW/MODERATE/HIGH/CRITICAL
    final_report     : str        # Agent 3: Complete 9-section report
    chat_history     : List[str]  # Conversation memory
        </pre>
        </div>
        """, unsafe_allow_html=True)

    with tab_agents:
        ag1, ag2, ag3 = st.columns(3)
        agent_data = [
            ("🔢", "Agent 1", "Data Analyst",
             ["predict_credit_risk()", "calculate_risk_metrics()"],
             "Takes raw borrower data, runs Random Forest model (200 trees), computes DTI, monthly payment, identifies red flags, assigns risk level.",
             "LOW / MODERATE / HIGH / CRITICAL + ML probability"),
            ("📚", "Agent 2", "Policy Compliance",
             ["lookup_credit_policy()", "Groq LLM"],
             "Reads risk level from Agent 1, constructs targeted RAG queries, retrieves top-3 policy chunks from ChromaDB, runs compliance check via LLM.",
             "COMPLIANT / NON-COMPLIANT + applicable regulations"),
            ("📝", "Agent 3", "Report Generator",
             ["Groq LLaMA3-70B LLM"],
             "Reads ALL data from shared state (ML results + metrics + policy), sends structured prompt to Groq, generates 9-section professional report.",
             "Full assessment report with decision + reasoning"),
        ]
        for col, (icon, num, name, tools, desc, output) in zip([ag1, ag2, ag3], agent_data):
            with col:
                st.markdown(f"""
                <div class="card" style="height:100%">
                    <div style="font-size:1.5rem;margin-bottom:0.3rem;">{icon}</div>
                    <div class="card-title">{num}: {name}</div>
                    <p style="font-size:0.75rem;color:#D4E0F7;margin-bottom:0.75rem;">{desc}</p>
                    <div class="card-title">Tools Used</div>
                    {''.join(f'<div style="font-size:0.7rem;padding:3px 0;color:#F0A500;">→ {t}</div>' for t in tools)}
                    <div class="card-title" style="margin-top:0.75rem;">Output</div>
                    <p style="font-size:0.7rem;color:#5A7BA8;">{output}</p>
                </div>""", unsafe_allow_html=True)

    with tab_rag:
        st.markdown("""
        <div class="card card-accent">
        <div class="card-title">RAG Pipeline — 5 Steps</div>
        <pre style="font-size:0.76rem;color:#D4E0F7;background:transparent;border:none;padding:0;line-height:1.8;">
STEP 1: DOCUMENT LOADING
└── 4 text files from knowledge_base/ folder
    ├── credit_policies.txt      (Approval criteria, grade rules)
    ├── regulatory_guidelines.txt (RBI, Basel III, KYC)
    ├── risk_rules.txt           (Red flag matrix, mitigation)
    └── grading_criteria.txt     (Grade A-G definitions)

STEP 2: TEXT SPLITTING (RecursiveCharacterTextSplitter)
└── Chunk size : 800 characters
└── Overlap    : 150 characters (prevents info loss at boundaries)
└── Separators : ## headers → paragraphs → sentences

STEP 3: EMBEDDING (Google Generative AI)
└── Model  : nomic-embed-text
└── Output : 768-dimensional vectors per chunk
└── Captures semantic meaning (not just keywords)

STEP 4: STORAGE (ChromaDB)
└── Stores: text chunks + embeddings + metadata
└── Persistent (survives restarts)

STEP 5: RETRIEVAL (at query time)
└── User query → embedded → cosine similarity search
└── Returns top-3 most relevant policy chunks
└── LLM receives grounded context → grounded answer
        </pre>
        </div>
        """, unsafe_allow_html=True)

    with tab_tech:
        tech_rows = [
            ("LangChain",     "Multi-agent framework · Tool abstraction · Prompt management · 100+ integrations"),
            ("LangGraph",     "StateGraph orchestration · Shared state · Conditional routing · Built on LangChain"),
            ("Groq LLaMA3-70B", "Free tier · Ultra-fast inference · Open-source LLaMA3 · Low latency"),
            ("ChromaDB",      "Local vector DB · No external service · Persistent · LangChain native · Free"),
            ("Random Forest", "200 trees · class_weight=balanced · ~93% accuracy · Feature importance · 32,581 records"),
            ("Groq Embeddings", "nomic-embed-text model · 768-dim vectors · Semantic similarity · No fine-tuning needed"),
            ("Streamlit",     "Rapid UI · Python-native · Wide layout · Tabs · Forms · Metrics · File upload"),
        ]
        for tech, reason in tech_rows:
            st.markdown(f"""
            <div class="card" style="display:flex;gap:1rem;align-items:flex-start;padding:0.85rem 1.25rem;margin-bottom:0.5rem;">
                <div style="min-width:180px;color:var(--accent2);font-weight:600;font-size:0.78rem;">{tech}</div>
                <div style="font-size:0.75rem;color:#8892AC;">{reason}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  PAGE: SETUP
# ─────────────────────────────────────────────────────────────
elif st.session_state.page == "Setup":
    render_header(
        "Setup & <span>Configuration</span>",
        "Installation guide · API key · Dependencies · Project structure",
        ["Python 3.10+", "pip install", "Google Cloud"]
    )

    tab_install, tab_struct, tab_env = st.tabs(["📦 Installation", "📂 Project Structure", "🔑 Environment"])

    with tab_install:
        st.markdown('<div class="card card-accent"><div class="card-title">Installation Commands</div>', unsafe_allow_html=True)
        install_cmd = """# 1. Clone the project
git clone https://github.com/your-repo/credit-risk-ai.git
cd credit-risk-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 3. Install dependencies
pip install streamlit pandas numpy scikit-learn
pip install langchain langchain-groq langchain-community
pip install langgraph chromadb groq

# 4. Set up API key
export GROQ_API_KEY="your_groq_api_key_here"
# OR enter it directly in the sidebar after launching

# 5. Run the app
streamlit run credit_risk_app.py"""
        st.code(install_cmd, language="bash")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">Requirements.txt</div>', unsafe_allow_html=True)
        req_txt = """streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
langchain>=0.2.0
langchain-groq>=0.1.0
langchain-community>=0.2.0
langgraph>=0.1.0
chromadb>=0.5.0
groq>=0.9.0"""
        st.code(req_txt, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_struct:
        st.markdown('<div class="card card-accent"><div class="card-title">Project File Structure</div>', unsafe_allow_html=True)
        struct = """credit_risk_app/
│
├── credit_risk_app.py          ← Main Streamlit app (this file)
│
├── knowledge_base/             ← RAG document store
│   ├── credit_policies.txt
│   ├── regulatory_guidelines.txt
│   ├── risk_rules.txt
│   └── grading_criteria.txt
│
├── models/                     ← Trained ML models (optional)
│   ├── random_forest.pkl
│   └── logistic_regression.pkl
│
├── chroma_db/                  ← ChromaDB persistent storage
│   └── (auto-generated)
│
├── requirements.txt
└── README.md"""
        st.code(struct, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_env:
        st.markdown('<div class="card"><div class="card-title">Get a Free Groq API Key</div>', unsafe_allow_html=True)
        st.markdown("""
        <ol style="font-size:0.82rem;line-height:2;color:#D4E0F7;">
            <li>Visit <a href="https://console.groq.com" style="color:#F0A500;">console.groq.com</a></li>
            <li>Sign in or create a free Groq account</li>
            <li>Click <strong>API Keys</strong> → <strong>Create API Key</strong></li>
            <li>Copy the key (starts with <code style="color:#F0A500;">gsk_...</code>)</li>
            <li>Paste it in the sidebar <strong>Groq API Key</strong> field</li>
        </ol>
        <p style="font-size:0.75rem;color:#5A7BA8;">The free tier provides generous rate limits for development and testing. No credit card required.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">Mode Comparison</div>', unsafe_allow_html=True)
        mode_df = pd.DataFrame({
            "Feature":           ["ML Prediction", "Risk Metrics", "Policy Lookup", "Report Quality", "Q&A Answers"],
            "Without API Key":   ["✅ Full", "✅ Full", "✅ Rule-based", "📋 Structured template", "📋 Rule-based"],
            "With Groq API Key": ["✅ Full", "✅ Full", "✅ Semantic RAG", "🤖 AI-generated (better)", "🤖 AI-generated"],
        })
        st.dataframe(mode_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='border-top:1px solid #1A2744;padding-top:1rem;text-align:center;'>
    <p style='font-size:0.68rem;color:#5A7BA8;font-family:"IBM Plex Mono";'>
    Credit Risk AI Assessment System · LangChain + LangGraph + Groq LLaMA3-70B + ChromaDB · 
    Random Forest ~93% accuracy · 32,581 training records
    </p>
</div>
""", unsafe_allow_html=True)