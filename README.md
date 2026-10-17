# FairScreen: AI-Powered Fair Resume Screening

> **IBM Z Datathon 2026 — Theme 2 (AI Secured: Sensitive Data Protection & Identity)**  
> An auditable, privacy-preserving resume screening engine designed for IBM LinuxONE (s390x).

---

## 🏗️ Repository Architecture

The codebase is partitioned into three distinct layers:

```
FairScreen/
├── frontend/                     # FRONTEND (UI & Dashboards)
│   ├── app.py                    # Multi-tab Streamlit dashboard
│   └── components/               # Visual metrics, cards & widgets
│
├── backend/                      # BACKEND (Ingestion & Orchestration)
│   ├── __init__.py
│   ├── ingest.py                 # In-memory PDF parser (pypdf + pdfplumber fallback)
│   └── pipeline.py               # End-to-end FairScreen orchestrator
│
├── ml/                           # ML & FAIRNESS ENGINE (The 6 Roles)
│   ├── __init__.py
│   ├── synth.py                  # Role 2: Biased synthetic resume dataset generator
│   ├── redact.py                 # Role 5: Demographic & identity signal scrubber
│   ├── features.py               # Role 3: TF-IDF & merit-based feature engineering
│   ├── model.py                  # Role 3: Baseline & clean screener classifiers
│   ├── audit.py                  # Role 4: AIF360 parity metrics & gender swap test
│   ├── mitigate.py               # Role 4: AIF360 sample reweighing mitigation
│   ├── explain.py                # Role 5: Merit-based token coefficient explainer
│   ├── jd_checker.py             # Role 5: Gaucher et al. gender-coded wording audit
│   └── train_all.py              # Single command to train and serialize all models
│
├── models/                       # SERIALIZED ARTIFACTS (.pkl)
│   ├── baseline_model.pkl        # Biased baseline model
│   ├── baseline_vectorizer.pkl
│   ├── redacted_model.pkl        # Identity-neutral model
│   ├── redacted_vectorizer.pkl
│   ├── mitigated_model.pkl       # Reweighed parity model
│   └── mitigated_vectorizer.pkl
│
├── data/                         # DATASETS & WORD LISTS
│   ├── gendered_words.json       # Gaucher et al. masculine/feminine stems
│   ├── name_pools.json           # Diverse male/female names for swap tests
│   └── results/audit_summary.json# Cached audit parity benchmarks
│
├── app.py                        # Root launcher delegating to frontend/app.py
├── requirements.txt              # Frozen Python dependencies
├── Dockerfile                    # Container definition for LinuxONE
├── CREDITS.md                    # Research and open-source citations
└── LICENSE                       # MIT License
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train and Serialize Models (Fixed Seed 42)
```bash
python -m ml.train_all
```

### 3. Launch the Dashboard
```bash
python -m streamlit run app.py
```
Visit **`http://localhost:8501`** in your browser.

---

## 🛡️ Core Fairness & Privacy Guarantees

1. **Strict Attribute Separation**: Protected attributes (`gender`, `race_eth`, `age`) **never** enter the feature matrix or model training. They reside exclusively in the separate audit bank.
2. **Identity Redaction**: Candidate names, contact information, honorifics, and gendered pronouns are scrubbed prior to scoring.
3. **Counterfactual Swap Stability**: Demonstrates that changing a candidate's gender from female to male yields an identical score under the mitigated model (zero shift), eliminating baseline discrimination.
4. **Local Air-Gapped Execution**: All inference, audit, and explainability happen entirely in-memory with zero external API calls.
