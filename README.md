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
├── ml/                           # ML & FAIRNESS ENGINE (100% Authentic Data)
│   ├── __init__.py
│   ├── train_recruitment_dataset.py # Trains on 10,174 real Kaggle recruitment resumes
│   ├── real_benchmark.py         # Evaluates 962 authentic multi-domain resumes
│   ├── redact.py                 # Demographic & identity signal scrubber
│   ├── features.py               # TF-IDF & merit-based feature engineering
│   ├── model.py                  # Baseline & clean screener classifiers
│   ├── audit.py                  # AIF360 parity metrics & counterfactual swap test
│   ├── mitigate.py               # AIF360 sample reweighing mitigation
│   ├── explain.py                # Merit-based token coefficient explainer
│   ├── jd_checker.py             # Gaucher et al. gender-coded wording audit
│   └── train_all.py              # Single command to train all authentic models
│
├── models/                       # SERIALIZED REAL ARTIFACTS (.pkl)
│   ├── recruitment_baseline_model.pkl
│   ├── recruitment_baseline_vectorizer.pkl
│   ├── recruitment_redacted_model.pkl
│   ├── recruitment_redacted_vectorizer.pkl
│   ├── recruitment_mitigated_model.pkl
│   └── recruitment_mitigated_vectorizer.pkl
│
├── data/                         # DATASETS & WORD LISTS
│   ├── kaggle_recruitment/       # 10,174 authentic candidate resumes & decisions
│   ├── gendered_words.json       # Gaucher et al. masculine/feminine stems
│   ├── name_pools.json           # Diverse male/female names for swap tests
│   └── results/                  # Cached audit parity benchmarks & role statistics
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
