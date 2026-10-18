# FairScreen: AI-Powered Fair Resume Screening

> **IBM Z Datathon 2026 — Theme 2 (AI Secured: Sensitive Data Protection & Identity)**  
> An auditable, privacy-preserving resume screening engine designed for IBM LinuxONE (s390x).

---

## 🏗️ Repository Architecture

The codebase is partitioned into three distinct layers:

```
FairScreen/
├── frontend/                     # FRONTEND (UI & Dashboards)
│   └── app.py                    # Multi-tab Streamlit dashboard (Screening, Audits, Benchmarks)
│
├── backend/                      # BACKEND (Ingestion & Orchestration)
│   ├── __init__.py
│   ├── ingest.py                 # In-memory PDF parser (pypdf + pdfplumber fallback)
│   └── pipeline.py               # End-to-end FairScreen multi-model orchestrator
│
├── ml/                           # ML & FAIRNESS ENGINE (100% Authentic Data)
│   ├── __init__.py
│   ├── bias_audit.py             # Audits Bias in Bios (138k bios) & PopResume (2.5k resumes)
│   ├── build_popresume.py        # Population-representative resume dataset generator
│   ├── real_benchmark.py         # Evaluates 962 authentic multi-domain resumes
│   ├── redact.py                 # Demographic & identity signal scrubber
│   ├── features.py               # TF-IDF & merit-based feature engineering
│   ├── model.py                  # Baseline & clean screener classifiers
│   ├── audit.py                  # AIF360 parity metrics & counterfactual swap test
│   ├── mitigate.py               # AIF360 sample reweighing mitigation
│   ├── explain.py                # Merit-based token coefficient explainer
│   ├── project_analyzer.py       # Projects section parser & ATS role alignment engine
│   └── train_all.py              # Single command to train and audit all models
│
├── models/                       # SERIALIZED ARTIFACTS (.pkl)
│   ├── bias_in_bios_baseline_model.pkl
│   ├── bias_in_bios_baseline_vectorizer.pkl
│   ├── pop_resume_model.pkl
│   ├── pop_resume_vectorizer.pkl
│   ├── bias_in_bios_redacted_model.pkl
│   ├── bias_in_bios_redacted_vectorizer.pkl
│   ├── bias_in_bios_mitigated_model.pkl
│   └── bias_in_bios_mitigated_vectorizer.pkl
│
├── data/                         # DATASETS & BENCHMARKS (Local Only, Excluded via .gitignore)
│   ├── bias_in_bios/             # LabHC/bias_in_bios (138,711 real biographies, 28 professions)
│   ├── pop_resume/               # sumin-yu/PopResume (2,500 population-representative resumes)
│   ├── real_resumes.csv          # 962 authentic resumes for utility benchmarking
│   ├── name_pools.json           # Diverse male/female names for swap tests
│   └── results/                  # Cached bias audit benchmarks & role statistics
│
├── app.py                        # Root launcher delegating to frontend/app.py
├── requirements.txt              # Frozen Python dependencies
├── Dockerfile                    # Container definition for LinuxONE
├── PRODUCT_OVERVIEW.md           # Comprehensive architectural & algorithmic specification
└── CREDITS.md                    # Research and open-source citations
```

## 📊 Datasets Used & Their Purpose

FairScreen is built on **100% authentic, peer-reviewed benchmarks** to measure, isolate, and mitigate algorithmic hiring bias. Raw data files reside strictly within the local environment and are excluded from Git version control (via `.gitignore`) to protect data storage and integrity.

### 1. `LabHC/bias_in_bios` — Empirical Bias Benchmark & Baseline Training
- **Source**: Hugging Face [`LabHC/bias_in_bios`](https://huggingface.co/datasets/LabHC/bias_in_bios) (De-Arteaga et al., Microsoft Research, *ACM FAT\* 2019*).
- **Volume**: **138,711 real-world online biographies** spanning 28 distinct professions (`dev.parquet` [39k] & `test.parquet` [99k]).
- **Purpose**:
  - **Historical Biased AI (Baseline Model)**: Demonstrates how standard AI models trained on unredacted text absorb systemic occupational stereotypes (e.g., Software Engineering is 15.9% female with Disparate Impact Ratio = 0.189; Nursing is 90.8% female with DIR = 9.897).
  - **AIF360 Mitigation Training**: Serves as the training ground for FairScreen's demographic-blind, sample-reweighed model to correct historical selection disparities.
  - **EEOC Four-Fifths Compliance Audit**: Evaluates the disparate impact ratio across all 28 career professions in Tab 2.

### 2. `sumin-yu/PopResume` — Calibrated Selection Percentage & Causal Fairness
- **Source**: Hugging Face [`sumin-yu/PopResume`](https://huggingface.co/datasets/sumin-yu/PopResume) (Yu et al., *EMNLP Oral 2024*).
- **Volume**: **2,500 population-representative resumes** across 12 major occupational groups (Software Developers, Registered Nurses, Accountants & Auditors, Teachers, Construction Laborers, etc.).
- **Purpose**:
  - **PopResume Selection Predictor**: Powers Pillar 2 of candidate screening, generating an exact **calibrated Percentage of Getting Selected** based on legitimate qualifications, degrees, and tenure.
  - **Cohort Benchmarks & Hiring Distributions**: Powers role-specific selection criteria insights (comparing candidate scores against real applicant pools).
  - **Causal Path Decomposition**: Isolates the *direct qualification path* (skills, degrees, credentials) from the *indirect demographic proxy path* (names, gender pronouns).

### 3. Real Unstructured Resumes — Utility Preservation Benchmark
- **Source**: Authentic Multi-Domain Resume Corpus (Bhawal et al., Kaggle Authentic Career Corpus).
- **Volume**: **962 authentic resumes** across 25 career domains (Data Science, DevOps, HR, Java Developer, etc.).
- **Purpose**:
  - **0.0% Utility Loss Validation**: Empirically proves in Tab 3 that scrubbing candidate names, emails, and demographic identifiers results in **zero loss of predictive accuracy** (retains 100% of merit signal, ROC-AUC 0.999).

### 4. SSA Demographic Name Pools — Counterfactual Swap Testing
- **Source**: United States Social Security Administration (SSA) demographic frequency registries.
- **Volume**: Curated male and female given name registries across diverse demographics.
- **Purpose**:
  - **Counterfactual Name Swap Robustness**: Evaluates identical authentic candidate credentials with swapped first names (Male ↔ Female) in Tab 2 to verify that FairScreen locks discriminatory variance down to **0.0% score swing**.
  - **Deterministic Entity Redaction**: Feeds the zero-leakage in-memory redaction engine to strip candidate identities before feature extraction.

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

---

## 📖 Deep-Dive Documentation

For detailed technical specifications, refer to:
- **[`PRODUCT_OVERVIEW.md`](file:///c:/Projects/FairScreen/PRODUCT_OVERVIEW.md)**: Full end-to-end product architecture, algorithmic explanations of the 3-pillar screening engine, mathematical formulas for parity metrics (DIR, SPD, EOD), 6-role team distribution, and legal/regulatory compliance (EEOC Title VII, NYC Local Law 144, EU AI Act).
- **[`CREDITS.md`](file:///c:/Projects/FairScreen/CREDITS.md)**: Academic citations, peer-reviewed literature, and open-source packages utilized in the project.
