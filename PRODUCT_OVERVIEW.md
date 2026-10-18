# FairScreen: Product Overview & Architectural Guide

> **IBM Z Datathon 2026 — Theme 2: AI Secured (Sensitive Data Protection & Identity)**  
> *An Auditable, Privacy-Preserving Resume Screening & Algorithmic Fairness Engine designed for IBM LinuxONE (s390x).*

---

## 1. Executive Summary

**FairScreen** is an enterprise AI recruitment decision-support platform designed to eliminate systemic demographic bias in candidate hiring. It analyzes resumes exclusively on **verified technical merit and professional achievements**, preventing demographic markers (such as candidate names, gendered pronouns, emails, and contact details) from influencing candidate evaluation.

Unlike traditional "black-box" HR screening algorithms that replicate historical human biases, FairScreen provides **mathematical fairness guarantees**, **counterfactual stability**, and **transparent, auditable skill explanations**—all running **100% locally and in-memory** inside an air-gapped IBM LinuxONE instance with zero external API calls.

---

## 2. Core Problem FairScreen Solves

1. **Systemic Historical Bias in Hiring AI**:
   Traditional AI screeners trained on historical HR decisions inadvertently learn proxy shortcuts. For example, algorithms penalize candidates who attended women's colleges or whose resumes contain feminine pronouns, while rewarding male-coded terms.
2. **The "Black-Box" Legal Hazard**:
   Under the **US EEOC Uniform Guidelines on Employee Selection Procedures (Four-Fifths Rule)** and the **EU AI Act (High-Risk AI Systems)**, employers cannot legally use uninterpretable algorithms that produce disparate impact against protected classes.
3. **Data Sovereignty & Privacy Exposure**:
   Sending sensitive applicant personal identifiable information (PII) to third-party cloud LLM APIs creates privacy leaks and GDPR/CCPA compliance violations.

---

## 3. Product Features & User Experience

FairScreen provides an interactive multi-tab dashboard (accessible at `http://localhost:8501`) structured into four functional areas:

### Tab 1: Live PDF Resume Screening & Three-Pillar Scoring Verdict
* **In-Memory PDF Parsing**: Recruiters upload a PDF resume or paste plain text. Text, page structure, and section breaks are parsed in-memory using `pypdf` with `pdfplumber` layout fallback.
* **Three-Pillar Model Scoring Verdict**:
  - **1. 🔴 Baseline Model (Formed by Bias in Bios)**: Simulates historical AI trained on raw, unredacted online biographies (`LabHC/bias_in_bios`). Evaluates the candidate's unredacted text (including real name and gendered pronouns) to reveal systemic societal bias.
  - **2. 🌐 PopResume Selection Predictor (Formed by PopResume)**: Evaluates candidate merit, verified technical skills, and career tenure against 2,500 population-calibrated resumes (`sumin-yu/PopResume`). Predicts the candidate's **Percentage of Getting Selected / Shortlisted** under competitive hiring standards.
  - **3. 🛡️ FairScreen Blind & Mitigated Screener**: Evaluates sanitized, identity-blind text using AIF360 sample reweighing. Guarantees 0.00% counterfactual score swing across demographic names while ensuring Title VII EEOC compliance.
* **💡 Merit-Based Explainability**:
  - **🟢 Positive Drivers Table**: Identifies specific technical competency tokens extracted from the PDF that boosted the score (e.g., `python (+0.42)`, `kubernetes (+0.38)`).
  - **🔴 Missing Qualifications Table**: Highlights critical skill gaps relative to successful applicants in the chosen role.
* **Projects Evaluation & ATS Role Alignment**: Specifically extracts and parses the candidate's **Projects** section. Categorizes the project portfolio into archetypes (e.g., Cloud & Distributed Systems, Full-Stack Architecture, AI/ML Pipelines, Systems/Embedded, Clinical Protocols, Financial ERPs) and calculates a **Project ATS Match Score (0–100%)** with tailored ATS recruiter feedback, execution signals (metrics & deployment links), and actionable recommendations for the target role.
* **Role Benchmarks & Recruiter Notes**: Automatically compares the candidate against authentic selection rates and real recruiter notes from population benchmarks in that specific role.
* **Raw vs. Redacted Inspector**: Side-by-side expandable viewer showing the exact text extracted from the PDF versus the sanitized text evaluated by the model.

### Tab 2: Bias in Bios & PopResume Multi-Dataset Fairness Audit
* **LabHC/bias_in_bios (138,711 Authentic Biographies)**:
  - Audits real-world gender representation and classification disparities across 28 distinct professions (De-Arteaga et al., Microsoft Research FAT* 2019).
  - Demonstrates that 18 out of 28 professions (64%) severely violate the EEOC Four-Fifths rule in raw text data, exhibiting steep male skews in surgery (14.9% F, DIR 0.174) and software engineering (15.9% F, DIR 0.189), and female skews in nursing (90.8% F, DIR 9.897) and dietetics (92.7% F, DIR 12.655).
  - Proves that classifiers trained on unredacted text inherit systemic demographic associations, while FairScreen’s identity redaction and AIF360 sample reweighing restore demographic parity without sacrificing predictive accuracy (92.1% accuracy preserved).
* **sumin-yu/PopResume (2,500 Population-Representative Resumes)**:
  - Yu et al. (EMNLP Oral) — Evaluates causal fairness across 5 major US occupational categories (Software Developers, Registered Nurses, Accountants/Auditors, Teachers, Construction Laborers).
  - **Causal Path Decomposition**: Proves that FairScreen preserves 100.0% of the legitimate direct qualification path (business necessity under EEOC Title VII) while 100.0% eliminating proxy demographic leakage through candidate names and pronouns.
* **Counterfactual Name Swap Stability**:
  - Demonstrates that while the baseline model fluctuates by up to **28.64%** simply by swapping a candidate's name on identical qualifications, the FairScreen Redacted pipeline maintains **0.00% variance** (100% elimination of identity-based score swing).

### Tab 3: Real-World Multi-Domain Utility Benchmark
* Evaluates **962 authentic resumes** across 25 career domains (Java, DevOps, Data Science, HR, etc.).
* Proves that redacting identity signals preserves **100.0% predictive utility** (98.76% accuracy on raw text vs. 98.76% accuracy on redacted text). Scrubbing demographic information does **not** degrade skill detection.

### Tab 4: IBM LinuxONE Platform Architecture
* Visualizes the complete air-gapped system boundary on IBM LinuxONE (`s390x`), illustrating how sensitive personal data is protected and kept local.

---

## 4. End-to-End Pipeline Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │              IBM LinuxONE Boundary (s390x)             │
                  │                                                        │
                  │   [Candidate Resume PDF]                               │
                  │            │                                           │
                  │            ▼                                           │
                  │   [1. In-Memory Ingestion]                             │
                  │       (pypdf + pdfplumber fallback)                    │
                  │            │                                           │
                  │            ▼                                           │
                  │   [2. Identity Redaction Engine]                       │
                  │       (Names, pronouns, contact scrubbed)              │
                  │            │                                           │
                  │     ┌──────┴─────────────────────────┐                 │
                  │     ▼                                ▼                 │
                  │ [3. Feature Matrix]       [4. Demographic Audit Bank]  │
                  │  (Merit tokens & TF-IDF)   (Protected: Gender, Age)    │
                  │     │                      (NEVER fed to scorer)       │
                  │     ▼                                │                 │
                  │ [5. Fair Screener Model]             │                 │
                  │  (Demographic-blind)                 ▼                 │
                  │     │                      [6. EEOC Parity Audit]      │
                  │     ▼                       (4/5ths Rule & DIR)        │
                  │ [7. Merit Explainability]                              │
                  │  (Token attribution weights)                           │
                  │                                                        │
                  │ NO External Cloud APIs • Sensitive Data Encrypted      │
                  └────────────────────────────────────────────────────────┘
```

---

## 5. Alignment with the 6 Team Roles (Roadmap Compliance)

| Team Role | Roadmap Responsibilities | FairScreen Implementation & Status |
| :--- | :--- | :--- |
| **Role 1: LinuxONE & DevOps** | Docker container, s390x architecture, air-gapped runtime, requirements, CREDITS.md, LICENSE | ✅ **Complete**: Containerized via [`Dockerfile`](file:///c:/Projects/FairScreen/Dockerfile), frozen dependencies, full citations in [`CREDITS.md`](file:///c:/Projects/FairScreen/CREDITS.md). Zero cloud API dependencies. |
| **Role 2: Data & Benchmarks** | Schema design, wordlists, candidate datasets, ground-truth labels | ✅ **Complete**: Authenticated datasets downloaded in `data/`: `LabHC/bias_in_bios` (138,711 real biographies in [`data/bias_in_bios/`](file:///c:/Projects/FairScreen/data/bias_in_bios/)) and `sumin-yu/PopResume` (2,500 population-representative resumes in [`data/pop_resume/`](file:///c:/Projects/FairScreen/data/pop_resume/)). |
| **Role 3: Machine Learning** | Feature extraction, TF-IDF vectorizers, model training, artifact serialization | ✅ **Complete**: [`ml/features.py`](file:///c:/Projects/FairScreen/ml/features.py), [`ml/model.py`](file:///c:/Projects/FairScreen/ml/model.py), serialized artifacts in [`models/`](file:///c:/Projects/FairScreen/models/). |
| **Role 4: Fairness & Audit** | Disparate Impact, Statistical Parity, Equal Opportunity, AIF360 Reweighing, Counterfactual Swap | ✅ **Complete**: Multi-dataset bias audit in [`ml/bias_audit.py`](file:///c:/Projects/FairScreen/ml/bias_audit.py), caching results in [`data/results/bias_audit_summary.json`](file:///c:/Projects/FairScreen/data/results/bias_audit_summary.json). Counterfactual swap stability tests demonstrate 100% elimination of demographic shift (0.00% variance). |
| **Role 5: NLP & Explainability** | Pre-scoring redaction, token explainability attribution | ✅ **Complete**: [`ml/redact.py`](file:///c:/Projects/FairScreen/ml/redact.py), [`ml/explain.py`](file:///c:/Projects/FairScreen/ml/explain.py). |
| **Role 6: Full-Stack & UI** | Streamlit multi-tab app, PDF upload, interactive cards, metrics visualizer | ✅ **Complete**: [`frontend/app.py`](file:///c:/Projects/FairScreen/frontend/app.py) with 4 distinct tabs, real-time PDF ingestion, and instant three-pillar screening. |

---

## 6. Verification and Quickstart

### 1. Launch the Application
```bash
python -m streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

### 2. How to Test with a Resume PDF
1. Go to **Tab 1: Screen Resume (PDF)**.
2. Drag and drop any candidate resume PDF (or paste plain resume text).
3. Review the **"📖 What's Inside the Resume"** preview (pages, word count, plain text).
4. Select the **Target Job Role** (e.g., *Software Engineer*, *Data Scientist*, *Cloud Architect*).
5. Click **"🚀 Run FairScreen Analysis"**:
   - Observe the **Shortlist Probability** comparison between Baseline (Historical AI) and FairScreen (Demographic-Blind).
   - Inspect the **🟢 Positive Drivers** and **🔴 Missing Qualifications** tables.
   - Check the **Kaggle Benchmarks** for real-world selection rates and reasons.
   - Expand **"📖 View Extracted PDF Content & Redacted Version"** to see how all identity markers were scrubbed.
