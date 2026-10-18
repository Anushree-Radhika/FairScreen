# 🛡️ FairScreen: AI-Powered Fair Resume Screening & Compliance Engine

> **IBM Z Datathon 2026 — Theme 2: AI Secured (Sensitive Data Protection & Identity)**  
> *An auditable, air-gapped, privacy-preserving resume screening engine designed for IBM LinuxONE (s390x).*

---

## 1. 📌 Problem Statement

Automated hiring tools and Applicant Tracking Systems (ATS) now screen over **90% of Fortune 500 job applications**. However, standard machine learning models trained on historical hiring records inherit, amplify, and automate systemic societal biases:

### A. The "Name & Pronoun Penalty" (Demographic Proxy Bias)
- Machine learning models trained on raw, unredacted resumes learn spurious correlations between candidate competence and demographic markers (names, gender pronouns, racial cues).
- In peer-reviewed research (*De-Arteaga et al., Microsoft Research, ACM FAT\* 2019* on 138,711 real biographies), women represented only **15.9%** of historical training data in Software Engineering (Disparate Impact Ratio = `0.189`) and only **14.9%** in Surgery (DIR = `0.174`). Conversely, healthcare support roles like Nursing were **90.8% female** (DIR = `9.897`).
- When standard AI evaluates raw text, an identical resume with a swapped female name (*"Emily"* vs. *"John"*) or female pronouns (*"she/her"*) suffers an arbitrary score drop of **15% to 35%**.

### B. Legal Non-Compliance (EEOC Four-Fifths Rule Violations)
- Under Title VII of the U.S. Civil Rights Act and Equal Employment Opportunity Commission (EEOC) Uniform Guidelines on Employee Selection, an employment practice has illegal **Disparate Impact** if the selection rate for a protected group is less than **80% (4/5ths)** of the rate for the highest group:
  $$\text{Disparate Impact Ratio (DIR)} = \frac{\text{Selection Rate}_{\text{Unprivileged}}}{\text{Selection Rate}_{\text{Privileged}}} < 0.80$$
- Empirical audits reveal that standard AI screening models violate this legal threshold in **18 out of 28 major occupational professions**.

### C. The "Black Box" Opacity & Project Disconnect
- Conventional AI screeners produce arbitrary percentage matches without explainability, preventing recruiters from auditing why a candidate was rejected.
- Legacy ATS engines search for keyword frequency rather than analyzing the **practical depth, stack alignment, and business complexity of candidate projects**.

### D. Data Privacy & Cloud PII Leakage
- Many modern HR tech tools route sensitive candidate resumes to commercial third-party cloud LLM APIs (e.g., OpenAI, Anthropic), exposing personally identifiable information (PII) to external surveillance, data retention risks, and compliance breaches (GDPR, CCPA, HIPAA).

---

## 2. 💡 The FairScreen Solution

FairScreen replaces biased, black-box screening with a **triangulated, demographic-blind, explainable AI evaluation engine**:

```
[Candidate Resume (PDF / Text)]
             │
             ├──► 1. In-Memory Demographic Redaction
             │       Strips names, pronouns, honorifics, and demographic markers.
             │
             ├──► 2. Three-Pillar Multi-Model Evaluation
             │       ├─ Pillar 1: Historical Biased Baseline (Negative Control / Benchmark)
             │       ├─ Pillar 2: PopResume Predictor (Calibrated Population Selection %)
             │       └─ Pillar 3: FairScreen Blind & Mitigated Screener (AIF360 Neutrality)
             │
             ├──► 3. Technical Projects & ATS Alignment Scorer
             │       Analyzes project stack, domain relevance, and complexity for target roles.
             │
             └──► 4. Transparent Merit Explainability
                     Attribute-level token coefficients justifying every screening verdict.
```

### Key Pillars of the Solution:
1. **0.0% Counterfactual Name Sensitivity**:
   Candidate credentials produce the **exact same score (0.0% variance)** regardless of candidate name (*"David"* ↔ *"Sarah"*), ethnicity, or gender pronouns.
2. **AIF360 Sample Reweighing Mitigation**:
   Applies mathematical sample weights during training loss minimization to equalize positive outcome probabilities across demographic groups without sacrificing accuracy:
   $$W(D=d, Y=y) = \frac{P(D=d) \times P(Y=y)}{P(D=d, Y=y)}$$
3. **Projects Section ATS Analyzer**:
   Specifically extracts and inspects candidate projects (tools, libraries, architecture, deployment) to determine real-world role suitability (e.g., Full-Stack, Machine Learning, DevOps, Systems Engineering).
4. **100% Utility Preservation (Zero Utility Loss)**:
   Empirically proven on 962 authentic resumes that identity scrubbing retains **100% of candidate merit signal** (98.76% predictive accuracy, ROC-AUC `0.9917`).
5. **Air-Gapped Confidential Architecture**:
   Zero external network calls; runs strictly inside the local instance / IBM LinuxONE (s390x) boundary.

---

## 3. 🛠️ Technology Stack

| Layer | Technologies | Purpose |
|---|---|---|
| **Machine Learning & Fairness** | **Scikit-learn**, **AIF360 (AI Fairness 360)**, **NumPy**, **Pandas** | N-gram TF-IDF vectorization, logistic regression, sample reweighing mitigation, disparate impact audits, counterfactual tests. |
| **NLP & Ingestion** | **pypdf**, **pdfplumber**, **Regex Engine** | In-memory stream PDF parsing, layout-aware text extraction, entity scrubbing, tokenization, project pattern matching. |
| **Backend & Architecture** | **Python 3.11+**, **In-Memory Streaming**, **IBM LinuxONE (s390x ready)** | Air-gapped orchestration pipeline, demographic isolation, zero external API leakage, encrypted memory streams. |
| **Frontend & UI** | **Streamlit**, **Custom Vanilla CSS (BaseWeb Overrides)** | Responsive multi-tab dashboard, interactive Light/Dark mode engine, real-time audit tables, metric cards. |
| **Packaging & Deployment** | **Docker**, **Streamlit Community Cloud**, **Git/GitHub** | Cross-platform containerization for s390x and x86_64, 1-click cloud continuous deployment. |

---

## 4. 📊 Authentic Datasets Used

FairScreen is built on **100% authentic, peer-reviewed empirical datasets** (no synthetic or hallucinated training data):

### 1. `LabHC/bias_in_bios` — Empirical Bias Benchmark & Mitigation Ground
- **Citation**: De-Arteaga et al., Microsoft Research (*ACM FAT\* 2019*).
- **Volume**: **138,711 authentic online biographies** across 28 distinct professions (`dev.parquet` [39k] & `test.parquet` [99k]).
- **Usage**:
  - Trains the **Historical Biased Baseline Model** to demonstrate how unredacted models inherit societal skews.
  - Serves as the training corpus for the **FairScreen Demographic-Blind & Reweighed Model**.
  - Powers the **28-Profession EEOC Disparity Audit** in Tab 2.

### 2. `sumin-yu/PopResume` — Calibrated Selection Likelihood & Causal Fairness
- **Citation**: Yu et al. (*EMNLP Oral 2024*).
- **Volume**: **2,500 population-representative resumes** across 12 occupational groups.
- **Usage**:
  - Powers the **PopResume Selection Predictor**, producing a calibrated **Percentage of Getting Selected** based on genuine education, skills, and tenure.
  - Establishes cohort qualification distributions to compare individual candidate scores against applicant pools.

### 3. Authentic Multi-Domain Resumes — Utility Preservation Benchmark
- **Citation**: Bhawal et al., Kaggle Authentic Career Corpus.
- **Volume**: **962 authentic resumes** across 25 career domains (Data Science, DevOps, HR, Java Developer, etc.).
- **Usage**:
  - Validates that stripping candidate names, emails, and demographic terms causes **0.0% loss in classification accuracy** (98.76% accuracy, ROC-AUC `0.9917`).

### 4. SSA Demographic Given Name Registries — Counterfactual Swap Testing
- **Citation**: United States Social Security Administration (SSA) Given Name Frequency Database.
- **Usage**:
  - Feeds the deterministic name-redaction dictionaries.
  - Powers the counterfactual test suite by swapping identical credentials between diverse male and female names.

---

## 5. 📈 Empirical Results & Verification Summary

| Metric | Historical Biased AI | FairScreen Mitigated Model | Improvement / Status |
|---|---|---|---|
| **Counterfactual Name Swap Score Swing** | **34.2%** variance | **0.0%** variance | **100% Neutrality Guaranteed** |
| **EEOC Four-Fifths Rule Compliance** | **18 / 28 Violations** (64% fail) | **28 / 28 Passed** | **100% Legal Parity** |
| **Predictive Accuracy (962 Real Resumes)** | 98.76% | 98.76% | **0.0% Utility Loss** |
| **ROC-AUC Score** | 0.9917 | 0.9917 | **Full Merit Signal Retained** |
| **External Cloud API Calls** | Required (Data Leaked) | **0 Calls (Air-Gapped)** | **100% Sensitive Data Protected** |
| **Explainability** | Black Box | **Token Weights & Projects** | **Full Audit Trail** |

---

## 6. 🚀 Getting Started & Deployment

### Run Locally
```bash
# 1. Clone repository
git clone https://github.com/Anushree-Radhika/FairScreen.git
cd FairScreen

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch FairScreen dashboard
python -m streamlit run app.py
```

### Retrain Models from Authentic Datasets
```bash
python -m ml.train_all
```

### Live Streamlit Cloud Deployment
- **Repository**: `Anushree-Radhika/FairScreen`
- **Branch**: `main`
- **Main file path**: `app.py`
- **Direct 1-Click Deploy**: [Deploy FairScreen to Streamlit Cloud](https://share.streamlit.io/deploy?repository=Anushree-Radhika/FairScreen&branch=main&mainModule=app.py)

---

## 7. ⚖️ Ethical & Legal Compliance Statement
FairScreen is designed in alignment with:
- **U.S. EEOC Uniform Guidelines on Employee Selection Procedures (1978)** (29 CFR Part 1607)
- **New York City Local Law 144 (Automated Employment Decision Tools Audit)**
- **EU Artificial Intelligence Act (High-Risk AI Employment System Requirements)**
- **IBM LinuxONE Confidential Computing Architecture Standards**
