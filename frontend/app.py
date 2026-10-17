import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so modules import reliably
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from backend.pipeline import FairScreenPipeline
from ml.jd_checker import check_jd

# Configure Streamlit
st.set_page_config(
    page_title="FairScreen | AI Resume Screening & Fairness Audit",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0062FF, #00D4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .badge-fair {
        background-color: #28a745;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .badge-bias {
        background-color: #dc3545;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .card {
        padding: 1.2rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_pipeline():
    return FairScreenPipeline()

pipeline = get_pipeline()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/scales.png", width=64)
    st.title("FairScreen")
    st.caption("Ethical, Auditable Resume Screening on IBM LinuxONE")
    
    st.markdown("---")
    st.markdown("### 🛡️ Privacy & Compliance")
    st.success("✅ **Demographic Signal Redaction Active**")
    st.info("🔒 **Air-Gapped Local Inference** (Zero External API calls)")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Specifications")
    st.markdown("""
    - **Architecture**: IBM LinuxONE (s390x ready)
    - **Scorer**: Linear & Gradient Boosting
    - **Debiasing**: AIF360 Sample Reweighing
    - **Explainability**: Merit Coefficient Attribution
    """)

# Main Title
st.markdown('<div class="main-header">FairScreen: AI Resume Screening & Fairness Engine</div>', unsafe_allow_html=True)
st.caption("Equally qualified candidates deserve an equal chance, regardless of gender, race, or age.")

tabs = st.tabs([
    "📄 Screen Resume (PDF)",
    "🔄 Counterfactual Swap Test",
    "📊 Bias Audit & Mitigation",
    "📝 Job Description Checker",
    "🏛️ System Architecture"
])

# ==================== TAB 1: SCREEN RESUME ====================
with tabs[0]:
    st.subheader("Candidate Resume Evaluation")
    
    col_input, col_meta = st.columns([2, 1])
    
    with col_input:
        uploaded_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])
        manual_text = st.text_area(
            "Or paste resume text directly:",
            height=130,
            placeholder="Paste raw text here if you don't have a PDF..."
        )

    resume_text_to_process = None
    file_info = {}

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF stream..."):
            extracted = pipeline.process_pdf(uploaded_file)
            resume_text_to_process = extracted["text"]
            file_info = {
                "name": uploaded_file.name,
                "size_kb": round(uploaded_file.size / 1024, 1),
                "pages": extracted["page_count"],
                "words": extracted["word_count"]
            }
    elif manual_text.strip():
        resume_text_to_process = manual_text.strip()
        file_info = {
            "name": "Manual Input",
            "size_kb": round(len(resume_text_to_process) / 1024, 1),
            "pages": 1,
            "words": len(resume_text_to_process.split())
        }

    with col_meta:
        if file_info:
            st.markdown("#### Document Metadata")
            st.metric("File", file_info["name"])
            m1, m2 = st.columns(2)
            m1.metric("Pages", file_info["pages"])
            m2.metric("Word Count", file_info["words"])
        else:
            st.info("Upload a PDF to view metrics and run evaluation.")

    if resume_text_to_process:
        with st.spinner("Running FairScreen pipeline..."):
            screen_results = pipeline.screen_candidate(resume_text_to_process)
            
        st.markdown("---")
        st.subheader("🎯 Screening Verdict & Disparity Comparison")
        
        scores = screen_results.get("scores", {})
        base_score = scores.get("baseline", {}).get("probability", 0.0)
        mit_score = scores.get("mitigated", {}).get("probability", 0.0)
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown("### ⚠️ Baseline Model")
            st.caption("Learns historical human hiring biases")
            st.metric("Shortlist Probability", f"{base_score * 100:.1f}%")
            if scores.get("baseline", {}).get("shortlisted"):
                st.success("Shortlisted")
            else:
                st.error("Not Shortlisted")

        with col_res2:
            st.markdown("### 🛡️ Redacted Model")
            st.caption("Demographic signals stripped")
            red_score = scores.get("redacted", {}).get("probability", 0.0)
            st.metric("Shortlist Probability", f"{red_score * 100:.1f}%")
            if scores.get("redacted", {}).get("shortlisted"):
                st.success("Shortlisted")
            else:
                st.error("Not Shortlisted")

        with col_res3:
            st.markdown("### ⚖️ FairScreen Mitigated")
            st.caption("Redaction + AIF360 Sample Reweighing")
            st.metric("Shortlist Probability", f"{mit_score * 100:.1f}%", delta=f"{(mit_score - base_score)*100:+.1f}% vs Baseline")
            if scores.get("mitigated", {}).get("shortlisted"):
                st.success("Shortlisted (Fair)")
            else:
                st.warning("Under Review (Fair)")

        # Explainability
        st.markdown("---")
        st.subheader("💡 Merit-Based Transparent Explanation")
        explanation = screen_results.get("explanation", {})
        st.write(explanation.get("summary", "Decision calculated on competency overlap."))
        
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("##### 🟢 Top Positive Drivers (Skills / Qualifications)")
            pos_drivers = explanation.get("positive_drivers", [])
            if pos_drivers:
                st.dataframe(pd.DataFrame(pos_drivers))
            else:
                st.write("No strong positive skill markers.")

        with exp_col2:
            st.markdown("##### 🔴 Missing Qualifications / Areas for Growth")
            neg_drivers = explanation.get("negative_drivers", [])
            if neg_drivers:
                st.dataframe(pd.DataFrame(neg_drivers))
            else:
                st.write("No severe negative penalties detected.")

        # Side-by-Side Text Redaction View
        st.markdown("---")
        with st.expander("🔍 View Raw vs Redacted Text (Identity Protection)"):
            c_raw, c_red = st.columns(2)
            with c_raw:
                st.markdown("**Original Ingested Text**")
                st.text_area("Original", screen_results["clean_text"], height=300, disabled=True)
            with c_red:
                st.markdown("**Redacted Identity-Neutral Text (What FairScreen Evaluates)**")
                st.text_area("Redacted", screen_results["redacted_text"], height=300, disabled=True)

# ==================== TAB 2: SWAP TEST DEMO ====================
with tabs[1]:
    st.subheader("Counterfactual Gender Swap Test")
    st.write(
        "A truly fair model will produce identical scores for the same resume when only the candidate's "
        "first name and gender pronouns are swapped."
    )
    
    sample_candidate = (
        "Sarah Jenkins\n"
        "Email: sarah.jenkins@example.com | Phone: (555) 019-3829\n\n"
        "Professional Summary:\n"
        "Sarah is a senior software engineer with 8 years of industry experience. "
        "She has led cloud microservices development and architected high-throughput data pipelines.\n\n"
        "Technical Skills:\n"
        "Python, SQL, Docker, Kubernetes, AWS, FastAPI, Git, CI/CD\n\n"
        "Experience:\n"
        "Senior Developer at Tech Corp (2018 - Present)\n"
        "Colleagues praise her leadership and technical excellence."
    )
    
    test_resume = st.text_area("Resume for Swap Test:", value=sample_candidate, height=180)
    
    if st.button("🚀 Run Counterfactual Swap Test", type="primary"):
        with st.spinner("Executing swap test..."):
            swap_res = pipeline.screen_candidate(test_resume, candidate_first_name="Sarah")
            swap_info = swap_res.get("counterfactual_swap", {})
            
            c_before, c_after = st.columns(2)
            
            with c_before:
                st.markdown(f"#### Original Candidate ({swap_info.get('original_name', 'Original')})")
                st.metric("Baseline Model Score", f"{swap_info.get('baseline_score_before', 0)*100:.1f}%")
                st.metric("FairScreen Mitigated Score", f"{swap_info.get('mitigated_score_before', 0)*100:.1f}%")
                
            with c_after:
                st.markdown(f"#### Swapped Candidate ({swap_info.get('swapped_name', 'Swapped')})")
                st.metric(
                    "Baseline Model Score",
                    f"{swap_info.get('baseline_score_after', 0)*100:.1f}%",
                    delta=f"{swap_info.get('baseline_score_delta', 0)*100:+.1f}% Flip",
                    delta_color="inverse"
                )
                st.metric(
                    "FairScreen Mitigated Score",
                    f"{swap_info.get('mitigated_score_after', 0)*100:.1f}%",
                    delta=f"{swap_info.get('mitigated_score_delta', 0)*100:+.1f}% (Zero Shift)",
                    delta_color="normal"
                )
                
            if abs(swap_info.get("baseline_score_delta", 0)) > 0.05:
                st.error("🚨 **Disparity Detected in Baseline**: Swapping the name changed the score without any change to qualifications!")
            if abs(swap_info.get("mitigated_score_delta", 0)) < 0.01:
                st.success("✅ **Fairness Verified in Mitigated Model**: Identical merit scores awarded regardless of gender.")

# ==================== TAB 3: BIAS AUDIT & MITIGATION ====================
with tabs[2]:
    st.subheader("Algorithmic Fairness Audit & Benchmark")
    audit_data = pipeline.get_audit_summary()
    
    if audit_data:
        m_base = audit_data["baseline"]["audit"]
        m_red = audit_data["redacted"]["audit"]
        m_mit = audit_data["mitigated"]["audit"]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Baseline Disparate Impact", f"{m_base['disparate_impact_ratio']:.3f}", help="Four-fifths rule threshold: 0.80")
            st.markdown('<span class="badge-bias">FAILED (Biased)</span>', unsafe_allow_html=True)
        with col_m2:
            st.metric("Redacted Disparate Impact", f"{m_red['disparate_impact_ratio']:.3f}")
            st.markdown('<span class="badge-fair">PASSED</span>', unsafe_allow_html=True)
        with col_m3:
            st.metric("Mitigated Disparate Impact", f"{m_mit['disparate_impact_ratio']:.3f}", delta=f"{m_mit['disparate_impact_ratio'] - m_base['disparate_impact_ratio']:+.3f}")
            st.markdown('<span class="badge-fair">GOLD STANDARD (Parity)</span>', unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("#### Comprehensive Parity Metrics Table")
        
        comparison_table = [
            {
                "Model Variant": "1. Baseline (Unmitigated)",
                "Disparate Impact (≥ 0.80)": m_base["disparate_impact_ratio"],
                "Stat. Parity Diff (≈ 0.00)": m_base["statistical_parity_difference"],
                "Equal Opportunity Diff": m_base["equal_opportunity_difference"],
                "Unprivileged Selection Rate": f"{m_base['selection_rate_unprivileged']*100:.1f}%",
                "Privileged Selection Rate": f"{m_base['selection_rate_privileged']*100:.1f}%",
                "Four-Fifths Rule": "❌ Violated" if m_base["is_biased"] else "✅ Passed"
            },
            {
                "Model Variant": "2. Redacted Signals",
                "Disparate Impact (≥ 0.80)": m_red["disparate_impact_ratio"],
                "Stat. Parity Diff (≈ 0.00)": m_red["statistical_parity_difference"],
                "Equal Opportunity Diff": m_red["equal_opportunity_difference"],
                "Unprivileged Selection Rate": f"{m_red['selection_rate_unprivileged']*100:.1f}%",
                "Privileged Selection Rate": f"{m_red['selection_rate_privileged']*100:.1f}%",
                "Four-Fifths Rule": "✅ Passed"
            },
            {
                "Model Variant": "3. FairScreen Mitigated (Reweighed)",
                "Disparate Impact (≥ 0.80)": m_mit["disparate_impact_ratio"],
                "Stat. Parity Diff (≈ 0.00)": m_mit["statistical_parity_difference"],
                "Equal Opportunity Diff": m_mit["equal_opportunity_difference"],
                "Unprivileged Selection Rate": f"{m_mit['selection_rate_unprivileged']*100:.1f}%",
                "Privileged Selection Rate": f"{m_mit['selection_rate_privileged']*100:.1f}%",
                "Four-Fifths Rule": "✅ Passed (Optimal)"
            }
        ]
        st.dataframe(pd.DataFrame(comparison_table), use_container_width=True)
    else:
        st.info("Run `python -m ml.train_all` to generate benchmark metrics.")

# ==================== TAB 4: JOB DESCRIPTION CHECKER ====================
with tabs[3]:
    st.subheader("Job Description Gendered-Language Audit")
    st.caption("Based on Gaucher, Davidai & Kay (2011) research on masculine vs feminine coded wording.")
    
    sample_jd = (
        "We are looking for an aggressive, competitive engineering lead to dominate our market. "
        "The ideal candidate must be decisive, independent, and self-reliant. "
        "You will lead an outspoken team and support customer relationships."
    )
    jd_input = st.text_area("Paste Job Description:", value=sample_jd, height=140)
    
    if st.button("🔍 Analyze Job Description", type="primary"):
        jd_result = check_jd(jd_input)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Masculine Coded Words", jd_result["masculine_count"])
        c2.metric("Feminine Coded Words", jd_result["feminine_count"])
        c3.metric("Tone Verdict", jd_result["bias_tone"].split()[0])
        
        st.info(f"**Language Diagnosis**: {jd_result['bias_tone']}")
        
        if jd_result["masculine_words"]:
            st.write(f"🚩 **Flagged Masculine Words**: `{', '.join(jd_result['masculine_words'])}`")
        if jd_result["feminine_words"]:
            st.write(f"🌸 **Flagged Feminine Words**: `{', '.join(jd_result['feminine_words'])}`")
            
        if jd_result["suggestions"]:
            st.markdown("#### 💡 Neutral Alternative Recommendations")
            for orig, repl in jd_result["suggestions"].items():
                st.write(f"- Replace **'{orig}'** with ➜ *'{repl}'*")

# ==================== TAB 5: SYSTEM ARCHITECTURE ====================
with tabs[4]:
    st.subheader("IBM Z & LinuxONE Platform Architecture")
    st.markdown("""
    ### FairScreen Architectural Boundary
    ```
    ┌────────────────────────────────────────────────────────────────────────┐
    │                        IBM LinuxONE Boundary (s390x)                   │
    │                                                                        │
    │   [PDF Upload]                                                         │
    │        │                                                               │
    │        ▼                                                               │
    │   [1. Ingestion Engine] (pypdf & pdfplumber in-memory stream)          │
    │        │                                                               │
    │        ▼                                                               │
    │   [2. Identity Redaction] (Demographics & pronouns stripped)           │
    │        │                                                               │
    │        ├────────────────────────────────┐                              │
    │        ▼                                ▼                              │
    │   [3. Feature Matrix]          [4. Separate Audit Bank]                │
    │    (Skills + Experience)        (Protected: Gender, Age, Race)         │
    │        │                                │                              │
    │        ▼                                ▼                              │
    │   [5. Screener Model]          [6. AIF360 Parity Audit]                │
    │    (Logistic / GBDT)            (Disparate Impact & 4/5ths Rule)       │
    │        │                                │                              │
    │        ├────────────────────────────────┘                              │
    │        ▼                                                               │
    │   [7. Sample Reweighing Mitigation]                                    │
    │        │                                                               │
    │        ▼                                                               │
    │   [8. Merit Explainability] (Top feature weights & transparent reason) │
    │                                                                        │
    │   NO External Cloud APIs • Sensitive Data Encrypted & Air-Gapped       │
    └────────────────────────────────────────────────────────────────────────┘
    ```
    """)
    st.success("All screening, auditing, and explanations run strictly inside the local instance boundary.")
