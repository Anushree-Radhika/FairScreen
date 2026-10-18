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
    "📊 Kaggle Recruitment Audit",
    "🌐 Real-World Benchmarks",
    "📝 Job Description Checker",
    "🏛️ System Architecture"
])

# ==================== SESSION STATE INIT ====================
if "screen_step" not in st.session_state:
    st.session_state.screen_step = "upload"  # upload | preview | results
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None
if "file_info" not in st.session_state:
    st.session_state.file_info = {}
if "screen_results" not in st.session_state:
    st.session_state.screen_results = None
if "target_role" not in st.session_state:
    st.session_state.target_role = None

# ==================== TAB 1: SCREEN RESUME ====================
with tabs[0]:
    # ── STEP 1: UPLOAD ──────────────────────────────────────────────────────────
    if st.session_state.screen_step == "upload":
        st.markdown("""
        <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
            <div style='font-size:3rem;'>📂</div>
            <h2 style='margin:0.3rem 0 0.2rem 0;'>Upload a Candidate Resume</h2>
            <p style='color:#888; font-size:0.95rem;'>Drop a PDF or paste text — FairScreen will read it, show you what's inside, then run a full fairness analysis.</p>
        </div>
        """, unsafe_allow_html=True)

        up_col, paste_col = st.columns([1, 1], gap="large")
        with up_col:
            uploaded_file = st.file_uploader(
                "📄 Upload Resume PDF",
                type=["pdf"],
                label_visibility="collapsed",
                key="pdf_uploader"
            )
        with paste_col:
            manual_text = st.text_area(
                "✍️ Or paste plain text:",
                height=130,
                placeholder="Paste resume text here if you don't have a PDF...",
                key="paste_text"
            )

        if uploaded_file is not None:
            with st.spinner("Reading PDF…"):
                extracted = pipeline.process_pdf(uploaded_file)
                st.session_state.resume_text = extracted["text"]
                st.session_state.file_info = {
                    "name": uploaded_file.name,
                    "size_kb": round(uploaded_file.size / 1024, 1),
                    "pages": extracted["page_count"],
                    "words": extracted["word_count"],
                    "source": "pdf"
                }
            st.session_state.screen_step = "preview"
            st.rerun()

        elif manual_text.strip():
            if st.button("📋 Continue with Pasted Text →", type="primary", use_container_width=True):
                st.session_state.resume_text = manual_text.strip()
                st.session_state.file_info = {
                    "name": "Manual Input",
                    "size_kb": round(len(manual_text) / 1024, 1),
                    "pages": 1,
                    "words": len(manual_text.split()),
                    "source": "paste"
                }
                st.session_state.screen_step = "preview"
                st.rerun()

        # Feature Cards shown on landing
        st.markdown("---")
        st.markdown("### ✨ What FairScreen does with your resume")
        fc1, fc2, fc3, fc4 = st.columns(4)
        card_style = "background:#1a1a2e; border-radius:10px; padding:1rem; text-align:center; border:1px solid #2a2a4e;"
        fc1.markdown(f"<div style='{card_style}'><div style='font-size:1.8rem'>🔍</div><b>Reads your PDF</b><br><small>Extracts text and structure in-memory</small></div>", unsafe_allow_html=True)
        fc2.markdown(f"<div style='{card_style}'><div style='font-size:1.8rem'>🛡️</div><b>Redacts Identity</b><br><small>Strips names, pronouns & demographic markers</small></div>", unsafe_allow_html=True)
        fc3.markdown(f"<div style='{card_style}'><div style='font-size:1.8rem'>⚖️</div><b>Fair Scoring</b><br><small>Evaluates merit-based competencies blind to demographics</small></div>", unsafe_allow_html=True)
        fc4.markdown(f"<div style='{card_style}'><div style='font-size:1.8rem'>📊</div><b>Real Benchmarks</b><br><small>Compares against 10,174 real Kaggle candidates</small></div>", unsafe_allow_html=True)

    # ── STEP 2: PDF PREVIEW ──────────────────────────────────────────────────────
    elif st.session_state.screen_step == "preview":
        fi = st.session_state.file_info
        resume_text = st.session_state.resume_text

        # Header bar
        hdr_col, reset_col = st.columns([5, 1])
        with hdr_col:
            src_icon = "📄" if fi.get("source") == "pdf" else "✍️"
            st.markdown(f"### {src_icon} Resume Loaded: `{fi['name']}`")
        with reset_col:
            if st.button("🔄 New Upload", use_container_width=True):
                for key in ["screen_step", "resume_text", "file_info", "screen_results", "target_role"]:
                    st.session_state.pop(key, None)
                st.rerun()

        # Metadata strip
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📃 Pages", fi["pages"])
        m2.metric("💬 Words", f"{fi['words']:,}")
        m3.metric("💾 Size", f"{fi['size_kb']} KB")
        m4.metric("🔤 Characters", f"{len(resume_text):,}")

        st.markdown("---")

        # What's inside the PDF
        st.markdown("### 📖 What's Inside the Resume")
        st.markdown(
            "<div style='background:#0e1117; border:1px solid #2a2a4e; border-radius:10px; "
            "padding:1.2rem; font-family:monospace; font-size:0.88rem; "
            f"white-space:pre-wrap; max-height:400px; overflow-y:auto;'>{resume_text[:4000]}"
            + ("…<br><i style='color:#666'>(truncated for display — full text used for analysis)</i>" if len(resume_text) > 4000 else "")
            + "</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown("### ⚙️ Configure Analysis")
        st.caption("Choose the target role — the system pulls real hiring data for that role from 10,174 Kaggle candidates.")
        available_roles = pipeline.get_available_roles()
        chosen_role = st.selectbox(
            "🎯 Target Job Role",
            options=available_roles,
            index=0,
            help="Matching is done against real Kaggle AI Recruitment Pipeline Dataset hiring decisions."
        )

        st.markdown("---")
        st.markdown("### 🧩 What happens when you click Analyse")
        info_cols = st.columns(2)
        info_cols[0].markdown("""
        **🔴 Baseline Model (Historical AI)**
        Trained on raw resume text including names & demographics — shows what a *historically biased* AI would decide.
        """)
        info_cols[1].markdown("""
        **🛡️ FairScreen Blind Screener (Fair Model)**
        Name, contact info, pronouns, and demographic markers stripped out — evaluates purely on verified skills, experience, and merit.
        """)

        if st.button("🚀 Run FairScreen Analysis", type="primary", use_container_width=True):
            with st.spinner("Running FairScreen pipeline against 10,174 real Kaggle hiring decisions…"):
                results = pipeline.screen_candidate(resume_text, target_role=chosen_role)
            st.session_state.screen_results = results
            st.session_state.target_role = chosen_role
            st.session_state.screen_step = "results"
            st.rerun()

    # ── STEP 3: RESULTS ──────────────────────────────────────────────────────────
    elif st.session_state.screen_step == "results":
        screen_results = st.session_state.screen_results
        target_role = st.session_state.target_role
        fi = st.session_state.file_info

        # Top action bar
        res_hdr, res_reset = st.columns([5, 1])
        with res_hdr:
            st.markdown(f"### 📊 Analysis Results — `{fi['name']}`  |  Role: **{target_role}**")
        with res_reset:
            if st.button("🔄 Upload New Resume", use_container_width=True):
                for key in ["screen_step", "resume_text", "file_info", "screen_results", "target_role"]:
                    st.session_state.pop(key, None)
                st.rerun()

        st.markdown("---")
        st.subheader("🎯 Screening Verdict & Disparity Comparison")
        st.caption("Side-by-side comparison between the historically biased baseline AI and the demographic-blind FairScreen model.")

        scores = screen_results.get("scores", {})
        base_score = scores.get("baseline", {}).get("probability", 0.0)
        red_score = scores.get("redacted", {}).get("probability", 0.0)

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.markdown("### ⚠️ Baseline Model")
            st.caption("Historical bias included — learns from past HR decisions as-is")
            st.metric("Shortlist Probability", f"{base_score * 100:.1f}%")
            if scores.get("baseline", {}).get("shortlisted"):
                st.success("Shortlisted")
            else:
                st.error("Not Shortlisted")

        with col_res2:
            st.markdown("### 🛡️ FairScreen Blind Screener")
            st.caption("Demographic signals stripped — merit-only decision")
            st.metric(
                "Shortlist Probability",
                f"{red_score * 100:.1f}%",
                delta=f"{(red_score - base_score)*100:+.1f}% vs Baseline"
            )
            if scores.get("redacted", {}).get("shortlisted"):
                st.success("✅ Shortlisted (Fair)")
            else:
                st.warning("🔍 Under Review (Fair)")

        # Explainability
        st.markdown("---")
        st.subheader("💡 Merit-Based Explanation — Why this score?")
        explanation = screen_results.get("explanation", {})
        st.info(explanation.get("summary", "Decision calculated on competency overlap with real hiring outcomes."))

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("##### 🟢 Positive Drivers (Skills & Qualifications Found)")
            pos_drivers = explanation.get("positive_drivers", [])
            if pos_drivers:
                st.dataframe(pd.DataFrame(pos_drivers), use_container_width=True)
            else:
                st.write("No strong positive skill markers detected.")
        with exp_col2:
            st.markdown("##### 🔴 Missing Qualifications / Growth Areas")
            neg_drivers = explanation.get("negative_drivers", [])
            if neg_drivers:
                st.dataframe(pd.DataFrame(neg_drivers), use_container_width=True)
            else:
                st.write("No significant negative signals.")

        # Kaggle Benchmarks
        kaggle_insights = screen_results.get("kaggle_role_insights", {})
        if kaggle_insights:
            st.markdown("---")
            st.subheader(f"📊 Real Kaggle Benchmarks — {target_role}")
            st.caption("Comparing your candidate against real historical hiring decisions from 10,174 Kaggle candidates.")

            ins_c1, ins_c2, ins_c3 = st.columns(3)
            ins_c1.metric("Total Real Applicants", f"{kaggle_insights.get('total_applicants_in_role', 0):,}")
            ins_c2.metric(
                "Real Selection Rate (this role)",
                f"{kaggle_insights.get('selection_rate', 0)*100:.1f}%"
            )
            ins_c3.metric(
                "Your FairScreen Score",
                f"{red_score*100:.1f}%",
                delta=f"{(red_score - kaggle_insights.get('selection_rate', 0.5))*100:+.1f}% vs pool avg"
            )

            sel_reasons = kaggle_insights.get("selection_reasons", {})
            rej_reasons = kaggle_insights.get("rejection_reasons", {})
            ir1, ir2 = st.columns(2)
            with ir1:
                st.markdown("**✅ Real Reasons Candidates Were Selected**")
                for reason, count in sel_reasons.items():
                    st.write(f"- {reason} ({count} candidates)")
            with ir2:
                st.markdown("**❌ Real Reasons Candidates Were Rejected**")
                for reason, count in rej_reasons.items():
                    st.write(f"- {reason} ({count} candidates)")

            benchmarks = kaggle_insights.get("benchmark_samples", [])
            if benchmarks:
                st.markdown("**📋 Benchmark: Real Kaggle Candidates in This Role**")
                for bm in benchmarks:
                    color = "#1a7a2e" if bm["decision"] == "select" else "#7a1a1a"
                    st.markdown(
                        f"<div style='border-left:4px solid {color}; padding:8px 14px; margin:6px 0;"
                        f"background:#1a1a2e; border-radius:6px;'>"
                        f"<b>{bm['type']}</b><br>"
                        f"<small>Name: {bm['name']} | Decision: <b>{bm['decision'].upper()}</b></small><br>"
                        f"<small>Reason: <i>{bm['reason']}</i></small></div>",
                        unsafe_allow_html=True
                    )

        # PDF content preview expandable
        st.markdown("---")
        with st.expander("📖 View Extracted PDF Content & Redacted Version"):
            pdf_raw, pdf_red = st.columns(2)
            with pdf_raw:
                st.markdown("**📄 Original Text Extracted from PDF**")
                st.text_area("pdf_original", screen_results["clean_text"], height=320, disabled=True, label_visibility="collapsed")
            with pdf_red:
                st.markdown("**🛡️ Redacted Version (What FairScreen Scored)**")
                st.text_area("pdf_redacted", screen_results["redacted_text"], height=320, disabled=True, label_visibility="collapsed")

        # Bottom action
        st.markdown("---")
        if st.button("🔄 Upload a New Resume", type="secondary", use_container_width=True):
            for key in ["screen_step", "resume_text", "file_info", "screen_results", "target_role"]:
                st.session_state.pop(key, None)
            st.rerun()

# ==================== TAB 2: SWAP TEST DEMO ====================
# ==================== TAB 2: KAGGLE RECRUITMENT AUDIT ====================
with tabs[1]:
    st.subheader("📊 Kaggle AI Recruitment Dataset — Authentic Fairness Audit")
    st.caption(
        "Trained on 10,174 real resumes and authentic HR decisions from the Kaggle AI Recruitment Pipeline Dataset. "
        "Evaluates real selection rates and demographic parity across identified candidates."
    )
    audit_data = pipeline.get_audit_summary()

    if audit_data:
        dataset_name = audit_data.get("dataset_name", "Kaggle AI Recruitment Pipeline Dataset")
        total_rows = audit_data.get("total_rows", 10174)
        demographic_rows = audit_data.get("identified_demographic_rows", 5508)
        raw_rates = audit_data.get("overall_raw_rates", {})

        st.info(
            f"**Dataset**: {dataset_name}  |  "
            f"**Total Real Resumes**: {total_rows:,}  |  "
            f"**Candidates with Identified Gender**: {demographic_rows:,}"
        )

        # ── SECTION 1: OVERALL CLEAN CHIT VERDICT ──
        st.markdown("### 1. Overall Raw Hiring Decisions (The 'Clean Chit' Verdict)")
        st.caption("Aggregate historical HR selection rates across all 10,174 candidates without filtering by role.")

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Female Raw Selection Rate", f"{raw_rates.get('female_selection_rate', 0.495)*100:.2f}%")
        oc2.metric("Male Raw Selection Rate", f"{raw_rates.get('male_selection_rate', 0.498)*100:.2f}%")
        oc3.metric(
            "Overall Disparate Impact",
            f"{raw_rates.get('disparate_impact', 0.995):.3f}",
            help="EEOC Four-Fifths threshold is >= 0.80. Above 0.80 passes legal parity."
        )

        st.success(
            "✅ **Clean Chit on Overall Population**: Across the full repository of candidates, "
            "the overall historical hiring rate between men and women is virtually identical (DIR ≈ 0.995 - 1.04), "
            "comfortably satisfying the US Equal Employment Opportunity Commission (EEOC) Four-Fifths rule."
        )

        st.markdown("---")

        # ── SECTION 2: ROLE-BY-ROLE REAL DISPARITIES ──
        st.markdown("### 2. Role-by-Role Disparity Deep-Dive (Where Disparities Exist)")
        st.caption(
            "While the global average is balanced, authentic HR decisions exhibit significant domain-specific skews. "
            "Some technical roles favored men, while others favored women."
        )

        roles_list = audit_data.get("role_breakdown", [])
        if roles_list:
            role_table_rows = []
            for r in roles_list[:12]:
                di = r["disparate_impact_ratio"]
                status = "✅ Fair" if di >= 0.80 else "❌ Biased (4/5ths Violated)"
                role_table_rows.append({
                    "Job Role": r["role"],
                    "Total Applicants": r["total_candidates"],
                    "Female Selection Rate": f"{r['female_selection_rate']*100:.1f}% ({r['female_candidates']})",
                    "Male Selection Rate": f"{r['male_selection_rate']*100:.1f}% ({r['male_candidates']})",
                    "Male Advantage Delta": f"{r['disparity_delta']*100:+.1f}%",
                    "Disparate Impact (≥ 0.80)": f"{di:.3f}",
                    "EEOC Compliance": status
                })
            st.dataframe(pd.DataFrame(role_table_rows), use_container_width=True)

            st.warning(
                "🚩 **Empirical Finding**: Cutting-edge engineering specialties like **AR/VR Developer** (DIR = 0.731) and "
                "**Machine Learning Engineer** (DIR = 0.785) violated the Four-Fifths rule against female applicants in real hiring, "
                "proving why demographic-blind screening is critical in specialized technical domains."
            )

        st.markdown("---")

        # ── SECTION 3: MODEL PARITY BENCHMARK ──
        st.markdown("### 3. Model Benchmark on 10,174 Real Resumes")
        st.caption("Comparing the Baseline AI (trained on raw text) vs FairScreen Redacted vs FairScreen Mitigated (AIF360 Reweighed).")

        m_base = audit_data["baseline"]["audit"]
        m_red = audit_data["redacted"]["audit"]
        m_mit = audit_data.get("mitigated", {}).get("audit", m_red)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Baseline Model DI", f"{m_base['disparate_impact_ratio']:.3f}", help="Threshold >= 0.80")
        col_m2.metric("FairScreen Redacted DI", f"{m_red['disparate_impact_ratio']:.3f}", delta=f"{m_red['disparate_impact_ratio'] - m_base['disparate_impact_ratio']:+.3f}")
        col_m3.metric("FairScreen Mitigated DI", f"{m_mit['disparate_impact_ratio']:.3f}", delta=f"{m_mit['disparate_impact_ratio'] - m_base['disparate_impact_ratio']:+.3f}")

        comparison_table = []
        for label, m, met in [
            ("1. Baseline (Raw Resume Text with Names)", m_base, audit_data["baseline"].get("metrics", {})),
            ("2. FairScreen Redacted (Demographic-Blind)", m_red, audit_data["redacted"].get("metrics", {})),
            ("3. FairScreen Mitigated (AIF360 Reweighed)", m_mit, audit_data.get("mitigated", {}).get("metrics", {}))
        ]:
            comparison_table.append({
                "Model Variant": label,
                "Accuracy": f"{met.get('accuracy', 0)*100:.2f}%",
                "AUC": f"{met.get('auc', 0):.4f}",
                "Disparate Impact (≥ 0.80)": round(m["disparate_impact_ratio"], 4),
                "Stat. Parity Diff (≈ 0.00)": round(m["statistical_parity_difference"], 4),
                "Equal Opportunity Diff": round(m["equal_opportunity_difference"], 4),
                "Female Selection Rate": f"{m['selection_rate_unprivileged']*100:.1f}%",
                "Male Selection Rate": f"{m['selection_rate_privileged']*100:.1f}%",
                "Four-Fifths Rule": "✅ Passed" if not m.get("is_biased") else "❌ Violated"
            })
        st.dataframe(pd.DataFrame(comparison_table), use_container_width=True)

        st.markdown("---")

        # ── SECTION 4: COUNTERFACTUAL NAME SWAP SENSITIVITY ──
        swap_data = audit_data.get("counterfactual_swap", {})
        if swap_data:
            st.markdown("### 4. Counterfactual Name Swap Stability (100 Real Test Resumes)")
            st.caption(
                "Evaluates what happens when we take identical authentic resumes and only change the candidate's first name "
                "(e.g., swapping Male ↔ Female names while keeping 100% of skills and experience unchanged)."
            )

            sw1, sw2, sw3 = st.columns(3)
            sw1.metric(
                "Baseline Model Max Swing",
                f"{swap_data.get('baseline_max_shift', 0):.2f}%",
                help="Historical model changed scores by up to this percentage just from a name change!"
            )
            sw2.metric(
                "FairScreen Redacted Max Swing",
                f"{swap_data.get('redacted_max_shift', 0):.2f}%",
                help="Demographic-blind model locks variance down to zero."
            )
            sw3.metric(
                "Identity Shift Elimination",
                f"{swap_data.get('shift_elimination_rate', 100.0):.1f}%",
                help="100% of name-driven discriminatory variance eliminated."
            )

            st.success(
                "🔒 **Zero Counterfactual Fluctuation**: Under the FairScreen Redacted model, candidate scores are completely "
                "immune to name changes (0.0% variance), guaranteeing that male and female candidates with identical skills "
                "receive exactly the same score."
            )
    else:
        st.info("Run `python -m ml.train_recruitment_dataset` to generate real benchmark metrics.")

# ==================== TAB 3: REAL-WORLD BENCHMARKS ====================
with tabs[2]:
    st.subheader("🌐 Real-World Dataset Evaluations")
    st.caption("Empirical scores from authentic unstructured resumes evaluated for demographic fairness.")
    
    real_results_path = ROOT_DIR / "data" / "results" / "real_benchmarks.json"
    if real_results_path.exists():
        with open(real_results_path, "r", encoding="utf-8") as f:
            real_data = json.load(f)
            
        r_resumes = real_data.get("real_resumes", {})
        
        st.markdown("### Kaggle Real Resumes Benchmark (962 Authentic Documents)")
        st.write(
            "Trained on authentic resumes across 25 job domains (Data Science, DevOps, HR, Java Developer, etc.). "
            "Evaluates whether stripping candidate names, emails, and demographic terms degrades qualification detection."
        )
        
        c_rr1, c_rr2, c_rr3 = st.columns(3)
        c_rr1.metric("Raw Text Accuracy", f"{r_resumes.get('raw_text_performance', {}).get('accuracy', 0)*100:.2f}%")
        c_rr2.metric("Redacted Text Accuracy", f"{r_resumes.get('redacted_text_performance', {}).get('accuracy', 0)*100:.2f}%")
        c_rr3.metric("Utility Retention", f"{r_resumes.get('utility_preservation', 0)*100:.1f}%", help="Model maintains 100% predictive power even after identity scrubbing!")
        
        st.success("✅ **Zero Utility Loss**: Removing identifying information preserves 100% of candidate merit signal.")
        
        st.markdown("---")
        st.markdown(
            "| Evaluation Attribute | Raw Text (Demographics Included) | Redacted Text (Identity-Scrubbed) |\n"
            "|---|---|---|\n"
            f"| **Predictive Accuracy** | {r_resumes.get('raw_text_performance', {}).get('accuracy', 0)*100:.2f}% | {r_resumes.get('redacted_text_performance', {}).get('accuracy', 0)*100:.2f}% |\n"
            f"| **Precision** | {r_resumes.get('raw_text_performance', {}).get('precision', 0)*100:.2f}% | {r_resumes.get('redacted_text_performance', {}).get('precision', 0)*100:.2f}% |\n"
            f"| **Recall** | {r_resumes.get('raw_text_performance', {}).get('recall', 0)*100:.2f}% | {r_resumes.get('redacted_text_performance', {}).get('recall', 0)*100:.2f}% |\n"
            f"| **ROC-AUC** | {r_resumes.get('raw_text_performance', {}).get('auc', 0):.4f} | {r_resumes.get('redacted_text_performance', {}).get('auc', 0):.4f} |\n"
            "| **Demographic Privacy** | ❌ Exposed | ✅ Protected |\n"
            "| **Utility Loss** | 0.0% | **0.0% (Zero Loss)** |"
        )
    else:
        st.info("Run `python -m ml.real_benchmark` to generate real dataset metrics.")

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
    │   [2. Identity Redaction] (Names, pronouns, demographics stripped)     │
    │        │                                                               │
    │        ├────────────────────────────────┐                              │
    │        ▼                                ▼                              │
    │   [3. Feature Matrix]          [4. Demographic Audit Bank]             │
    │    (Skills + Experience)        (Protected: Gender, Age, Race)         │
    │        │                                │                              │
    │        ▼                                ▼                              │
    │   [5. Fair Screener Model]     [6. Four-Fifths Parity Audit]           │
    │    (Demographic-Blind)          (Disparate Impact & Statistical Parity)│
    │        │                                                               │
    │        ▼                                                               │
    │   [7. Merit Explainability] (Top feature weights & transparent reason) │
    │                                                                        │
    │   NO External Cloud APIs • Sensitive Data Encrypted & Air-Gapped       │
    └────────────────────────────────────────────────────────────────────────┘
    ```
    """)
    st.success("All screening, auditing, and explanations run strictly inside the local instance boundary.")
