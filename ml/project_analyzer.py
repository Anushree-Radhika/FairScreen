"""
FairScreen Project Analyzer & ATS Role Alignment Engine.
Analyzes the Projects section of candidate resumes, categorizes the project archetypes,
and evaluates role-specific ATS alignment with actionable feedback.
100% In-Memory, Air-Gapped, and Deterministic (Zero Cloud APIs).
"""

import re
from typing import Dict, Any, List, Optional

# Technology & domain dictionaries for project categorization
PROJECT_ARCHETYPES = {
    "Cloud & Distributed Systems": {
        "keywords": [
            "docker", "kubernetes", "k8s", "aws", "gcp", "azure", "terraform",
            "microservices", "ci/cd", "jenkins", "helm", "kafka", "rabbitmq",
            "redis", "nginx", "load balancer", "serverless", "lambda", "cloud"
        ],
        "weight_for_roles": {
            "software developers": 0.95,
            "devops engineer": 1.0,
            "cloud architect": 1.0,
            "data scientist": 0.70
        },
        "description": "Scalable cloud infrastructure, containerization, and distributed event-driven architectures."
    },
    "Full-Stack & Web Architecture": {
        "keywords": [
            "react", "angular", "vue", "next.js", "django", "flask", "fastapi",
            "spring boot", "node.js", "express", "rest api", "graphql", "html",
            "css", "javascript", "typescript", "tailwind", "backend", "frontend",
            "api", "web application", "crud", "mvc"
        ],
        "weight_for_roles": {
            "software developers": 1.0,
            "devops engineer": 0.75,
            "cloud architect": 0.80,
            "data scientist": 0.60
        },
        "description": "Interactive client-facing web applications and robust backend service integration."
    },
    "AI, Machine Learning & Data Science": {
        "keywords": [
            "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn",
            "pandas", "numpy", "nlp", "llm", "transformers", "computer vision", "cnn",
            "rnn", "clustering", "classification", "regression", "data pipeline",
            "spark", "hadoop", "eda", "feature engineering", "model training", "predictive"
        ],
        "weight_for_roles": {
            "data scientist": 1.0,
            "software developers": 0.85,
            "devops engineer": 0.65,
            "cloud architect": 0.75
        },
        "description": "Statistical learning models, neural networks, ETL pipelines, and data-driven intelligence."
    },
    "Mobile & Cross-Platform Development": {
        "keywords": [
            "flutter", "react native", "swift", "kotlin", "ios", "android", "xcode",
            "mobile app", "dart", "mobile ui", "app store", "play store"
        ],
        "weight_for_roles": {
            "software developers": 0.90,
            "devops engineer": 0.50,
            "cloud architect": 0.60,
            "data scientist": 0.40
        },
        "description": "Native and multiplatform mobile applications optimized for handheld devices."
    },
    "Systems, Embedded & IoT": {
        "keywords": [
            "c++", "c language", "rust", "embedded", "rtos", "linux kernel", "firmware",
            "microcontroller", "arduino", "raspberry pi", "socket programming", "device driver",
            "memory management", "multithreading", "low-level"
        ],
        "weight_for_roles": {
            "software developers": 0.90,
            "devops engineer": 0.70,
            "cloud architect": 0.65,
            "data scientist": 0.50
        },
        "description": "High-performance systems programming, memory optimization, and hardware interfacing."
    },
    "Enterprise Data & Database Engineering": {
        "keywords": [
            "postgresql", "postgres", "mysql", "mongodb", "oracle", "sql server",
            "sqlite", "database design", "indexing", "query optimization", "etl",
            "snowflake", "bigquery", "data warehouse", "schema"
        ],
        "weight_for_roles": {
            "software developers": 0.92,
            "data scientist": 0.92,
            "accountants auditors": 0.75,
            "cloud architect": 0.88,
            "devops engineer": 0.80
        },
        "description": "Relational data modeling, ACID compliance, high-volume query tuning, and warehousing."
    },
    "Healthcare & Clinical Protocols": {
        "keywords": [
            "ehr", "emr", "epic", "cerner", "hl7", "fhir", "patient care", "triage",
            "clinical trial", "hospital protocol", "vital signs", "hipaa", "medication",
            "nursing care plan", "icu", "telehealth"
        ],
        "weight_for_roles": {
            "registered nurses": 1.0,
            "physician": 1.0,
            "surgeon": 1.0,
            "software developers": 0.45
        },
        "description": "Clinical care implementation, patient outcome workflows, and health informatics."
    },
    "Finance, Audit & Accounting Systems": {
        "keywords": [
            "financial modeling", "gaap", "ifrs", "ledger", "balance sheet", "p&l",
            "reconciliation", "tax compliance", "internal audit", "erp", "sap",
            "quickbooks", "financial forecasting", "variance analysis"
        ],
        "weight_for_roles": {
            "accountants auditors": 1.0,
            "software developers": 0.40,
            "data scientist": 0.70
        },
        "description": "Fiscal reconciliation, statutory accounting controls, and ERP financial system modeling."
    }
}

# Impact and complexity indicators
QUANTITATIVE_METRICS_REGEX = re.compile(
    r"\b(\d+%\s*(?:increase|reduction|improvement|growth|decrease|faster|boost)|"
    r"(?:handled|processed|served)\s*\d+[\w\+]*|"
    r"(?:reduced|improved|increased|saved)\s*(?:latency|time|cost|downtime)\s*(?:by\s*)?\d+|"
    r"\d+\+?\s*(?:users|requests|qps|rps|queries|records|transactions))\b",
    re.IGNORECASE
)

DEPLOYMENT_INDICATORS = [
    "github.com", "gitlab.com", "deployed", "live at", "production", "hosted on",
    "docker hub", "pypi", "npm", "aws s3", "vercel", "netlify", "heroku"
]


def extract_project_text(full_text: str, sections: Dict[str, str]) -> str:
    """
    Extracts text belonging to the projects section.
    If section dictionary has 'projects', uses that; otherwise heuristic regex extraction.
    """
    if "projects" in sections and len(sections["projects"].strip()) > 30:
        return sections["projects"].strip()

    # Heuristic fallback: look for block starting with Projects heading
    pattern = re.compile(
        r"(?:^|\n)\s*(?:key\s+|academic\s+|personal\s+|technical\s+)?projects\s*[:\-]?\s*\n(.*?)(?=\n\s*(?:education|experience|skills|certifications|awards|\Z))",
        re.DOTALL | re.IGNORECASE
    )
    m = pattern.search(full_text)
    if m:
        return m.group(1).strip()

    # Search for project-like bullet points
    lines = full_text.splitlines()
    project_lines = []
    in_proj = False
    for line in lines:
        lower = line.lower()
        if any(h in lower for h in ["project:", "project 1", "project 2", "github.com", "capstone"]):
            in_proj = True
        if in_proj:
            project_lines.append(line)
            if len(project_lines) > 25:
                break
    if project_lines:
        return "\n".join(project_lines).strip()

    return ""


def analyze_candidate_projects(
    full_text: str,
    sections: Dict[str, str],
    target_role: str = "Software Developers"
) -> Dict[str, Any]:
    """
    Evaluates candidate projects against target role expectations:
    1. Detects project presence and extracts project details.
    2. Classifies project archetypes (Full-Stack, Cloud, AI/ML, Embedded, etc.).
    3. Calculates ATS Project Relevance Score (0-100%).
    4. Detects quantitative impact metrics & deployment links.
    5. Formulates actionable ATS recruiter feedback tailored to the target role.
    """
    proj_text = extract_project_text(full_text, sections)
    has_projects_section = len(proj_text) > 40
    
    # If no separate projects section, search overall text for project mentions
    eval_text = proj_text if has_projects_section else full_text
    eval_text_lower = eval_text.lower()
    target_role_lower = target_role.lower().strip()

    # Score each archetype
    detected_archetypes: List[Dict[str, Any]] = []
    for arch_name, info in PROJECT_ARCHETYPES.items():
        found_kw = []
        for kw in info["keywords"]:
            if re.search(r"\b" + re.escape(kw) + r"\b", eval_text_lower):
                found_kw.append(kw)

        if found_kw:
            # Determine role weight
            role_weight = 0.50
            for r_key, w in info["weight_for_roles"].items():
                if r_key in target_role_lower or target_role_lower in r_key:
                    role_weight = w
                    break

            relevance = min(len(found_kw) * 18, 100) * role_weight
            detected_archetypes.append({
                "archetype": arch_name,
                "matched_keywords": found_kw,
                "keyword_count": len(found_kw),
                "role_weight": role_weight,
                "relevance_score": round(relevance, 1),
                "description": info["description"]
            })

    # Sort archetypes by relevance
    detected_archetypes.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Primary archetype
    primary_archetype = detected_archetypes[0]["archetype"] if detected_archetypes else "General Engineering / Practical Implementation"
    matched_tools = []
    for arch in detected_archetypes[:3]:
        matched_tools.extend(arch["matched_keywords"])
    matched_tools = list(dict.fromkeys(matched_tools))[:10]

    # Metrics and deployment signals
    metric_matches = QUANTITATIVE_METRICS_REGEX.findall(eval_text)
    has_metrics = len(metric_matches) > 0
    metric_samples = metric_matches[:3]

    has_deployment = any(d in eval_text_lower for d in DEPLOYMENT_INDICATORS)

    # Compute overall ATS Project Relevance Score (0 - 100%)
    if detected_archetypes:
        base_rel = detected_archetypes[0]["relevance_score"]
        # Bonus for secondary complementary archetype
        if len(detected_archetypes) > 1:
            base_rel += min(detected_archetypes[1]["relevance_score"] * 0.25, 20)
        # Bonus for metrics
        if has_metrics:
            base_rel += 10
        # Bonus for live links / deployment
        if has_deployment:
            base_rel += 8
        # Penalty if projects were only inferred from general text rather than clean section
        if not has_projects_section:
            base_rel *= 0.85
        final_project_score = min(max(round(base_rel, 1), 15.0), 98.0)
    else:
        final_project_score = 30.0 if has_projects_section else 15.0

    # ATS Tier
    if final_project_score >= 75.0:
        alignment_tier = "High Alignment"
        tier_color = "#16a34a"
        badge_bg = "#dcfce7"
    elif final_project_score >= 50.0:
        alignment_tier = "Moderate Alignment"
        tier_color = "#ca8a04"
        badge_bg = "#fef9c3"
    else:
        alignment_tier = "Needs Domain Alignment"
        tier_color = "#dc2626"
        badge_bg = "#fee2e2"

    # Generate custom ATS Recruiter Reply tailored to role and project type
    ats_reply = _generate_ats_reply(
        target_role=target_role,
        primary_archetype=primary_archetype,
        score=final_project_score,
        tools=matched_tools,
        has_metrics=has_metrics,
        has_deployment=has_deployment,
        has_section=has_projects_section
    )

    return {
        "has_projects_section": has_projects_section,
        "project_text_snippet": proj_text[:500] if proj_text else "",
        "primary_archetype": primary_archetype,
        "archetypes": detected_archetypes,
        "matched_technologies": matched_tools,
        "project_ats_score": final_project_score,
        "alignment_tier": alignment_tier,
        "tier_color": tier_color,
        "badge_bg": badge_bg,
        "has_metrics": has_metrics,
        "metric_samples": metric_samples,
        "has_deployment": has_deployment,
        "ats_recruiter_reply": ats_reply["summary"],
        "ats_recommendations": ats_reply["recommendations"],
        "role_suitability": ats_reply["suitability"]
    }


def _generate_ats_reply(
    target_role: str,
    primary_archetype: str,
    score: float,
    tools: List[str],
    has_metrics: bool,
    has_deployment: bool,
    has_section: bool
) -> Dict[str, Any]:
    """Crafts human-readable, tailored ATS recruiter feedback."""
    tools_str = ", ".join(f"`{t}`" for t in tools[:6]) if tools else "core industry tools"
    
    if score >= 75.0:
        suitability = f"Strong Match for {target_role}"
        summary = (
            f"Candidate's project portfolio demonstrates verified practical competence in **{primary_archetype}**. "
            f"Using tools like {tools_str}, the projects reflect the technical depth and hands-on complexity "
            f"typically required of competitive **{target_role}** hires."
        )
        recommendations = [
            "Ensure GitHub / portfolio repositories have clean READMEs with architecture diagrams.",
            "Highlight test coverage (unit/integration tests) and CI/CD automation pipelines.",
            "Quantify scale and user impact directly in project headers."
        ]
    elif score >= 50.0:
        suitability = f"Promising Foundational Match for {target_role}"
        summary = (
            f"Candidate built projects centered around **{primary_archetype}** ({tools_str}). "
            f"While foundational concepts are demonstrated, an ATS screener for **{target_role}** looks for "
            f"deeper end-to-end architecture, such as cloud deployment, database scaling, or production monitoring."
        )
        recommendations = [
            f"Incorporate complementary {target_role} stack components (e.g., Docker containerization, cloud deployment).",
            "Add measurable impact metrics (e.g., 'reduced API response latency by 35%').",
            "Provide live demo URLs or active repository links to verify implementation quality."
        ]
    else:
        suitability = f"Divergent / Developing Project Stack for {target_role}"
        summary = (
            f"The candidate's project work is currently classified under **{primary_archetype}**. "
            f"For the **{target_role}** track, ATS screening algorithms look for direct hands-on evidence with "
            f"role-critical toolsets. The current project scope presents a qualification gap relative to cohort benchmarks."
        )
        recommendations = [
            f"Build a flagship, production-grade project directly solving a problem in **{target_role}**.",
            "Explicitly document the technical architecture, problem statement, and libraries used.",
            "Organize project descriptions into clear Problem → Solution → Tech Stack → Outcome format."
        ]

    if not has_section:
        recommendations.insert(0, "Add an explicit **Projects** section heading in your resume to ensure ATS parsers index your projects.")

    return {
        "summary": summary,
        "recommendations": recommendations,
        "suitability": suitability
    }
