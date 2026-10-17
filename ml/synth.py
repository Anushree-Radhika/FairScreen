import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

SKILL_SETS = [
    "Python", "SQL", "Machine Learning", "Docker", "Kubernetes", "AWS", "Git",
    "Java", "React", "TypeScript", "TensorFlow", "PyTorch", "Linux", "Data Modeling",
    "PostgreSQL", "CI/CD", "FastAPI", "Pandas", "Scikit-Learn", "System Design"
]

ROLES = ["Data Scientist", "Software Engineer", "MLOps Engineer", "Backend Developer"]

def load_names() -> Tuple[List[str], List[str]]:
    """Loads male and female names from data/name_pools.json or falls back to built-ins."""
    names_path = Path(__file__).resolve().parent.parent / "data" / "name_pools.json"
    if names_path.exists():
        with open(names_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("male_names", []), data.get("female_names", [])
    
    # Fallback default names
    males = ["James", "John", "Robert", "Michael", "David", "William", "Richard", "Joseph"]
    females = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan", "Jessica", "Sarah"]
    return males, females

def make_candidates(n: int = 1000, bias_group: str = "Female", bias_strength: float = 0.35, seed: int = 42) -> pd.DataFrame:
    """
    Role 2 Contract:
    Generates a realistic candidate dataset with controlled ground-truth qualification
    and an intentionally injected historical hiring bias against `bias_group`.
    
    Rule: Sensitive attributes (gender, race_eth, age) are generated for audit purposes only.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    male_names, female_names = load_names()
    genders = rng.choice(["Male", "Female"], size=n, p=[0.5, 0.5])
    races = rng.choice(["Asian", "Black", "Hispanic", "White", "Other"], size=n, p=[0.25, 0.15, 0.15, 0.40, 0.05])
    ages = rng.integers(22, 62, size=n)
    
    records = []
    
    for i in range(n):
        gender = genders[i]
        first_name = random.choice(female_names if gender == "Female" else male_names)
        last_name = f"Candidate_{i:04d}"
        full_name = f"{first_name} {last_name}"
        
        # Merit attributes
        years_exp = int(rng.integers(1, 18))
        num_skills = int(rng.integers(3, 9))
        candidate_skills = list(rng.choice(SKILL_SETS, size=num_skills, replace=False))
        skills_match = round(min(1.0, (num_skills / 10.0) + (years_exp / 30.0)), 3)
        target_role = random.choice(ROLES)
        
        pronoun_subject = "She" if gender == "Female" else "He"
        pronoun_possessive = "Her" if gender == "Female" else "His"
        pronoun_object = "her" if gender == "Female" else "him"
        
        # Resume Text Generation
        resume_text = (
            f"{full_name}\n"
            f"Email: {first_name.lower()}.{last_name.lower()}@example.com | Phone: (555) 019-{i:04d}\n\n"
            f"Professional Summary:\n"
            f"{first_name} is an experienced {target_role} with {years_exp} years of industry track record. "
            f"{pronoun_subject} has driven scalable engineering solutions and led high-impact technical initiatives.\n\n"
            f"Core Technical Competencies:\n"
            f"{', '.join(candidate_skills)}\n\n"
            f"Professional Experience:\n"
            f"Senior Engineer at Tech Solutions ({2026 - years_exp} - Present)\n"
            f"- Built scalable architectures and delivered production software.\n"
            f"- Colleagues commend {pronoun_object} for strong teamwork and technical leadership.\n"
            f"- Managed projects aligning with {pronoun_possessive.lower()} team's quarterly milestones.\n\n"
            f"Education:\n"
            f"B.S. in Computer Science, University of Technology\n"
        )
        
        # Ground Truth Qualification (Merit-based)
        # 1 if candidate meets merit threshold, else 0
        merit_score = (years_exp * 0.05) + (skills_match * 0.5) + rng.normal(0, 0.05)
        true_qualified = int(merit_score >= 0.55)
        
        # Biased Historical Hiring Outcome (Injected Bias)
        # If candidate belongs to bias_group, artificially depress selection probability
        hire_prob = merit_score
        if gender == bias_group:
            hire_prob -= bias_strength
        
        hire_prob = np.clip(hire_prob, 0.05, 0.95)
        hired = int(rng.random() < hire_prob)
        
        records.append({
            "candidate_id": f"CAN_{i:05d}",
            "full_name": full_name,
            "first_name": first_name,
            "gender": gender,
            "race_eth": races[i],
            "age": int(ages[i]),
            "years_exp": years_exp,
            "skills": candidate_skills,
            "skills_match": skills_match,
            "resume_text": resume_text,
            "true_qualified": true_qualified,
            "hired": hired
        })
        
    return pd.DataFrame(records)
