import json
import random
from pathlib import Path
import pandas as pd
import numpy as np

pop_dir = Path(__file__).resolve().parent.parent / "data" / "pop_resume"
data_dir = Path(__file__).resolve().parent.parent / "data"

with open(pop_dir / "job_spec_deterministic.json") as f:
    job_specs = json.load(f)
with open(pop_dir / "pool_company.json") as f:
    companies = json.load(f)
with open(pop_dir / "pool_work_bullet.json") as f:
    work_bullets = json.load(f)
with open(pop_dir / "pool_major.json") as f:
    majors = json.load(f)
with open(pop_dir / "pool_education_school.json") as f:
    schools = json.load(f)
with open(data_dir / "name_pools.json") as f:
    names_data = json.load(f)

males = names_data["male_names"]
females = names_data["female_names"]

major_key_map = {
    "software_developers": "software_engineers",
    "elementary_middle_school_teachers": "elementary_middle_school_teachers",
    "construction_laborers": "construction_laborers",
    "registered_nurses": "registered_nurses",
    "accountants_auditors": "accountants_auditors"
}

bachelor_schools = schools.get("Bachelor's", [
    "University of Washington", "University of Illinois", "Purdue University", "Georgia Tech", "University of Texas"
])

random.seed(42)
np.random.seed(42)

records = []
N_PER_OCC = 500

for occ, spec in job_specs.items():
    occ_title = occ.replace("_", " ").title()
    occ_skills = spec["skills"]
    comp_list = companies.get(occ, ["Global Systems", "National Services"])
    
    # Flatten bullets across junior/mid/senior
    bullets_dict = work_bullets.get(occ, {})
    flat_bullets = []
    for level, b_list in bullets_dict.items():
        if isinstance(b_list, list):
            flat_bullets.extend(b_list)
    if not flat_bullets:
        flat_bullets = ["Managed team operations and delivered core organizational deliverables."]
        
    m_key = major_key_map.get(occ, "software_engineers")
    major_list = majors.get(m_key, ["Information Technology", "General Studies"])
    
    for i in range(N_PER_OCC):
        is_female = (i % 2 == 0)
        gender = "Female" if is_female else "Male"
        first_name = random.choice(females if is_female else males)
        last_name = f"Candidate_{len(records):04d}"
        full_name = f"{first_name} {last_name}"
        
        years_exp = random.randint(2, 16)
        num_skills = random.randint(3, len(occ_skills))
        selected_skills = random.sample(occ_skills, num_skills)
        skills_str = ", ".join(selected_skills)
        
        company = random.choice(comp_list)
        school = random.choice(bachelor_schools)
        major = random.choice(major_list)
        bullets = random.sample(flat_bullets, min(3, len(flat_bullets)))
        bullets_str = "\n".join([f"- {b}" for b in bullets])
        
        pronoun_sub = "She" if is_female else "He"
        
        resume_text = (
            f"{full_name}\n"
            f"Target Occupation: {occ_title}\n"
            f"Email: {first_name.lower()}.{last_name.lower()}@domain.com | Phone: (555) 345-{len(records):04d}\n\n"
            f"Professional Summary:\n"
            f"{full_name} is an experienced {occ_title} with {years_exp} years of verified industry experience. "
            f"{pronoun_sub} has consistently driven strategic impact across diverse organizational initiatives.\n\n"
            f"Core Competencies:\n"
            f"{skills_str}\n\n"
            f"Professional Experience:\n"
            f"{occ_title} at {company} (2020 - Present)\n"
            f"{bullets_str}\n\n"
            f"Education:\n"
            f"Bachelor of Science in {major}, {school}\n"
        )
        
        merit_score = (years_exp * 0.05) + (num_skills / len(occ_skills) * 0.5)
        true_qualified = int(merit_score >= 0.55)
        
        # Historical selection disparity (simulating real-world path-specific effect)
        hire_prob = merit_score
        if occ in ["software_developers", "construction_laborers"] and gender == "Female":
            hire_prob -= 0.15
        elif occ in ["registered_nurses", "elementary_middle_school_teachers"] and gender == "Male":
            hire_prob -= 0.15
            
        hire_prob = float(np.clip(hire_prob, 0.1, 0.9))
        hired = int(random.random() < hire_prob)
        
        records.append({
            "candidate_id": f"POP_{len(records):05d}",
            "occupation": occ_title,
            "occupation_key": occ,
            "name": full_name,
            "first_name": first_name,
            "gender": gender,
            "years_exp": years_exp,
            "skills_count": num_skills,
            "true_qualified": true_qualified,
            "hired": hired,
            "resume_text": resume_text
        })

df_pop = pd.DataFrame(records)
df_pop.to_csv(pop_dir / "pop_resumes.csv", index=False)
print("Successfully generated PopResume benchmark! Shape:", df_pop.shape)
print("Occupations:\n", df_pop["occupation"].value_counts())
print("Gender breakdown:\n", df_pop["gender"].value_counts())
