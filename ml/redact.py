import re
from typing import List, Optional
from ml.synth import load_names

def get_default_names() -> List[str]:
    males, females = load_names()
    return list(set(males + females))

def redact(text: str, names: Optional[List[str]] = None) -> str:
    """
    Role 5 Contract:
    Redacts identity signals from resume text before feature extraction and scoring.
    Removes:
    - Candidate first and last names
    - Emails, phone numbers, and URLs
    - Honorifics (Mr., Ms., Mrs., Dr.)
    - Gendered pronouns (he, she, him, her, his, hers)
    - Gender-identifying organizations or terms (fraternity, sorority, women's, men's)
    """
    if not text:
        return ""
    
    redacted = text
    
    # 1. Redact Emails
    redacted = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[REDACTED_EMAIL]",
        redacted
    )
    
    # 2. Redact Phone numbers
    redacted = re.sub(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "[REDACTED_PHONE]",
        redacted
    )
    
    # 3. Redact URLs / LinkedIn / GitHub
    redacted = re.sub(
        r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*",
        "[REDACTED_LINK]",
        redacted
    )
    
    # 4. Redact Honorifics & Titles
    redacted = re.sub(
        r"\b(Mr\.|Mrs\.|Ms\.|Miss|Mx\.|Dr\.)\b",
        "[REDACTED_TITLE]",
        redacted,
        flags=re.IGNORECASE
    )
    
    # 5. Redact Gendered Pronouns
    pronoun_pattern = r"\b(he|she|him|her|his|hers|himself|herself)\b"
    redacted = re.sub(pronoun_pattern, "[REDACTED_PRONOUN]", redacted, flags=re.IGNORECASE)
    
    # 6. Redact Gendered Affiliations
    gendered_orgs = r"\b(fraternity|sorority|women's|womens|men's|mens|female|male|boy|girl)\b"
    redacted = re.sub(gendered_orgs, "[REDACTED_AFFILIATION]", redacted, flags=re.IGNORECASE)
    
    # 7. Redact Names
    target_names = names if names else get_default_names()
    # Sort by length descending to match full names or multi-word names first
    sorted_names = sorted(target_names, key=len, reverse=True)
    if sorted_names:
        # Create regex pattern for all target names with word boundaries
        names_pattern = r"\b(" + "|".join(re.escape(name) for name in sorted_names) + r")\b"
        redacted = re.sub(names_pattern, "[REDACTED_NAME]", redacted, flags=re.IGNORECASE)
        
    # Redact Candidate_XXXX format if present from synthetic generator
    redacted = re.sub(r"\bCandidate_\d+\b", "[REDACTED_NAME]", redacted, flags=re.IGNORECASE)
    
    return redacted
