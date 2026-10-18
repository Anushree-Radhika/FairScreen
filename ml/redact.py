import functools
import json
from pathlib import Path
import re
from typing import List, Optional, Tuple

# Pre-compile static regexes for fast processing
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
URL_REGEX = re.compile(r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*")
TITLE_REGEX = re.compile(r"\b(Mr\.|Mrs\.|Ms\.|Miss|Mx\.|Dr\.)\b", flags=re.IGNORECASE)
PRONOUN_REGEX = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", flags=re.IGNORECASE)
ORGS_REGEX = re.compile(r"\b(fraternity|sorority|women's|womens|men's|mens|female|male|boy|girl)\b", flags=re.IGNORECASE)
CANDIDATE_TAG_REGEX = re.compile(r"\bCandidate_\d+\b", flags=re.IGNORECASE)

def load_names() -> Tuple[List[str], List[str]]:
    """Loads male and female names from data/name_pools.json or falls back to built-ins."""
    names_path = Path(__file__).resolve().parent.parent / "data" / "name_pools.json"
    if names_path.exists():
        with open(names_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("male_names", []), data.get("female_names", [])
    
    males = ["James", "John", "Robert", "Michael", "David", "William", "Richard", "Joseph"]
    females = ["Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan", "Jessica", "Sarah"]
    return males, females

def get_default_names() -> List[str]:
    males, females = load_names()
    return list(set(males + females))

DEFAULT_NAMES_TUPLE = tuple(get_default_names())

@functools.lru_cache(maxsize=16)
def _compile_names_regex(names_tuple: tuple) -> Optional[re.Pattern]:
    if not names_tuple:
        return None
    sorted_names = sorted(names_tuple, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(name) for name in sorted_names) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)

def redact(text: str, names: Optional[List[str]] = None) -> str:
    """
    Role 5 Contract:
    Redacts identity signals from resume text before feature extraction and scoring.
    """
    if not text:
        return ""
    
    redacted = EMAIL_REGEX.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_REGEX.sub("[REDACTED_PHONE]", redacted)
    redacted = URL_REGEX.sub("[REDACTED_LINK]", redacted)
    redacted = TITLE_REGEX.sub("[REDACTED_TITLE]", redacted)
    redacted = PRONOUN_REGEX.sub("[REDACTED_PRONOUN]", redacted)
    redacted = ORGS_REGEX.sub("[REDACTED_AFFILIATION]", redacted)
    
    names_tuple = tuple(names) if names else DEFAULT_NAMES_TUPLE
    names_regex = _compile_names_regex(names_tuple)
    if names_regex:
        redacted = names_regex.sub("[REDACTED_NAME]", redacted)
        
    redacted = CANDIDATE_TAG_REGEX.sub("[REDACTED_NAME]", redacted)
    return redacted
