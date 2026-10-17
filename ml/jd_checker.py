import json
import re
from pathlib import Path
from typing import Any, Dict, List

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gendered_words.json"

NEUTRAL_REPLACEMENTS = {
    "competitive": "high-standard / ambitious",
    "lead": "guide / coordinate",
    "dominant": "prominent / established",
    "aggressive": "proactive / focused",
    "assertive": "clear / decisive",
    "support": "partner with / assist",
    "collaborate": "work cross-functionally",
    "interpersonal": "communication"
}

def load_gendered_words() -> Dict[str, List[str]]:
    """Loads gender-coded words from data/gendered_words.json."""
    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"masculine_coded": [], "feminine_coded": []}

def check_jd(text: str) -> Dict[str, Any]:
    """
    Role 5 Contract:
    Analyzes job description text for gender-coded language (Gaucher et al. 2011).
    Returns word counts, flagged terms, and suggested neutral alternatives.
    """
    word_dict = load_gendered_words()
    masculine_stems = word_dict.get("masculine_coded", [])
    feminine_stems = word_dict.get("feminine_coded", [])
    
    clean_words = re.findall(r"\b[a-zA-Z\-]+\b", text.lower())
    
    masculine_found = []
    feminine_found = []
    
    for word in clean_words:
        for stem in masculine_stems:
            if word.startswith(stem):
                masculine_found.append(word)
                break
        for stem in feminine_stems:
            if word.startswith(stem):
                feminine_found.append(word)
                break
                
    masc_count = len(masculine_found)
    fem_count = len(feminine_found)
    
    if masc_count > fem_count + 2:
        bias_tone = "Strongly Masculine-Coded (may deter qualified female applicants)"
    elif fem_count > masc_count + 2:
        bias_tone = "Strongly Feminine-Coded (may deter qualified male applicants)"
    else:
        bias_tone = "Balanced / Neutral Language"
        
    suggestions = {}
    for word in set(masculine_found + feminine_found):
        if word in NEUTRAL_REPLACEMENTS:
            suggestions[word] = NEUTRAL_REPLACEMENTS[word]
            
    return {
        "masculine_count": masc_count,
        "feminine_count": fem_count,
        "masculine_words": list(set(masculine_found)),
        "feminine_words": list(set(feminine_found)),
        "bias_tone": bias_tone,
        "suggestions": suggestions
    }
