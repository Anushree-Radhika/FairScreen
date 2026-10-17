from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer(ngram_range=(1, 2), min_df=2, max_features=1200) -> TfidfVectorizer:
    """Creates a standard TF-IDF vectorizer for resume text representation."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z0-9_\-\[\]]+\b" # preserves [REDACTED_*] tokens
    )

def extract_features(
    texts: list,
    vectorizer: Optional[TfidfVectorizer] = None,
    fit: bool = False,
    structured_df: Optional[pd.DataFrame] = None
) -> Tuple[csr_matrix, TfidfVectorizer]:
    """
    Role 3 Contract:
    Extracts numerical feature representations from resume text.
    Combines TF-IDF tokens with structured merit features (years_exp, skills_match).
    
    CRITICAL PRIVACY RULE:
    Protected attributes (gender, race_eth, age) are strictly excluded from features.
    """
    if vectorizer is None:
        vectorizer = create_vectorizer()
        fit = True
        
    if fit:
        text_features = vectorizer.fit_transform(texts)
    else:
        text_features = vectorizer.transform(texts)
        
    if structured_df is not None and {"years_exp", "skills_match"}.issubset(structured_df.columns):
        # Normalize structured merit attributes
        exp_norm = (structured_df["years_exp"].values / 20.0).reshape(-1, 1)
        skills_norm = structured_df["skills_match"].values.reshape(-1, 1)
        structured_matrix = csr_matrix(np.hstack([exp_norm, skills_norm]))
        combined_features = hstack([text_features, structured_matrix]).tocsr()
        return combined_features, vectorizer

    return text_features, vectorizer
