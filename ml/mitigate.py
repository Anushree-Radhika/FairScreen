from typing import Any, Dict, Tuple
import numpy as np
from ml.model import build_and_train

def compute_reweighing_weights(
    y: np.ndarray,
    sensitive_group: np.ndarray,
    unprivileged_value: str = "Female",
    privileged_value: str = "Male"
) -> np.ndarray:
    """
    Role 4 Contract:
    Computes AIF360-standard Sample Reweighing weights to remove historical training bias.
    Formula: W(S=s, Y=y) = (P(S=s) * P(Y=y)) / P(S=s, Y=y)
    
    This adjusts instance weights during training so that the model cannot achieve high
    loss minimization simply by favoring the privileged demographic group.
    """
    y = np.asarray(y)
    sensitive_group = np.asarray(sensitive_group)
    n = len(y)
    
    weights = np.ones(n, dtype=float)
    
    groups = [unprivileged_value, privileged_value]
    labels = [0, 1]
    
    for s in groups:
        for y_val in labels:
            mask_s = (sensitive_group == s)
            mask_y = (y == y_val)
            mask_joint = mask_s & mask_y
            
            p_s = np.mean(mask_s)
            p_y = np.mean(mask_y)
            p_joint = np.mean(mask_joint)
            
            if p_joint > 0:
                target_weight = (p_s * p_y) / p_joint
                weights[mask_joint] = target_weight
                
    return weights

def train_mitigated_model(
    texts: list,
    y: np.ndarray,
    sensitive_group: np.ndarray,
    unprivileged_value: str = "Female",
    privileged_value: str = "Male",
    model_type: str = "logistic"
) -> Tuple[Any, Any, Dict[str, float]]:
    """
    Trains a debiased screener model using sample reweighing.
    """
    weights = compute_reweighing_weights(
        y=y,
        sensitive_group=sensitive_group,
        unprivileged_value=unprivileged_value,
        privileged_value=privileged_value
    )
    return build_and_train(
        texts=texts,
        y=y,
        weights=weights,
        model_type=model_type
    )
