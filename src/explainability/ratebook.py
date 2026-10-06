import numpy as np
import pandas as pd


def extract_glm_ratebook(freq_glm_results) -> pd.DataFrame:
    params = freq_glm_results.params
    conf = freq_glm_results.conf_int()
    pvals = freq_glm_results.pvalues
    
    ratebook = pd.DataFrame({
        "Variable_Term": params.index,
        "GLM_Beta": params.values,
        "GLM_Relativity": np.exp(params.values),
        "CI_Lower": np.exp(conf[0].values),
        "CI_Upper": np.exp(conf[1].values),
        "p_value": pvals.values,
    })
    return ratebook


def build_binned_relativities(
    df: pd.DataFrame,
    pred_glm_col: str,
    pred_xgb_col: str,
    group_col: str,
    exposure_col: str = "Exposure_Capped",
) -> pd.DataFrame:
    base_glm = np.average(df[pred_glm_col], weights=df[exposure_col])
    base_xgb = np.average(df[pred_xgb_col], weights=df[exposure_col])
    
    summary = []
    for val, grp in df.groupby(group_col, observed=True):
        exp = grp[exposure_col].sum()
        mean_glm = np.average(grp[pred_glm_col], weights=grp[exposure_col])
        mean_xgb = np.average(grp[pred_xgb_col], weights=grp[exposure_col])
        
        summary.append({
            group_col: val,
            "Policies": len(grp),
            "Exposure": exp,
            "Exposure_Pct": (exp / df[exposure_col].sum()) * 100,
            "GLM_Pure_Premium": mean_glm,
            "GLM_Relativity": mean_glm / base_glm,
            "XGB_Pure_Premium": mean_xgb,
            "XGB_Relativity": mean_xgb / base_xgb,
        })
        
    return pd.DataFrame(summary)
