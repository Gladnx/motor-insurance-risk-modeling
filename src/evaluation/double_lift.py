import numpy as np
import pandas as pd


def compute_double_lift(
    df: pd.DataFrame,
    pred_challenger_col: str,
    pred_baseline_col: str,
    target_loss_col: str = "TotalClaimAmount",
    exposure_col: str = "Exposure_Capped",
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Computes the Actuarial Double Lift Table.
    Sorts policies by the ratio (Challenger / Baseline) and evaluates
    the empirical observed losses vs baseline and challenger predictions in each bucket.
    """
    data = df.copy()
    data["ratio"] = data[pred_challenger_col] / np.clip(data[pred_baseline_col], 1e-6, None)
    
    # Sort and bucket into quantiles by exposure
    data["bucket"] = pd.qcut(data["ratio"], q=n_bins, labels=False) + 1
    
    summary = []
    for b in range(1, n_bins + 1):
        subset = data[data["bucket"] == b]
        exp_sum = subset[exposure_col].sum()
        loss_sum = subset[target_loss_col].sum()
        
        obs_pure_prem = loss_sum / exp_sum if exp_sum > 0 else 0.0
        challenger_pure_prem = np.average(subset[pred_challenger_col], weights=subset[exposure_col])
        baseline_pure_prem = np.average(subset[pred_baseline_col], weights=subset[exposure_col])
        avg_ratio = subset["ratio"].mean()
        
        summary.append({
            "Bucket": b,
            "Policies": len(subset),
            "Exposure": exp_sum,
            "Avg_Ratio_XGB_over_GLM": avg_ratio,
            "Observed_Pure_Premium": obs_pure_prem,
            "Challenger_Pure_Premium": challenger_pure_prem,
            "Baseline_Pure_Premium": baseline_pure_prem,
            "Loss_Ratio_Challenger": obs_pure_prem / challenger_pure_prem if challenger_pure_prem > 0 else 0.0,
            "Loss_Ratio_Baseline": obs_pure_prem / baseline_pure_prem if baseline_pure_prem > 0 else 0.0,
        })
        
    return pd.DataFrame(summary)
