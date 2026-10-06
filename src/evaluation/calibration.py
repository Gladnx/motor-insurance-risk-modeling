import numpy as np
import pandas as pd


def compute_calibration_table(
    df: pd.DataFrame,
    pred_col: str,
    target_loss_col: str = "TotalClaimAmount",
    exposure_col: str = "Exposure_Capped",
    n_deciles: int = 10,
) -> pd.DataFrame:
    data = df.copy()
    data["decile"] = pd.qcut(data[pred_col], q=n_deciles, labels=False) + 1
    
    rows = []
    for d in range(1, n_deciles + 1):
        sub = data[data["decile"] == d]
        exp = sub[exposure_col].sum()
        loss = sub[target_loss_col].sum()
        obs = loss / exp if exp > 0 else 0.0
        pred = np.average(sub[pred_col], weights=sub[exposure_col])
        rows.append({
            "Decile": d,
            "Policies": len(sub),
            "Exposure": exp,
            "Observed_Pure_Premium": obs,
            "Predicted_Pure_Premium": pred,
            "Abs_Error": abs(obs - pred),
        })
    return pd.DataFrame(rows)


def compute_rmsce(calib_table: pd.DataFrame) -> float:
    # Root Mean Squared Calibration Error weighted by exposure
    weights = calib_table["Exposure"] / calib_table["Exposure"].sum()
    sq_err = (calib_table["Observed_Pure_Premium"] - calib_table["Predicted_Pure_Premium"]) ** 2
    return float(np.sqrt(np.sum(weights * sq_err)))


def compute_lorenz_curve(y_actual, y_pred, exposure=None) -> tuple[np.ndarray, np.ndarray]:
    y_act = np.asarray(y_actual, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    weights = np.asarray(exposure, dtype=float) if exposure is not None else np.ones(len(y_act))
    
    order = np.argsort(y_p)
    sorted_act = y_act[order]
    sorted_w = weights[order]
    
    cum_act = np.insert(np.cumsum(sorted_act * sorted_w), 0, 0.0)
    cum_w = np.insert(np.cumsum(sorted_w), 0, 0.0)
    
    cum_act_norm = cum_act / cum_act[-1]
    cum_w_norm = cum_w / cum_w[-1]
    return cum_w_norm, cum_act_norm
