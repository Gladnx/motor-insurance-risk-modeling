import numpy as np
import pandas as pd


def poisson_deviance(y_true, y_pred, exposure=None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-9, None)
    
    # Safely evaluate log only where y_true > 0
    safe_y_true = np.where(y_true > 0, y_true, 1.0)
    dev = np.where(
        y_true > 0,
        2 * (y_true * np.log(safe_y_true / y_pred) - (y_true - y_pred)),
        2 * y_pred
    )
    if exposure is not None:
        exposure = np.asarray(exposure, dtype=float)
        return float(np.sum(dev) / np.sum(exposure))
    return float(np.mean(dev))


def gamma_deviance(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-9, None)
    safe_y_true = np.clip(y_true, 1e-9, None)
    
    dev = 2 * ((y_true - y_pred) / y_pred - np.log(safe_y_true / y_pred))
    return float(np.mean(dev))


def tweedie_deviance(y_true, y_pred, p: float = 1.5, weights=None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-9, None)
    
    if p == 1.0:
        safe_y_true = np.where(y_true > 0, y_true, 1.0)
        dev = np.where(
            y_true > 0,
            2 * (y_true * np.log(safe_y_true / y_pred) - (y_true - y_pred)),
            2 * y_pred
        )
    elif p == 2.0:
        safe_y_true = np.clip(y_true, 1e-9, None)
        dev = 2 * (np.log(y_pred / safe_y_true) + y_true / y_pred - 1)
    else:
        term1 = (np.power(np.clip(y_true, 0, None), 2 - p)) / ((1 - p) * (2 - p))
        term2 = (y_true * np.power(y_pred, 1 - p)) / (1 - p)
        term3 = (np.power(y_pred, 2 - p)) / (2 - p)
        dev = 2 * (term1 - term2 + term3)
        
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        return float(np.sum(weights * dev) / np.sum(weights))
    return float(np.mean(dev))


def gini_coefficient(y_actual, y_pred, exposure=None) -> float:
    y_actual = np.asarray(y_actual, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_actual)
    
    if exposure is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(exposure, dtype=float)
        
    order = np.argsort(y_pred)
    sorted_actual = y_actual[order]
    sorted_weights = weights[order]
    
    cum_actual = np.cumsum(sorted_actual * sorted_weights)
    total_actual = cum_actual[-1]
    if total_actual == 0:
        return 0.0
    cum_actual_norm = cum_actual / total_actual
    
    cum_weights = np.cumsum(sorted_weights)
    cum_weights_norm = cum_weights / cum_weights[-1]
    
    lorenz_area = np.sum((cum_actual_norm[:-1] + cum_actual_norm[1:]) * np.diff(cum_weights_norm)) / 2.0
    gini = 1.0 - 2.0 * lorenz_area
    return float(np.abs(gini))
