import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_dispersion(df: pd.DataFrame, formula: str = "ClaimNb ~ 1", offset_col: str = "LogExposure") -> dict:
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        offset=df[offset_col]
    ).fit()
    
    mu_hat = model.fittedvalues
    pearson_chi2 = np.sum(((df["ClaimNb"] - mu_hat) ** 2) / mu_hat)
    phi = pearson_chi2 / model.df_resid
    return {
        "llf": model.llf,
        "aic": model.aic,
        "df_resid": model.df_resid,
        "pearson_chi2": pearson_chi2,
        "dispersion_phi": phi,
    }


def fit_severity_distributions(claims: pd.Series) -> pd.DataFrame:
    clean_claims = claims.dropna()
    clean_claims = clean_claims[clean_claims > 0]

    # Gamma
    g_shape, g_loc, g_scale = stats.gamma.fit(clean_claims, floc=0)
    g_ll = np.sum(stats.gamma.logpdf(clean_claims, g_shape, loc=0, scale=g_scale))
    g_aic = 4 - 2 * g_ll
    g_bic = np.log(len(clean_claims)) * 2 - 2 * g_ll
    g_ks, g_p = stats.kstest(clean_claims, 'gamma', args=(g_shape, 0, g_scale))

    # Log-Normal
    ln_s, ln_loc, ln_scale = stats.lognorm.fit(clean_claims, floc=0)
    ln_ll = np.sum(stats.lognorm.logpdf(clean_claims, ln_s, loc=0, scale=ln_scale))
    ln_aic = 4 - 2 * ln_ll
    ln_bic = np.log(len(clean_claims)) * 2 - 2 * ln_ll
    ln_ks, ln_p = stats.kstest(clean_claims, 'lognorm', args=(ln_s, 0, ln_scale))

    # Weibull
    wb_c, wb_loc, wb_scale = stats.weibull_min.fit(clean_claims, floc=0)
    wb_ll = np.sum(stats.weibull_min.logpdf(clean_claims, wb_c, loc=0, scale=wb_scale))
    wb_aic = 4 - 2 * wb_ll
    wb_bic = np.log(len(clean_claims)) * 2 - 2 * wb_ll
    wb_ks, wb_p = stats.kstest(clean_claims, 'weibull_min', args=(wb_c, 0, wb_scale))

    return pd.DataFrame({
        "Distribution": ["Gamma", "Log-Normal", "Weibull"],
        "Log-Likelihood": [g_ll, ln_ll, wb_ll],
        "AIC": [g_aic, ln_aic, wb_aic],
        "BIC": [g_bic, ln_bic, wb_bic],
        "KS-Stat": [g_ks, ln_ks, wb_ks],
        "p-value": [g_p, ln_p, wb_p],
    }).sort_values("AIC")


def compute_vif(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy().dropna()
    X_const = sm.add_constant(X)
    vifs = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    res = pd.DataFrame({"Feature": X_const.columns, "VIF": vifs})
    return res[res["Feature"] != "const"].sort_values("VIF", ascending=False).reset_index(drop=True)
