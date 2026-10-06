import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


class FrequencyPoissonGLM:
    def __init__(self, formula: str = None):
        if formula is None:
            self.formula = (
                "ClaimNb ~ bs(DrivAge, df=4) + bs(VehAge, df=3) + VehPower + "
                "BonusMalus + LogDensity + C(Area) + C(VehGas) + C(VehBrand) + C(Region)"
            )
        else:
            self.formula = formula
        self.model = None
        self.results = None

    def fit(self, df: pd.DataFrame, offset_col: str = "LogExposure"):
        self.model = smf.glm(
            formula=self.formula,
            data=df,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=df[offset_col]
        )
        self.results = self.model.fit()
        return self

    def predict(self, df: pd.DataFrame, offset_col: str = None) -> np.ndarray:
        if offset_col is not None:
            return self.results.predict(df, offset=df[offset_col])
        # Return annualized frequency (exposure = 1.0, offset = 0.0)
        zero_offset = np.zeros(len(df))
        return self.results.predict(df, offset=zero_offset)


class SeverityGammaGLM:
    def __init__(self, formula: str = None, large_loss_threshold: float = 100000.0):
        if formula is None:
            self.formula = (
                "AvgClaimSeverity ~ DrivAge + VehAge + VehPower + "
                "BonusMalus + LogDensity + C(Area) + C(VehGas)"
            )
        else:
            self.formula = formula
        self.large_loss_threshold = large_loss_threshold
        self.model = None
        self.results = None

    def fit(self, df: pd.DataFrame):
        claims_df = df[
            (df["HasClaim"] == 1) & 
            (df["AvgClaimSeverity"] > 0) & 
            (df["AvgClaimSeverity"] < self.large_loss_threshold)
        ].copy()
        
        weights = claims_df["SevRecordCount"].clip(lower=1)
        self.model = smf.glm(
            formula=self.formula,
            data=claims_df,
            family=sm.families.Gamma(link=sm.families.links.Log()),
            var_weights=weights
        )
        self.results = self.model.fit()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.results.predict(df)


class TweedieGLM:
    def __init__(self, formula: str = None, var_power: float = 1.5):
        if formula is None:
            self.formula = (
                "PurePremium ~ DrivAge + VehAge + VehPower + "
                "BonusMalus + LogDensity + C(Area) + C(VehGas)"
            )
        else:
            self.formula = formula
        self.var_power = var_power
        self.model = None
        self.results = None

    def fit(self, df: pd.DataFrame, exposure_col: str = "Exposure_Capped"):
        weights = df[exposure_col].clip(lower=0.001)
        self.model = smf.glm(
            formula=self.formula,
            data=df,
            family=sm.families.Tweedie(
                link=sm.families.links.Log(),
                var_power=self.var_power
            ),
            var_weights=weights
        )
        self.results = self.model.fit()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.results.predict(df)


class TwoPartGLM:
    def __init__(self, freq_glm: FrequencyPoissonGLM = None, sev_glm: SeverityGammaGLM = None):
        self.freq_glm = freq_glm if freq_glm is not None else FrequencyPoissonGLM()
        self.sev_glm = sev_glm if sev_glm is not None else SeverityGammaGLM()

    def fit(self, df_train: pd.DataFrame):
        self.freq_glm.fit(df_train)
        self.sev_glm.fit(df_train)
        return self

    def predict_pure_premium(self, df: pd.DataFrame) -> np.ndarray:
        # Annualized frequency (claims/year) * expected severity (€/claim) = pure premium (€/year)
        pred_freq = self.freq_glm.predict(df, offset_col=None)
        pred_sev = self.sev_glm.predict(df)
        return pred_freq * pred_sev
