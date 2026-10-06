import numpy as np
import pandas as pd
import xgboost as xgb


FEATURE_COLS = [
    "DrivAge",
    "VehAge",
    "VehPower",
    "BonusMalus",
    "LogDensity",
    "Area",
    "VehGas",
    "VehBrand",
    "Region",
]

CATEGORICAL_COLS = ["Area", "VehGas", "VehBrand", "Region"]


def prepare_xgb_dataframe(df: pd.DataFrame, feature_cols: list[str] = None) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    X = df[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


# Alias for backwards compatibility
prepare_xgb_features = prepare_xgb_dataframe


class FrequencyPoissonXGB:
    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 20.0,
        random_state: int = 42,
        enforce_monotonic_bm: bool = True,
    ):
        self.feature_cols = FEATURE_COLS
        self.n_estimators = n_estimators
        
        monotone_tuple = [0] * len(self.feature_cols)
        if enforce_monotonic_bm and "BonusMalus" in self.feature_cols:
            idx = self.feature_cols.index("BonusMalus")
            monotone_tuple[idx] = 1

        self.params = {
            "objective": "count:poisson",
            "eval_metric": "poisson-nloglik",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "seed": random_state,
            "tree_method": "hist",
            "monotone_constraints": tuple(monotone_tuple),
        }
        self.booster = None

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None, early_stopping_rounds: int = 25):
        X_tr = prepare_xgb_dataframe(df_train, self.feature_cols)
        dtrain = xgb.DMatrix(
            X_tr,
            label=df_train["ClaimNb"].values,
            base_margin=df_train["LogExposure"].values,
            enable_categorical=True,
        )

        evals = [(dtrain, "train")]
        if df_val is not None:
            X_val = prepare_xgb_dataframe(df_val, self.feature_cols)
            dval = xgb.DMatrix(
                X_val,
                label=df_val["ClaimNb"].values,
                base_margin=df_val["LogExposure"].values,
                enable_categorical=True,
            )
            evals.append((dval, "val"))
            self.booster = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )
        else:
            self.booster = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.n_estimators,
                evals=evals,
                verbose_eval=False,
            )
        return self

    def predict(self, df: pd.DataFrame, offset_col: str = None) -> np.ndarray:
        X = prepare_xgb_dataframe(df, self.feature_cols)
        if offset_col is not None:
            base_margin = df[offset_col].values
        else:
            # Annualized frequency (exposure = 1.0, offset = 0.0)
            base_margin = np.zeros(len(df))
            
        dmat = xgb.DMatrix(X, base_margin=base_margin, enable_categorical=True)
        return self.booster.predict(dmat)


class SeverityGammaXGB:
    def __init__(
        self,
        max_depth: int = 3,
        learning_rate: float = 0.03,
        n_estimators: int = 150,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 15.0,
        random_state: int = 42,
        large_loss_threshold: float = 100000.0,
    ):
        self.feature_cols = FEATURE_COLS
        self.n_estimators = n_estimators
        self.large_loss_threshold = large_loss_threshold
        self.params = {
            "objective": "reg:gamma",
            "eval_metric": "gamma-deviance",
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "seed": random_state,
            "tree_method": "hist",
        }
        self.booster = None

    def fit(self, df_train: pd.DataFrame):
        claims_df = df_train[
            (df_train["HasClaim"] == 1) &
            (df_train["AvgClaimSeverity"] > 0) &
            (df_train["AvgClaimSeverity"] < self.large_loss_threshold)
        ].copy()

        X_tr = prepare_xgb_dataframe(claims_df, self.feature_cols)
        weights = claims_df["SevRecordCount"].clip(lower=1).values
        dtrain = xgb.DMatrix(
            X_tr,
            label=claims_df["AvgClaimSeverity"].values,
            weight=weights,
            enable_categorical=True,
        )
        self.booster = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = prepare_xgb_dataframe(df, self.feature_cols)
        dmat = xgb.DMatrix(X, enable_categorical=True)
        return self.booster.predict(dmat)


class TweedieXGB:
    def __init__(
        self,
        tweedie_variance_power: float = 1.5,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        n_estimators: int = 250,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 20.0,
        random_state: int = 42,
    ):
        self.feature_cols = FEATURE_COLS
        self.n_estimators = n_estimators
        self.params = {
            "objective": "reg:tweedie",
            "tweedie_variance_power": tweedie_variance_power,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "seed": random_state,
            "tree_method": "hist",
        }
        self.booster = None

    def fit(self, df_train: pd.DataFrame):
        X_tr = prepare_xgb_dataframe(df_train, self.feature_cols)
        weights = df_train["Exposure_Capped"].values
        dtrain = xgb.DMatrix(
            X_tr,
            label=df_train["PurePremium"].values,
            weight=weights,
            enable_categorical=True,
        )
        self.booster = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators, verbose_eval=False)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = prepare_xgb_dataframe(df, self.feature_cols)
        dmat = xgb.DMatrix(X, enable_categorical=True)
        return self.booster.predict(dmat)


class TwoPartXGB:
    def __init__(self, freq_xgb: FrequencyPoissonXGB = None, sev_xgb: SeverityGammaXGB = None):
        self.freq_xgb = freq_xgb if freq_xgb is not None else FrequencyPoissonXGB()
        self.sev_xgb = sev_xgb if sev_xgb is not None else SeverityGammaXGB()

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None):
        self.freq_xgb.fit(df_train, df_val=df_val)
        self.sev_xgb.fit(df_train)
        return self

    def predict_pure_premium(self, df: pd.DataFrame) -> np.ndarray:
        pred_freq = self.freq_xgb.predict(df, offset_col=None)
        pred_sev = self.sev_xgb.predict(df)
        return pred_freq * pred_sev
