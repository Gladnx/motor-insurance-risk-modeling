import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from src.models.xgb_models import prepare_xgb_dataframe, FEATURE_COLS


class TreeSHAPExplainer:
    def __init__(self, booster, feature_cols: list[str] = None):
        self.feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS
        self.booster = booster
        self.explainer = shap.TreeExplainer(self.booster)
        self.shap_values = None
        self.explanation = None
        self.feature_data = None

    def compute_shap_values(
        self,
        df: pd.DataFrame,
        sample_size: int = 10000,
        random_state: int = 42,
    ) -> shap.Explanation:
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=random_state)
        else:
            sample_df = df.copy()

        self.feature_data = prepare_xgb_dataframe(sample_df, self.feature_cols)
        dmat = xgb.DMatrix(self.feature_data, enable_categorical=True)
        
        # Calculate raw SHAP values via TreeExplainer
        raw_shap = self.explainer.shap_values(dmat)
        self.shap_values = raw_shap
        
        # Build standard shap.Explanation object for plotting (beeswarm, waterfall, dependence)
        expected_val = self.explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) == 1:
            expected_val = float(expected_val[0])
            
        self.explanation = shap.Explanation(
            values=raw_shap,
            base_values=np.full(len(self.feature_data), expected_val),
            data=self.feature_data.values,
            feature_names=self.feature_cols,
        )
        return self.explanation

    def get_feature_importance_df(self) -> pd.DataFrame:
        if self.shap_values is None:
            raise ValueError("Must call compute_shap_values first.")
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        df_imp = pd.DataFrame({
            "Feature": self.feature_cols,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        df_imp["Relative_Importance_Pct"] = (df_imp["Mean_Abs_SHAP"] / df_imp["Mean_Abs_SHAP"].sum()) * 100
        return df_imp

