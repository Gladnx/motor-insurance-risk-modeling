# Motor Insurance Frequency-Severity Pricing (freMTPL2)

Built to price French motor insurance properly from raw data to ratebook, using freMTPL2. The point was to do it the way it is done in practice: clean portfolio data, fit GLM baselines, then see if XGBoost actually adds anything on holdout, and finish with something explainable.

Scope: main models cover attritional claims only. Claims over 100k (about 14.7M total) are excluded from fitting and need a separate loading. That keeps the base rates stable.

## Results

Tested on policies the models had not seen. XGBoost beat the GLM in both places that matter.

Frequency: 31.08% vs 27.92% Gini. It is better at ranking, so high risk policies actually end up with higher predicted frequency.

Pure premium: 33.96% vs 31.36% Gini. Same story on the final price, plus the decile check showed its predicted cost stayed closer to what was actually paid out.

Both models are scaled to the observed total first, so this is about who prices better, not who guessed the overall total closer.

## Modelling Pipeline - Notebook Execution Order

Run 01 to 06 in order:

1. 01_data_ingestion_and_pipeline - fetch data, clean it, make the main table
2. 02_statistical_analysis_and_hypothesis_testing - some EDA and distribution checks
3. 03_actuarial_baseline_glms - baseline GLMs
4. 04_xgboost_frequency_severity - xgboost models
5. 05_model_comparison_actuarial_metrics - GLM vs xgboost comparison
6. 06_shap_explainability_ratebook - SHAP and ratebook

## Environment Setup (uv)

Using uv:

```
uv venv
.venv\Scripts\activate
uv pip install -r requirements.txt
jupyter notebook
```

Config is in config/config.yaml.

## Repository Structure

```
config/ - settings
notebooks/ - the analysis
data/ - ignored, gets created when you run 01
models/ - ignored, saved models
```