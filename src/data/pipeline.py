import os
import logging
import yaml
import duckdb
import pandas as pd
from sklearn.datasets import fetch_openml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path="config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_directories(config: dict):
    os.makedirs(config["data"]["paths"]["raw_dir"], exist_ok=True)
    os.makedirs(config["data"]["paths"]["processed_dir"], exist_ok=True)
    os.makedirs(config["data"]["paths"]["database_dir"], exist_ok=True)


def download_fremtpl2_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_freq_path = config["data"]["paths"]["raw_freq_file"]
    raw_sev_path = config["data"]["paths"]["raw_sev_file"]

    if os.path.exists(raw_freq_path) and os.path.exists(raw_sev_path):
        logger.info("Loading cached raw datasets from %s", config["data"]["paths"]["raw_dir"])
        df_freq = pd.read_parquet(raw_freq_path)
        df_sev = pd.read_parquet(raw_sev_path)
        # Clean quotes if present in cached data
        df_freq["VehGas"] = df_freq["VehGas"].astype(str).str.replace("'", "").str.strip()
        return df_freq, df_sev

    logger.info("Fetching freMTPL2freq from OpenML (ID: %s)...", config["data"]["openml"]["freq_data_id"])
    freq_data = fetch_openml(data_id=config["data"]["openml"]["freq_data_id"], as_frame=True, parser="auto")
    df_freq = freq_data.frame

    logger.info("Fetching freMTPL2sev from OpenML (ID: %s)...", config["data"]["openml"]["sev_data_id"])
    sev_data = fetch_openml(data_id=config["data"]["openml"]["sev_data_id"], as_frame=True, parser="auto")
    df_sev = sev_data.frame

    df_freq["IDpol"] = df_freq["IDpol"].astype("int64")
    df_freq["ClaimNb"] = df_freq["ClaimNb"].astype("int32")
    df_freq["Exposure"] = df_freq["Exposure"].astype("float64")
    df_freq["VehPower"] = df_freq["VehPower"].astype("int32")
    df_freq["VehAge"] = df_freq["VehAge"].astype("int32")
    df_freq["DrivAge"] = df_freq["DrivAge"].astype("int32")
    df_freq["BonusMalus"] = df_freq["BonusMalus"].astype("int32")
    df_freq["Density"] = df_freq["Density"].astype("int32")

    for cat_col in ["Area", "VehBrand", "VehGas", "Region"]:
        df_freq[cat_col] = df_freq[cat_col].astype(str).str.replace("'", "").str.strip()

    df_sev["IDpol"] = df_sev["IDpol"].astype("int64")
    df_sev["ClaimAmount"] = df_sev["ClaimAmount"].astype("float64")

    df_freq.to_parquet(raw_freq_path, index=False)
    df_sev.to_parquet(raw_sev_path, index=False)
    logger.info("Cached raw parquet files to %s and %s", raw_freq_path, raw_sev_path)

    return df_freq, df_sev


def build_duckdb_abt(config: dict, df_freq: pd.DataFrame, df_sev: pd.DataFrame) -> pd.DataFrame:
    db_path = config["data"]["paths"]["duckdb_path"]
    logger.info("Connecting to DuckDB at %s", db_path)
    con = duckdb.connect(db_path)

    con.register("df_freq_view", df_freq)
    con.register("df_sev_view", df_sev)

    logger.info("Executing sql/01_create_schema.sql")
    with open("sql/01_create_schema.sql", "r") as f:
        con.execute(f.read())

    con.execute("INSERT INTO raw_freq SELECT * FROM df_freq_view")
    con.execute("INSERT INTO raw_sev SELECT * FROM df_sev_view")

    logger.info("Executing sql/02_aggregate_severity.sql")
    with open("sql/02_aggregate_severity.sql", "r") as f:
        con.execute(f.read())

    logger.info("Executing sql/03_build_abt.sql")
    with open("sql/03_build_abt.sql", "r") as f:
        con.execute(f.read())

    df_abt = con.execute("SELECT * FROM abt_motor_portfolio").df()
    abt_path = config["data"]["paths"]["abt_file"]
    df_abt.to_parquet(abt_path, index=False)
    logger.info("Saved Analytic Base Table to %s (%d rows)", abt_path, len(df_abt))

    con.close()
    return df_abt


def run_pipeline(config_path="config/config.yaml"):
    config = load_config(config_path)
    ensure_directories(config)
    df_freq, df_sev = download_fremtpl2_data(config)
    df_abt = build_duckdb_abt(config, df_freq, df_sev)
    return df_abt


if __name__ == "__main__":
    run_pipeline()
