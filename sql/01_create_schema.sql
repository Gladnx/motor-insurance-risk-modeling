CREATE OR REPLACE TABLE raw_freq (
    IDpol BIGINT PRIMARY KEY,
    ClaimNb INTEGER,
    Exposure DOUBLE,
    Area VARCHAR,
    VehPower INTEGER,
    VehAge INTEGER,
    DrivAge INTEGER,
    BonusMalus INTEGER,
    VehBrand VARCHAR,
    VehGas VARCHAR,
    Density INTEGER,
    Region VARCHAR
);

CREATE OR REPLACE TABLE raw_sev (
    IDpol BIGINT,
    ClaimAmount DOUBLE
);
