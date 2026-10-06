CREATE OR REPLACE TABLE abt_motor_portfolio AS
SELECT
    f.IDpol,
    f.ClaimNb,
    LEAST(f.ClaimNb, 4) AS ClaimNb_Capped,
    f.Exposure,
    LEAST(f.Exposure, 1.0) AS Exposure_Capped,
    LN(GREATEST(LEAST(f.Exposure, 1.0), 0.0027397)) AS LogExposure,
    COALESCE(s.SevRecordCount, 0) AS SevRecordCount,
    COALESCE(s.TotalClaimAmount, 0.0) AS TotalClaimAmount,
    CASE 
        WHEN COALESCE(s.SevRecordCount, 0) > 0 THEN s.TotalClaimAmount / s.SevRecordCount
        ELSE 0.0 
    END AS AvgClaimSeverity,
    CASE 
        WHEN f.Exposure > 0 THEN COALESCE(s.TotalClaimAmount, 0.0) / LEAST(f.Exposure, 1.0)
        ELSE 0.0 
    END AS PurePremium,
    CASE WHEN f.ClaimNb > 0 OR COALESCE(s.TotalClaimAmount, 0.0) > 0 THEN 1 ELSE 0 END AS HasClaim,
    CASE WHEN COALESCE(s.TotalClaimAmount, 0.0) >= 100000.0 THEN 1 ELSE 0 END AS IsLargeLoss,
    f.Area,
    f.VehPower,
    f.VehAge,
    f.DrivAge,
    f.BonusMalus,
    f.VehBrand,
    f.VehGas,
    f.Density,
    LN(GREATEST(f.Density, 1)) AS LogDensity,
    f.Region
FROM raw_freq f
LEFT JOIN policy_severity_agg s
    ON f.IDpol = s.IDpol
WHERE f.Exposure > 0;
