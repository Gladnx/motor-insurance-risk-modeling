CREATE OR REPLACE TABLE policy_severity_agg AS
SELECT
    IDpol,
    COUNT(*) AS SevRecordCount,
    SUM(ClaimAmount) AS TotalClaimAmount,
    AVG(ClaimAmount) AS AvgClaimAmount,
    MAX(ClaimAmount) AS MaxClaimAmount
FROM raw_sev
WHERE ClaimAmount > 0
GROUP BY IDpol;
