-- Purpose: High-level executive KPI snapshot from the Customer 360 table.
-- One row output. All revenue in GBP (£).
-- Portable to BigQuery with minor dialect adjustments (APPROX_QUANTILES for median).

SELECT
    COUNT(*)                                                        AS total_customers,
    ROUND(SUM(total_revenue), 2)                                    AS total_revenue,
    ROUND(AVG(total_revenue), 2)                                    AS avg_revenue_per_customer,
    ROUND(MEDIAN(total_revenue), 2)                                 AS median_revenue_per_customer,
    SUM(total_invoices)                                             AS total_invoices,
    ROUND(AVG(total_invoices), 2)                                   AS avg_invoices_per_customer,
    SUM(retention_target_flag)                                      AS retention_targets,
    SUM(vip_flag)                                                   AS vip_customers,
    SUM(CASE WHEN model_scored = 1 THEN 1 ELSE 0 END)              AS model_scored_customers,
    SUM(CASE WHEN model_scored = 0 THEN 1 ELSE 0 END)              AS not_scored_customers,
    SUM(CASE WHEN health_tier IN ('Excellent', 'Healthy')
             THEN 1 ELSE 0 END)                                     AS excellent_healthy_customers,
    ROUND(SUM(CASE WHEN health_tier IN ('Excellent', 'Healthy')
                   THEN total_revenue ELSE 0 END), 2)              AS excellent_healthy_revenue,
    ROUND(
        SUM(CASE WHEN health_tier IN ('Excellent', 'Healthy')
                 THEN total_revenue ELSE 0 END)
        / NULLIF(SUM(total_revenue), 0) * 100, 2
    )                                                               AS excellent_healthy_revenue_pct,
    SUM(CASE WHEN churn_risk_tier IN ('Critical Risk', 'High Risk')
             THEN 1 ELSE 0 END)                                     AS critical_high_risk_customers,
    ROUND(SUM(CASE WHEN churn_risk_tier IN ('Critical Risk', 'High Risk')
                   THEN total_revenue ELSE 0 END), 2)              AS critical_high_risk_revenue
FROM customer_360;
