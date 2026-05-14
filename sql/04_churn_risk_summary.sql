-- Purpose: Customer and revenue breakdown by churn risk tier.
-- Includes Not Scored as a distinct tier. avg_churn_probability is NULL-safe.

SELECT
    churn_risk_tier,
    COUNT(*)                                                        AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)            AS customer_pct,
    ROUND(SUM(total_revenue), 2)                                    AS total_revenue,
    ROUND(SUM(total_revenue) * 100.0
          / NULLIF(SUM(SUM(total_revenue)) OVER (), 0), 2)         AS revenue_pct,
    ROUND(AVG(churn_probability), 4)                               AS avg_churn_probability,
    ROUND(AVG(customer_health_score), 2)                           AS avg_health_score,
    SUM(retention_target_flag)                                      AS retention_target_count,
    SUM(vip_flag)                                                   AS vip_count
FROM customer_360
GROUP BY churn_risk_tier
ORDER BY
    CASE churn_risk_tier
        WHEN 'Critical Risk' THEN 1
        WHEN 'High Risk'     THEN 2
        WHEN 'Medium Risk'   THEN 3
        WHEN 'Low Risk'      THEN 4
        WHEN 'Not Scored'    THEN 5
        ELSE 6
    END;
