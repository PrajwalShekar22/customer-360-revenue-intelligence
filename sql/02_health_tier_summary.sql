-- Purpose: Customer and revenue breakdown by health tier.
-- avg_churn_probability excludes NULL values (Not Scored customers).

SELECT
    health_tier,
    COUNT(*)                                                        AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)            AS customer_pct,
    ROUND(SUM(total_revenue), 2)                                    AS total_revenue,
    ROUND(SUM(total_revenue) * 100.0
          / NULLIF(SUM(SUM(total_revenue)) OVER (), 0), 2)         AS revenue_pct,
    ROUND(AVG(total_revenue), 2)                                    AS avg_revenue_per_customer,
    ROUND(AVG(customer_health_score), 2)                           AS avg_health_score,
    ROUND(AVG(churn_probability), 4)                               AS avg_churn_probability,
    SUM(retention_target_flag)                                      AS retention_target_count,
    SUM(vip_flag)                                                   AS vip_count
FROM customer_360
GROUP BY health_tier
ORDER BY
    CASE health_tier
        WHEN 'Excellent' THEN 1
        WHEN 'Healthy'   THEN 2
        WHEN 'Watchlist' THEN 3
        WHEN 'At Risk'   THEN 4
        WHEN 'Critical'  THEN 5
        ELSE 6
    END;
