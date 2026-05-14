-- Purpose: Revenue and customer breakdown by customer_value_tier.
-- Tiers: Top 10% Value, High Value, Mid Value, Low Value.
-- avg_churn_probability excludes NULLs (Not Scored customers).

SELECT
    customer_value_tier,
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
GROUP BY customer_value_tier
ORDER BY
    CASE customer_value_tier
        WHEN 'Top 10% Value' THEN 1
        WHEN 'High Value'    THEN 2
        WHEN 'Mid Value'     THEN 3
        WHEN 'Low Value'     THEN 4
        ELSE 5
    END;
