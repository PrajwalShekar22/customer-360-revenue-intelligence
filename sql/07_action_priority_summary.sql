-- Purpose: Business action planning summary by action_priority.
-- Sorted Priority 1 → Priority 5.
-- avg_churn_probability excludes NULLs (Not Scored customers).

SELECT
    action_priority,
    COUNT(*)                                                        AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)            AS customer_pct,
    ROUND(SUM(total_revenue), 2)                                    AS total_revenue,
    ROUND(SUM(total_revenue) * 100.0
          / NULLIF(SUM(SUM(total_revenue)) OVER (), 0), 2)         AS revenue_pct,
    ROUND(AVG(total_revenue), 2)                                    AS avg_revenue_per_customer,
    ROUND(AVG(churn_probability), 4)                               AS avg_churn_probability,
    SUM(retention_target_flag)                                      AS retention_target_count,
    SUM(vip_flag)                                                   AS vip_count
FROM customer_360
GROUP BY action_priority
ORDER BY
    CASE action_priority
        WHEN 'Priority 1' THEN 1
        WHEN 'Priority 2' THEN 2
        WHEN 'Priority 3' THEN 3
        WHEN 'Priority 4' THEN 4
        WHEN 'Priority 5' THEN 5
        ELSE 6
    END;
