-- Purpose: Customer and revenue breakdown by RFM segment.
-- avg_churn_probability excludes NULLs (Not Scored customers).

SELECT
    rfm_segment,
    COUNT(*)                                                        AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)            AS customer_pct,
    ROUND(SUM(total_revenue), 2)                                    AS total_revenue,
    ROUND(SUM(total_revenue) * 100.0
          / NULLIF(SUM(SUM(total_revenue)) OVER (), 0), 2)         AS revenue_pct,
    ROUND(AVG(total_revenue), 2)                                    AS avg_revenue_per_customer,
    ROUND(AVG(recency_days), 1)                                    AS avg_recency_days,
    ROUND(AVG(total_invoices), 2)                                  AS avg_total_invoices,
    ROUND(AVG(churn_probability), 4)                               AS avg_churn_probability,
    SUM(retention_target_flag)                                      AS retention_target_count,
    SUM(vip_flag)                                                   AS vip_count
FROM customer_360
GROUP BY rfm_segment
ORDER BY SUM(total_revenue) DESC;
