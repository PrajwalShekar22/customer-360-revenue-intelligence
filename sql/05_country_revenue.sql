-- Purpose: Revenue and customer concentration by country (from customer_360).
-- country_mode is the most frequent transaction country per customer.
-- All monetary values in GBP (£).

SELECT
    country_mode                                                    AS country,
    COUNT(*)                                                        AS customer_count,
    ROUND(SUM(total_revenue), 2)                                    AS total_revenue,
    ROUND(SUM(total_revenue) * 100.0
          / NULLIF(SUM(SUM(total_revenue)) OVER (), 0), 2)         AS revenue_pct,
    ROUND(AVG(total_revenue), 2)                                    AS avg_revenue_per_customer,
    SUM(retention_target_flag)                                      AS retention_target_count,
    SUM(vip_flag)                                                   AS vip_count,
    ROUND(AVG(churn_probability), 4)                               AS avg_churn_probability
FROM customer_360
GROUP BY country_mode
ORDER BY total_revenue DESC;
