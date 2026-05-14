-- Purpose: Top 50 customers ranked by total revenue.
-- Includes full Customer 360 profile for each customer.

SELECT
    customer_id,
    country_mode                                                    AS country,
    ROUND(total_revenue, 2)                                        AS total_revenue,
    total_invoices,
    recency_days,
    rfm_segment,
    health_tier,
    ROUND(customer_health_score, 2)                                AS customer_health_score,
    ROUND(churn_probability, 4)                                    AS churn_probability,
    churn_risk_tier,
    action_priority,
    customer_value_tier,
    retention_target_flag,
    vip_flag
FROM customer_360
ORDER BY total_revenue DESC
LIMIT 50;
