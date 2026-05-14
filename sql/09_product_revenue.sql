-- Purpose: Top 50 products by total revenue across all transactions.
-- Uses clean_transactions. Excludes blank descriptions.

SELECT
    stock_code,
    description,
    ROUND(SUM(revenue), 2)                                        AS total_revenue,
    SUM(quantity)                                                 AS total_quantity,
    COUNT(DISTINCT invoice_id)                                    AS total_invoices,
    COUNT(DISTINCT customer_id)                                   AS unique_customers,
    ROUND(AVG(unit_price), 4)                                     AS avg_unit_price
FROM clean_transactions
WHERE description IS NOT NULL
  AND TRIM(description) <> ''
GROUP BY stock_code, description
ORDER BY total_revenue DESC
LIMIT 50;
