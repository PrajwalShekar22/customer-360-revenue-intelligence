-- Purpose: Monthly revenue trend with active customers and order metrics.
-- Uses clean_transactions. invoice_yearmonth is YYYY-MM string (e.g. '2010-11').
-- avg_order_value = total revenue / distinct invoices in that month.

SELECT
    invoice_yearmonth                                              AS invoice_month,
    ROUND(SUM(revenue), 2)                                        AS total_revenue,
    COUNT(DISTINCT invoice_id)                                    AS total_invoices,
    COUNT(DISTINCT customer_id)                                   AS active_customers,
    SUM(quantity)                                                 AS total_quantity,
    ROUND(SUM(revenue) / NULLIF(COUNT(DISTINCT invoice_id), 0), 2) AS avg_order_value
FROM clean_transactions
GROUP BY invoice_yearmonth
ORDER BY invoice_yearmonth;
