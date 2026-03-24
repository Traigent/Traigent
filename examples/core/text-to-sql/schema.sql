-- Telecom Customer Database Schema
-- Used by the text-to-SQL example

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         VARCHAR(100),
    city         VARCHAR(100),
    plan_type    VARCHAR(50),   -- e.g. 'basic', 'premium', 'enterprise'
    status       VARCHAR(20)    -- 'active', 'suspended', 'churned'
);

CREATE TABLE subscriptions (
    sub_id        INTEGER PRIMARY KEY,
    customer_id   INTEGER REFERENCES customers(customer_id),
    plan_name     VARCHAR(50),
    start_date    DATE,
    end_date      DATE,
    monthly_rate  DECIMAL(10, 2)
);

CREATE TABLE billing (
    bill_id      INTEGER PRIMARY KEY,
    customer_id  INTEGER REFERENCES customers(customer_id),
    amount       DECIMAL(10, 2),
    due_date     DATE,
    paid_date    DATE,
    status       VARCHAR(20)    -- 'paid', 'unpaid', 'overdue'
);

CREATE TABLE network_usage (
    usage_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER REFERENCES customers(customer_id),
    data_gb       DECIMAL(8, 2),
    call_minutes  INTEGER,
    sms_count     INTEGER,
    month         VARCHAR(7)    -- 'YYYY-MM'
);
