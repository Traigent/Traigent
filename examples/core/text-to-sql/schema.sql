-- Telecom customer database schema (example for text-to-sql optimization)

CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    city          TEXT,
    status        TEXT DEFAULT 'active'
);

CREATE TABLE subscriptions (
    subscription_id  INTEGER PRIMARY KEY,
    customer_id      INTEGER REFERENCES customers(customer_id),
    plan_name        TEXT NOT NULL,
    monthly_rate     REAL,
    start_date       DATE
);

CREATE TABLE billing (
    billing_id    INTEGER PRIMARY KEY,
    customer_id   INTEGER REFERENCES customers(customer_id),
    amount        REAL NOT NULL,
    status        TEXT DEFAULT 'paid',
    due_date      DATE
);

CREATE TABLE network_usage (
    usage_id      INTEGER PRIMARY KEY,
    customer_id   INTEGER REFERENCES customers(customer_id),
    call_minutes  REAL DEFAULT 0,
    data_gb       REAL DEFAULT 0,
    record_date   DATE
);
