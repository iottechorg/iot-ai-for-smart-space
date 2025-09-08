# clickhouse/init.sql
-- ClickHouse database initialization

CREATE DATABASE IF NOT EXISTS smartcity;

USE smartcity;

-- Device data table
CREATE TABLE IF NOT EXISTS device_data (
    timestamp DateTime64(3),
    device_type String,
    device_id String,
    metric_name String,
    metric_value Float64,
    metadata String DEFAULT ''
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (device_type, device_id, metric_name, timestamp)
TTL timestamp + INTERVAL 1 YEAR;

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    timestamp DateTime64(3),
    device_type String,
    device_id String,
    model_name String,
    prediction String,
    confidence Float64,
    metadata String DEFAULT ''
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (device_type, device_id, model_name, timestamp)
TTL timestamp + INTERVAL 1 YEAR;

-- Materialized view for latest device metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS latest_device_metrics
ENGINE = ReplacingMergeTree(timestamp)
ORDER BY (device_type, device_id, metric_name)
AS SELECT
    device_type,
    device_id,
    metric_name,
    metric_value,
    timestamp
FROM device_data;

-- Materialized view for hourly aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_device_metrics
ENGINE = SummingMergeTree()
ORDER BY (device_type, device_id, metric_name, hour)
AS SELECT
    device_type,
    device_id,
    metric_name,
    toStartOfHour(timestamp) as hour,
    avg(metric_value) as avg_value,
    min(metric_value) as min_value,
    max(metric_value) as max_value,
    count() as count
FROM device_data
GROUP BY device_type, device_id, metric_name, hour;

