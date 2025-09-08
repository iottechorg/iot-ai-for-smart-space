# startup_emqx.sh - Updated startup script for EMQX

#!/bin/bash

echo "🏙️ Starting Smart City IoT System with EMQX..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed."
    exit 1
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p config/telegraf config/emqx clickhouse
mkdir -p ml_service/app/config dashboard/app

# Create EMQX config if it doesn't exist
if [ ! -f config/emqx/emqx.conf ]; then
    echo "📝 Creating EMQX configuration..."
    # Config is provided above - copy it to config/emqx/emqx.conf
fi

# Create ClickHouse init script if it doesn't exist
if [ ! -f clickhouse/init.sql ]; then
    echo "📝 Creating ClickHouse initialization script..."
    cat > clickhouse/init.sql << 'EOF'
CREATE DATABASE IF NOT EXISTS smartcity;

USE smartcity;

CREATE TABLE IF NOT EXISTS device_data (
    timestamp DateTime64(3) DEFAULT now(),
    device_type String,
    device_id String,
    metric_name String,
    metric_value Float64,
    metadata String DEFAULT ''
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (device_type, device_id, metric_name, timestamp)
TTL timestamp + INTERVAL 1 YEAR;

CREATE TABLE IF NOT EXISTS predictions (
    timestamp DateTime64(3) DEFAULT now(),
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
EOF
fi

# Start services in the correct order
echo "🚀 Starting services..."

# Start core infrastructure first
echo "1️⃣ Starting core infrastructure..."
docker-compose up -d emqx clickhouse minio

# Wait for core services to be healthy
echo "⏳ Waiting for core services to be ready..."
sleep 45  # EMQX takes a bit longer to start than Mosquitto

# Initialize MinIO buckets
echo "2️⃣ Initializing storage..."
docker-compose up minio-init

# Start data pipeline
echo "3️⃣ Starting data pipeline..."
docker-compose up -d telegraf

# Start application services
echo "4️⃣ Starting application services..."
docker-compose up -d gateway ml-service dashboard

# Show status
echo "📊 Checking service status..."
sleep 15
docker-compose ps

echo ""
echo "✅ Smart City IoT System with EMQX started!"
echo ""
echo "🌐 Access points:"
echo "   Dashboard: http://localhost:8501"
echo "   ML Service API: http://localhost:8000/docs"
echo "   ClickHouse: http://localhost:8123"
echo "   MinIO Console: http://localhost:9091 (minioadmin/minioadmin123)"
echo "   EMQX Dashboard: http://localhost:18086 (admin/smartcity123)"
echo ""
echo "📡 MQTT Access:"
echo "   MQTT Port: 11883"
echo "   WebSocket: 18084"
echo "   SSL MQTT: 18883 (if configured)"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose logs -f [service-name]"
echo "   Stop system: docker-compose down"
echo "   Restart service: docker-compose restart [service-name]"
echo "   Check health: docker-compose ps"
echo "   EMQX CLI: docker-compose exec emqx emqx ctl status"
echo ""
echo "🔍 Monitor startup with: docker-compose logs -f"