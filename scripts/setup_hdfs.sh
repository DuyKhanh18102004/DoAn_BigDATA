#!/bin/bash
# Setup HDFS Directories
# Initialize HDFS directory structure

echo "🔧 Setting up HDFS directories..."

docker exec namenode hdfs dfs -mkdir -p /user/data/raw/train/REAL
docker exec namenode hdfs dfs -mkdir -p /user/data/raw/train/FAKE
docker exec namenode hdfs dfs -mkdir -p /user/data/raw/test/REAL
docker exec namenode hdfs dfs -mkdir -p /user/data/raw/test/FAKE

docker exec namenode hdfs dfs -mkdir -p /user/data/features
docker exec namenode hdfs dfs -mkdir -p /user/data/results
docker exec namenode hdfs dfs -mkdir -p /user/models
docker exec namenode hdfs dfs -mkdir -p /spark-logs

echo "🔐 Setting permissions..."
docker exec namenode hdfs dfs -chmod -R 777 /user
docker exec namenode hdfs dfs -chmod -R 777 /spark-logs

echo "✅ HDFS directories created successfully"
echo ""
echo "📊 Directory structure:"
docker exec namenode hdfs dfs -ls -R /user
