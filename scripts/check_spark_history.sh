#!/bin/bash
# Check Spark History Server
# View Spark application history

echo "📊 Opening Spark History Server..."
echo ""
echo "🌐 Spark History Server: http://localhost:18080"
echo "🌐 Spark Master UI: http://localhost:8080"
echo "🌐 HDFS NameNode UI: http://localhost:9870"
echo ""
echo "📋 Recent Spark applications:"
curl -s http://localhost:18080/api/v1/applications | python -m json.tool

echo ""
echo "💡 Tip: Open http://localhost:18080 in your browser to view details"
