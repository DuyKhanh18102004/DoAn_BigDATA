#!/bin/bash

export SPARK_HOME=/spark

. "/spark/sbin/spark-config.sh"

. "/spark/bin/load-spark-env.sh"

SPARK_HS_LOG_DIR=$SPARK_HOME/spark-hs-logs

mkdir -p $SPARK_HS_LOG_DIR

LOG=$SPARK_HS_LOG_DIR/spark-hs.out

ln -sf /dev/stdout $LOG

# Custom configuration for HDFS log directory
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=hdfs://namenode:8020/spark-logs -Dspark.history.ui.port=18080"

cd /spark/bin && /spark/sbin/../bin/spark-class org.apache.spark.deploy.history.HistoryServer >> $LOG
