"""
Spark Utilities
Helper functions cho Spark session management
"""

from pyspark.sql import SparkSession
from ..config.spark_config import SparkConfig
import logging

logger = logging.getLogger(__name__)


class SparkUtils:
    """Spark utility functions"""
    
    @staticmethod
    def create_spark_session(app_name=None):
        """
        Create Spark session với configuration
        Args:
            app_name: Application name (optional)
        Returns:
            SparkSession
        """
        if app_name is None:
            app_name = SparkConfig.APP_NAME
        
        logger.info(f"Creating Spark session: {app_name}")
        
        builder = SparkSession.builder.appName(app_name)
        
        # Apply configurations
        for key, value in SparkConfig.get_spark_conf().items():
            builder = builder.config(key, value)
        
        spark = builder.getOrCreate()
        
        logger.info(f"Spark session created successfully")
        logger.info(f"Spark version: {spark.version}")
        logger.info(f"Spark master: {spark.sparkContext.master}")
        
        return spark
    
    @staticmethod
    def stop_spark_session(spark):
        """
        Stop Spark session
        Args:
            spark: SparkSession to stop
        """
        if spark:
            spark.stop()
            logger.info("Spark session stopped")
    
    @staticmethod
    def get_executor_info(spark):
        """
        Get executor information
        Args:
            spark: SparkSession
        Returns:
            Dict with executor info
        """
        sc = spark.sparkContext
        executors = sc._jsc.sc().getExecutorMemoryStatus()
        
        info = {
            'app_id': sc.applicationId,
            'app_name': sc.appName,
            'master': sc.master,
            'num_executors': len(executors),
            'default_parallelism': sc.defaultParallelism,
        }
        
        return info
