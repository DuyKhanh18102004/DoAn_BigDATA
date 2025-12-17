"""
Spark Configuration
Spark session settings, resource allocation
"""

class SparkConfig:
    """Spark configuration settings"""
    
    # Spark Master
    MASTER_URL = "spark://spark-master:7077"
    APP_NAME = "DeepfakeDetection"
    
    # Resource allocation
    EXECUTOR_MEMORY = "4g"
    DRIVER_MEMORY = "4g"
    EXECUTOR_CORES = 2
    EXECUTOR_INSTANCES = 2
    
    # Performance tuning
    DEFAULT_PARALLELISM = 200
    SQL_SHUFFLE_PARTITIONS = 200
    
    # Memory management
    MEMORY_FRACTION = 0.6
    MEMORY_STORAGE_FRACTION = 0.5
    SHUFFLE_SPILL = True
    
    # Off-heap memory
    OFFHEAP_ENABLED = True
    OFFHEAP_SIZE = "2g"
    
    # Serialization
    SERIALIZER = "org.apache.spark.serializer.KryoSerializer"
    
    # Event logging
    EVENT_LOG_ENABLED = True
    EVENT_LOG_DIR = "hdfs://namenode:8020/spark-logs"
    
    @classmethod
    def get_spark_conf(cls):
        """Get Spark configuration dictionary"""
        return {
            "spark.master": cls.MASTER_URL,
            "spark.app.name": cls.APP_NAME,
            "spark.executor.memory": cls.EXECUTOR_MEMORY,
            "spark.driver.memory": cls.DRIVER_MEMORY,
            "spark.executor.cores": cls.EXECUTOR_CORES,
            "spark.executor.instances": cls.EXECUTOR_INSTANCES,
            "spark.default.parallelism": cls.DEFAULT_PARALLELISM,
            "spark.sql.shuffle.partitions": cls.SQL_SHUFFLE_PARTITIONS,
            "spark.memory.fraction": cls.MEMORY_FRACTION,
            "spark.memory.storageFraction": cls.MEMORY_STORAGE_FRACTION,
            "spark.shuffle.spill": cls.SHUFFLE_SPILL,
            "spark.memory.offHeap.enabled": cls.OFFHEAP_ENABLED,
            "spark.memory.offHeap.size": cls.OFFHEAP_SIZE,
            "spark.serializer": cls.SERIALIZER,
            "spark.eventLog.enabled": cls.EVENT_LOG_ENABLED,
            "spark.eventLog.dir": cls.EVENT_LOG_DIR,
        }
