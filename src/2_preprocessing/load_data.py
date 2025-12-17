"""
Load Data from HDFS
Sử dụng Spark binaryFile format
"""

import logging
from pyspark.sql.functions import col, udf, regexp_extract
from pyspark.sql.types import StringType
from ..config.hdfs_config import HDFSConfig
from ..utils.spark_utils import SparkUtils
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def load_images_from_hdfs(spark, hdfs_path):
    """
    Load images from HDFS using Spark binaryFile format
    Args:
        spark: SparkSession
        hdfs_path: HDFS path to load from
    Returns:
        DataFrame with columns: [path, content, modificationTime, length]
    """
    logger.info(f"Loading images from HDFS: {hdfs_path}")
    
    df = spark.read.format("binaryFile") \
        .option("pathGlobFilter", "*.jpg") \
        .option("recursiveFileLookup", "true") \
        .load(HDFSConfig.get_full_path(hdfs_path))
    
    logger.info(f"Loaded {df.count()} images")
    logger.info(f"Schema: {df.schema}")
    
    return df


def extract_label_from_path(df):
    """
    Extract label from HDFS path
    Path format: hdfs://.../train/REAL/img_001.jpg -> label: REAL
    Args:
        df: DataFrame with 'path' column
    Returns:
        DataFrame with 'label' column added
    """
    logger.info("Extracting labels from paths...")
    
    # Extract label using regex
    df = df.withColumn(
        "label",
        regexp_extract(col("path"), r"/(REAL|FAKE)/", 1)
    )
    
    return df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load data from HDFS')
    parser.add_argument('--hdfs_path', required=True, help='HDFS path to load from')
    args = parser.parse_args()
    
    # Create Spark session
    spark = SparkUtils.create_spark_session("DataLoading")
    
    try:
        # Load images
        df = load_images_from_hdfs(spark, args.hdfs_path)
        
        # Extract labels
        df = extract_label_from_path(df)
        
        # Show sample
        logger.info("Sample data:")
        df.select("path", "label", "length").show(5, truncate=False)
        
        # Statistics
        logger.info("Label distribution:")
        df.groupBy("label").count().show()
        
    finally:
        SparkUtils.stop_spark_session(spark)


if __name__ == "__main__":
    main()
