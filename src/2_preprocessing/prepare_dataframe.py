"""
Prepare DataFrame
Create labeled DataFrame ready for feature extraction
"""

import logging
from pyspark.sql.functions import col
from .load_data import load_images_from_hdfs, extract_label_from_path
from .validate_images import validate_images
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def prepare_dataframe(spark, hdfs_path):
    """
    Prepare complete DataFrame with labels and validation
    Args:
        spark: SparkSession
        hdfs_path: HDFS path to load from
    Returns:
        DataFrame[path, content, label, is_valid]
    """
    logger.info("Preparing DataFrame...")
    
    # Load images
    df = load_images_from_hdfs(spark, hdfs_path)
    
    # Extract labels
    df = extract_label_from_path(df)
    
    # Validate images
    df = validate_images(df)
    
    # Filter only valid images
    df_valid = df.filter(col("is_valid") == True)
    
    logger.info(f"Prepared {df_valid.count()} valid images")
    
    return df_valid


def main():
    """Main execution"""
    # TODO: Implement standalone preparation
    pass


if __name__ == "__main__":
    main()
