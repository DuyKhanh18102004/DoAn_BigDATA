"""HDFS Configuration.

Defines paths and connection settings for HDFS operations.
"""


class HDFSConfig:
    """HDFS configuration constants."""

    NAMENODE_HOST = "namenode"
    NAMENODE_PORT = 8020
    NAMENODE_URL = f"hdfs://{NAMENODE_HOST}:{NAMENODE_PORT}"

    BASE_PATH = "/user/data"

    RAW_PATH = f"{BASE_PATH}/raw"
    RAW_TRAIN_REAL = f"{RAW_PATH}/train/REAL"
    RAW_TRAIN_FAKE = f"{RAW_PATH}/train/FAKE"
    RAW_TEST_REAL = f"{RAW_PATH}/test/REAL"
    RAW_TEST_FAKE = f"{RAW_PATH}/test/FAKE"

    FEATURES_PATH = f"{BASE_PATH}/features"
    FEATURES_TRAIN_REAL = f"{FEATURES_PATH}/train/REAL"
    FEATURES_TRAIN_FAKE = f"{FEATURES_PATH}/train/FAKE"
    FEATURES_TEST_REAL = f"{FEATURES_PATH}/test/REAL"
    FEATURES_TEST_FAKE = f"{FEATURES_PATH}/test/FAKE"

    MODELS_PATH = "/user/models"

    RESULTS_PATH = f"{BASE_PATH}/results"
    PREDICTIONS_PATH = f"{RESULTS_PATH}/predictions"
    METRICS_PATH = f"{RESULTS_PATH}/metrics"

    @classmethod
    def get_full_path(cls, path):
        """Get full HDFS URL.
        
        Args:
            path: HDFS path
            
        Returns:
            str: Full HDFS URL
        """
        return f"{cls.NAMENODE_URL}{path}"

