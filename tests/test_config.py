# Unit Tests
# Test suite cho các modules

import unittest
from src.config import HDFSConfig, SparkConfig, ModelConfig


class TestConfig(unittest.TestCase):
    """Test configuration modules"""
    
    def test_hdfs_config(self):
        """Test HDFS configuration"""
        self.assertEqual(HDFSConfig.NAMENODE_HOST, "namenode")
        self.assertEqual(HDFSConfig.NAMENODE_PORT, 8020)
        
        full_path = HDFSConfig.get_full_path("/test")
        self.assertTrue(full_path.startswith("hdfs://"))
    
    def test_spark_config(self):
        """Test Spark configuration"""
        conf = SparkConfig.get_spark_conf()
        self.assertIsInstance(conf, dict)
        self.assertIn("spark.master", conf)
    
    def test_model_config(self):
        """Test Model configuration"""
        self.assertEqual(ModelConfig.FEATURE_EXTRACTOR, "resnet50")
        self.assertEqual(ModelConfig.FEATURE_DIM, 2048)


if __name__ == "__main__":
    unittest.main()
