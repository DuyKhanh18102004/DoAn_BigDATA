"""
Verify HDFS Upload
Check data integrity after upload
"""

import logging
from ..config.hdfs_config import HDFSConfig
from ..utils.hdfs_utils import HDFSUtils
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class UploadVerifier:
    """Verify HDFS upload integrity"""
    
    def __init__(self, hdfs_base_path=None):
        self.hdfs_base_path = hdfs_base_path or HDFSConfig.RAW_PATH
    
    def verify(self):
        """
        Verify upload integrity
        Returns:
            Dict with verification results
        """
        logger.info("Verifying HDFS upload...")
        
        results = {
            'train_real': self._verify_path(HDFSConfig.RAW_TRAIN_REAL),
            'train_fake': self._verify_path(HDFSConfig.RAW_TRAIN_FAKE),
            'test_real': self._verify_path(HDFSConfig.RAW_TEST_REAL),
            'test_fake': self._verify_path(HDFSConfig.RAW_TEST_FAKE),
        }
        
        return results
    
    def _verify_path(self, hdfs_path):
        """Verify single HDFS path"""
        try:
            count = HDFSUtils.count_files(hdfs_path)
            logger.info(f"{hdfs_path}: {count} files")
            return count
        except Exception as e:
            logger.error(f"Error verifying {hdfs_path}: {e}")
            return 0


def main():
    """Main execution"""
    verifier = UploadVerifier()
    results = verifier.verify()
    
    total = sum(results.values())
    logger.info(f"Total files verified: {total}")
    
    for category, count in results.items():
        logger.info(f"  {category}: {count}")


if __name__ == "__main__":
    main()
