"""
HDFS Utilities
Helper functions cho HDFS operations
"""

import subprocess
import logging

logger = logging.getLogger(__name__)


class HDFSUtils:
    """HDFS utility functions"""
    
    @staticmethod
    def create_directory(hdfs_path):
        """
        Create directory on HDFS
        Args:
            hdfs_path: HDFS path to create
        """
        cmd = f"docker exec namenode hdfs dfs -mkdir -p {hdfs_path}"
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Created HDFS directory: {hdfs_path}")
    
    @staticmethod
    def list_files(hdfs_path):
        """
        List files in HDFS directory
        Args:
            hdfs_path: HDFS path to list
        Returns:
            List of files
        """
        cmd = f"docker exec namenode hdfs dfs -ls {hdfs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    
    @staticmethod
    def count_files(hdfs_path):
        """
        Count files in HDFS directory
        Args:
            hdfs_path: HDFS path
        Returns:
            File count
        """
        cmd = f"docker exec namenode hdfs dfs -count {hdfs_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # Parse output: directories, files, bytes, path
        parts = result.stdout.strip().split()
        if len(parts) >= 2:
            return int(parts[1])
        return 0
    
    @staticmethod
    def remove_directory(hdfs_path):
        """
        Remove directory from HDFS
        Args:
            hdfs_path: HDFS path to remove
        """
        cmd = f"docker exec namenode hdfs dfs -rm -r {hdfs_path}"
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Removed HDFS directory: {hdfs_path}")
    
    @staticmethod
    def upload_file(local_path, hdfs_path):
        """
        Upload file to HDFS
        Args:
            local_path: Local file path
            hdfs_path: HDFS destination path
        """
        cmd = f"docker exec namenode hdfs dfs -put {local_path} {hdfs_path}"
        subprocess.run(cmd, shell=True, check=True)
        logger.info(f"Uploaded {local_path} to {hdfs_path}")
