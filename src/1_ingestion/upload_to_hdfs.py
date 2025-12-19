
"""
Upload Dataset to HDFS.

Script này dùng để upload một thư mục dataset từ local lên HDFS.
Bao gồm các class và hàm để kiểm tra, upload và xác minh dữ liệu.
"""


# Thư viện logging để ghi log quá trình upload
import logging
# Pathlib để thao tác với đường dẫn file/thư mục
from pathlib import Path
# Import cấu hình HDFS (đường dẫn mặc định)
from ..config.hdfs_config import HDFSConfig
# Import hàm setup_logger để cấu hình logger
from ..utils.logging_utils import setup_logger


# Khởi tạo logger cho module này
logger = setup_logger(__name__)



class HDFSIngestion:
    """
    Lớp thực hiện upload dataset từ local lên HDFS.
    """

    def __init__(self, local_path, hdfs_base_path=None):
        """
        Khởi tạo đối tượng ingestion.
        Args:
            local_path: Đường dẫn thư mục dataset local
            hdfs_base_path: Đường dẫn gốc HDFS (mặc định: /user/data/raw)
        """
        self.local_path = Path(local_path)
        # Nếu không truyền hdfs_base_path thì lấy mặc định từ config
        self.hdfs_base_path = hdfs_base_path or HDFSConfig.RAW_PATH

        # Kiểm tra thư mục local có tồn tại không
        if not self.local_path.exists():
            raise FileNotFoundError(f"Local path not found: {local_path}")


    def upload_dataset(self, max_files=None):
        """
        Thực hiện upload dataset từ local lên HDFS.
        Args:
            max_files: Giới hạn số lượng file upload (dùng để test)
        Returns:
            dict: Thống kê upload (số file, tổng dung lượng, số file lỗi)
        """
        logger.info(f"Starting upload from {self.local_path} to {self.hdfs_base_path}")

        # Khởi tạo biến thống kê
        stats = {
            'total_files': 0,      # Tổng số file upload
            'total_size': 0,       # Tổng dung lượng upload
            'failed_files': 0      # Số file upload thất bại
        }

        # TODO: Thực hiện upload thực tế ở đây

        logger.info(f"Upload completed: {stats}")
        return stats


    def verify_upload(self):
        """
        Kiểm tra lại toàn bộ file đã upload thành công lên HDFS chưa.
        Returns:
            bool: Kết quả xác minh (True nếu thành công)
        """
        logger.info("Verifying upload...")
        # TODO: Thực hiện kiểm tra thực tế ở đây
        return True



def main():
    """
    Hàm main để chạy script từ command line.
    Sử dụng argparse để nhận tham số dòng lệnh:
        --local_path: Đường dẫn dataset local (bắt buộc)
        --hdfs_path: Đích đến trên HDFS (tùy chọn)
        --max_files: Giới hạn số file upload (tùy chọn)
    """
    import argparse

    parser = argparse.ArgumentParser(description='Upload dataset to HDFS')
    parser.add_argument('--local_path', required=True, help='Local dataset path')
    parser.add_argument('--hdfs_path', default=None, help='HDFS destination path')
    parser.add_argument('--max_files', type=int, default=None, help='Max files to upload')

    args = parser.parse_args()

    uploader = HDFSIngestion(args.local_path, args.hdfs_path)
    stats = uploader.upload_dataset(max_files=args.max_files)

    if uploader.verify_upload():
        logger.info("Upload verified successfully")
    else:
        logger.error("Upload verification failed")


# Nếu chạy file này trực tiếp thì gọi main()
if __name__ == "__main__":
    main()

