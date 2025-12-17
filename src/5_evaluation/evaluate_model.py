"""
Evaluate Model
Calculate evaluation metrics
"""

import logging
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from ..config.hdfs_config import HDFSConfig
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Model evaluation class"""
    
    def __init__(self, spark):
        self.spark = spark
    
    def evaluate(self, model, test_df):
        """
        Evaluate model on test set
        Args:
            model: Trained model
            test_df: Test DataFrame[features_vec, label_idx]
        Returns:
            Dict of metrics
        """
        logger.info("Evaluating model...")
        
        # Generate predictions
        predictions = model.transform(test_df)
        
        # Save predictions to HDFS
        predictions.write.mode("overwrite").parquet(
            HDFSConfig.get_full_path(HDFSConfig.PREDICTIONS_PATH)
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions)
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self, predictions):
        """Calculate all metrics"""
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label_idx",
            predictionCol="prediction"
        )
        
        accuracy = evaluator.evaluate(
            predictions,
            {evaluator.metricName: "accuracy"}
        )
        
        precision = evaluator.evaluate(
            predictions,
            {evaluator.metricName: "weightedPrecision"}
        )
        
        recall = evaluator.evaluate(
            predictions,
            {evaluator.metricName: "weightedRecall"}
        )
        
        f1 = evaluator.evaluate(
            predictions,
            {evaluator.metricName: "f1"}
        )
        
        # AUC
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label_idx",
            metricName="areaUnderROC"
        )
        auc = binary_evaluator.evaluate(predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        logger.info("Metrics calculated:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def _save_metrics(self, metrics):
        """Save metrics to HDFS"""
        metrics_df = self.spark.createDataFrame([metrics])
        metrics_df.write.mode("overwrite").parquet(
            HDFSConfig.get_full_path(HDFSConfig.METRICS_PATH)
        )
        logger.info("Metrics saved to HDFS")


def main():
    """Main execution"""
    # TODO: Implement standalone evaluation
    pass


if __name__ == "__main__":
    main()
