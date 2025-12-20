#!/usr/bin/env python3
"""Model Management Utilities.

Provides functions to load pre-trained models and make predictions.
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Manage model loading and prediction."""
    
    # Default model paths
    MODELS_BASE = "hdfs://namenode:8020/user/models"
    TF_MODEL_PATH = f"{MODELS_BASE}/logistic_regression_tf"
    
    @staticmethod
    def load_tf_model(model_path: str = TF_MODEL_PATH) -> LogisticRegressionModel:
        """Load pre-trained TensorFlow MobileNetV2 Logistic Regression model.
        
        Args:
            model_path: Path to the model on HDFS
            
        Returns:
            Loaded LogisticRegressionModel
            
        Raises:
            Exception: If model cannot be loaded
        """
        try:
            model = LogisticRegressionModel.load(model_path)
            logger.info(f"✅ Model loaded from: {model_path}")
            return model
        except Exception as e:
            logger.error(f"❌ Error loading model from {model_path}: {e}")
            raise
    
    @staticmethod
    def predict(model: LogisticRegressionModel, df_features):
        """Make predictions using loaded model.
        
        Args:
            model: LogisticRegressionModel instance
            df_features: DataFrame with 'features' column
            
        Returns:
            DataFrame with 'prediction' and 'probability' columns added
        """
        return model.transform(df_features)
    
    @staticmethod
    def get_model_info(model: LogisticRegressionModel) -> dict:
        """Get model metadata.
        
        Args:
            model: LogisticRegressionModel instance
            
        Returns:
            Dictionary with model information
        """
        return {
            "features_col": model.featuresCol,
            "label_col": model.labelCol,
            "prediction_col": model.predictionCol,
            "coefficients_dim": len(model.coefficients),
            "intercept": float(model.intercept),
            "threshold": float(model.threshold)
        }


def load_model_and_predict(spark: SparkSession, 
                          df_features,
                          model_path: str = ModelManager.TF_MODEL_PATH):
    """Convenience function to load model and make predictions.
    
    Args:
        spark: SparkSession instance
        df_features: DataFrame with features
        model_path: Path to model on HDFS
        
    Returns:
        DataFrame with predictions
    """
    manager = ModelManager()
    model = manager.load_tf_model(model_path)
    predictions = manager.predict(model, df_features)
    return predictions


if __name__ == "__main__":
    # Test model loading
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder \
        .appName("Test_ModelManager") \
        .getOrCreate()
    
    try:
        manager = ModelManager()
        model = manager.load_tf_model()
        info = manager.get_model_info(model)
        
        print("✅ Model loaded successfully!")
        print(f"Model Info: {info}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        spark.stop()
