"""
Confusion Matrix
Generate confusion matrix and visualization
"""

import logging
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def generate_confusion_matrix(predictions):
    """
    Generate confusion matrix from predictions
    Args:
        predictions: DataFrame with predictions and labels
    Returns:
        Confusion matrix as dict
    """
    logger.info("Generating confusion matrix...")
    
    # Collect predictions and labels
    y_true = predictions.select("label_idx").rdd.map(lambda x: x[0]).collect()
    y_pred = predictions.select("prediction").rdd.map(lambda x: x[0]).collect()
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    result = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'confusion_matrix': cm.tolist()
    }
    
    logger.info(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    
    return result


def main():
    """Main execution"""
    # TODO: Implement standalone confusion matrix generation
    pass


if __name__ == "__main__":
    main()
