"""
Generate Report
Business insights and evaluation report
"""

import logging
from pathlib import Path
from datetime import datetime
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def generate_evaluation_report(metrics, confusion_matrix_data, output_path):
    """
    Generate comprehensive evaluation report
    Args:
        metrics: Dict of evaluation metrics
        confusion_matrix_data: Confusion matrix data
        output_path: Path to save report
    """
    logger.info("Generating evaluation report...")
    
    report_path = Path(output_path) / "evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Deepfake Detection - Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Performance Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric, value in metrics.items():
            f.write(f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n")
        
        f.write("\n## Confusion Matrix\n\n")
        f.write("```\n")
        f.write("         Predicted\n")
        f.write("         REAL  FAKE\n")
        f.write("Actual\n")
        tn = confusion_matrix_data['true_negatives']
        fp = confusion_matrix_data['false_positives']
        fn = confusion_matrix_data['false_negatives']
        tp = confusion_matrix_data['true_positives']
        f.write(f"REAL     {tn:4d}  {fp:4d}\n")
        f.write(f"FAKE     {fn:4d}  {tp:4d}\n")
        f.write("```\n\n")
        
        f.write("## Analysis\n\n")
        f.write(f"- **Accuracy:** {metrics['accuracy']:.2%}\n")
        f.write(f"- **Precision:** {metrics['precision']:.2%}\n")
        f.write(f"- **Recall:** {metrics['recall']:.2%}\n")
        f.write(f"- **F1-Score:** {metrics['f1_score']:.2%}\n")
        f.write(f"- **AUC:** {metrics['auc']:.4f}\n\n")
        
        # False rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        f.write(f"- **False Positive Rate:** {fpr:.2%}\n")
        f.write(f"- **False Negative Rate:** {fnr:.2%}\n\n")
        
        f.write("## Business Insights\n\n")
        f.write("### Question: Liệu ResNet50 features có đủ để phát hiện Deepfake?\n\n")
        
        if metrics['accuracy'] > 0.85:
            f.write("**Answer:** ResNet50 features RẤT HIỆU QUẢ cho bài toán deepfake detection.\n\n")
        elif metrics['accuracy'] > 0.70:
            f.write("**Answer:** ResNet50 features TỐT nhưng có thể cải thiện thêm.\n\n")
        else:
            f.write("**Answer:** ResNet50 features CHƯA ĐỦ, cần xem xét models khác.\n\n")
        
        f.write("### Recommendations\n\n")
        f.write("- Fine-tune ResNet50 on deepfake dataset\n")
        f.write("- Experiment with other architectures (EfficientNet, ViT)\n")
        f.write("- Ensemble multiple models\n")
        f.write("- Augment training data\n")
    
    logger.info(f"Report saved to {report_path}")


def main():
    """Main execution"""
    # TODO: Implement standalone report generation
    pass


if __name__ == "__main__":
    main()
