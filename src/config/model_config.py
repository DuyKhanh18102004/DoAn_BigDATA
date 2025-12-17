"""
Model Configuration
ML model hyperparameters và settings
"""

class ModelConfig:
    """ML Model configuration"""
    
    # Feature Extractor
    FEATURE_EXTRACTOR = "resnet50"  # hoặc "mobilenetv2"
    PRETRAINED = True
    FEATURE_DIM = 2048  # ResNet50 output dimension
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Random Forest Classifier
    RF_NUM_TREES = 100
    RF_MAX_DEPTH = 10
    RF_MIN_INSTANCES_PER_NODE = 1
    
    # Logistic Regression
    LR_MAX_ITER = 100
    LR_REG_PARAM = 0.1
    LR_ELASTIC_NET_PARAM = 0.8
    
    # Cross-Validation
    CV_NUM_FOLDS = 3
    CV_METRIC = "areaUnderROC"
    
    # Label mapping
    LABEL_REAL = 0
    LABEL_FAKE = 1
    LABEL_STRING_TO_INT = {"REAL": 0, "FAKE": 1}
    LABEL_INT_TO_STRING = {0: "REAL", 1: "FAKE"}
