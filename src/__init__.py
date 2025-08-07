"""
GeoAI驱动的多暴露预测与健康不平等分析平台

该平台利用遥感数据、气象数据和社会统计数据，结合机器学习/深度学习模型，
对城市空气污染和绿地等环境暴露进行长期预测，并评估不同人群的暴露不平等。
"""

__version__ = "0.1.0"
__author__ = "GeoAI Team"
__email__ = "contact@geoai-platform.org"

from . import data_processing
from . import models
from . import classification
from . import prediction
from . import analysis
from . import visualization
from . import api

__all__ = [
    "data_processing",
    "models", 
    "classification",
    "prediction",
    "analysis",
    "visualization",
    "api"
]