# API参考文档

本文档提供GeoAI平台各模块的详细API参考。

## 目录

- [数据处理模块](#数据处理模块)
- [机器学习模型](#机器学习模型)
- [分类系统](#分类系统)
- [预测引擎](#预测引擎)
- [分析模块](#分析模块)
- [可视化模块](#可视化模块)

## 数据处理模块

### RemoteSensingProcessor

遥感数据处理器，支持多种卫星数据源。

#### 类定义

```python
class RemoteSensingProcessor:
    def __init__(self, 
                 gee_service_account: Optional[str] = None,
                 data_dir: Union[str, Path] = "data/raw/remote_sensing")
```

**参数:**
- `gee_service_account`: Google Earth Engine服务账户路径
- `data_dir`: 数据存储目录

#### 主要方法

##### get_landsat_data()

```python
def get_landsat_data(self, 
                    bounds: Tuple[float, float, float, float],
                    start_date: str,
                    end_date: str,
                    cloud_threshold: float = 20) -> xr.Dataset
```

获取Landsat数据。

**参数:**
- `bounds`: 边界框 (min_lon, min_lat, max_lon, max_lat)
- `start_date`: 开始日期 'YYYY-MM-DD'
- `end_date`: 结束日期 'YYYY-MM-DD' 
- `cloud_threshold`: 云覆盖阈值

**返回:**
- `xr.Dataset`: Landsat数据集

**示例:**
```python
processor = RemoteSensingProcessor()
data = processor.get_landsat_data(
    bounds=(-120, 34, -118, 36),
    start_date="2020-01-01",
    end_date="2020-12-31",
    cloud_threshold=10
)
```

##### calculate_green_space_metrics()

```python
def calculate_green_space_metrics(self, 
                                ndvi_data: xr.Dataset,
                                threshold: float = 0.3) -> Dict[str, float]
```

计算绿地空间指标。

**参数:**
- `ndvi_data`: NDVI数据
- `threshold`: 绿地NDVI阈值

**返回:**
- `Dict[str, float]`: 绿地指标字典

### MeteorologicalProcessor

气象数据处理器。

#### 类定义

```python
class MeteorologicalProcessor:
    def __init__(self, 
                 cds_api_key: Optional[str] = None,
                 data_dir: Union[str, Path] = "data/raw/meteorological")
```

#### 主要方法

##### get_era5_data()

```python
def get_era5_data(self,
                 variables: List[str],
                 bounds: Tuple[float, float, float, float],
                 start_date: str,
                 end_date: str,
                 time_resolution: str = 'monthly') -> xr.Dataset
```

获取ERA5再分析数据。

**参数:**
- `variables`: 气象变量列表
- `bounds`: 边界框
- `start_date`: 开始日期
- `end_date`: 结束日期
- `time_resolution`: 时间分辨率

**返回:**
- `xr.Dataset`: ERA5数据集

### SocioeconomicProcessor

社会经济数据处理器。

#### 主要方法

##### get_census_data()

```python
def get_census_data(self, 
                   region: str,
                   variables: List[str],
                   year: int = 2020,
                   geographic_level: str = 'tract') -> gpd.GeoDataFrame
```

获取人口普查数据。

**参数:**
- `region`: 地区代码或名称
- `variables`: 人口统计变量列表
- `year`: 年份
- `geographic_level`: 地理级别

**返回:**
- `gpd.GeoDataFrame`: 人口普查数据

## 机器学习模型

### BaseModel

所有预测模型的基类。

#### 抽象方法

```python
@abstractmethod
def build_model(self, **kwargs) -> Any:
    """构建模型"""
    pass

@abstractmethod  
def fit(self, X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **kwargs) -> 'BaseModel':
    """训练模型"""
    pass

@abstractmethod
def predict(self, X: Union[np.ndarray, pd.DataFrame],
           **kwargs) -> np.ndarray:
    """进行预测"""
    pass
```

### RandomForestPredictor

随机森林预测器。

#### 类定义

```python
class RandomForestPredictor(BaseModel):
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 verbose: bool = True)
```

#### 主要方法

##### fit()

```python
def fit(self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sample_weight: Optional[np.ndarray] = None) -> 'RandomForestPredictor'
```

训练随机森林模型。

##### predict()

```python
def predict(self,
           X: Union[np.ndarray, pd.DataFrame],
           return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
```

进行预测，可选返回标准差。

##### get_feature_importance()

```python
def get_feature_importance(self,
                          importance_type: str = 'impurity') -> Dict[str, float]
```

获取特征重要性。

### SVRPredictor

支持向量回归预测器。

#### 类定义

```python
class SVRPredictor(BaseModel):
    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 cache_size: float = 200,
                 max_iter: int = -1,
                 random_state: int = 42,
                 verbose: bool = True)
```

### DNNPredictor

深度神经网络预测器。

#### 类定义

```python
class DNNPredictor(BaseModel):
    def __init__(self,
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 10,
                 reduce_lr_patience: int = 5,
                 random_state: int = 42,
                 verbose: bool = True)
```

### LSTMPredictor

LSTM预测器，用于时间序列预测。

#### 类定义

```python
class LSTMPredictor(BaseModel):
    def __init__(self,
                 lstm_units: List[int] = [64, 32],
                 dense_units: List[int] = [32],
                 dropout_rate: float = 0.2,
                 recurrent_dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 sequence_length: int = 10,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 10,
                 random_state: int = 42,
                 verbose: bool = True)
```

### LSTMXGBoostPredictor

LSTM-XGBoost混合预测器。

#### 类定义

```python
class LSTMXGBoostPredictor(BaseModel):
    def __init__(self,
                 # LSTM参数
                 lstm_units: List[int] = [64, 32],
                 lstm_dropout: float = 0.2,
                 sequence_length: int = 10,
                 lstm_epochs: int = 50,
                 
                 # XGBoost参数
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 
                 # 集成参数
                 ensemble_method: str = 'weighted',
                 lstm_weight: float = 0.6,
                 
                 random_state: int = 42,
                 verbose: bool = True)
```

### ModelEvaluator

模型评估器。

#### 主要方法

##### evaluate_single_model()

```python
def evaluate_single_model(self,
                         model,
                         X_test: Union[np.ndarray, pd.DataFrame],
                         y_test: Union[np.ndarray, pd.Series],
                         metrics: Optional[List[str]] = None) -> Dict[str, float]
```

评估单个模型。

##### compare_models()

```python
def compare_models(self,
                  models: Dict[str, Any],
                  X_test: Union[np.ndarray, pd.DataFrame],
                  y_test: Union[np.ndarray, pd.Series],
                  metrics: Optional[List[str]] = None) -> pd.DataFrame
```

比较多个模型。

### ModelSelector

模型选择器。

#### 主要方法

##### select_best_model()

```python
def select_best_model(self,
                     models: Dict[str, Any],
                     X: Union[np.ndarray, pd.DataFrame],
                     y: Union[np.ndarray, pd.Series],
                     param_grids: Optional[Dict[str, Dict]] = None,
                     search_method: str = 'grid',
                     n_iter: int = 50) -> Tuple[str, Any, Dict[str, float]]
```

选择最佳模型。

## 分类系统

### RUCAClassifier

RUCA分类器。

#### 主要方法

##### classify_by_population_and_commuting()

```python
def classify_by_population_and_commuting(self,
                                       census_data: gpd.GeoDataFrame,
                                       population_col: str = 'total_population',
                                       commuting_data: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame
```

基于人口和通勤数据进行RUCA分类。

### PopulationDensityClassifier

人口密度分类器。

#### 主要方法

##### classify_by_fixed_thresholds()

```python
def classify_by_fixed_thresholds(self,
                               data: gpd.GeoDataFrame,
                               population_col: str = 'total_population',
                               area_col: Optional[str] = None,
                               threshold_type: str = 'us_census',
                               categories: int = 2) -> gpd.GeoDataFrame
```

使用固定阈值进行分类。

##### classify_by_percentiles()

```python
def classify_by_percentiles(self,
                          data: gpd.GeoDataFrame,
                          population_col: str = 'total_population',
                          area_col: Optional[str] = None,
                          percentiles: List[float] = [25, 50, 75]) -> gpd.GeoDataFrame
```

使用百分位数进行分类。

### NightlightClassifier

夜光强度分类器。

#### 主要方法

##### load_nightlight_data()

```python
def load_nightlight_data(self,
                       data_path: Union[str, Path],
                       data_type: str = 'viirs') -> xr.DataArray
```

加载夜光数据。

##### classify_by_fixed_thresholds()

```python
def classify_by_fixed_thresholds(self,
                               nightlight_data: xr.DataArray,
                               data_type: str = 'viirs',
                               categories: int = 3) -> xr.DataArray
```

使用固定阈值进行分类。

### LCZClassifier

地方气候区划分类器。

#### 主要方法

##### classify_from_features()

```python
def classify_from_features(self,
                         data: gpd.GeoDataFrame,
                         feature_mapping: Dict[str, str]) -> gpd.GeoDataFrame
```

从特征数据进行LCZ分类。

### ClassificationComparator

分类系统比较器。

#### 主要方法

##### compare_classifications()

```python
def compare_classifications(self,
                          data: gpd.GeoDataFrame,
                          classification_columns: Dict[str, str],
                          reference_column: Optional[str] = None) -> Dict[str, Any]
```

比较多种分类系统。

## 预测引擎

### ExposurePredictor

环境暴露预测器基类。

#### 抽象方法

```python
@abstractmethod
def prepare_training_data(self,
                        environmental_data: Dict[str, xr.Dataset],
                        socioeconomic_data: gpd.GeoDataFrame,
                        target_data: Union[xr.Dataset, gpd.GeoDataFrame],
                        time_range: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]

@abstractmethod
def train_model(self,
               X: np.ndarray,
               y: np.ndarray,
               model_type: str = 'ensemble') -> None

@abstractmethod
def predict_exposure(self,
                    prediction_data: Dict[str, Any],
                    target_years: List[int],
                    uncertainty: bool = True) -> xr.Dataset
```

### AirQualityPredictor

空气质量预测器。

#### 类定义

```python
class AirQualityPredictor(ExposurePredictor):
    def __init__(self,
                 pollutant: str = 'pm25',
                 temporal_resolution: str = 'monthly',
                 spatial_resolution: float = 0.01,
                 prediction_horizon: int = 10,
                 model_type: str = 'ensemble',
                 verbose: bool = True)
```

#### 支持的污染物

- `pm25`: PM2.5
- `no2`: 二氧化氮
- `o3`: 臭氧
- `so2`: 二氧化硫

## 使用示例

### 完整工作流程示例

```python
import numpy as np
import pandas as pd
import geopandas as gpd
from src.data_processing import RemoteSensingProcessor, SocioeconomicProcessor
from src.models import RandomForestPredictor, ModelEvaluator
from src.classification import PopulationDensityClassifier
from src.prediction import AirQualityPredictor

# 1. 数据获取
rs_processor = RemoteSensingProcessor()
socio_processor = SocioeconomicProcessor()

# 获取遥感数据
bounds = (-120, 34, -118, 36)
landsat_data = rs_processor.get_landsat_data(
    bounds=bounds,
    start_date="2020-01-01", 
    end_date="2020-12-31"
)

# 获取人口数据
census_data = socio_processor.get_census_data(
    region="CA_LA",
    variables=['total_population', 'median_income'],
    year=2020
)

# 2. 城乡分类
classifier = PopulationDensityClassifier()
classified_data = classifier.classify_by_fixed_thresholds(
    census_data, 
    categories=3
)

# 3. 空气质量预测
air_predictor = AirQualityPredictor(pollutant='pm25')

# 准备训练数据
environmental_data = {'remote_sensing': landsat_data}
X_train, y_train = air_predictor.prepare_training_data(
    environmental_data=environmental_data,
    socioeconomic_data=classified_data,
    target_data=pm25_observations,
    time_range=('2020-01-01', '2022-12-31')
)

# 训练模型
air_predictor.train_model(X_train, y_train)

# 进行预测
predictions = air_predictor.predict_exposure(
    prediction_data={'features': X_train[:100]},
    target_years=[2025, 2030],
    uncertainty=True
)

# 4. 模型评估
evaluator = ModelEvaluator()
evaluation_results = evaluator.evaluate_single_model(
    air_predictor.model, X_test, y_test
)

print(f"模型R²: {evaluation_results['r2']:.3f}")
print(f"RMSE: {evaluation_results['rmse']:.3f}")
```

## 错误处理

所有API都会抛出适当的异常：

- `ValueError`: 参数值错误
- `FileNotFoundError`: 文件未找到
- `ImportError`: 缺少依赖包
- `RuntimeError`: 运行时错误

建议使用try-except块处理异常：

```python
try:
    result = processor.process_data(data)
except ValueError as e:
    print(f"参数错误: {e}")
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```