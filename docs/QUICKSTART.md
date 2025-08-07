# 快速开始指南

本指南将帮助您快速开始使用GeoAI驱动的多暴露预测与健康不平等分析平台。

## 安装

### 环境要求

- Python 3.8 或更高版本
- GDAL/OGR地理空间库
- 至少4GB RAM（推荐8GB以上）

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/your-username/GeoAI.git
cd GeoAI
```

2. **创建虚拟环境**

```bash
# 使用conda（推荐）
conda create -n geoai python=3.8
conda activate geoai

# 或使用venv
python -m venv geoai_env
source geoai_env/bin/activate  # Linux/Mac
# 或
geoai_env\Scripts\activate  # Windows
```

3. **安装依赖**

```bash
# 安装基础依赖
pip install -r requirements.txt

# 安装开发依赖（可选）
pip install -r requirements-dev.txt
```

4. **验证安装**

```bash
python src/main.py
```

如果看到以下输出，说明安装成功：

```
GeoAI平台启动
配置文件加载完成
=== 数据处理演示 ===
...
演示完成
```

## 基础使用

### 1. 数据处理

#### 遥感数据处理

```python
from src.data_processing import RemoteSensingProcessor

# 初始化处理器
rs_processor = RemoteSensingProcessor()

# 获取Landsat数据（需要Google Earth Engine认证）
bounds = (-120, 34, -118, 36)  # 洛杉矶地区
landsat_data = rs_processor.get_landsat_data(
    bounds=bounds,
    start_date="2020-01-01",
    end_date="2020-12-31",
    cloud_threshold=20
)

# 计算绿地指标
green_metrics = rs_processor.calculate_green_space_metrics(landsat_data)
print(f"绿地覆盖率: {green_metrics['green_cover_percentage']:.1f}%")
```

#### 气象数据处理

```python
from src.data_processing import MeteorologicalProcessor

# 初始化处理器
met_processor = MeteorologicalProcessor()

# 获取气象站数据
station_data = met_processor.get_station_data(
    station_ids=['001', '002', '003'],
    variables=['temperature', 'precipitation', 'humidity'],
    start_date='2020-01-01',
    end_date='2020-12-31'
)

# 计算气候指数
climate_indices = met_processor.calculate_climate_indices(station_data)
```

#### 社会经济数据处理

```python
from src.data_processing import SocioeconomicProcessor

# 初始化处理器
socio_processor = SocioeconomicProcessor()

# 获取人口普查数据
census_data = socio_processor.get_census_data(
    region='CA_LA',
    variables=['total_population', 'median_income', 'poverty_rate'],
    year=2020
)

# 计算脆弱性指数
vulnerability_data = socio_processor.calculate_vulnerability_index(
    census_data, 
    health_data=None
)
```

### 2. 城乡分类

#### RUCA分类

```python
from src.classification import RUCAClassifier

# 初始化分类器
ruca_classifier = RUCAClassifier()

# 进行RUCA分类
ruca_result = ruca_classifier.classify_by_population_and_commuting(
    census_data, 
    population_col='total_population'
)

# 查看分类结果
print(ruca_result['ruca_type'].value_counts())
```

#### 人口密度分类

```python
from src.classification import PopulationDensityClassifier

# 初始化分类器
density_classifier = PopulationDensityClassifier()

# 固定阈值分类
density_result = density_classifier.classify_by_fixed_thresholds(
    census_data,
    population_col='total_population',
    threshold_type='us_census',
    categories=3
)

# 百分位数分类
percentile_result = density_classifier.classify_by_percentiles(
    census_data,
    population_col='total_population',
    percentiles=[25, 75]
)
```

#### 分类系统比较

```python
from src.classification import ClassificationComparator

# 准备比较数据
comparison_data = census_data.copy()
comparison_data['ruca_type'] = ruca_result['ruca_type']
comparison_data['density_class'] = density_result['density_class']

# 初始化比较器
comparator = ClassificationComparator()

# 比较分类系统
comparison_results = comparator.compare_classifications(
    comparison_data,
    classification_columns={
        'RUCA': 'ruca_type',
        'Density': 'density_class'
    }
)

print(f"分类一致性: {comparison_results['consistency_metrics']['mean_agreement']:.3f}")
```

### 3. 机器学习建模

#### 随机森林模型

```python
from src.models import RandomForestPredictor
import numpy as np

# 准备示例数据
X = np.random.normal(0, 1, (1000, 10))
y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.normal(0, 0.5, 1000)

# 初始化模型
rf_model = RandomForestPredictor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# 训练模型
rf_model.fit(X, y)

# 进行预测
predictions = rf_model.predict(X[:100])

# 获取特征重要性
importance = rf_model.get_feature_importance()
print("特征重要性:", importance)
```

#### 神经网络模型

```python
from src.models import DNNPredictor

# 初始化神经网络
dnn_model = DNNPredictor(
    hidden_layers=[128, 64, 32],
    dropout_rate=0.2,
    learning_rate=0.001,
    epochs=50
)

# 训练模型
dnn_model.fit(X, y)

# 预测（包含不确定性）
pred_mean, pred_lower, pred_upper = dnn_model.predict_with_uncertainty(X[:100])
```

#### 模型比较与选择

```python
from src.models import ModelSelector, SVRPredictor

# 定义候选模型
models = {
    'RandomForest': RandomForestPredictor(n_estimators=50, random_state=42),
    'SVR': SVRPredictor(C=1.0, epsilon=0.1, random_state=42),
    'DNN': DNNPredictor(hidden_layers=[64, 32], epochs=20, random_state=42)
}

# 初始化模型选择器
selector = ModelSelector()

# 选择最佳模型
best_name, best_model, scores = selector.select_best_model(models, X, y)
print(f"最佳模型: {best_name}")
```

### 4. 环境暴露预测

#### 空气质量预测

```python
from src.prediction import AirQualityPredictor
import xarray as xr
import pandas as pd

# 初始化预测器
air_predictor = AirQualityPredictor(
    pollutant='pm25',
    temporal_resolution='monthly',
    prediction_horizon=5
)

# 准备模拟数据
lons = np.linspace(-120, -118, 20)
lats = np.linspace(34, 36, 20)
times = pd.date_range('2020-01-01', '2022-12-31', freq='MS')

# 模拟气象数据
met_data = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], 
                   np.random.normal(20, 5, (len(times), len(lats), len(lons)))),
    'humidity': (['time', 'lat', 'lon'], 
                np.random.normal(60, 15, (len(times), len(lats), len(lons))))
}, coords={'time': times, 'lat': lats, 'lon': lons})

# 模拟PM2.5观测数据
pm25_data = xr.Dataset({
    'pm25': (['time', 'lat', 'lon'], 
            np.random.lognormal(2.5, 0.5, (len(times), len(lats), len(lons))))
}, coords={'time': times, 'lat': lats, 'lon': lons})

# 准备训练数据
environmental_data = {'meteorological': met_data}
socio_data = census_data  # 使用之前获取的人口数据

X_train, y_train = air_predictor.prepare_training_data(
    environmental_data=environmental_data,
    socioeconomic_data=socio_data,
    target_data=pm25_data,
    time_range=('2020-01-01', '2022-12-31')
)

# 训练模型
air_predictor.train_model(X_train, y_train, model_type='random_forest')

# 进行预测
predictions = air_predictor.predict_exposure(
    prediction_data={'features': X_train[:100]},
    target_years=[2025, 2030],
    uncertainty=True
)

print(f"预测完成，未来PM2.5浓度预测范围: {predictions.pm25.min().values:.1f} - {predictions.pm25.max().values:.1f} μg/m³")
```

### 5. 模型评估

```python
from src.models import ModelEvaluator

# 初始化评估器
evaluator = ModelEvaluator()

# 准备测试数据
X_test = X[-200:]
y_test = y[-200:]

# 评估单个模型
evaluation_results = evaluator.evaluate_single_model(
    rf_model, X_test, y_test,
    metrics=['rmse', 'mae', 'r2', 'mape']
)

print("模型评估结果:")
for metric, value in evaluation_results.items():
    print(f"  {metric.upper()}: {value:.3f}")

# 比较多个模型
trained_models = {
    'RandomForest': rf_model,
    'DNN': dnn_model
}

comparison_df = evaluator.compare_models(trained_models, X_test, y_test)
print("\n模型比较:")
print(comparison_df)
```

## 配置文件

平台使用YAML配置文件进行设置。主要配置文件位于 `config/default_config.yaml`。

### 修改配置

```yaml
# 数据配置
data:
  spatial:
    default_resolution: 0.01  # 空间分辨率（度）
  temporal:
    prediction_horizon: 10    # 预测时间范围（年）

# 模型配置
models:
  random_forest:
    n_estimators: 200        # 增加树的数量
    max_depth: 15           # 增加最大深度

# 预测配置
prediction:
  air_quality:
    model_type: "lstm_xgboost"  # 使用混合模型
    include_uncertainty: true
```

## 常见问题

### Q1: Google Earth Engine认证失败

**A:** 需要先注册GEE账户并设置认证：

```bash
# 安装Earth Engine API
pip install earthengine-api

# 认证
earthengine authenticate
```

### Q2: 内存不足错误

**A:** 减少数据处理的块大小或空间分辨率：

```python
# 降低分辨率
air_predictor = AirQualityPredictor(spatial_resolution=0.05)

# 或分块处理大数据
chunk_size = 1000
for i in range(0, len(X), chunk_size):
    X_chunk = X[i:i+chunk_size]
    predictions_chunk = model.predict(X_chunk)
```

### Q3: 缺少依赖包

**A:** 安装特定的地理空间库：

```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# 或使用conda
conda install -c conda-forge gdal geopandas rasterio
```

### Q4: 模型训练时间过长

**A:** 使用更简单的模型或减少数据量：

```python
# 使用更少的估计器
rf_model = RandomForestPredictor(n_estimators=50)

# 或使用数据子集
X_subset = X[:5000]  # 只使用前5000个样本
y_subset = y[:5000]
```

## 下一步

- 查看 [API参考文档](API_REFERENCE.md) 了解详细的函数说明
- 浏览 `notebooks/` 目录中的Jupyter示例
- 参考 [用户指南](USER_GUIDE.md) 获取高级用法
- 加入项目讨论：[GitHub Discussions](https://github.com/your-username/GeoAI/discussions)

## 获取帮助

如果遇到问题：

1. 查看常见问题部分
2. 搜索 [GitHub Issues](https://github.com/your-username/GeoAI/issues)
3. 创建新的Issue描述问题
4. 联系维护者

欢迎使用GeoAI平台！