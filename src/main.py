# -*- coding: utf-8 -*-
"""
主程序入口
用来演示各个功能模块
"""

import sys
import logging
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')  # 忽略一些烦人的警告

# 把项目根目录加到path里，这样才能import
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_processing import RemoteSensingProcessor, MeteorologicalProcessor, SocioeconomicProcessor, DataIntegrator
from src.models import RandomForestPredictor, ModelEvaluator, ModelSelector
from src.classification import RUCAClassifier, PopulationDensityClassifier, ClassificationComparator
from src.prediction import AirQualityPredictor
from src.data_processing.utils import DataUtils


def setup_logging():
    # 配置日志输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('geoai_platform.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/default_config.yaml"):
    # 读取配置文件
    config_file = project_root / config_path
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        logger.warning(f"找不到配置文件: {config_file}，使用默认设置")
        return {}


def demo_data_processing(logger, config):
    # 演示数据处理部分
    logger.info("=== 数据处理演示 ===")
    
    # 先试试遥感数据
    logger.info("1. 遥感数据处理")
    rs_processor = RemoteSensingProcessor(verbose=True)
    
    # 用洛杉矶地区做测试
    bounds = (-120, 34, -118, 36)
    try:
        # Google Earth Engine需要认证，实际使用时需要先设置
        # landsat_data = rs_processor.get_landsat_data(bounds, "2020-01-01", "2020-12-31")
        # logger.info(f"Landsat数据OK，尺寸: {landsat_data.sizes}")
        logger.info("Landsat数据获取（需要GEE认证，暂时跳过）")
    except Exception as e:
        logger.warning(f"遥感数据获取失败: {e}")
    
    # 气象数据处理
    logger.info("2. 气象数据处理")
    met_processor = MeteorologicalProcessor(verbose=True)
    
    # 获取几个气象站的数据试试
    station_data = met_processor.get_station_data(
        station_ids=['001', '002', '003'],  # 这几个是测试用的站点ID
        variables=['temperature', 'precipitation', 'humidity'],
        start_date='2020-01-01',
        end_date='2020-12-31'
    )
    logger.info(f"气象站数据OK，记录数: {len(station_data)}")
    
    # 社会经济数据
    logger.info("3. 社会经济数据处理")
    socio_processor = SocioeconomicProcessor(verbose=True)
    
    # 人口普查数据（目前是模拟的）
    census_data = socio_processor.get_census_data(
        region='CA_LA',  # TODO: 改成真实的地区代码
        variables=['total_population', 'median_income', 'poverty_rate'],
        year=2020
    )
    logger.info(f"人口普查数据OK，区域数: {len(census_data)}")
    
    return station_data, census_data


def demo_classification_systems(logger, config, census_data):
    """演示分类系统功能"""
    logger.info("=== 分类系统演示 ===")
    
    # 1. RUCA分类
    logger.info("1. RUCA分类系统")
    ruca_classifier = RUCAClassifier(verbose=True)
    
    ruca_result = ruca_classifier.classify_by_population_and_commuting(
        census_data, population_col='total_population'
    )
    logger.info(f"RUCA分类完成，分类结果: {ruca_result['ruca_type'].value_counts().to_dict()}")
    
    # 2. 人口密度分类
    logger.info("2. 人口密度分类系统")
    density_classifier = PopulationDensityClassifier(verbose=True)
    
    density_result = density_classifier.classify_by_fixed_thresholds(
        census_data, population_col='total_population', categories=3
    )
    logger.info(f"密度分类完成，分类结果: {density_result['density_class'].value_counts().to_dict()}")
    
    # 3. 分类系统比较
    logger.info("3. 分类系统比较")
    comparator = ClassificationComparator(verbose=True)
    
    # 准备比较数据
    comparison_data = census_data.copy()
    comparison_data['ruca_type'] = ruca_result['ruca_type']
    comparison_data['density_class'] = density_result['density_class']
    
    comparison_results = comparator.compare_classifications(
        comparison_data,
        classification_columns={
            'RUCA': 'ruca_type',
            'Density': 'density_class'
        }
    )
    
    logger.info("分类系统比较完成")
    logger.info(f"平均一致性: {comparison_results['consistency_metrics'].get('mean_agreement', 'N/A'):.3f}")
    
    return ruca_result, density_result, comparison_results


def demo_machine_learning(logger, config, environmental_data, target_data):
    """演示机器学习功能"""
    logger.info("=== 机器学习演示 ===")
    
    # 1. 模型训练
    logger.info("1. 随机森林模型训练")
    
    # 准备训练数据（模拟）
    import numpy as np
    n_samples = 1000
    n_features = 10
    
    X = np.random.normal(0, 1, (n_samples, n_features))
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.normal(0, 0.5, n_samples)
    
    # 训练模型
    rf_model = RandomForestPredictor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        verbose=True
    )
    
    rf_model.fit(X, y)
    logger.info("随机森林模型训练完成")
    
    # 2. 模型评估
    logger.info("2. 模型评估")
    evaluator = ModelEvaluator(verbose=True)
    
    # 分割验证数据
    X_test = X[-200:]
    y_test = y[-200:]
    
    evaluation_results = evaluator.evaluate_single_model(rf_model, X_test, y_test)
    logger.info(f"模型评估结果: RMSE={evaluation_results['rmse']:.3f}, R²={evaluation_results['r2']:.3f}")
    
    # 3. 模型选择
    logger.info("3. 模型选择")
    from src.models import SVRPredictor
    
    models = {
        'RandomForest': RandomForestPredictor(n_estimators=50, random_state=42, verbose=False),
        'SVR': SVRPredictor(C=1.0, epsilon=0.1, random_state=42, verbose=False)
    }
    
    selector = ModelSelector(verbose=True)
    best_name, best_model, model_scores = selector.select_best_model(
        models, X[:-200], y[:-200], search_method='none'
    )
    
    logger.info(f"最佳模型: {best_name}")
    
    return rf_model, evaluation_results


def demo_prediction(logger, config):
    """演示预测功能"""
    logger.info("=== 环境暴露预测演示 ===")
    
    # 空气质量预测
    logger.info("1. PM2.5预测")
    air_predictor = AirQualityPredictor(
        pollutant='pm25',
        temporal_resolution='monthly',
        prediction_horizon=5,
        verbose=True
    )
    
    # 模拟训练数据
    import numpy as np
    import xarray as xr
    import geopandas as gpd
    from shapely.geometry import Point
    
    # 创建模拟环境数据
    lons = np.linspace(-120, -118, 20)
    lats = np.linspace(34, 36, 20)
    times = pd.date_range('2020-01-01', '2022-12-31', freq='MS')
    
    # 模拟气象数据
    met_data = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], 
                       np.random.normal(20, 5, (len(times), len(lats), len(lons)))),
        'humidity': (['time', 'lat', 'lon'], 
                    np.random.normal(60, 15, (len(times), len(lats), len(lons)))),
        'wind_speed': (['time', 'lat', 'lon'], 
                      np.random.exponential(3, (len(times), len(lats), len(lons))))
    }, coords={'time': times, 'lat': lats, 'lon': lons})
    
    # 模拟社会经济数据
    points = [Point(lon, lat) for lon in lons[::4] for lat in lats[::4]]
    socio_data = gpd.GeoDataFrame({
        'population_density': np.random.lognormal(6, 1, len(points)),
        'building_density': np.random.beta(2, 5, len(points)) * 100
    }, geometry=points, crs='EPSG:4326')
    
    # 模拟PM2.5观测数据
    pm25_data = xr.Dataset({
        'pm25': (['time', 'lat', 'lon'], 
                np.random.lognormal(2.5, 0.5, (len(times), len(lats), len(lons))))
    }, coords={'time': times, 'lat': lats, 'lon': lons})
    
    # 准备训练数据
    try:
        environmental_data = {'meteorological': met_data}
        
        X_train, y_train = air_predictor.prepare_training_data(
            environmental_data=environmental_data,
            socioeconomic_data=socio_data,
            target_data=pm25_data,
            time_range=('2020-01-01', '2022-12-31')
        )
        
        logger.info(f"训练数据准备完成: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        
        # 训练模型
        air_predictor.train_model(X_train, y_train, model_type='random_forest')
        
        # 进行预测
        prediction_data = {'features': X_train[:100]}  # 使用部分训练数据作为预测数据
        predictions = air_predictor.predict_exposure(
            prediction_data=prediction_data,
            target_years=[2025, 2030],
            uncertainty=True
        )
        
        logger.info(f"PM2.5预测完成，预测结果形状: {predictions[air_predictor.pollutant].shape}")
        logger.info(f"预测均值: {float(predictions[air_predictor.pollutant].mean()):.2f} μg/m³")
        
    except Exception as e:
        logger.error(f"预测演示出错: {e}")
        predictions = None
    
    return predictions


def demo_analysis(logger, config):
    """演示分析功能"""
    logger.info("=== 健康不平等分析演示 ===")
    
    # 模拟暴露和人口数据
    import numpy as np
    import pandas as pd
    
    n_areas = 100
    
    # 模拟数据
    exposure_data = np.random.lognormal(2.5, 0.5, n_areas)  # PM2.5暴露
    income_data = np.random.lognormal(10.5, 0.8, n_areas)   # 收入
    age_data = np.random.normal(40, 15, n_areas)            # 年龄
    
    analysis_data = pd.DataFrame({
        'area_id': range(n_areas),
        'pm25_exposure': exposure_data,
        'median_income': income_data,
        'median_age': age_data
    })
    
    # 计算基本不平等指标
    logger.info("1. 基本不平等指标计算")
    
    # 按收入分组
    income_quartiles = pd.qcut(analysis_data['median_income'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    analysis_data['income_quartile'] = income_quartiles
    
    # 计算各收入组的平均暴露
    exposure_by_income = analysis_data.groupby('income_quartile')['pm25_exposure'].mean()
    
    # 不平等比率
    inequality_ratio = exposure_by_income['Q1'] / exposure_by_income['Q4']
    
    logger.info(f"收入不平等比率 (最低/最高收入组): {inequality_ratio:.2f}")
    logger.info("各收入组平均暴露:")
    for quartile, exposure in exposure_by_income.items():
        logger.info(f"  {quartile}: {exposure:.2f} μg/m³")
    
    # 2. 基尼系数计算
    def calculate_gini(values):
        """计算基尼系数"""
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n
    
    gini_coefficient = calculate_gini(analysis_data['pm25_exposure'])
    logger.info(f"暴露基尼系数: {gini_coefficient:.3f}")
    
    # 3. 脆弱性分析
    logger.info("2. 脆弱性分析")
    
    # 计算脆弱性指数
    # 标准化指标
    income_norm = (analysis_data['median_income'] - analysis_data['median_income'].min()) / \
                  (analysis_data['median_income'].max() - analysis_data['median_income'].min())
    age_norm = analysis_data['median_age'] / analysis_data['median_age'].max()
    
    # 脆弱性指数（低收入和高年龄增加脆弱性）
    vulnerability_index = (1 - income_norm) * 0.6 + age_norm * 0.4
    analysis_data['vulnerability_index'] = vulnerability_index
    
    # 高脆弱性地区
    high_vulnerability = analysis_data[analysis_data['vulnerability_index'] > 0.7]
    logger.info(f"高脆弱性地区数量: {len(high_vulnerability)} ({len(high_vulnerability)/len(analysis_data)*100:.1f}%)")
    logger.info(f"高脆弱性地区平均暴露: {high_vulnerability['pm25_exposure'].mean():.2f} μg/m³")
    
    return analysis_data


def main():
    # 主程序入口
    logger = setup_logging()
    logger.info("开始运行演示程序")
    
    # 读取配置
    config = load_config()
    logger.info("配置文件读取完成")
    
    try:
        # 1. 先演示数据处理
        station_data, census_data = demo_data_processing(logger, config)
        
        # 2. 然后是分类系统
        ruca_result, density_result, comparison_results = demo_classification_systems(
            logger, config, census_data
        )
        
        # 3. 机器学习部分
        model, evaluation_results = demo_machine_learning(
            logger, config, None, None
        )
        
        # 4. 预测功能
        predictions = demo_prediction(logger, config)
        
        # 5. 最后是分析
        analysis_results = demo_analysis(logger, config)
        
        logger.info("=== 演示完成 ===")
        logger.info("所有功能都跑了一遍！")
        
        # 简单总结一下
        logger.info("演示总结:")
        logger.info(f"- 处理了 {len(census_data)} 个人口普查区域")
        logger.info(f"- 机器学习模型R² = {evaluation_results['r2']:.3f}")
        logger.info(f"- 完成了环境暴露预测")
        logger.info(f"- 分析了 {len(analysis_results)} 个区域的不平等情况")
        
    except Exception as e:
        logger.error(f"运行出错了: {e}")
        raise


if __name__ == "__main__":
    main()