# -*- coding: utf-8 -*-
"""
简化版演示
主要是为了在没有复杂依赖的情况下测试基本功能
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# 设置项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def setup_logging():
    # 设置日志输出
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def demo_basic_ml():
    # 基础机器学习演示
    logger = logging.getLogger(__name__)
    logger.info("=== 基础机器学习演示 ===")
    
    # 随便生成点数据试试
    np.random.seed(42)  # 固定随机种子，这样结果可复现
    n_samples = 1000
    n_features = 10
    
    # 特征数据，标准正态分布
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # 目标变量，模拟PM2.5浓度
    # 假设前几个特征跟PM2.5有关系
    y = (X[:, 0] * 2.5 +           # 假设这是温度影响
         X[:, 1] * -1.5 +          # 这是风速（风大污染低）
         X[:, 2] * 1.0 +           # 湿度
         X[:, 3] * 0.8 +           # 人口密度
         np.random.normal(0, 0.5, n_samples))  # 加点噪声
    
    # PM2.5不能是负数
    y = np.maximum(y + 15, 0.1)  # 基准15μg/m³，最小0.1
    
    logger.info(f"生成了 {n_samples} 个样本, {n_features} 个特征")
    logger.info(f"PM2.5浓度范围: {y.min():.1f} - {y.max():.1f} μg/m³")
    
    # 8:2分割训练测试集
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def demo_sklearn_models(X_train, X_test, y_train, y_test):
    # 试试几个不同的模型
    logger = logging.getLogger(__name__)
    logger.info("=== 机器学习模型对比 ===")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    
    # SVR对数据比较敏感，需要标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 几个常用的模型
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1)  # SVR调参比较麻烦，先用默认的
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"正在训练 {name}...")
        
        # SVR用标准化的数据，其他的用原始数据
        if name == 'SVR':
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # 训练
        model.fit(X_tr, y_train)
        
        # 预测
        y_pred = model.predict(X_te)
        
        # 计算几个评估指标
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        logger.info(f"{name} 结果 - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
        
        # 随机森林可以看特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_features = np.argsort(importances)[-3:][::-1]  # 取前3个重要的
            logger.info(f"  重要特征排序: {top_features.tolist()}")
    
    return results

def demo_data_analysis(X_train, y_train):
    """演示数据分析功能"""
    logger = logging.getLogger(__name__)
    logger.info("=== 数据分析演示 ===")
    
    # 创建DataFrame便于分析
    feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    df = pd.DataFrame(X_train, columns=feature_names)
    df['PM25'] = y_train
    
    # 基础统计
    logger.info("PM2.5浓度统计:")
    logger.info(f"  平均值: {y_train.mean():.2f} μg/m³")
    logger.info(f"  标准差: {y_train.std():.2f} μg/m³")
    logger.info(f"  中位数: {np.median(y_train):.2f} μg/m³")
    
    # 相关性分析
    correlations = df.corr()['PM25'].abs().sort_values(ascending=False)
    logger.info("与PM2.5相关性最高的特征:")
    for feature, corr in correlations.head(4).items():
        if feature != 'PM25':
            logger.info(f"  {feature}: {corr:.3f}")
    
    # 分组分析（模拟不同地区）
    # 将数据分为高、中、低污染区
    pollution_levels = pd.qcut(y_train, 3, labels=['低污染', '中污染', '高污染'])
    
    logger.info("不同污染水平区域的特征对比:")
    for level in ['低污染', '中污染', '高污染']:
        mask = pollution_levels == level
        level_data = df[mask]
        logger.info(f"  {level}区域 (n={mask.sum()}):")
        logger.info(f"    平均PM2.5: {level_data['PM25'].mean():.2f} μg/m³")
        logger.info(f"    主要特征均值: {level_data[feature_names[:3]].mean().values}")

def demo_inequality_analysis():
    """演示健康不平等分析"""
    logger = logging.getLogger(__name__)
    logger.info("=== 健康不平等分析演示 ===")
    
    # 生成模拟的社区数据
    np.random.seed(42)
    n_communities = 200
    
    # 社区特征
    income = np.random.lognormal(mean=10.5, sigma=0.8, size=n_communities)  # 收入
    age = np.random.normal(40, 12, n_communities)  # 平均年龄
    education = np.random.beta(3, 2, n_communities) * 100  # 教育水平百分比
    
    # PM2.5暴露（与收入负相关，与年龄正相关）
    pm25_exposure = (
        25 +  # 基准水平
        -5 * (income - income.mean()) / income.std() +  # 收入效应
        2 * (age - age.mean()) / age.std() +  # 年龄效应
        np.random.normal(0, 3, n_communities)  # 随机变异
    )
    pm25_exposure = np.maximum(pm25_exposure, 1)  # 确保为正
    
    # 创建分析数据
    analysis_data = pd.DataFrame({
        'income': income,
        'age': age,
        'education': education,
        'pm25_exposure': pm25_exposure
    })
    
    # 按收入分组分析
    income_quartiles = pd.qcut(income, 4, labels=['Q1(最低)', 'Q2', 'Q3', 'Q4(最高)'])
    exposure_by_income = analysis_data.groupby(income_quartiles)['pm25_exposure'].agg(['mean', 'std'])
    
    logger.info("按收入分组的PM2.5暴露水平:")
    for quartile in exposure_by_income.index:
        mean_exp = exposure_by_income.loc[quartile, 'mean']
        std_exp = exposure_by_income.loc[quartile, 'std']
        logger.info(f"  {quartile}: {mean_exp:.2f} ± {std_exp:.2f} μg/m³")
    
    # 不平等指标
    q1_exposure = exposure_by_income.loc['Q1(最低)', 'mean']
    q4_exposure = exposure_by_income.loc['Q4(最高)', 'mean']
    inequality_ratio = q1_exposure / q4_exposure
    
    logger.info(f"不平等比率 (最低收入/最高收入): {inequality_ratio:.2f}")
    
    # 基尼系数
    def gini_coefficient(values):
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * cumsum.sum() / cumsum[-1]) / n
    
    gini = gini_coefficient(pm25_exposure)
    logger.info(f"PM2.5暴露基尼系数: {gini:.3f}")
    
    # 脆弱性分析
    # 标准化指标
    income_norm = (income - income.min()) / (income.max() - income.min())
    age_norm = age / age.max()
    
    # 脆弱性指数（低收入和高年龄增加脆弱性）
    vulnerability = (1 - income_norm) * 0.6 + age_norm * 0.4
    
    # 高脆弱性社区
    high_vulnerability = vulnerability > np.percentile(vulnerability, 75)
    high_vuln_exposure = pm25_exposure[high_vulnerability].mean()
    low_vuln_exposure = pm25_exposure[~high_vulnerability].mean()
    
    logger.info(f"高脆弱性社区数量: {high_vulnerability.sum()} ({high_vulnerability.mean()*100:.1f}%)")
    logger.info(f"高脆弱性社区平均暴露: {high_vuln_exposure:.2f} μg/m³")
    logger.info(f"低脆弱性社区平均暴露: {low_vuln_exposure:.2f} μg/m³")
    logger.info(f"脆弱性差异: {high_vuln_exposure - low_vuln_exposure:.2f} μg/m³")

def main():
    """主函数"""
    logger = setup_logging()
    
    logger.info("GeoAI平台简化演示开始")
    logger.info("=" * 50)
    
    try:
        # 1. 生成示例数据
        X_train, X_test, y_train, y_test = demo_basic_ml()
        
        # 2. 机器学习模型演示
        model_results = demo_sklearn_models(X_train, X_test, y_train, y_test)
        
        # 3. 数据分析演示
        demo_data_analysis(X_train, y_train)
        
        # 4. 不平等分析演示
        demo_inequality_analysis()
        
        # 总结
        logger.info("=" * 50)
        logger.info("演示完成！主要结果总结:")
        
        # 找到最佳模型
        best_model = min(model_results.items(), key=lambda x: x[1]['RMSE'])
        logger.info(f"最佳模型: {best_model[0]} (RMSE: {best_model[1]['RMSE']:.3f})")
        
        logger.info("演示功能:")
        logger.info("✓ 机器学习模型训练和评估")
        logger.info("✓ 特征重要性分析")
        logger.info("✓ 环境暴露数据分析")
        logger.info("✓ 健康不平等指标计算")
        logger.info("✓ 脆弱性群体识别")
        
        logger.info("\n要体验完整功能，请安装所有依赖包并运行 python src/main.py")
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()