"""
社会经济数据处理模块

支持人口统计、社会经济状况、健康数据等的获取和预处理。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import requests
import json


class SocioeconomicProcessor:
    """社会经济数据处理器"""
    
    def __init__(self, data_dir: Union[str, Path] = "data/raw/socioeconomic"):
        """
        初始化社会经济数据处理器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_census_data(self, 
                       region: str,
                       variables: List[str],
                       year: int = 2020,
                       geographic_level: str = 'tract') -> gpd.GeoDataFrame:
        """
        获取人口普查数据
        
        Args:
            region: 地区代码或名称
            variables: 人口统计变量列表
            year: 年份
            geographic_level: 地理级别 ('tract', 'block_group', 'county')
            
        Returns:
            包含人口统计数据的GeoDataFrame
        """
        # 变量映射 (以美国人口普查为例)
        variable_mapping = {
            'total_population': 'B01003_001E',
            'median_age': 'B01002_001E',
            'median_income': 'B19013_001E',
            'poverty_rate': 'B17001_002E',
            'education_bachelors': 'B15003_022E',
            'unemployment_rate': 'B23025_005E',
            'white_population': 'B02001_002E',
            'black_population': 'B02001_003E',
            'asian_population': 'B02001_005E',
            'hispanic_population': 'B03003_003E',
            'owner_occupied_housing': 'B25003_002E',
            'renter_occupied_housing': 'B25003_003E'
        }
        
        # 这里应该连接到具体的API (如美国人口普查局API)
        # 实际实现需要API密钥和具体的API调用
        
        # 模拟数据生成
        np.random.seed(42)
        n_units = 100  # 模拟100个地理单元
        
        data = {
            'GEOID': [f"{region}_{i:06d}" for i in range(n_units)],
            'NAME': [f"Census {geographic_level} {i}" for i in range(n_units)]
        }
        
        # 生成模拟的社会经济数据
        for var in variables:
            if var == 'total_population':
                data[var] = np.random.randint(500, 5000, n_units)
            elif var == 'median_age':
                data[var] = np.random.normal(38, 8, n_units)
            elif var == 'median_income':
                data[var] = np.random.lognormal(10.5, 0.5, n_units)
            elif var == 'poverty_rate':
                data[var] = np.random.beta(2, 8, n_units) * 100
            elif var == 'education_bachelors':
                data[var] = np.random.beta(3, 5, n_units) * 100
            elif var == 'unemployment_rate':
                data[var] = np.random.beta(1, 10, n_units) * 100
            elif 'population' in var:
                total_pop = data.get('total_population', np.random.randint(500, 5000, n_units))
                data[var] = np.random.beta(2, 5, n_units) * total_pop
            else:
                data[var] = np.random.normal(0, 1, n_units)
        
        # 生成模拟几何
        geometries = []
        base_lon, base_lat = -118.2437, 34.0522  # 洛杉矶坐标
        
        for i in range(n_units):
            # 创建小的矩形多边形
            lon_offset = (i % 10) * 0.01
            lat_offset = (i // 10) * 0.01
            
            from shapely.geometry import Polygon
            poly = Polygon([
                (base_lon + lon_offset, base_lat + lat_offset),
                (base_lon + lon_offset + 0.008, base_lat + lat_offset),
                (base_lon + lon_offset + 0.008, base_lat + lat_offset + 0.008),
                (base_lon + lon_offset, base_lat + lat_offset + 0.008)
            ])
            geometries.append(poly)
        
        df = pd.DataFrame(data)
        gdf = gpd.GeoDataFrame(df, geometry=geometries, crs='EPSG:4326')
        
        return gdf
    
    def get_health_data(self, 
                       region: str,
                       indicators: List[str],
                       year: int = 2020) -> pd.DataFrame:
        """
        获取健康指标数据
        
        Args:
            region: 地区代码
            indicators: 健康指标列表
            year: 年份
            
        Returns:
            健康数据DataFrame
        """
        # 健康指标映射
        health_indicators = {
            'life_expectancy': 'years',
            'infant_mortality': 'per_1000_births',
            'diabetes_prevalence': 'percentage',
            'obesity_prevalence': 'percentage',
            'asthma_prevalence': 'percentage',
            'cardiovascular_disease': 'age_adjusted_rate',
            'respiratory_disease': 'age_adjusted_rate',
            'cancer_incidence': 'age_adjusted_rate',
            'mental_health_days': 'average_days',
            'preventable_hospitalizations': 'rate_per_1000'
        }
        
        # 模拟健康数据
        np.random.seed(42)
        n_areas = 50
        
        data = {
            'area_id': [f"{region}_health_{i:03d}" for i in range(n_areas)],
            'year': [year] * n_areas
        }
        
        for indicator in indicators:
            if indicator == 'life_expectancy':
                data[indicator] = np.random.normal(78, 3, n_areas)
            elif indicator == 'infant_mortality':
                data[indicator] = np.random.gamma(2, 2, n_areas)
            elif 'prevalence' in indicator:
                data[indicator] = np.random.beta(2, 8, n_areas) * 100
            elif 'disease' in indicator or 'cancer' in indicator:
                data[indicator] = np.random.gamma(3, 50, n_areas)
            elif indicator == 'mental_health_days':
                data[indicator] = np.random.poisson(5, n_areas)
            elif 'hospitalizations' in indicator:
                data[indicator] = np.random.gamma(2, 20, n_areas)
            else:
                data[indicator] = np.random.normal(0, 1, n_areas)
        
        return pd.DataFrame(data)
    
    def calculate_vulnerability_index(self, 
                                    socioeconomic_data: gpd.GeoDataFrame,
                                    health_data: pd.DataFrame,
                                    weights: Optional[Dict[str, float]] = None) -> gpd.GeoDataFrame:
        """
        计算社会脆弱性指数
        
        Args:
            socioeconomic_data: 社会经济数据
            health_data: 健康数据
            weights: 各指标权重
            
        Returns:
            包含脆弱性指数的GeoDataFrame
        """
        # 默认权重
        if weights is None:
            weights = {
                'poverty_rate': 0.25,
                'unemployment_rate': 0.15,
                'education_bachelors': -0.20,  # 负权重，教育水平高则脆弱性低
                'median_income': -0.15,
                'median_age': 0.10,
                'minority_population': 0.15
            }
        
        # 复制数据
        result = socioeconomic_data.copy()
        
        # 计算少数族裔比例
        if all(col in result.columns for col in ['black_population', 'asian_population', 'hispanic_population', 'total_population']):
            result['minority_population'] = (
                (result['black_population'] + 
                 result['asian_population'] + 
                 result['hispanic_population']) / 
                result['total_population'] * 100
            )
        
        # 标准化指标
        vulnerability_score = 0
        
        for indicator, weight in weights.items():
            if indicator in result.columns:
                # Z-score标准化
                standardized = (result[indicator] - result[indicator].mean()) / result[indicator].std()
                vulnerability_score += weight * standardized
        
        result['vulnerability_index'] = vulnerability_score
        
        # 分级
        result['vulnerability_level'] = pd.cut(
            result['vulnerability_index'],
            bins=5,
            labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        )
        
        return result
    
    def calculate_environmental_justice_metrics(self, 
                                              exposure_data: gpd.GeoDataFrame,
                                              socioeconomic_data: gpd.GeoDataFrame) -> Dict[str, float]:
        """
        计算环境正义指标
        
        Args:
            exposure_data: 环境暴露数据
            socioeconomic_data: 社会经济数据
            
        Returns:
            环境正义指标字典
        """
        # 合并数据
        merged = gpd.sjoin(exposure_data, socioeconomic_data, how='inner', predicate='intersects')
        
        metrics = {}
        
        # 计算不同收入群体的暴露差异
        if 'median_income' in merged.columns and 'pm25' in merged.columns:
            # 按收入分组
            merged['income_quartile'] = pd.qcut(merged['median_income'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            # 计算各收入群体的平均暴露
            exposure_by_income = merged.groupby('income_quartile')['pm25'].mean()
            
            # 不平等比率 (最低收入vs最高收入)
            inequality_ratio = exposure_by_income['Q1'] / exposure_by_income['Q4']
            metrics['income_inequality_ratio'] = inequality_ratio
            
            # 基尼系数
            metrics['exposure_gini'] = self._calculate_gini_coefficient(merged['pm25'])
        
        # 计算种族/族裔暴露差异
        if 'minority_population' in merged.columns:
            # 按少数族裔比例分组
            merged['minority_high'] = merged['minority_population'] > merged['minority_population'].median()
            
            minority_exposure = merged[merged['minority_high']]['pm25'].mean()
            majority_exposure = merged[~merged['minority_high']]['pm25'].mean()
            
            metrics['racial_disparity'] = minority_exposure - majority_exposure
            metrics['racial_disparity_ratio'] = minority_exposure / majority_exposure
        
        # 累积暴露负担
        if all(col in merged.columns for col in ['pm25', 'poverty_rate']):
            # 高贫困率地区的暴露
            high_poverty = merged['poverty_rate'] > merged['poverty_rate'].quantile(0.75)
            cumulative_burden = merged[high_poverty]['pm25'].mean()
            overall_mean = merged['pm25'].mean()
            
            metrics['cumulative_burden_ratio'] = cumulative_burden / overall_mean
        
        return metrics
    
    def _calculate_gini_coefficient(self, values: pd.Series) -> float:
        """
        计算基尼系数
        
        Args:
            values: 数值序列
            
        Returns:
            基尼系数
        """
        values = values.dropna().sort_values()
        n = len(values)
        
        if n == 0:
            return 0
        
        cumsum = values.cumsum()
        return (n + 1 - 2 * cumsum.sum() / cumsum.iloc[-1]) / n
    
    def get_demographic_projections(self, 
                                  current_data: gpd.GeoDataFrame,
                                  target_year: int,
                                  growth_scenarios: Optional[Dict[str, float]] = None) -> gpd.GeoDataFrame:
        """
        获取人口统计预测数据
        
        Args:
            current_data: 当前人口数据
            target_year: 目标年份
            growth_scenarios: 增长情景
            
        Returns:
            预测的人口数据
        """
        if growth_scenarios is None:
            # 默认增长情景
            growth_scenarios = {
                'total_population': 0.01,  # 年增长1%
                'median_age': 0.2,         # 年增长0.2岁
                'median_income': 0.02,     # 年增长2%
                'education_bachelors': 0.5 # 年增长0.5%
            }
        
        # 计算年数差
        current_year = 2020  # 假设当前年份
        years_diff = target_year - current_year
        
        # 复制数据
        projected = current_data.copy()
        
        # 应用增长率
        for variable, growth_rate in growth_scenarios.items():
            if variable in projected.columns:
                projected[variable] = projected[variable] * (1 + growth_rate) ** years_diff
        
        # 添加不确定性
        np.random.seed(42)
        for variable in growth_scenarios.keys():
            if variable in projected.columns:
                uncertainty = np.random.normal(1, 0.05, len(projected))
                projected[variable] = projected[variable] * uncertainty
        
        projected['projection_year'] = target_year
        projected['projection_scenario'] = 'baseline'
        
        return projected
    
    def create_synthetic_population(self, 
                                   census_data: gpd.GeoDataFrame,
                                   n_individuals: int = 10000) -> pd.DataFrame:
        """
        创建合成人口数据
        
        Args:
            census_data: 人口普查数据
            n_individuals: 个体数量
            
        Returns:
            合成人口DataFrame
        """
        np.random.seed(42)
        
        individuals = []
        
        for _, tract in census_data.iterrows():
            # 根据人口密度分配个体数量
            tract_population = int(tract.get('total_population', 1000))
            n_tract_individuals = min(
                max(1, int(n_individuals * tract_population / census_data['total_population'].sum())),
                tract_population
            )
            
            for i in range(n_tract_individuals):
                individual = {
                    'individual_id': f"{tract['GEOID']}_{i:04d}",
                    'tract_id': tract['GEOID'],
                    'geometry_tract': tract['geometry']
                }
                
                # 年龄分布
                median_age = tract.get('median_age', 38)
                individual['age'] = max(0, int(np.random.normal(median_age, 15)))
                
                # 收入分布
                median_income = tract.get('median_income', 50000)
                individual['income'] = max(0, np.random.lognormal(
                    np.log(median_income), 0.5
                ))
                
                # 教育水平
                education_prob = tract.get('education_bachelors', 30) / 100
                individual['college_educated'] = np.random.random() < education_prob
                
                # 就业状态
                unemployment_prob = tract.get('unemployment_rate', 5) / 100
                individual['employed'] = np.random.random() > unemployment_prob
                
                # 种族/族裔 (简化)
                minority_prob = tract.get('minority_population', 30) / 100
                individual['minority'] = np.random.random() < minority_prob
                
                individuals.append(individual)
        
        return pd.DataFrame(individuals)