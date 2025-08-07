"""
人口密度分类器

基于人口密度的城乡分类系统。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from sklearn.cluster import KMeans
from scipy import stats


class PopulationDensityClassifier:
    """人口密度分类器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化人口密度分类器
        
        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # 预定义的密度阈值（人/km²）
        self.standard_thresholds = {
            'who_urban': 1500,  # WHO城市定义
            'us_census': 1000,  # 美国人口普查城市定义
            'eu_urban': 300,    # 欧盟城市定义
            'oecd_urban': 1500  # OECD城市定义
        }
        
        # 多级分类阈值
        self.multi_level_thresholds = {
            'very_high': 5000,
            'high': 1500,
            'medium': 500,
            'low': 100,
            'very_low': 0
        }
    
    def classify_by_fixed_thresholds(self,
                                   data: gpd.GeoDataFrame,
                                   population_col: str = 'total_population',
                                   area_col: Optional[str] = None,
                                   threshold_type: str = 'us_census',
                                   categories: int = 2) -> gpd.GeoDataFrame:
        """
        使用固定阈值进行分类
        
        Args:
            data: 地理数据
            population_col: 人口列名
            area_col: 面积列名
            threshold_type: 阈值类型
            categories: 分类数量 (2, 3, 5)
            
        Returns:
            分类结果
        """
        if self.verbose:
            self.logger.info(f"Classifying by fixed thresholds ({threshold_type})...")
        
        result = data.copy()
        
        # 计算人口密度
        if 'population_density' not in result.columns:
            result['population_density'] = self._calculate_population_density(
                result, population_col, area_col
            )
        
        # 应用分类
        if categories == 2:
            threshold = self.standard_thresholds[threshold_type]
            result['density_class'] = np.where(
                result['population_density'] >= threshold, 'Urban', 'Rural'
            )
            result['density_code'] = np.where(
                result['population_density'] >= threshold, 1, 0
            )
        
        elif categories == 3:
            threshold = self.standard_thresholds[threshold_type]
            result['density_class'] = np.where(
                result['population_density'] >= threshold, 'Urban',
                np.where(
                    result['population_density'] >= threshold/3, 'Suburban', 'Rural'
                )
            )
            result['density_code'] = np.where(
                result['population_density'] >= threshold, 2,
                np.where(
                    result['population_density'] >= threshold/3, 1, 0
                )
            )
        
        elif categories == 5:
            thresholds = self.multi_level_thresholds
            conditions = [
                result['population_density'] >= thresholds['very_high'],
                result['population_density'] >= thresholds['high'],
                result['population_density'] >= thresholds['medium'],
                result['population_density'] >= thresholds['low']
            ]
            choices = ['Very High', 'High', 'Medium', 'Low']
            result['density_class'] = np.select(conditions, choices, default='Very Low')
            
            # 对应的数值代码
            code_conditions = [
                result['population_density'] >= thresholds['very_high'],
                result['population_density'] >= thresholds['high'], 
                result['population_density'] >= thresholds['medium'],
                result['population_density'] >= thresholds['low']
            ]
            result['density_code'] = np.select(code_conditions, [4, 3, 2, 1], default=0)
        
        if self.verbose:
            self._log_classification_summary(result, 'density_class')
        
        return result
    
    def classify_by_percentiles(self,
                              data: gpd.GeoDataFrame,
                              population_col: str = 'total_population',
                              area_col: Optional[str] = None,
                              percentiles: List[float] = [25, 50, 75]) -> gpd.GeoDataFrame:
        """
        使用百分位数进行分类
        
        Args:
            data: 地理数据
            population_col: 人口列名
            area_col: 面积列名
            percentiles: 百分位数列表
            
        Returns:
            分类结果
        """
        if self.verbose:
            self.logger.info("Classifying by percentiles...")
        
        result = data.copy()
        
        # 计算人口密度
        if 'population_density' not in result.columns:
            result['population_density'] = self._calculate_population_density(
                result, population_col, area_col
            )
        
        # 计算百分位数阈值
        thresholds = np.percentile(result['population_density'], percentiles)
        
        # 创建分类标签
        n_categories = len(percentiles) + 1
        if n_categories == 2:
            labels = ['Low', 'High']
        elif n_categories == 3:
            labels = ['Low', 'Medium', 'High'] 
        elif n_categories == 4:
            labels = ['Very Low', 'Low', 'Medium', 'High']
        elif n_categories == 5:
            labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        else:
            labels = [f'Level_{i}' for i in range(n_categories)]
        
        # 应用分类
        result['density_class'] = pd.cut(
            result['population_density'],
            bins=[-np.inf] + thresholds.tolist() + [np.inf],
            labels=labels,
            include_lowest=True
        )
        
        result['density_code'] = pd.cut(
            result['population_density'],
            bins=[-np.inf] + thresholds.tolist() + [np.inf],
            labels=range(n_categories),
            include_lowest=True
        ).astype(int)
        
        # 存储阈值信息
        result.attrs['percentile_thresholds'] = dict(zip(percentiles, thresholds))
        
        if self.verbose:
            self.logger.info(f"Percentile thresholds: {dict(zip(percentiles, thresholds))}")
            self._log_classification_summary(result, 'density_class')
        
        return result
    
    def classify_by_clustering(self,
                             data: gpd.GeoDataFrame,
                             population_col: str = 'total_population',
                             area_col: Optional[str] = None,
                             n_clusters: int = 3,
                             method: str = 'kmeans') -> gpd.GeoDataFrame:
        """
        使用聚类进行分类
        
        Args:
            data: 地理数据
            population_col: 人口列名
            area_col: 面积列名
            n_clusters: 聚类数量
            method: 聚类方法
            
        Returns:
            分类结果
        """
        if self.verbose:
            self.logger.info(f"Classifying by clustering ({method})...")
        
        result = data.copy()
        
        # 计算人口密度
        if 'population_density' not in result.columns:
            result['population_density'] = self._calculate_population_density(
                result, population_col, area_col
            )
        
        # 准备聚类数据（log变换以处理偏斜分布）
        density_log = np.log1p(result['population_density']).values.reshape(-1, 1)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(density_log)
            
            # 获取聚类中心
            cluster_centers = clusterer.cluster_centers_.ravel()
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # 按密度排序聚类标签
        center_density_pairs = list(zip(range(n_clusters), cluster_centers))
        center_density_pairs.sort(key=lambda x: x[1])
        
        # 创建标签映射
        label_mapping = {}
        if n_clusters == 2:
            labels = ['Rural', 'Urban']
        elif n_clusters == 3:
            labels = ['Rural', 'Suburban', 'Urban']
        elif n_clusters == 4:
            labels = ['Rural', 'Low Urban', 'Medium Urban', 'High Urban']
        elif n_clusters == 5:
            labels = ['Rural', 'Low Rural', 'Suburban', 'Urban', 'High Urban']
        else:
            labels = [f'Cluster_{i}' for i in range(n_clusters)]
        
        for i, (original_label, _) in enumerate(center_density_pairs):
            label_mapping[original_label] = labels[i]
        
        # 应用标签
        result['density_class'] = pd.Series(cluster_labels).map(label_mapping)
        result['density_code'] = pd.Series(cluster_labels).map(
            {original: i for i, (original, _) in enumerate(center_density_pairs)}
        )
        
        # 存储聚类信息
        result.attrs['cluster_centers'] = {labels[i]: np.expm1(center) 
                                         for i, (_, center) in enumerate(center_density_pairs)}
        
        if self.verbose:
            self.logger.info(f"Cluster centers (density): {result.attrs['cluster_centers']}")
            self._log_classification_summary(result, 'density_class')
        
        return result
    
    def classify_by_natural_breaks(self,
                                 data: gpd.GeoDataFrame,
                                 population_col: str = 'total_population',
                                 area_col: Optional[str] = None,
                                 n_classes: int = 5) -> gpd.GeoDataFrame:
        """
        使用自然分级法进行分类
        
        Args:
            data: 地理数据
            population_col: 人口列名
            area_col: 面积列名
            n_classes: 分类数量
            
        Returns:
            分类结果
        """
        try:
            import mapclassify
        except ImportError:
            self.logger.warning("mapclassify not available, using percentile method instead")
            percentiles = [i * 100/n_classes for i in range(1, n_classes)]
            return self.classify_by_percentiles(data, population_col, area_col, percentiles)
        
        if self.verbose:
            self.logger.info("Classifying by natural breaks...")
        
        result = data.copy()
        
        # 计算人口密度
        if 'population_density' not in result.columns:
            result['population_density'] = self._calculate_population_density(
                result, population_col, area_col
            )
        
        # 应用自然分级
        classifier = mapclassify.NaturalBreaks(result['population_density'], k=n_classes)
        
        # 创建标签
        if n_classes == 2:
            labels = ['Rural', 'Urban']
        elif n_classes == 3:
            labels = ['Rural', 'Suburban', 'Urban']
        elif n_classes == 5:
            labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        else:
            labels = [f'Class_{i+1}' for i in range(n_classes)]
        
        result['density_class'] = pd.Categorical.from_codes(
            classifier.yb, categories=labels
        )
        result['density_code'] = classifier.yb
        
        # 存储分界点
        result.attrs['natural_breaks'] = classifier.bins.tolist()
        
        if self.verbose:
            self.logger.info(f"Natural breaks: {classifier.bins}")
            self._log_classification_summary(result, 'density_class')
        
        return result
    
    def _calculate_population_density(self,
                                    data: gpd.GeoDataFrame,
                                    population_col: str,
                                    area_col: Optional[str] = None) -> pd.Series:
        """
        计算人口密度
        
        Args:
            data: 地理数据
            population_col: 人口列名
            area_col: 面积列名
            
        Returns:
            人口密度
        """
        if area_col and area_col in data.columns:
            # 使用提供的面积列
            area = data[area_col]
        else:
            # 计算几何面积
            if data.crs.is_geographic:
                # 转换到等面积投影
                area_gdf = data.to_crs('EPSG:3857')  # Web Mercator
                area = area_gdf.geometry.area / 1e6  # 转换为km²
            else:
                area = data.geometry.area / 1e6
        
        # 避免除零
        area = area.replace(0, np.nan)
        density = data[population_col] / area
        
        return density.fillna(0)
    
    def create_density_surface(self,
                             point_data: gpd.GeoDataFrame,
                             bounds: Tuple[float, float, float, float],
                             resolution: float = 0.01,
                             population_col: str = 'population',
                             method: str = 'kernel') -> np.ndarray:
        """
        创建人口密度表面
        
        Args:
            point_data: 点数据
            bounds: 边界框
            resolution: 分辨率
            population_col: 人口列名
            method: 插值方法
            
        Returns:
            密度表面数组
        """
        from scipy.spatial import distance
        from scipy.stats import gaussian_kde
        
        if self.verbose:
            self.logger.info(f"Creating density surface using {method} method...")
        
        # 创建网格
        min_x, min_y, max_x, max_y = bounds
        x_coords = np.arange(min_x, max_x + resolution, resolution)
        y_coords = np.arange(min_y, max_y + resolution, resolution)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        if method == 'kernel':
            # 核密度估计
            points = np.vstack([point_data.geometry.x, point_data.geometry.y])
            weights = point_data[population_col].values
            
            # 创建加权核密度估计
            kde = gaussian_kde(points, weights=weights)
            grid_points = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(grid_points).reshape(xx.shape)
            
        elif method == 'idw':
            # 反距离权重插值
            density = np.zeros_like(xx)
            
            for i in range(len(x_coords)):
                for j in range(len(y_coords)):
                    grid_point = np.array([x_coords[i], y_coords[j]])
                    
                    # 计算到所有点的距离
                    distances = np.array([
                        distance.euclidean(grid_point, [pt.x, pt.y])
                        for pt in point_data.geometry
                    ])
                    
                    # 避免除零
                    distances = np.maximum(distances, 1e-10)
                    
                    # IDW插值
                    weights = 1 / (distances ** 2)
                    weighted_values = weights * point_data[population_col].values
                    density[j, i] = np.sum(weighted_values) / np.sum(weights)
        
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return density
    
    def analyze_density_patterns(self,
                               classified_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        分析密度模式
        
        Args:
            classified_data: 分类结果
            
        Returns:
            分析结果
        """
        analysis = {}
        
        # 基本统计
        density_stats = classified_data['population_density'].describe()
        analysis['basic_stats'] = density_stats.to_dict()
        
        # 按类别统计
        if 'density_class' in classified_data.columns:
            class_stats = classified_data.groupby('density_class')['population_density'].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ])
            analysis['class_statistics'] = class_stats.to_dict()
            
            # 类别分布
            class_dist = classified_data['density_class'].value_counts(normalize=True) * 100
            analysis['class_distribution'] = class_dist.to_dict()
        
        # 空间分布指标
        try:
            # 莫兰指数（空间自相关）
            from libpysal.weights import Queen
            from esda import Moran
            
            w = Queen.from_dataframe(classified_data)
            moran = Moran(classified_data['population_density'], w)
            
            analysis['spatial_autocorrelation'] = {
                'moran_i': moran.I,
                'p_value': moran.p_norm,
                'z_score': moran.z_norm
            }
        except ImportError:
            self.logger.warning("libpysal not available for spatial analysis")
        
        # 分布特征
        skewness = stats.skew(classified_data['population_density'])
        kurtosis = stats.kurtosis(classified_data['population_density'])
        
        analysis['distribution_shape'] = {
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        return analysis
    
    def _log_classification_summary(self, 
                                  data: gpd.GeoDataFrame, 
                                  class_col: str) -> None:
        """
        记录分类摘要
        
        Args:
            data: 分类数据
            class_col: 分类列名
        """
        summary = data[class_col].value_counts()
        self.logger.info("Population Density Classification Summary:")
        for category, count in summary.items():
            percentage = (count / len(data)) * 100
            self.logger.info(f"  {category}: {count} ({percentage:.1f}%)")
    
    def compare_classification_methods(self,
                                     data: gpd.GeoDataFrame,
                                     population_col: str = 'total_population') -> pd.DataFrame:
        """
        比较不同分类方法
        
        Args:
            data: 地理数据
            population_col: 人口列名
            
        Returns:
            比较结果
        """
        if self.verbose:
            self.logger.info("Comparing classification methods...")
        
        # 应用不同分类方法
        methods = {}
        
        # 固定阈值法
        fixed_result = self.classify_by_fixed_thresholds(data, population_col, categories=3)
        methods['Fixed_Threshold'] = fixed_result['density_code']
        
        # 百分位数法
        percentile_result = self.classify_by_percentiles(data, population_col, [33, 67])
        methods['Percentile'] = percentile_result['density_code']
        
        # 聚类法
        cluster_result = self.classify_by_clustering(data, population_col, n_clusters=3)
        methods['Clustering'] = cluster_result['density_code']
        
        # 创建比较表
        comparison_df = pd.DataFrame(methods)
        
        # 计算一致性
        consistency_matrix = {}
        method_names = list(methods.keys())
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i <= j:
                    agreement = (comparison_df[method1] == comparison_df[method2]).mean()
                    consistency_matrix[f"{method1}_vs_{method2}"] = agreement
        
        comparison_df.attrs['consistency'] = consistency_matrix
        
        if self.verbose:
            self.logger.info("Method consistency (agreement rates):")
            for pair, agreement in consistency_matrix.items():
                self.logger.info(f"  {pair}: {agreement:.3f}")
        
        return comparison_df