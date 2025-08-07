"""
数据处理工具函数

提供数据处理中常用的工具函数和辅助方法。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import rasterio
from rasterio.features import rasterize
from shapely.geometry import Point, Polygon
import logging


class DataUtils:
    """数据处理工具类"""
    
    @staticmethod
    def setup_logging(log_level: str = "INFO") -> logging.Logger:
        """
        设置日志记录
        
        Args:
            log_level: 日志级别
            
        Returns:
            配置好的logger
        """
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('geoai_platform.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    @staticmethod
    def validate_bounds(bounds: Tuple[float, float, float, float]) -> bool:
        """
        验证边界框有效性
        
        Args:
            bounds: 边界框 (min_lon, min_lat, max_lon, max_lat)
            
        Returns:
            是否有效
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # 检查经纬度范围
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            return False
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            return False
        
        # 检查最小值小于最大值
        if min_lon >= max_lon or min_lat >= max_lat:
            return False
        
        return True
    
    @staticmethod
    def calculate_resolution_from_bounds(bounds: Tuple[float, float, float, float],
                                       target_pixels: int = 1000) -> float:
        """
        根据边界框和目标像素数计算合适的分辨率
        
        Args:
            bounds: 边界框
            target_pixels: 目标像素数
            
        Returns:
            推荐分辨率
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        width = max_lon - min_lon
        height = max_lat - min_lat
        
        # 计算每个维度的分辨率
        res_x = width / np.sqrt(target_pixels)
        res_y = height / np.sqrt(target_pixels)
        
        # 返回较小的分辨率以确保不超过目标像素数
        return min(res_x, res_y)
    
    @staticmethod
    def detect_outliers(data: Union[np.ndarray, pd.Series],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> np.ndarray:
        """
        检测异常值
        
        Args:
            data: 输入数据
            method: 检测方法 ('iqr', 'zscore', 'modified_zscore')
            threshold: 阈值
            
        Returns:
            异常值掩码 (True表示异常值)
        """
        data_array = np.asarray(data)
        
        if method == 'iqr':
            Q1 = np.percentile(data_array, 25)
            Q3 = np.percentile(data_array, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data_array < lower_bound) | (data_array > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data_array, nan_policy='omit'))
            outliers = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = np.median(data_array)
            mad = np.median(np.abs(data_array - median))
            modified_z_scores = 0.6745 * (data_array - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outliers
    
    @staticmethod
    def remove_outliers(data: Union[np.ndarray, pd.Series, xr.DataArray],
                       method: str = 'iqr',
                       threshold: float = 1.5,
                       replace_with: str = 'nan') -> Union[np.ndarray, pd.Series, xr.DataArray]:
        """
        移除或替换异常值
        
        Args:
            data: 输入数据
            method: 检测方法
            threshold: 阈值
            replace_with: 替换方式 ('nan', 'median', 'mean')
            
        Returns:
            处理后的数据
        """
        if isinstance(data, xr.DataArray):
            outliers = DataUtils.detect_outliers(data.values.ravel(), method, threshold)
            outliers = outliers.reshape(data.shape)
            
            if replace_with == 'nan':
                result = data.where(~outliers)
            elif replace_with == 'median':
                result = data.where(~outliers, data.median())
            elif replace_with == 'mean':
                result = data.where(~outliers, data.mean())
            else:
                raise ValueError(f"Unknown replace_with: {replace_with}")
            
            return result
        
        else:
            outliers = DataUtils.detect_outliers(data, method, threshold)
            data_clean = np.array(data, copy=True)
            
            if replace_with == 'nan':
                data_clean[outliers] = np.nan
            elif replace_with == 'median':
                data_clean[outliers] = np.nanmedian(data_clean)
            elif replace_with == 'mean':
                data_clean[outliers] = np.nanmean(data_clean)
            
            if isinstance(data, pd.Series):
                return pd.Series(data_clean, index=data.index)
            else:
                return data_clean
    
    @staticmethod
    def standardize_data(data: Union[np.ndarray, pd.DataFrame, xr.Dataset],
                        method: str = 'zscore',
                        feature_range: Tuple[float, float] = (0, 1)) -> Union[np.ndarray, pd.DataFrame, xr.Dataset]:
        """
        数据标准化
        
        Args:
            data: 输入数据
            method: 标准化方法 ('zscore', 'minmax', 'robust')
            feature_range: MinMax标准化的范围
            
        Returns:
            标准化后的数据
        """
        if isinstance(data, xr.Dataset):
            standardized_vars = {}
            
            for var in data.data_vars:
                values = data[var].values
                
                if method == 'zscore':
                    standardized = (values - np.nanmean(values)) / np.nanstd(values)
                elif method == 'minmax':
                    min_val, max_val = np.nanmin(values), np.nanmax(values)
                    standardized = (values - min_val) / (max_val - min_val)
                    standardized = standardized * (feature_range[1] - feature_range[0]) + feature_range[0]
                elif method == 'robust':
                    median = np.nanmedian(values)
                    mad = np.nanmedian(np.abs(values - median))
                    standardized = (values - median) / mad
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                standardized_vars[var] = (data[var].dims, standardized)
            
            return xr.Dataset(standardized_vars, coords=data.coords)
        
        elif isinstance(data, pd.DataFrame):
            if method == 'zscore':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler(feature_range=feature_range)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            data_scaled = data.copy()
            data_scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])
            
            return data_scaled
        
        else:
            values = np.asarray(data)
            
            if method == 'zscore':
                return (values - np.nanmean(values)) / np.nanstd(values)
            elif method == 'minmax':
                min_val, max_val = np.nanmin(values), np.nanmax(values)
                normalized = (values - min_val) / (max_val - min_val)
                return normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
            elif method == 'robust':
                median = np.nanmedian(values)
                mad = np.nanmedian(np.abs(values - median))
                return (values - median) / mad
    
    @staticmethod
    def spatial_join_with_aggregation(left_gdf: gpd.GeoDataFrame,
                                    right_gdf: gpd.GeoDataFrame,
                                    agg_functions: Dict[str, str],
                                    how: str = 'inner') -> gpd.GeoDataFrame:
        """
        带聚合的空间连接
        
        Args:
            left_gdf: 左侧GeoDataFrame
            right_gdf: 右侧GeoDataFrame
            agg_functions: 聚合函数字典 {column: function}
            how: 连接方式
            
        Returns:
            聚合后的GeoDataFrame
        """
        # 执行空间连接
        joined = gpd.sjoin(left_gdf, right_gdf, how=how, predicate='intersects')
        
        # 按左侧几何聚合
        left_cols = [col for col in left_gdf.columns if col != 'geometry']
        group_cols = left_cols + ['geometry']
        
        # 聚合数据
        agg_data = []
        
        for idx, group in joined.groupby(left_gdf.index):
            row_data = {}
            
            # 保留左侧数据
            for col in left_cols:
                row_data[col] = left_gdf.loc[idx, col]
            
            row_data['geometry'] = left_gdf.loc[idx, 'geometry']
            
            # 聚合右侧数据
            for col, func in agg_functions.items():
                if col in group.columns:
                    if func == 'mean':
                        row_data[col] = group[col].mean()
                    elif func == 'sum':
                        row_data[col] = group[col].sum()
                    elif func == 'count':
                        row_data[col] = len(group)
                    elif func == 'max':
                        row_data[col] = group[col].max()
                    elif func == 'min':
                        row_data[col] = group[col].min()
                    elif func == 'median':
                        row_data[col] = group[col].median()
                    else:
                        row_data[col] = group[col].iloc[0]  # 默认取第一个值
            
            agg_data.append(row_data)
        
        return gpd.GeoDataFrame(agg_data, crs=left_gdf.crs)
    
    @staticmethod
    def calculate_spatial_weights(geometries: gpd.GeoSeries,
                                method: str = 'queen',
                                k: int = 8) -> Dict[int, List[int]]:
        """
        计算空间权重矩阵
        
        Args:
            geometries: 几何对象
            method: 权重方法 ('queen', 'rook', 'knn', 'distance')
            k: KNN的k值或距离阈值
            
        Returns:
            权重字典
        """
        try:
            import libpysal
        except ImportError:
            raise ImportError("libpysal is required for spatial weights calculation")
        
        if method == 'queen':
            w = libpysal.weights.Queen.from_dataframe(gpd.GeoDataFrame(geometry=geometries))
        elif method == 'rook':
            w = libpysal.weights.Rook.from_dataframe(gpd.GeoDataFrame(geometry=geometries))
        elif method == 'knn':
            w = libpysal.weights.KNN.from_dataframe(gpd.GeoDataFrame(geometry=geometries), k=k)
        elif method == 'distance':
            w = libpysal.weights.DistanceBand.from_dataframe(
                gpd.GeoDataFrame(geometry=geometries), threshold=k
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return dict(w.neighbors)
    
    @staticmethod
    def calculate_spatial_autocorrelation(values: np.ndarray,
                                        weights: Dict[int, List[int]],
                                        method: str = 'moran') -> Dict[str, float]:
        """
        计算空间自相关
        
        Args:
            values: 数值数组
            weights: 空间权重
            method: 方法 ('moran', 'geary')
            
        Returns:
            自相关统计量
        """
        try:
            import libpysal
            from esda import Moran, Geary_C
        except ImportError:
            raise ImportError("libpysal and esda are required for spatial autocorrelation")
        
        # 创建权重对象
        w = libpysal.weights.W(weights)
        
        if method == 'moran':
            moran = Moran(values, w)
            return {
                'I': moran.I,
                'p_value': moran.p_norm,
                'z_score': moran.z_norm
            }
        elif method == 'geary':
            geary = Geary_C(values, w)
            return {
                'C': geary.C,
                'p_value': geary.p_norm,
                'z_score': geary.z_norm
            }
        else:
            raise ValueError(f"Unknown method: {method}")
    
    @staticmethod
    def create_buffer_zones(geometries: gpd.GeoSeries,
                          distances: Union[float, List[float]],
                          crs: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        创建缓冲区
        
        Args:
            geometries: 几何对象
            distances: 缓冲距离(米)
            crs: 坐标系
            
        Returns:
            缓冲区GeoDataFrame
        """
        if crs and geometries.crs != crs:
            geometries = geometries.to_crs(crs)
        
        # 转换到适合距离计算的投影坐标系
        if geometries.crs.is_geographic:
            # 使用Web Mercator进行近似计算
            geometries_proj = geometries.to_crs('EPSG:3857')
        else:
            geometries_proj = geometries
        
        if isinstance(distances, (int, float)):
            distances = [distances]
        
        buffer_data = []
        
        for i, geom in enumerate(geometries_proj):
            for dist in distances:
                buffer_geom = geom.buffer(dist)
                buffer_data.append({
                    'original_index': i,
                    'buffer_distance': dist,
                    'geometry': buffer_geom
                })
        
        buffer_gdf = gpd.GeoDataFrame(buffer_data, crs=geometries_proj.crs)
        
        # 转换回原始坐标系
        if geometries.crs != geometries_proj.crs:
            buffer_gdf = buffer_gdf.to_crs(geometries.crs)
        
        return buffer_gdf
    
    @staticmethod
    def rasterize_vector(vector_data: gpd.GeoDataFrame,
                        target_grid: Tuple[np.ndarray, np.ndarray],
                        value_column: str,
                        all_touched: bool = False) -> np.ndarray:
        """
        矢量数据栅格化
        
        Args:
            vector_data: 矢量数据
            target_grid: 目标网格 (lons, lats)
            value_column: 值列名
            all_touched: 是否包含所有接触的像素
            
        Returns:
            栅格化数组
        """
        lons, lats = target_grid
        
        # 创建仿射变换
        transform = rasterio.transform.from_bounds(
            lons.min(), lats.min(), lons.max(), lats.max(),
            lons.shape[1], lons.shape[0]
        )
        
        # 准备几何和值
        shapes = [(geom, value) for geom, value in 
                 zip(vector_data.geometry, vector_data[value_column])]
        
        # 栅格化
        raster = rasterize(
            shapes,
            out_shape=lons.shape,
            transform=transform,
            all_touched=all_touched,
            dtype=np.float64,
            fill=np.nan
        )
        
        return raster
    
    @staticmethod
    def calculate_distance_to_features(points: gpd.GeoDataFrame,
                                     features: gpd.GeoDataFrame,
                                     max_distance: Optional[float] = None) -> np.ndarray:
        """
        计算点到要素的距离
        
        Args:
            points: 点数据
            features: 要素数据
            max_distance: 最大距离限制
            
        Returns:
            距离数组
        """
        from scipy.spatial import cKDTree
        
        # 确保在同一坐标系
        if points.crs != features.crs:
            features = features.to_crs(points.crs)
        
        # 提取坐标
        point_coords = np.column_stack([points.geometry.x, points.geometry.y])
        
        # 对于线和面要素，提取代表点
        if features.geometry.iloc[0].geom_type in ['LineString', 'MultiLineString']:
            feature_coords = np.column_stack([
                features.geometry.centroid.x,
                features.geometry.centroid.y
            ])
        elif features.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
            feature_coords = np.column_stack([
                features.geometry.centroid.x,
                features.geometry.centroid.y
            ])
        else:
            feature_coords = np.column_stack([
                features.geometry.x,
                features.geometry.y
            ])
        
        # 构建KD树
        tree = cKDTree(feature_coords)
        
        # 计算最近距离
        distances, _ = tree.query(point_coords)
        
        if max_distance is not None:
            distances = np.where(distances > max_distance, np.nan, distances)
        
        return distances
    
    @staticmethod
    def create_hexagonal_grid(bounds: Tuple[float, float, float, float],
                            cell_size: float) -> gpd.GeoDataFrame:
        """
        创建六边形网格
        
        Args:
            bounds: 边界框
            cell_size: 网格大小
            
        Returns:
            六边形网格GeoDataFrame
        """
        from shapely.geometry import Polygon
        import math
        
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # 六边形几何参数
        width = cell_size
        height = cell_size * math.sqrt(3) / 2
        
        hexagons = []
        
        y = min_lat
        row = 0
        
        while y < max_lat:
            x_offset = (width * 3/4) * (row % 2)
            x = min_lon + x_offset
            
            while x < max_lon:
                # 创建六边形
                angles = np.linspace(0, 2*np.pi, 7)
                hex_x = x + width/2 * np.cos(angles)
                hex_y = y + height/2 * np.sin(angles)
                
                hexagon = Polygon(zip(hex_x, hex_y))
                hexagons.append({
                    'geometry': hexagon,
                    'row': row,
                    'col': int((x - min_lon) / (width * 3/4)),
                    'center_x': x,
                    'center_y': y
                })
                
                x += width * 3/4
            
            y += height
            row += 1
        
        return gpd.GeoDataFrame(hexagons, crs='EPSG:4326')