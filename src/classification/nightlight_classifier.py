"""
夜光强度分类器

基于夜间灯光数据的城乡分类系统。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from sklearn.cluster import KMeans
from scipy import ndimage, stats
import warnings


class NightlightClassifier:
    """夜光强度分类器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化夜光分类器
        
        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # 夜光强度阈值（基于不同数据源）
        self.viirs_thresholds = {
            'urban_core': 10.0,     # 城市核心区
            'urban_area': 2.0,      # 城市区域
            'peri_urban': 0.5,      # 城郊区域
            'rural_lit': 0.1,       # 有灯光的农村
            'rural_dark': 0.0       # 无灯光农村
        }
        
        # DMSP-OLS阈值（不同数据源有不同范围）
        self.dmsp_thresholds = {
            'urban_core': 55,
            'urban_area': 30,
            'peri_urban': 10,
            'rural_lit': 3,
            'rural_dark': 0
        }
    
    def load_nightlight_data(self,
                           data_path: Union[str, Path],
                           data_type: str = 'viirs') -> xr.DataArray:
        """
        加载夜光数据
        
        Args:
            data_path: 数据路径
            data_type: 数据类型 ('viirs', 'dmsp')
            
        Returns:
            夜光数据
        """
        if self.verbose:
            self.logger.info(f"Loading {data_type.upper()} nightlight data...")
        
        try:
            # 尝试加载不同格式的数据
            if str(data_path).endswith('.nc'):
                data = xr.open_dataarray(data_path)
            elif str(data_path).endswith('.tif'):
                import rioxarray
                data = rioxarray.open_rasterio(data_path).squeeze()
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            # 数据预处理
            data = self._preprocess_nightlight_data(data, data_type)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load nightlight data: {e}")
            # 返回模拟数据
            return self._create_mock_nightlight_data()
    
    def _preprocess_nightlight_data(self,
                                  data: xr.DataArray,
                                  data_type: str) -> xr.DataArray:
        """
        预处理夜光数据
        
        Args:
            data: 原始夜光数据
            data_type: 数据类型
            
        Returns:
            预处理后的数据
        """
        # 处理无效值
        if data_type == 'viirs':
            # VIIRS数据通常包含负值和异常高值
            data = data.where(data >= 0)  # 移除负值
            data = data.where(data <= 1000)  # 移除异常高值
        elif data_type == 'dmsp':
            # DMSP数据范围通常是0-63
            data = data.where((data >= 0) & (data <= 63))
        
        # 平滑处理（去除噪声）
        if data.ndim == 2:
            smoothed = ndimage.gaussian_filter(data.values, sigma=1)
            data.values = smoothed
        
        return data
    
    def _create_mock_nightlight_data(self) -> xr.DataArray:
        """创建模拟夜光数据"""
        # 创建简单的模拟数据
        lons = np.linspace(-120, -119, 100)
        lats = np.linspace(35, 36, 100)
        
        # 创建具有城市模式的夜光数据
        xx, yy = np.meshgrid(lons, lats)
        
        # 创建多个"城市中心"
        centers = [(-119.5, 35.5), (-119.3, 35.7), (-119.7, 35.3)]
        nightlight = np.zeros_like(xx)
        
        for center_lon, center_lat in centers:
            dist = np.sqrt((xx - center_lon)**2 + (yy - center_lat)**2)
            intensity = 20 * np.exp(-dist * 50)  # 指数衰减
            nightlight += intensity
        
        # 添加噪声
        nightlight += np.random.gamma(0.5, 0.1, nightlight.shape)
        
        data = xr.DataArray(
            nightlight,
            coords={'lat': lats, 'lon': lons},
            dims=['lat', 'lon']
        )
        
        return data
    
    def classify_by_fixed_thresholds(self,
                                   nightlight_data: xr.DataArray,
                                   data_type: str = 'viirs',
                                   categories: int = 3) -> xr.DataArray:
        """
        使用固定阈值进行分类
        
        Args:
            nightlight_data: 夜光数据
            data_type: 数据类型
            categories: 分类数量
            
        Returns:
            分类结果
        """
        if self.verbose:
            self.logger.info(f"Classifying {data_type.upper()} data with fixed thresholds...")
        
        # 选择阈值
        if data_type == 'viirs':
            thresholds = self.viirs_thresholds
        elif data_type == 'dmsp':
            thresholds = self.dmsp_thresholds
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        if categories == 2:
            # 城市 vs 农村
            urban_threshold = thresholds['urban_area']
            classification = xr.where(
                nightlight_data >= urban_threshold, 1, 0
            )
            classification.attrs['labels'] = {0: 'Rural', 1: 'Urban'}
            
        elif categories == 3:
            # 城市、郊区、农村
            urban_threshold = thresholds['urban_area']
            peri_urban_threshold = thresholds['peri_urban']
            
            classification = xr.where(
                nightlight_data >= urban_threshold, 2,
                xr.where(
                    nightlight_data >= peri_urban_threshold, 1, 0
                )
            )
            classification.attrs['labels'] = {0: 'Rural', 1: 'Peri-urban', 2: 'Urban'}
            
        elif categories == 5:
            # 五级分类
            conditions = [
                nightlight_data >= thresholds['urban_core'],
                nightlight_data >= thresholds['urban_area'],
                nightlight_data >= thresholds['peri_urban'],
                nightlight_data >= thresholds['rural_lit']
            ]
            
            classification = xr.where(conditions[0], 4,
                            xr.where(conditions[1], 3,
                            xr.where(conditions[2], 2,
                            xr.where(conditions[3], 1, 0))))
            
            classification.attrs['labels'] = {
                0: 'Rural Dark', 1: 'Rural Lit', 2: 'Peri-urban',
                3: 'Urban', 4: 'Urban Core'
            }
        
        classification.attrs['thresholds'] = thresholds
        classification.attrs['data_type'] = data_type
        
        return classification
    
    def classify_by_clustering(self,
                             nightlight_data: xr.DataArray,
                             n_clusters: int = 3,
                             method: str = 'kmeans') -> xr.DataArray:
        """
        使用聚类进行分类
        
        Args:
            nightlight_data: 夜光数据
            n_clusters: 聚类数量
            method: 聚类方法
            
        Returns:
            分类结果
        """
        if self.verbose:
            self.logger.info(f"Classifying with {method} clustering...")
        
        # 准备数据
        valid_mask = ~np.isnan(nightlight_data.values)
        valid_data = nightlight_data.values[valid_mask]
        
        # 对数变换以处理偏斜分布
        log_data = np.log1p(valid_data).reshape(-1, 1)
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(log_data)
            cluster_centers = clusterer.cluster_centers_.ravel()
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        # 创建分类数组
        classification = np.full_like(nightlight_data.values, -1, dtype=int)
        classification[valid_mask] = cluster_labels
        
        # 按强度排序聚类标签
        center_intensity_pairs = list(zip(range(n_clusters), cluster_centers))
        center_intensity_pairs.sort(key=lambda x: x[1])
        
        # 重新映射标签
        label_mapping = {old: new for new, (old, _) in enumerate(center_intensity_pairs)}
        
        for old_label, new_label in label_mapping.items():
            classification[classification == old_label] = new_label + 100  # 临时偏移
        classification[classification >= 100] -= 100  # 还原
        
        # 创建xarray
        result = xr.DataArray(
            classification,
            coords=nightlight_data.coords,
            dims=nightlight_data.dims
        )
        
        # 添加标签
        if n_clusters == 2:
            labels = {0: 'Rural', 1: 'Urban'}
        elif n_clusters == 3:
            labels = {0: 'Rural', 1: 'Peri-urban', 2: 'Urban'}
        elif n_clusters == 4:
            labels = {0: 'Rural', 1: 'Suburban', 2: 'Urban', 3: 'Urban Core'}
        elif n_clusters == 5:
            labels = {0: 'Rural Dark', 1: 'Rural Lit', 2: 'Peri-urban', 3: 'Urban', 4: 'Urban Core'}
        else:
            labels = {i: f'Cluster_{i}' for i in range(n_clusters)}
        
        result.attrs['labels'] = labels
        result.attrs['cluster_centers'] = {
            labels[i]: np.expm1(center) for i, (_, center) in enumerate(center_intensity_pairs)
        }
        
        return result
    
    def classify_by_otsu_threshold(self,
                                 nightlight_data: xr.DataArray) -> xr.DataArray:
        """
        使用Otsu阈值进行二值分类
        
        Args:
            nightlight_data: 夜光数据
            
        Returns:
            分类结果
        """
        try:
            from skimage.filters import threshold_otsu
        except ImportError:
            self.logger.warning("scikit-image not available, using percentile method")
            threshold = np.percentile(nightlight_data.values[~np.isnan(nightlight_data.values)], 75)
        else:
            valid_data = nightlight_data.values[~np.isnan(nightlight_data.values)]
            threshold = threshold_otsu(valid_data)
        
        if self.verbose:
            self.logger.info(f"Otsu threshold: {threshold:.3f}")
        
        classification = xr.where(nightlight_data >= threshold, 1, 0)
        classification.attrs['labels'] = {0: 'Rural', 1: 'Urban'}
        classification.attrs['otsu_threshold'] = threshold
        
        return classification
    
    def extract_urban_areas(self,
                          nightlight_data: xr.DataArray,
                          min_area: float = 1.0,
                          connectivity: int = 2) -> gpd.GeoDataFrame:
        """
        提取城市区域面要素
        
        Args:
            nightlight_data: 夜光数据
            min_area: 最小面积阈值 (km²)
            connectivity: 连通性
            
        Returns:
            城市区域面要素
        """
        from skimage import measure
        from shapely.geometry import Polygon
        from rasterio.features import shapes
        from rasterio.transform import from_bounds
        
        if self.verbose:
            self.logger.info("Extracting urban areas...")
        
        # 二值化
        if 'labels' not in nightlight_data.attrs:
            # 如果没有分类，使用阈值
            urban_mask = nightlight_data > np.percentile(
                nightlight_data.values[~np.isnan(nightlight_data.values)], 80
            )
        else:
            # 使用已有分类
            urban_mask = nightlight_data >= 1
        
        # 形态学操作（填充小孔洞）
        from scipy.ndimage import binary_fill_holes, binary_opening
        
        cleaned_mask = binary_fill_holes(urban_mask.values)
        cleaned_mask = binary_opening(cleaned_mask, structure=np.ones((3, 3)))
        
        # 计算像素面积
        if nightlight_data.rio.crs and not nightlight_data.rio.crs.is_geographic:
            # 投影坐标系，直接计算面积
            pixel_area_m2 = abs(nightlight_data.rio.resolution()[0] * nightlight_data.rio.resolution()[1])
        else:
            # 地理坐标系，需要近似计算
            lat_center = float(nightlight_data.lat.mean())
            lon_res = float(nightlight_data.lon[1] - nightlight_data.lon[0])
            lat_res = float(nightlight_data.lat[1] - nightlight_data.lat[0])
            
            # 近似计算像素面积
            deg_to_m = 111000  # 1度约等于111km
            pixel_area_m2 = (lon_res * deg_to_m * np.cos(np.radians(lat_center))) * (lat_res * deg_to_m)
        
        pixel_area_km2 = pixel_area_m2 / 1e6
        min_pixels = int(min_area / pixel_area_km2)
        
        # 标记连通区域
        labeled_areas, num_features = ndimage.label(cleaned_mask, structure=np.ones((3, 3)))
        
        # 过滤小区域
        for region_id in range(1, num_features + 1):
            region_size = np.sum(labeled_areas == region_id)
            if region_size < min_pixels:
                labeled_areas[labeled_areas == region_id] = 0
        
        # 重新编号
        labeled_areas = measure.label(labeled_areas > 0, connectivity=connectivity)
        
        # 转换为几何对象
        geometries = []
        properties = []
        
        # 创建仿射变换
        bounds = (
            float(nightlight_data.lon.min()),
            float(nightlight_data.lat.min()),
            float(nightlight_data.lon.max()),
            float(nightlight_data.lat.max())
        )
        
        height, width = labeled_areas.shape
        transform = from_bounds(*bounds, width, height)
        
        # 提取形状
        for geom, value in shapes(labeled_areas.astype(np.int16), transform=transform):
            if value > 0:
                # 计算区域属性
                region_mask = labeled_areas == value
                mean_intensity = float(nightlight_data.values[region_mask].mean())
                max_intensity = float(nightlight_data.values[region_mask].max())
                area_km2 = np.sum(region_mask) * pixel_area_km2
                
                geometries.append(Polygon(geom['coordinates'][0]))
                properties.append({
                    'region_id': int(value),
                    'area_km2': area_km2,
                    'mean_intensity': mean_intensity,
                    'max_intensity': max_intensity,
                    'pixel_count': int(np.sum(region_mask))
                })
        
        # 创建GeoDataFrame
        if geometries:
            urban_areas = gpd.GeoDataFrame(properties, geometry=geometries)
            urban_areas.crs = nightlight_data.rio.crs if hasattr(nightlight_data, 'rio') else 'EPSG:4326'
        else:
            # 空的GeoDataFrame
            urban_areas = gpd.GeoDataFrame(columns=['region_id', 'area_km2', 'mean_intensity', 'max_intensity', 'pixel_count', 'geometry'])
            urban_areas.crs = 'EPSG:4326'
        
        if self.verbose:
            self.logger.info(f"Extracted {len(urban_areas)} urban areas")
        
        return urban_areas
    
    def calculate_light_pollution_index(self,
                                      nightlight_data: xr.DataArray,
                                      reference_level: float = None) -> xr.DataArray:
        """
        计算光污染指数
        
        Args:
            nightlight_data: 夜光数据
            reference_level: 参考光照水平
            
        Returns:
            光污染指数
        """
        if reference_level is None:
            # 使用10th百分位数作为自然背景光照水平
            reference_level = float(np.percentile(
                nightlight_data.values[~np.isnan(nightlight_data.values)], 10
            ))
        
        if self.verbose:
            self.logger.info(f"Calculating light pollution index (reference: {reference_level:.3f})...")
        
        # 光污染指数 = (观测值 - 背景值) / 背景值
        pollution_index = (nightlight_data - reference_level) / reference_level
        pollution_index = xr.where(pollution_index < 0, 0, pollution_index)
        
        pollution_index.attrs['reference_level'] = reference_level
        pollution_index.attrs['description'] = 'Light pollution index relative to natural background'
        
        return pollution_index
    
    def temporal_analysis(self,
                        nightlight_time_series: xr.DataArray,
                        analysis_type: str = 'trend') -> xr.DataArray:
        """
        时间序列分析
        
        Args:
            nightlight_time_series: 时间序列夜光数据
            analysis_type: 分析类型 ('trend', 'seasonality', 'change_detection')
            
        Returns:
            分析结果
        """
        if self.verbose:
            self.logger.info(f"Performing temporal analysis: {analysis_type}")
        
        if analysis_type == 'trend':
            # 线性趋势分析
            time_coords = nightlight_time_series.time
            time_numeric = np.arange(len(time_coords))
            
            def calculate_trend(ts):
                valid_mask = ~np.isnan(ts)
                if np.sum(valid_mask) < 3:
                    return np.nan
                
                x = time_numeric[valid_mask]
                y = ts[valid_mask]
                
                # 线性回归
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                return slope
            
            trend = xr.apply_ufunc(
                calculate_trend,
                nightlight_time_series,
                input_core_dims=[['time']],
                vectorize=True,
                dask='forbidden'
            )
            
            trend.attrs['description'] = 'Linear trend in nightlight intensity'
            trend.attrs['units'] = 'intensity_units_per_time_step'
            
            return trend
        
        elif analysis_type == 'change_detection':
            # 变化检测（比较第一年和最后一年）
            first_year = nightlight_time_series.isel(time=0)
            last_year = nightlight_time_series.isel(time=-1)
            
            change = last_year - first_year
            relative_change = change / (first_year + 1e-10) * 100  # 百分比变化
            
            relative_change.attrs['description'] = 'Relative change in nightlight intensity (%)'
            
            return relative_change
        
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
    
    def compare_with_other_data(self,
                              nightlight_classification: xr.DataArray,
                              comparison_data: gpd.GeoDataFrame,
                              comparison_column: str) -> pd.DataFrame:
        """
        与其他数据源比较
        
        Args:
            nightlight_classification: 夜光分类结果
            comparison_data: 比较数据
            comparison_column: 比较列名
            
        Returns:
            比较结果
        """
        if self.verbose:
            self.logger.info("Comparing with other data sources...")
        
        # 提取夜光分类到点或面要素
        comparison_results = []
        
        for idx, row in comparison_data.iterrows():
            geom = row.geometry
            
            # 提取该几何区域的夜光分类
            if geom.geom_type == 'Point':
                # 点要素：直接提取值
                try:
                    nightlight_value = float(nightlight_classification.sel(
                        lon=geom.x, lat=geom.y, method='nearest'
                    ).values)
                except:
                    nightlight_value = np.nan
            else:
                # 面要素：计算统计值
                try:
                    # 简化实现：使用边界框
                    bounds = geom.bounds
                    subset = nightlight_classification.sel(
                        lon=slice(bounds[0], bounds[2]),
                        lat=slice(bounds[1], bounds[3])
                    )
                    nightlight_value = float(subset.mean().values)
                except:
                    nightlight_value = np.nan
            
            comparison_results.append({
                'feature_id': idx,
                'nightlight_class': nightlight_value,
                'comparison_value': row[comparison_column],
                'geometry_type': geom.geom_type
            })
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # 计算一致性指标
        if not comparison_df.empty:
            # 计算相关性
            valid_mask = ~(comparison_df['nightlight_class'].isna() | comparison_df['comparison_value'].isna())
            if valid_mask.sum() > 0:
                correlation = np.corrcoef(
                    comparison_df.loc[valid_mask, 'nightlight_class'],
                    comparison_df.loc[valid_mask, 'comparison_value']
                )[0, 1]
                
                comparison_df.attrs = {'correlation': correlation}
                
                if self.verbose:
                    self.logger.info(f"Correlation with {comparison_column}: {correlation:.3f}")
        
        return comparison_df