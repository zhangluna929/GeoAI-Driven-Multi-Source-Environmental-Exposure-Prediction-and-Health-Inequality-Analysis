"""
遥感数据处理模块

支持多种卫星数据源的获取和预处理，包括Landsat、Sentinel、MODIS等。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import ee  # Google Earth Engine
from pathlib import Path


class RemoteSensingProcessor:
    """遥感数据处理器"""
    
    def __init__(self, 
                 gee_service_account: Optional[str] = None,
                 data_dir: Union[str, Path] = "data/raw/remote_sensing"):
        """
        初始化遥感数据处理器
        
        Args:
            gee_service_account: Google Earth Engine服务账户路径
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化Google Earth Engine
        try:
            if gee_service_account:
                credentials = ee.ServiceAccountCredentials(
                    email=None, 
                    key_file=gee_service_account
                )
                ee.Initialize(credentials)
            else:
                ee.Initialize()
            self.gee_available = True
        except Exception as e:
            print(f"Warning: Google Earth Engine initialization failed: {e}")
            self.gee_available = False
    
    def get_landsat_data(self, 
                        bounds: Tuple[float, float, float, float],
                        start_date: str,
                        end_date: str,
                        cloud_threshold: float = 20) -> xr.Dataset:
        """
        获取Landsat数据
        
        Args:
            bounds: 边界框 (min_lon, min_lat, max_lon, max_lat)
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            cloud_threshold: 云覆盖阈值
            
        Returns:
            Landsat数据集
        """
        if not self.gee_available:
            raise RuntimeError("Google Earth Engine not available")
        
        # 定义研究区域
        geometry = ee.Geometry.Rectangle(bounds)
        
        # 获取Landsat 8数据
        collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUD_COVER', cloud_threshold)))
        
        # 计算中位数合成
        image = collection.median().clip(geometry)
        
        # 选择波段并计算指数
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
        image = image.select(bands)
        
        # 计算NDVI
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        
        # 计算NDBI (Normalized Difference Built-up Index)
        ndbi = image.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
        
        # 计算裸土指数
        bare_soil = image.normalizedDifference(['SR_B6', 'SR_B7']).rename('BARE_SOIL')
        
        # 合并所有波段和指数
        final_image = image.addBands([ndvi, ndbi, bare_soil])
        
        return self._ee_to_xarray(final_image, geometry, scale=30)
    
    def get_sentinel_data(self,
                         bounds: Tuple[float, float, float, float],
                         start_date: str,
                         end_date: str,
                         cloud_threshold: float = 20) -> xr.Dataset:
        """
        获取Sentinel-2数据
        
        Args:
            bounds: 边界框
            start_date: 开始日期
            end_date: 结束日期
            cloud_threshold: 云覆盖阈值
            
        Returns:
            Sentinel-2数据集
        """
        if not self.gee_available:
            raise RuntimeError("Google Earth Engine not available")
        
        geometry = ee.Geometry.Rectangle(bounds)
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)))
        
        image = collection.median().clip(geometry)
        
        # 选择相关波段
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        image = image.select(bands)
        
        # 计算植被指数
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8'),
                'RED': image.select('B4'),
                'BLUE': image.select('B2')
            }
        ).rename('EVI')
        
        final_image = image.addBands([ndvi, evi])
        
        return self._ee_to_xarray(final_image, geometry, scale=10)
    
    def get_modis_data(self,
                       bounds: Tuple[float, float, float, float],
                       start_date: str,
                       end_date: str,
                       product: str = 'MOD13Q1') -> xr.Dataset:
        """
        获取MODIS数据
        
        Args:
            bounds: 边界框
            start_date: 开始日期
            end_date: 结束日期
            product: MODIS产品名称
            
        Returns:
            MODIS数据集
        """
        if not self.gee_available:
            raise RuntimeError("Google Earth Engine not available")
        
        geometry = ee.Geometry.Rectangle(bounds)
        
        # 根据产品选择数据集
        if product == 'MOD13Q1':  # NDVI产品
            collection = ee.ImageCollection('MODIS/006/MOD13Q1')
            scale = 250
        elif product == 'MOD11A1':  # 地表温度产品
            collection = ee.ImageCollection('MODIS/006/MOD11A1')
            scale = 1000
        elif product == 'MCD12Q1':  # 土地覆盖产品
            collection = ee.ImageCollection('MODIS/006/MCD12Q1')
            scale = 500
        elif product == 'MCD19A2':  # AOD产品
            collection = ee.ImageCollection('MODIS/061/MCD19A2_GRANULES')
            scale = 1000
        elif product == 'MOD04_3K':  # 另一种AOD产品
            collection = ee.ImageCollection('MODIS/006/MOD04_3K')
            scale = 3000
        else:
            raise ValueError(f"Unsupported MODIS product: {product}")
        
        collection = (collection
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date))
        
        image = collection.median().clip(geometry)
        
        return self._ee_to_xarray(image, geometry, scale=scale)
    
    def get_aod_data(self,
                     bounds: Tuple[float, float, float, float],
                     start_date: str,
                     end_date: str,
                     product: str = 'MCD19A2',
                     band: str = 'Optical_Depth_047') -> xr.Dataset:
        """
        获取NASA AOD（气溶胶光学厚度）数据

        Args:
            bounds: 边界框 (min_lon, min_lat, max_lon, max_lat)
            start_date: 开始日期
            end_date: 结束日期
            product: AOD产品 ('MCD19A2' 或 'MOD04_3K')
            band: 要提取的AOD波段

        Returns:
            AOD数据集
        """
        # MCD19A2_GRANULES 和 MOD04_3K 都包含AOD信息
        if product not in ['MCD19A2', 'MOD04_3K']:
            raise ValueError("Unsupported AOD product")
        
        collection_id = 'MODIS/061/MCD19A2_GRANULES' if product == 'MCD19A2' else 'MODIS/006/MOD04_3K'
        scale = 1000 if product == 'MCD19A2' else 3000
        
        geometry = ee.Geometry.Rectangle(bounds)
        collection = (ee.ImageCollection(collection_id)
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date))
        
        # 取时间平均值
        image = collection.mean().clip(geometry).select(band)
        
        return self._ee_to_xarray(image, geometry, scale=scale)

    def get_gpm_precipitation(self,
                              bounds: Tuple[float, float, float, float],
                              start_date: str,
                              end_date: str,
                              product: str = 'IMERG') -> xr.Dataset:
        """
        获取NASA GPM降水数据（IMERG）

        Args:
            bounds: 边界框
            start_date: 开始日期
            end_date: 结束日期
            product: 产品类型 ('IMERG' 支持日/半小时)

        Returns:
            降水数据集
        """
        if product != 'IMERG':
            raise ValueError("Currently only IMERG product is supported")
        
        geometry = ee.Geometry.Rectangle(bounds)
        collection = (ee.ImageCollection('NASA/GPM_L3/IMERG_V06')
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date))
        
        # 计算平均降水强度
        image = collection.mean().clip(geometry).select('precipitation')
        
        return self._ee_to_xarray(image, geometry, scale=10000)

    def calculate_green_space_metrics(self, 
                                    ndvi_data: xr.Dataset,
                                    threshold: float = 0.3) -> Dict[str, float]:
        """
        计算绿地空间指标
        
        Args:
            ndvi_data: NDVI数据
            threshold: 绿地NDVI阈值
            
        Returns:
            绿地指标字典
        """
        ndvi = ndvi_data.NDVI
        
        # 绿地覆盖率
        green_cover = (ndvi > threshold).sum() / ndvi.size * 100
        
        # 平均NDVI
        mean_ndvi = float(ndvi.mean())
        
        # NDVI标准差
        ndvi_std = float(ndvi.std())
        
        # 高质量绿地比例 (NDVI > 0.5)
        high_quality_green = (ndvi > 0.5).sum() / ndvi.size * 100
        
        return {
            'green_cover_percentage': float(green_cover),
            'mean_ndvi': mean_ndvi,
            'ndvi_std': ndvi_std,
            'high_quality_green_percentage': float(high_quality_green)
        }
    
    def calculate_urban_heat_island(self, 
                                  lst_data: xr.Dataset,
                                  urban_mask: Optional[xr.Dataset] = None) -> Dict[str, float]:
        """
        计算城市热岛效应指标
        
        Args:
            lst_data: 地表温度数据
            urban_mask: 城市区域掩模
            
        Returns:
            热岛效应指标
        """
        lst = lst_data.LST_Day_1km if 'LST_Day_1km' in lst_data else lst_data.LST
        
        if urban_mask is not None:
            urban_temp = lst.where(urban_mask == 1)
            rural_temp = lst.where(urban_mask == 0)
            
            uhi_intensity = float(urban_temp.mean() - rural_temp.mean())
            
            return {
                'uhi_intensity': uhi_intensity,
                'urban_mean_temp': float(urban_temp.mean()),
                'rural_mean_temp': float(rural_temp.mean()),
                'urban_temp_std': float(urban_temp.std()),
                'rural_temp_std': float(rural_temp.std())
            }
        else:
            return {
                'mean_temp': float(lst.mean()),
                'temp_std': float(lst.std()),
                'min_temp': float(lst.min()),
                'max_temp': float(lst.max())
            }
    
    def _ee_to_xarray(self, 
                     image: ee.Image, 
                     geometry: ee.Geometry, 
                     scale: int = 30) -> xr.Dataset:
        """
        将Earth Engine图像转换为xarray数据集
        
        Args:
            image: Earth Engine图像
            geometry: 几何边界
            scale: 像素尺度
            
        Returns:
            xarray数据集
        """
        # 获取图像信息
        proj = image.projection()
        
        # 导出为numpy数组
        try:
            # 这里需要实现具体的转换逻辑
            # 由于GEE的限制，实际应用中可能需要使用geemap等工具
            pass
        except Exception as e:
            print(f"Error converting EE image to xarray: {e}")
            # 返回模拟数据用于测试
            return self._create_mock_dataset()
    
    def _create_mock_dataset(self) -> xr.Dataset:
        """创建模拟数据集用于测试"""
        lons = np.linspace(-120, -119, 100)
        lats = np.linspace(35, 36, 100)
        
        data = np.random.rand(100, 100)
        
        ds = xr.Dataset({
            'NDVI': (['lat', 'lon'], data),
        }, coords={
            'lat': lats,
            'lon': lons
        })
        
        return ds
    
    def preprocess_time_series(self, 
                              data_list: List[xr.Dataset],
                              dates: List[datetime]) -> xr.Dataset:
        """
        预处理时间序列数据
        
        Args:
            data_list: 数据集列表
            dates: 对应日期列表
            
        Returns:
            时间序列数据集
        """
        # 添加时间维度
        for i, (data, date) in enumerate(zip(data_list, dates)):
            data_list[i] = data.expand_dims('time').assign_coords(time=[date])
        
        # 合并时间序列
        combined = xr.concat(data_list, dim='time')
        
        # 填充缺失值
        combined = combined.interpolate_na(dim='time', method='linear')
        
        return combined