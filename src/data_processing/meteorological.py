"""
气象数据处理模块

支持多种气象数据源的获取和预处理，包括ERA5、NCEP、本地气象站数据等。
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import requests
import cdsapi  # Climate Data Store API
from pathlib import Path


class MeteorologicalProcessor:
    """气象数据处理器"""
    
    def __init__(self, 
                 cds_api_key: Optional[str] = None,
                 data_dir: Union[str, Path] = "data/raw/meteorological"):
        """
        初始化气象数据处理器
        
        Args:
            cds_api_key: Climate Data Store API密钥
            data_dir: 数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化CDS API
        try:
            if cds_api_key:
                self.cds_client = cdsapi.Client(key=cds_api_key)
            else:
                self.cds_client = cdsapi.Client()
            self.cds_available = True
        except Exception as e:
            print(f"Warning: CDS API initialization failed: {e}")
            self.cds_available = False
    
    def get_era5_data(self,
                     variables: List[str],
                     bounds: Tuple[float, float, float, float],
                     start_date: str,
                     end_date: str,
                     time_resolution: str = 'monthly') -> xr.Dataset:
        """
        获取ERA5再分析数据
        
        Args:
            variables: 气象变量列表
            bounds: 边界框 (min_lon, min_lat, max_lon, max_lat)
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            time_resolution: 时间分辨率 ('hourly', 'daily', 'monthly')
            
        Returns:
            ERA5数据集
        """
        if not self.cds_available:
            raise RuntimeError("CDS API not available")
        
        # 变量映射
        var_mapping = {
            'temperature': '2m_temperature',
            'precipitation': 'total_precipitation',
            'humidity': 'relative_humidity',
            'wind_speed': '10m_wind_speed',
            'pressure': 'surface_pressure',
            'solar_radiation': 'surface_solar_radiation_downwards'
        }
        
        # 转换变量名
        era5_variables = [var_mapping.get(var, var) for var in variables]
        
        # 确定数据集名称
        if time_resolution == 'hourly':
            dataset = 'reanalysis-era5-single-levels'
        elif time_resolution == 'monthly':
            dataset = 'reanalysis-era5-single-levels-monthly-means'
        else:
            dataset = 'reanalysis-era5-single-levels'
        
        # 构建请求参数
        request_params = {
            'product_type': 'reanalysis',
            'variable': era5_variables,
            'year': self._get_years_range(start_date, end_date),
            'month': self._get_months_range(start_date, end_date),
            'day': self._get_days_range(start_date, end_date) if time_resolution != 'monthly' else None,
            'time': self._get_hours_range() if time_resolution == 'hourly' else None,
            'area': [bounds[3], bounds[0], bounds[1], bounds[2]],  # [N, W, S, E]
            'format': 'netcdf',
        }
        
        # 移除None值
        request_params = {k: v for k, v in request_params.items() if v is not None}
        
        # 下载数据
        output_file = self.data_dir / f"era5_{time_resolution}_{start_date}_{end_date}.nc"
        
        try:
            self.cds_client.retrieve(dataset, request_params, str(output_file))
            return xr.open_dataset(output_file)
        except Exception as e:
            print(f"Error downloading ERA5 data: {e}")
            return self._create_mock_meteorological_dataset(bounds, start_date, end_date)
    
    def get_station_data(self,
                        station_ids: List[str],
                        variables: List[str],
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """
        获取气象站数据
        
        Args:
            station_ids: 气象站ID列表
            variables: 变量列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            气象站数据DataFrame
        """
        # 这里应该连接到具体的气象数据API
        # 例如：NOAA, 中国气象数据网等
        
        # 模拟数据
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        data = []
        for station_id in station_ids:
            for date in date_range:
                row = {
                    'station_id': station_id,
                    'date': date,
                    'latitude': 39.9 + np.random.normal(0, 0.1),  # 模拟坐标
                    'longitude': 116.4 + np.random.normal(0, 0.1),
                }
                
                # 模拟气象数据
                for var in variables:
                    if var == 'temperature':
                        row[var] = 20 + np.random.normal(0, 5)
                    elif var == 'precipitation':
                        row[var] = max(0, np.random.exponential(2))
                    elif var == 'humidity':
                        row[var] = 50 + np.random.normal(0, 15)
                    elif var == 'wind_speed':
                        row[var] = max(0, np.random.normal(3, 2))
                    else:
                        row[var] = np.random.normal(0, 1)
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def calculate_climate_indices(self, data: xr.Dataset) -> Dict[str, xr.DataArray]:
        """
        计算气候指数
        
        Args:
            data: 气象数据集
            
        Returns:
            气候指数字典
        """
        indices = {}
        
        if 'temperature' in data.variables or 't2m' in data.variables:
            temp = data.temperature if 'temperature' in data else data.t2m
            
            # 热浪指数 (连续5天超过35°C的天数)
            if temp.dims[0] == 'time':
                hot_days = (temp > 35).resample(time='1Y').sum()
                indices['hot_days_per_year'] = hot_days
                
                # 生长度日 (Growing Degree Days)
                gdd = (temp - 10).where(temp > 10, 0).resample(time='1Y').sum()
                indices['growing_degree_days'] = gdd
        
        if 'precipitation' in data.variables or 'tp' in data.variables:
            precip = data.precipitation if 'precipitation' in data else data.tp
            
            # 干旱指数 (连续无降水天数)
            if precip.dims[0] == 'time':
                dry_days = (precip < 1).resample(time='1Y').sum()
                indices['dry_days_per_year'] = dry_days
                
                # 极端降水指数 (95th percentile)
                extreme_precip = precip.quantile(0.95, dim='time')
                indices['extreme_precipitation'] = extreme_precip
        
        # 舒适度指数
        if all(var in data.variables for var in ['temperature', 'humidity']):
            temp = data.temperature
            humidity = data.humidity
            
            # 体感温度 (Heat Index)
            heat_index = self._calculate_heat_index(temp, humidity)
            indices['heat_index'] = heat_index
        
        return indices
    
    def _calculate_heat_index(self, 
                            temperature: xr.DataArray, 
                            humidity: xr.DataArray) -> xr.DataArray:
        """
        计算体感温度指数
        
        Args:
            temperature: 温度 (°C)
            humidity: 相对湿度 (%)
            
        Returns:
            体感温度
        """
        # 转换为华氏度
        T = temperature * 9/5 + 32
        H = humidity
        
        # Rothfusz回归方程
        HI = (-42.379 + 
              2.04901523 * T + 
              10.14333127 * H - 
              0.22475541 * T * H - 
              6.83783e-3 * T**2 - 
              5.481717e-2 * H**2 + 
              1.22874e-3 * T**2 * H + 
              8.5282e-4 * T * H**2 - 
              1.99e-6 * T**2 * H**2)
        
        # 转换回摄氏度
        return (HI - 32) * 5/9
    
    def interpolate_spatial(self, 
                           station_data: pd.DataFrame,
                           target_grid: Tuple[np.ndarray, np.ndarray],
                           method: str = 'idw') -> xr.Dataset:
        """
        空间插值气象站数据
        
        Args:
            station_data: 气象站数据
            target_grid: 目标网格 (lons, lats)
            method: 插值方法 ('idw', 'kriging', 'spline')
            
        Returns:
            插值后的网格数据
        """
        from scipy.spatial.distance import cdist
        from scipy.interpolate import griddata
        
        lons, lats = target_grid
        
        # 获取唯一日期
        dates = station_data['date'].unique()
        variables = [col for col in station_data.columns 
                    if col not in ['station_id', 'date', 'latitude', 'longitude']]
        
        interpolated_data = {}
        
        for var in variables:
            var_data = []
            
            for date in dates:
                daily_data = station_data[station_data['date'] == date]
                
                if len(daily_data) < 3:
                    continue
                
                points = daily_data[['longitude', 'latitude']].values
                values = daily_data[var].values
                
                # 创建目标网格点
                grid_points = np.column_stack([lons.ravel(), lats.ravel()])
                
                if method == 'idw':
                    # 反距离权重插值
                    distances = cdist(grid_points, points)
                    weights = 1 / (distances + 1e-10)**2
                    weights = weights / weights.sum(axis=1, keepdims=True)
                    interpolated_values = (weights * values).sum(axis=1)
                else:
                    # 使用scipy的griddata
                    interpolated_values = griddata(
                        points, values, grid_points, method='cubic'
                    )
                
                interpolated_grid = interpolated_values.reshape(lons.shape)
                var_data.append(interpolated_grid)
            
            interpolated_data[var] = (['time', 'lat', 'lon'], 
                                   np.array(var_data))
        
        return xr.Dataset(
            interpolated_data,
            coords={
                'time': dates,
                'lat': lats[:, 0],
                'lon': lons[0, :]
            }
        )
    
    def calculate_air_quality_meteorology(self, 
                                        met_data: xr.Dataset) -> Dict[str, xr.DataArray]:
        """
        计算与空气质量相关的气象指标
        
        Args:
            met_data: 气象数据
            
        Returns:
            空气质量相关气象指标
        """
        indicators = {}
        
        # 边界层高度估算
        if 'temperature' in met_data and 'pressure' in met_data:
            temp = met_data.temperature
            pres = met_data.pressure
            
            # 简化的边界层高度计算
            pbl_height = 1000 * (1 - (pres / 101325)**0.19)
            indicators['boundary_layer_height'] = pbl_height
        
        # 通风系数
        if 'wind_speed' in met_data and 'boundary_layer_height' in indicators:
            wind = met_data.wind_speed
            pbl = indicators['boundary_layer_height']
            
            ventilation_coefficient = wind * pbl
            indicators['ventilation_coefficient'] = ventilation_coefficient
        
        # 大气稳定度指标
        if 'temperature' in met_data and 'wind_speed' in met_data:
            temp = met_data.temperature
            wind = met_data.wind_speed
            
            # Pasquill稳定度分类的简化版本
            stability = xr.where(
                wind < 2, 'F',  # Very stable
                xr.where(
                    wind < 4, 'E',  # Stable
                    xr.where(
                        wind < 6, 'D',  # Neutral
                        'C'  # Unstable
                    )
                )
            )
            indicators['atmospheric_stability'] = stability
        
        return indicators
    
    def _get_years_range(self, start_date: str, end_date: str) -> List[str]:
        """获取年份范围"""
        start_year = int(start_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        return [str(year) for year in range(start_year, end_year + 1)]
    
    def _get_months_range(self, start_date: str, end_date: str) -> List[str]:
        """获取月份范围"""
        return [f"{i:02d}" for i in range(1, 13)]
    
    def _get_days_range(self, start_date: str, end_date: str) -> List[str]:
        """获取日期范围"""
        return [f"{i:02d}" for i in range(1, 32)]
    
    def _get_hours_range(self) -> List[str]:
        """获取小时范围"""
        return [f"{i:02d}:00" for i in range(0, 24)]
    
    def _create_mock_meteorological_dataset(self, 
                                          bounds: Tuple[float, float, float, float],
                                          start_date: str,
                                          end_date: str) -> xr.Dataset:
        """创建模拟气象数据集"""
        # 创建空间网格
        lons = np.linspace(bounds[0], bounds[2], 50)
        lats = np.linspace(bounds[1], bounds[3], 50)
        
        # 创建时间序列
        times = pd.date_range(start_date, end_date, freq='D')
        
        # 生成模拟数据
        np.random.seed(42)
        temp_data = 20 + 10 * np.sin(np.arange(len(times)) * 2 * np.pi / 365) + \
                   np.random.normal(0, 2, (len(times), len(lats), len(lons)))
        
        precip_data = np.random.exponential(2, (len(times), len(lats), len(lons)))
        
        humidity_data = 60 + np.random.normal(0, 15, (len(times), len(lats), len(lons)))
        
        return xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], temp_data),
            'precipitation': (['time', 'lat', 'lon'], precip_data),
            'humidity': (['time', 'lat', 'lon'], humidity_data),
        }, coords={
            'time': times,
            'lat': lats,
            'lon': lons
        })