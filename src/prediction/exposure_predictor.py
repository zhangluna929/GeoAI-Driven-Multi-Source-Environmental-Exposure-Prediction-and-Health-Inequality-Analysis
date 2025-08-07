"""
环境暴露预测基类

定义环境暴露预测的通用接口和基础功能。
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
import warnings


class ExposurePredictor(ABC):
    """环境暴露预测器基类"""
    
    def __init__(self,
                 predictor_name: str,
                 target_variable: str,
                 temporal_resolution: str = 'monthly',
                 spatial_resolution: float = 0.01,
                 prediction_horizon: int = 10,
                 verbose: bool = True):
        """
        初始化暴露预测器
        
        Args:
            predictor_name: 预测器名称
            target_variable: 目标变量名称
            temporal_resolution: 时间分辨率
            spatial_resolution: 空间分辨率 (度)
            prediction_horizon: 预测时间范围 (年)
            verbose: 是否输出详细信息
        """
        self.predictor_name = predictor_name
        self.target_variable = target_variable
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.prediction_horizon = prediction_horizon
        self.verbose = verbose
        
        # 设置日志
        self.logger = logging.getLogger(f"{__name__}.{predictor_name}")
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # 模型和数据存储
        self.model = None
        self.is_fitted = False
        self.training_data = None
        self.feature_importance = None
        
        # 预测结果缓存
        self.prediction_cache = {}
        
    @abstractmethod
    def prepare_training_data(self,
                            environmental_data: Dict[str, xr.Dataset],
                            socioeconomic_data: gpd.GeoDataFrame,
                            target_data: Union[xr.Dataset, gpd.GeoDataFrame],
                            time_range: Tuple[str, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            environmental_data: 环境数据字典
            socioeconomic_data: 社会经济数据
            target_data: 目标变量数据
            time_range: 时间范围
            
        Returns:
            特征数组和目标数组
        """
        pass
    
    @abstractmethod
    def train_model(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   model_type: str = 'ensemble') -> None:
        """
        训练预测模型
        
        Args:
            X: 特征数据
            y: 目标数据
            model_type: 模型类型
        """
        pass
    
    @abstractmethod
    def predict_exposure(self,
                        prediction_data: Dict[str, Any],
                        target_years: List[int],
                        uncertainty: bool = True) -> xr.Dataset:
        """
        预测环境暴露
        
        Args:
            prediction_data: 预测所需数据
            target_years: 目标年份列表
            uncertainty: 是否包含不确定性
            
        Returns:
            预测结果
        """
        pass
    
    def validate_predictions(self,
                           validation_data: Dict[str, Any],
                           metrics: List[str] = ['rmse', 'mae', 'r2']) -> Dict[str, float]:
        """
        验证预测结果
        
        Args:
            validation_data: 验证数据
            metrics: 评估指标
            
        Returns:
            验证结果
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before validation")
        
        # 提取验证特征和目标
        X_val, y_val = self._extract_validation_data(validation_data)
        
        # 进行预测
        y_pred = self.model.predict(X_val)
        
        # 计算指标
        results = {}
        for metric in metrics:
            if metric == 'rmse':
                results['rmse'] = np.sqrt(np.mean((y_val - y_pred)**2))
            elif metric == 'mae':
                results['mae'] = np.mean(np.abs(y_val - y_pred))
            elif metric == 'r2':
                from sklearn.metrics import r2_score
                results['r2'] = r2_score(y_val, y_pred)
            elif metric == 'mape':
                mask = y_val != 0
                if np.any(mask):
                    results['mape'] = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
                else:
                    results['mape'] = np.inf
        
        return results
    
    def _extract_validation_data(self, validation_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从验证数据中提取特征和目标
        
        Args:
            validation_data: 验证数据
            
        Returns:
            特征和目标数组
        """
        # 这里需要根据具体的数据格式实现
        # 简化实现
        if 'X' in validation_data and 'y' in validation_data:
            return validation_data['X'], validation_data['y']
        else:
            raise NotImplementedError("Validation data extraction not implemented")
    
    def create_feature_matrix(self,
                            data_dict: Dict[str, Union[xr.Dataset, gpd.GeoDataFrame]],
                            spatial_grid: Tuple[np.ndarray, np.ndarray],
                            time_coords: Optional[pd.DatetimeIndex] = None) -> np.ndarray:
        """
        创建特征矩阵
        
        Args:
            data_dict: 数据字典
            spatial_grid: 空间网格
            time_coords: 时间坐标
            
        Returns:
            特征矩阵
        """
        features = []
        feature_names = []
        
        lons, lats = spatial_grid
        n_points = lons.size
        
        for data_name, data in data_dict.items():
            if isinstance(data, xr.Dataset):
                # 处理xarray数据
                for var_name in data.data_vars:
                    var_data = data[var_name]
                    
                    # 插值到目标网格
                    if time_coords is not None and 'time' in var_data.dims:
                        # 时空数据
                        interpolated = var_data.interp(
                            lon=lons.ravel(),
                            lat=lats.ravel(),
                            time=time_coords,
                            method='linear'
                        )
                        # 重塑为 (n_samples, n_features)
                        reshaped = interpolated.values.reshape(-1, n_points).T
                        features.append(reshaped)
                        
                        for t_idx, time in enumerate(time_coords):
                            feature_names.append(f"{data_name}_{var_name}_{time.strftime('%Y%m')}")
                    else:
                        # 只有空间数据
                        interpolated = var_data.interp(
                            lon=lons.ravel(),
                            lat=lats.ravel(),
                            method='linear'
                        )
                        features.append(interpolated.values.reshape(-1, 1))
                        feature_names.append(f"{data_name}_{var_name}")
            
            elif isinstance(data, gpd.GeoDataFrame):
                # 处理矢量数据
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    # 空间插值到网格点
                    gridded_values = self._interpolate_vector_to_grid(
                        data, col, lons, lats
                    )
                    features.append(gridded_values.reshape(-1, 1))
                    feature_names.append(f"{data_name}_{col}")
        
        if features:
            feature_matrix = np.concatenate(features, axis=1)
            self.feature_names = feature_names
            return feature_matrix
        else:
            raise ValueError("No features could be extracted from the provided data")
    
    def _interpolate_vector_to_grid(self,
                                  gdf: gpd.GeoDataFrame,
                                  column: str,
                                  lons: np.ndarray,
                                  lats: np.ndarray) -> np.ndarray:
        """
        将矢量数据插值到网格
        
        Args:
            gdf: 矢量数据
            column: 数据列
            lons: 经度网格
            lats: 纬度网格
            
        Returns:
            插值结果
        """
        from scipy.spatial import cKDTree
        from shapely.geometry import Point
        
        # 提取质心坐标和值
        centroids = gdf.geometry.centroid
        coords = np.array([[pt.x, pt.y] for pt in centroids])
        values = gdf[column].values
        
        # 创建KD树
        tree = cKDTree(coords)
        
        # 网格点坐标
        grid_coords = np.column_stack([lons.ravel(), lats.ravel()])
        
        # 最近邻插值
        distances, indices = tree.query(grid_coords)
        interpolated_values = values[indices]
        
        return interpolated_values
    
    def add_temporal_features(self,
                            data: np.ndarray,
                            time_coords: pd.DatetimeIndex) -> np.ndarray:
        """
        添加时间特征
        
        Args:
            data: 原始数据
            time_coords: 时间坐标
            
        Returns:
            包含时间特征的数据
        """
        temporal_features = []
        
        # 年份
        years = time_coords.year.values
        temporal_features.append(years.reshape(-1, 1))
        
        # 月份
        months = time_coords.month.values
        temporal_features.append(months.reshape(-1, 1))
        
        # 季节
        seasons = ((time_coords.month - 1) // 3 + 1).values
        temporal_features.append(seasons.reshape(-1, 1))
        
        # 趋势项（线性时间）
        time_trend = np.arange(len(time_coords)).reshape(-1, 1)
        temporal_features.append(time_trend)
        
        # 合并所有特征
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        temporal_matrix = np.concatenate(temporal_features, axis=1)
        
        # 广播到空间维度
        n_spatial = data.shape[0] if data.ndim > 1 else 1
        n_temporal = len(time_coords)
        
        if data.shape[0] == n_temporal:
            # 时间序列数据，需要扩展到空间
            expanded_temporal = np.tile(temporal_matrix, (n_spatial, 1))
            expanded_data = np.repeat(data, n_spatial, axis=0)
            return np.concatenate([expanded_data, expanded_temporal], axis=1)
        else:
            # 空间数据，添加时间特征
            return np.concatenate([data, temporal_matrix], axis=1)
    
    def calculate_prediction_confidence(self,
                                      predictions: np.ndarray,
                                      uncertainty_estimates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算预测置信度
        
        Args:
            predictions: 预测值
            uncertainty_estimates: 不确定性估计
            
        Returns:
            置信度分数
        """
        if uncertainty_estimates is not None:
            # 基于不确定性的置信度
            max_uncertainty = np.max(uncertainty_estimates)
            confidence = 1 - (uncertainty_estimates / max_uncertainty)
        else:
            # 基于预测值的置信度（简化）
            pred_std = np.std(predictions)
            if pred_std > 0:
                normalized_pred = np.abs(predictions - np.mean(predictions)) / pred_std
                confidence = np.exp(-normalized_pred / 2)  # 高斯权重
            else:
                confidence = np.ones_like(predictions)
        
        return np.clip(confidence, 0, 1)
    
    def save_predictions(self,
                        predictions: xr.Dataset,
                        output_path: Union[str, Path],
                        format: str = 'netcdf') -> None:
        """
        保存预测结果
        
        Args:
            predictions: 预测结果
            output_path: 输出路径
            format: 保存格式
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'netcdf':
            predictions.to_netcdf(output_path)
        elif format == 'zarr':
            predictions.to_zarr(output_path, mode='w')
        elif format == 'csv':
            # 转换为DataFrame保存
            df = predictions.to_dataframe().reset_index()
            df.to_csv(output_path.with_suffix('.csv'), index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.verbose:
            self.logger.info(f"Predictions saved to {output_path}")
    
    def load_predictions(self,
                        input_path: Union[str, Path],
                        format: str = 'netcdf') -> xr.Dataset:
        """
        加载预测结果
        
        Args:
            input_path: 输入路径
            format: 数据格式
            
        Returns:
            预测结果
        """
        if format == 'netcdf':
            predictions = xr.open_dataset(input_path)
        elif format == 'zarr':
            predictions = xr.open_zarr(input_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.verbose:
            self.logger.info(f"Predictions loaded from {input_path}")
        
        return predictions
    
    def get_predictor_summary(self) -> Dict[str, Any]:
        """
        获取预测器摘要信息
        
        Returns:
            摘要信息字典
        """
        summary = {
            'predictor_name': self.predictor_name,
            'target_variable': self.target_variable,
            'temporal_resolution': self.temporal_resolution,
            'spatial_resolution': self.spatial_resolution,
            'prediction_horizon': self.prediction_horizon,
            'is_fitted': self.is_fitted,
            'feature_count': len(getattr(self, 'feature_names', [])),
            'feature_names': getattr(self, 'feature_names', None)
        }
        
        if self.model and hasattr(self.model, 'get_model_summary'):
            summary['model_summary'] = self.model.get_model_summary()
        
        return summary
    
    def clear_cache(self) -> None:
        """清除预测缓存"""
        self.prediction_cache.clear()
        if self.verbose:
            self.logger.info("Prediction cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        获取缓存信息
        
        Returns:
            缓存信息
        """
        return {
            'cache_size': len(self.prediction_cache),
            'cached_keys': list(self.prediction_cache.keys())
        }