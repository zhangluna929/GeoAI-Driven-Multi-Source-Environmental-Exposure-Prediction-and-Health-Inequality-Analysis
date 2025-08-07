"""
基础模型类

定义所有预测模型的通用接口和基础功能。
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


class BaseModel(ABC):
    """所有预测模型的基类"""
    
    def __init__(self, 
                 model_name: str,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        初始化基础模型
        
        Args:
            model_name: 模型名称
            random_state: 随机种子
            verbose: 是否输出详细信息
        """
        self.model_name = model_name
        self.random_state = random_state
        self.verbose = verbose
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.target_scaler = None
        self.feature_scaler = None
        
        # 设置日志
        self.logger = logging.getLogger(f"{__name__}.{model_name}")
        if verbose:
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def build_model(self, **kwargs) -> Any:
        """
        构建模型
        
        Returns:
            构建的模型对象
        """
        pass
    
    @abstractmethod
    def fit(self, 
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            **kwargs) -> 'BaseModel':
        """
        训练模型
        
        Args:
            X: 特征数据
            y: 目标变量
            **kwargs: 其他参数
            
        Returns:
            训练后的模型实例
        """
        pass
    
    @abstractmethod
    def predict(self, 
               X: Union[np.ndarray, pd.DataFrame],
               **kwargs) -> np.ndarray:
        """
        进行预测
        
        Args:
            X: 特征数据
            **kwargs: 其他参数
            
        Returns:
            预测结果
        """
        pass
    
    def predict_with_uncertainty(self, 
                                X: Union[np.ndarray, pd.DataFrame],
                                n_estimators: int = 100,
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带不确定性的预测 (默认实现)
        
        Args:
            X: 特征数据
            n_estimators: 集成估计器数量
            confidence_level: 置信水平
            
        Returns:
            预测值、下界、上界
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 默认实现：返回点预测和标准误差估计
        predictions = self.predict(X)
        
        # 简单的不确定性估计
        alpha = 1 - confidence_level
        residual_std = getattr(self, 'residual_std_', 0.1)
        
        margin = stats.norm.ppf(1 - alpha/2) * residual_std
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        
        return predictions, lower_bound, upper_bound
    
    def evaluate(self, 
                X: Union[np.ndarray, pd.DataFrame],
                y_true: Union[np.ndarray, pd.Series],
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X: 特征数据
            y_true: 真实值
            metrics: 评估指标列表
            
        Returns:
            评估结果字典
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2', 'mape']
        
        y_pred = self.predict(X)
        
        results = {}
        
        for metric in metrics:
            if metric == 'rmse':
                results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            elif metric == 'mae':
                results['mae'] = mean_absolute_error(y_true, y_pred)
            elif metric == 'r2':
                results['r2'] = r2_score(y_true, y_pred)
            elif metric == 'mape':
                results['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            elif metric == 'mse':
                results['mse'] = mean_squared_error(y_true, y_pred)
            else:
                self.logger.warning(f"Unknown metric: {metric}")
        
        if self.verbose:
            self.logger.info(f"Model evaluation results: {results}")
        
        return results
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_scaler': self.target_scaler,
            'feature_scaler': self.feature_scaler,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        
        if self.verbose:
            self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> 'BaseModel':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']
        self.target_scaler = model_data.get('target_scaler')
        self.feature_scaler = model_data.get('feature_scaler')
        self.random_state = model_data.get('random_state', 42)
        
        if self.verbose:
            self.logger.info(f"Model loaded from {filepath}")
        
        return self
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            if self.feature_names is not None:
                return dict(zip(self.feature_names, importances))
            else:
                return dict(enumerate(importances))
        else:
            self.logger.warning("Model does not support feature importance")
            return None
    
    def cross_validate(self,
                      X: Union[np.ndarray, pd.DataFrame],
                      y: Union[np.ndarray, pd.Series],
                      cv: int = 5,
                      metrics: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """
        交叉验证
        
        Args:
            X: 特征数据
            y: 目标变量
            cv: 折数
            metrics: 评估指标
            
        Returns:
            交叉验证结果
        """
        from sklearn.model_selection import KFold
        
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2']
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cv_results = {metric: [] for metric in metrics}
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            if self.verbose:
                self.logger.info(f"Cross-validation fold {fold + 1}/{cv}")
            
            # 分割数据
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            
            if isinstance(y, pd.Series):
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练模型
            fold_model = self.__class__(
                model_name=f"{self.model_name}_fold_{fold}",
                random_state=self.random_state,
                verbose=False
            )
            
            fold_model.fit(X_train, y_train)
            
            # 评估
            fold_results = fold_model.evaluate(X_val, y_val, metrics)
            
            for metric in metrics:
                cv_results[metric].append(fold_results[metric])
        
        # 计算统计量
        cv_summary = {}
        for metric in metrics:
            cv_summary[f"{metric}_mean"] = np.mean(cv_results[metric])
            cv_summary[f"{metric}_std"] = np.std(cv_results[metric])
        
        if self.verbose:
            self.logger.info(f"Cross-validation results: {cv_summary}")
        
        return cv_results
    
    def _prepare_data(self, 
                     X: Union[np.ndarray, pd.DataFrame],
                     y: Optional[Union[np.ndarray, pd.Series]] = None,
                     fit_scalers: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        数据预处理
        
        Args:
            X: 特征数据
            y: 目标变量
            fit_scalers: 是否拟合缩放器
            
        Returns:
            处理后的X和y
        """
        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # 特征缩放
        if fit_scalers and self.feature_scaler is not None:
            X_array = self.feature_scaler.fit_transform(X_array)
        elif self.feature_scaler is not None:
            X_array = self.feature_scaler.transform(X_array)
        
        # 目标变量处理
        y_array = None
        if y is not None:
            y_array = np.array(y) if not isinstance(y, np.ndarray) else y
            
            if fit_scalers and self.target_scaler is not None:
                y_array = self.target_scaler.fit_transform(y_array.reshape(-1, 1)).ravel()
            elif self.target_scaler is not None:
                y_array = self.target_scaler.transform(y_array.reshape(-1, 1)).ravel()
        
        return X_array, y_array
    
    def _inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        目标变量逆变换
        
        Args:
            y: 变换后的目标变量
            
        Returns:
            原始尺度的目标变量
        """
        if self.target_scaler is not None:
            return self.target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()
        else:
            return y
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        summary = {
            'model_name': self.model_name,
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else None,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'has_feature_scaler': self.feature_scaler is not None,
            'has_target_scaler': self.target_scaler is not None
        }
        
        # 添加模型特定信息
        if hasattr(self.model, 'get_params'):
            summary['model_params'] = self.model.get_params()
        
        return summary