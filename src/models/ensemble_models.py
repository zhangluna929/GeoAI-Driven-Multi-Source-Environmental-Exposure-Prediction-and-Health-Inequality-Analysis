"""
集成模型

包含LSTM-XGBoost混合模型等集成学习方法。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from .base_model import BaseModel
from .neural_networks import LSTMPredictor
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. LSTM-XGBoost model will not work.")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class LSTMXGBoostPredictor(BaseModel):
    """LSTM-XGBoost混合预测器"""
    
    def __init__(self,
                 # LSTM参数
                 lstm_units: List[int] = [64, 32],
                 lstm_dropout: float = 0.2,
                 sequence_length: int = 10,
                 lstm_epochs: int = 50,
                 
                 # XGBoost参数
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 
                 # 集成参数
                 ensemble_method: str = 'weighted',
                 lstm_weight: float = 0.6,
                 
                 random_state: int = 42,
                 verbose: bool = True):
        """
        初始化LSTM-XGBoost混合预测器
        
        Args:
            lstm_units: LSTM层神经元数量
            lstm_dropout: LSTM dropout比率
            sequence_length: 序列长度
            lstm_epochs: LSTM训练轮数
            n_estimators: XGBoost树的数量
            max_depth: XGBoost最大深度
            learning_rate: XGBoost学习率
            subsample: XGBoost子采样比率
            colsample_bytree: XGBoost特征采样比率
            ensemble_method: 集成方法 ('weighted', 'stacking', 'adaptive')
            lstm_weight: LSTM权重（weighted方法使用）
            random_state: 随机种子
            verbose: 是否输出详细信息
        """
        if not (TF_AVAILABLE and XGBOOST_AVAILABLE):
            raise ImportError("Both TensorFlow and XGBoost are required for LSTM-XGBoost model")
        
        super().__init__(
            model_name="LSTM-XGBoost",
            random_state=random_state,
            verbose=verbose
        )
        
        # LSTM参数
        self.lstm_units = lstm_units
        self.lstm_dropout = lstm_dropout
        self.sequence_length = sequence_length
        self.lstm_epochs = lstm_epochs
        
        # XGBoost参数
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        
        # 集成参数
        self.ensemble_method = ensemble_method
        self.lstm_weight = lstm_weight
        
        # 初始化子模型
        self.lstm_model = None
        self.xgb_model = None
        self.meta_model = None  # 用于stacking方法
        
    def build_model(self) -> Tuple[Any, Any]:
        """
        构建LSTM和XGBoost模型
        
        Returns:
            LSTM和XGBoost模型
        """
        # 构建LSTM模型
        lstm_model = LSTMPredictor(
            lstm_units=self.lstm_units,
            dropout_rate=self.lstm_dropout,
            sequence_length=self.sequence_length,
            epochs=self.lstm_epochs,
            random_state=self.random_state,
            verbose=False
        )
        
        # 构建XGBoost模型
        xgb_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            verbosity=1 if self.verbose else 0
        )
        
        return lstm_model, xgb_model
    
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            validation_data: Optional[Tuple] = None) -> 'LSTMXGBoostPredictor':
        """
        训练LSTM-XGBoost混合模型
        
        Args:
            X: 特征数据
            y: 目标变量
            validation_data: 验证数据
            
        Returns:
            训练后的模型实例
        """
        if self.verbose:
            self.logger.info("Training LSTM-XGBoost hybrid model...")
        
        # 构建子模型
        self.lstm_model, self.xgb_model = self.build_model()
        
        # 准备数据
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            self.feature_names = X.columns.tolist()
        else:
            X_array = np.array(X)
        
        y_array = np.array(y)
        
        # 训练LSTM模型
        if self.verbose:
            self.logger.info("Training LSTM component...")
        
        self.lstm_model.fit(X_array, y_array, validation_data)
        
        # 获取LSTM预测作为XGBoost的额外特征
        lstm_predictions = self.lstm_model.predict(X_array)
        
        # 为XGBoost准备特征（原始特征 + LSTM预测）
        X_enhanced = self._enhance_features(X_array, lstm_predictions)
        
        # 训练XGBoost模型
        if self.verbose:
            self.logger.info("Training XGBoost component...")
        
        self.xgb_model.fit(X_enhanced, y_array)
        
        # 如果使用stacking方法，训练元学习器
        if self.ensemble_method == 'stacking':
            self._train_meta_model(X_array, y_array)
        
        self.is_fitted = True
        
        if self.verbose:
            self.logger.info("LSTM-XGBoost training completed")
        
        return self
    
    def _enhance_features(self,
                         X: np.ndarray,
                         lstm_pred: np.ndarray) -> np.ndarray:
        """
        增强特征（添加LSTM预测）
        
        Args:
            X: 原始特征
            lstm_pred: LSTM预测结果
            
        Returns:
            增强后的特征
        """
        # 确保LSTM预测和原始特征样本数匹配
        min_samples = min(len(X), len(lstm_pred))
        X_trimmed = X[-min_samples:]
        lstm_pred_trimmed = lstm_pred[-min_samples:]
        
        # 合并特征
        if len(lstm_pred_trimmed.shape) == 1:
            lstm_pred_trimmed = lstm_pred_trimmed.reshape(-1, 1)
        
        X_enhanced = np.column_stack([X_trimmed, lstm_pred_trimmed])
        
        return X_enhanced
    
    def _train_meta_model(self,
                         X: np.ndarray,
                         y: np.ndarray) -> None:
        """
        训练元学习器（用于stacking）
        
        Args:
            X: 特征数据
            y: 目标变量
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_predict
        
        # 使用交叉验证获取基模型预测
        lstm_cv_pred = self._get_lstm_cv_predictions(X, y)
        
        # 获取XGBoost交叉验证预测
        X_enhanced = self._enhance_features(X, lstm_cv_pred)
        xgb_cv_pred = cross_val_predict(
            self.xgb_model, X_enhanced, y, cv=5
        )
        
        # 训练元模型
        meta_features = np.column_stack([lstm_cv_pred, xgb_cv_pred])
        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_features, y)
    
    def _get_lstm_cv_predictions(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                cv: int = 5) -> np.ndarray:
        """
        获取LSTM的交叉验证预测
        
        Args:
            X: 特征数据
            y: 目标变量
            cv: 交叉验证折数
            
        Returns:
            交叉验证预测结果
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        cv_predictions = np.zeros(len(y))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练LSTM
            fold_lstm = LSTMPredictor(
                lstm_units=self.lstm_units,
                dropout_rate=self.lstm_dropout,
                sequence_length=self.sequence_length,
                epochs=self.lstm_epochs,
                random_state=self.random_state,
                verbose=False
            )
            
            fold_lstm.fit(X_train, y_train)
            val_pred = fold_lstm.predict(X_val)
            
            cv_predictions[val_idx] = val_pred
        
        return cv_predictions
    
    def predict(self,
               X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # 获取子模型预测
        lstm_pred = self.lstm_model.predict(X_array)
        
        # 为XGBoost准备增强特征
        X_enhanced = self._enhance_features(X_array, lstm_pred)
        xgb_pred = self.xgb_model.predict(X_enhanced)
        
        # 集成预测
        if self.ensemble_method == 'weighted':
            final_pred = (self.lstm_weight * lstm_pred + 
                         (1 - self.lstm_weight) * xgb_pred)
        
        elif self.ensemble_method == 'stacking':
            if self.meta_model is None:
                raise ValueError("Meta model not trained for stacking")
            
            meta_features = np.column_stack([lstm_pred, xgb_pred])
            final_pred = self.meta_model.predict(meta_features)
        
        elif self.ensemble_method == 'adaptive':
            # 自适应权重（基于预测一致性）
            final_pred = self._adaptive_ensemble(lstm_pred, xgb_pred)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return final_pred
    
    def _adaptive_ensemble(self,
                          lstm_pred: np.ndarray,
                          xgb_pred: np.ndarray) -> np.ndarray:
        """
        自适应集成（根据预测一致性调整权重）
        
        Args:
            lstm_pred: LSTM预测
            xgb_pred: XGBoost预测
            
        Returns:
            自适应集成预测
        """
        # 计算预测差异
        pred_diff = np.abs(lstm_pred - xgb_pred)
        
        # 计算自适应权重（差异越小，平均权重越大）
        max_diff = np.max(pred_diff)
        if max_diff > 0:
            consistency_score = 1 - (pred_diff / max_diff)
        else:
            consistency_score = np.ones_like(pred_diff)
        
        # 当一致性高时，使用平均权重；一致性低时，偏向历史表现更好的模型
        base_weight = 0.5 + 0.1 * consistency_score  # LSTM基础权重
        lstm_weights = np.clip(base_weight, 0.2, 0.8)
        
        final_pred = lstm_weights * lstm_pred + (1 - lstm_weights) * xgb_pred
        
        return final_pred
    
    def predict_with_uncertainty(self,
                                X: Union[np.ndarray, pd.DataFrame],
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带不确定性的预测
        
        Args:
            X: 特征数据
            confidence_level: 置信水平
            
        Returns:
            预测值、下界、上界
        """
        # 获取子模型的不确定性预测
        lstm_pred, lstm_lower, lstm_upper = self.lstm_model.predict_with_uncertainty(
            X, confidence_level=confidence_level
        )
        
        # XGBoost不确定性估计（使用quantile regression或bootstrap）
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        X_enhanced = self._enhance_features(X_array, lstm_pred)
        xgb_pred = self.xgb_model.predict(X_enhanced)
        
        # 简化的XGBoost不确定性估计
        # 实际应用中可以使用quantile regression或bootstrap
        xgb_std = np.std(xgb_pred) * 0.1  # 简化估计
        from scipy import stats
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        margin = z_score * xgb_std
        
        xgb_lower = xgb_pred - margin
        xgb_upper = xgb_pred + margin
        
        # 集成不确定性
        if self.ensemble_method == 'weighted':
            final_pred = (self.lstm_weight * lstm_pred + 
                         (1 - self.lstm_weight) * xgb_pred)
            final_lower = (self.lstm_weight * lstm_lower + 
                          (1 - self.lstm_weight) * xgb_lower)
            final_upper = (self.lstm_weight * lstm_upper + 
                          (1 - self.lstm_weight) * xgb_upper)
        else:
            # 其他方法的简化处理
            final_pred = self.predict(X)
            ensemble_std = np.sqrt(
                self.lstm_weight * (lstm_upper - lstm_lower)**2 + 
                (1 - self.lstm_weight) * (xgb_upper - xgb_lower)**2
            ) / (2 * z_score)
            
            margin = z_score * ensemble_std
            final_lower = final_pred - margin
            final_upper = final_pred + margin
        
        return final_pred, final_lower, final_upper
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        获取特征重要性（分别来自LSTM和XGBoost）
        
        Returns:
            特征重要性字典
        """
        importance_dict = {}
        
        # XGBoost特征重要性
        if hasattr(self.xgb_model, 'feature_importances_'):
            xgb_importance = self.xgb_model.feature_importances_
            
            # 原始特征重要性
            n_original_features = len(self.feature_names) if self.feature_names else len(xgb_importance) - 1
            original_importance = xgb_importance[:n_original_features]
            
            if self.feature_names:
                importance_dict['xgb_original'] = dict(zip(
                    self.feature_names, original_importance
                ))
            else:
                importance_dict['xgb_original'] = dict(enumerate(original_importance))
            
            # LSTM预测作为特征的重要性
            importance_dict['lstm_as_feature'] = xgb_importance[-1] if len(xgb_importance) > n_original_features else 0
        
        return importance_dict
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        summary = super().get_model_summary()
        
        if self.is_fitted:
            summary.update({
                'ensemble_method': self.ensemble_method,
                'lstm_weight': self.lstm_weight,
                'lstm_units': self.lstm_units,
                'sequence_length': self.sequence_length,
                'xgb_n_estimators': self.n_estimators,
                'xgb_max_depth': self.max_depth
            })
            
            # 添加子模型信息
            if self.lstm_model:
                summary['lstm_summary'] = self.lstm_model.get_model_summary()
            
            if self.xgb_model:
                summary['xgb_params'] = self.xgb_model.get_params()
        
        return summary