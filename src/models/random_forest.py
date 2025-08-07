# -*- coding: utf-8 -*-
"""
随机森林模型实现
用sklearn的RandomForestRegressor包装了一下，主要是为了环境暴露预测
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel


class RandomForestPredictor(BaseModel):
    # 随机森林预测器，主要用来预测PM2.5什么的
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float] = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 verbose: bool = True):
        # 初始化随机森林，参数基本就是sklearn那一套
        super().__init__(
            model_name="RandomForest",
            random_state=random_state,
            verbose=verbose
        )
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        
        # 构建模型
        self.model = self.build_model()
        
    def build_model(self) -> RandomForestRegressor:
        """
        构建随机森林模型
        
        Returns:
            RandomForestRegressor实例
        """
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        return model
    
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            sample_weight: Optional[np.ndarray] = None) -> 'RandomForestPredictor':
        # 训练随机森林，没什么特别的
        if self.verbose:
            self.logger.info("开始训练随机森林...")
        
        # 先处理一下数据
        X_processed, y_processed = self._prepare_data(X, y, fit_scalers=True)
        
        # 直接调用sklearn的fit
        self.model.fit(X_processed, y_processed, sample_weight=sample_weight)
        self.is_fitted = True
        
        # 如果开启了OOB，看看分数怎么样
        if self.oob_score and hasattr(self.model, 'oob_score_'):
            if self.verbose:
                self.logger.info(f"OOB分数: {self.model.oob_score_:.4f}")
        
        # 计算残差标准差，后面估计不确定性要用
        y_pred = self.model.predict(X_processed)
        residuals = y_processed - y_pred
        self.residual_std_ = np.std(residuals)
        
        if self.verbose:
            self.logger.info("Random Forest training completed")
        
        return self
    
    def predict(self,
               X: Union[np.ndarray, pd.DataFrame],
               return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        进行预测
        
        Args:
            X: 特征数据
            return_std: 是否返回预测的标准差
            
        Returns:
            预测结果，如果return_std=True则还返回标准差
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 数据预处理
        X_processed, _ = self._prepare_data(X)
        
        # 预测
        predictions = self.model.predict(X_processed)
        
        # 逆变换
        predictions = self._inverse_transform_target(predictions)
        
        if return_std:
            # 计算预测标准差
            if self.bootstrap:
                tree_predictions = np.array([
                    tree.predict(X_processed) for tree in self.model.estimators_
                ])
                pred_std = np.std(tree_predictions, axis=0)
                pred_std = self._inverse_transform_target(pred_std)
                return predictions, pred_std
            else:
                # 如果没有bootstrap，使用残差标准差作为近似
                pred_std = np.full(predictions.shape, self.residual_std_)
                return predictions, pred_std
        
        return predictions
    
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
        from scipy import stats
        
        predictions, pred_std = self.predict(X, return_std=True)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        margin = z_score * pred_std
        lower_bound = predictions - margin
        upper_bound = predictions + margin
        
        return predictions, lower_bound, upper_bound
    
    def get_feature_importance(self,
                              importance_type: str = 'impurity') -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 ('impurity', 'permutation')
            
        Returns:
            特征重要性字典
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if importance_type == 'impurity':
            importances = self.model.feature_importances_
        elif importance_type == 'permutation':
            from sklearn.inspection import permutation_importance
            # 需要验证数据，这里使用训练数据作为示例
            # 实际应用中应该使用验证集
            result = permutation_importance(
                self.model, self.X_train, self.y_train,
                n_repeats=10, random_state=self.random_state
            )
            importances = result.importances_mean
        else:
            raise ValueError(f"Unknown importance type: {importance_type}")
        
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        else:
            return dict(enumerate(importances))
    
    def plot_feature_importance(self,
                               top_n: int = 20,
                               figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前N个重要特征
            figsize: 图像大小
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("matplotlib and seaborn are required for plotting")
            return
        
        importance_dict = self.get_feature_importance()
        
        # 排序并选择前N个
        sorted_importance = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_importance)
        
        plt.figure(figsize=figsize)
        sns.barplot(x=list(importances), y=list(features))
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    
    def get_tree_predictions(self,
                           X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        获取所有树的预测结果
        
        Args:
            X: 特征数据
            
        Returns:
            所有树的预测结果 (n_trees, n_samples)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_processed, _ = self._prepare_data(X)
        
        tree_predictions = np.array([
            tree.predict(X_processed) for tree in self.model.estimators_
        ])
        
        return tree_predictions
    
    def analyze_prediction_variance(self,
                                  X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        分析预测方差
        
        Args:
            X: 特征数据
            
        Returns:
            方差分析结果
        """
        tree_predictions = self.get_tree_predictions(X)
        
        mean_pred = np.mean(tree_predictions, axis=0)
        var_pred = np.var(tree_predictions, axis=0)
        std_pred = np.std(tree_predictions, axis=0)
        
        # 计算预测一致性（变异系数）
        cv = std_pred / (mean_pred + 1e-8)  # 避免除零
        
        return {
            'mean_prediction': mean_pred,
            'prediction_variance': var_pred,
            'prediction_std': std_pred,
            'coefficient_of_variation': cv
        }
    
    def partial_dependence_analysis(self,
                                  X: Union[np.ndarray, pd.DataFrame],
                                  features: Union[int, str, List[Union[int, str]]],
                                  grid_resolution: int = 100) -> Dict[str, Any]:
        """
        部分依赖分析
        
        Args:
            X: 特征数据
            features: 要分析的特征
            grid_resolution: 网格分辨率
            
        Returns:
            部分依赖分析结果
        """
        try:
            from sklearn.inspection import partial_dependence
        except ImportError:
            self.logger.error("sklearn >= 0.22 is required for partial dependence")
            return {}
        
        if isinstance(features, (int, str)):
            features = [features]
        
        # 转换特征名为索引
        feature_indices = []
        for feature in features:
            if isinstance(feature, str):
                if self.feature_names is not None:
                    feature_indices.append(self.feature_names.index(feature))
                else:
                    raise ValueError("Feature names not available")
            else:
                feature_indices.append(feature)
        
        X_processed, _ = self._prepare_data(X)
        
        # 计算部分依赖
        pd_results = partial_dependence(
            self.model, X_processed, feature_indices,
            grid_resolution=grid_resolution
        )
        
        return {
            'partial_dependence': pd_results['average'],
            'grid_values': pd_results['values'],
            'feature_indices': feature_indices
        }
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        获取模型复杂度信息
        
        Returns:
            模型复杂度指标
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting complexity")
        
        total_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
        total_leaves = sum(tree.tree_.n_leaves for tree in self.model.estimators_)
        max_depth_actual = max(tree.tree_.max_depth for tree in self.model.estimators_)
        avg_depth = np.mean([tree.tree_.max_depth for tree in self.model.estimators_])
        
        return {
            'n_estimators': self.n_estimators,
            'total_nodes': total_nodes,
            'total_leaves': total_leaves,
            'max_depth_actual': max_depth_actual,
            'average_depth': avg_depth,
            'nodes_per_tree': total_nodes / self.n_estimators,
            'leaves_per_tree': total_leaves / self.n_estimators
        }