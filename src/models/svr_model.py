"""
支持向量回归模型

基于sklearn的SVR实现，用于环境暴露预测。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from .base_model import BaseModel


class SVRPredictor(BaseModel):
    """支持向量回归预测器"""
    
    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 epsilon: float = 0.1,
                 gamma: Union[str, float] = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 shrinking: bool = True,
                 cache_size: float = 200,
                 max_iter: int = -1,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        初始化SVR预测器
        
        Args:
            kernel: 核函数类型 ('linear', 'poly', 'rbf', 'sigmoid')
            C: 正则化参数
            epsilon: epsilon管道的宽度
            gamma: 核函数系数
            degree: 多项式核的度数
            coef0: 核函数中的独立项
            shrinking: 是否使用收缩启发式
            cache_size: 缓存大小(MB)
            max_iter: 最大迭代次数
            random_state: 随机种子
            verbose: 是否输出详细信息
        """
        super().__init__(
            model_name="SVR",
            random_state=random_state,
            verbose=verbose
        )
        
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter
        
        # SVR需要特征标准化
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 构建模型
        self.model = self.build_model()
        
    def build_model(self) -> SVR:
        """
        构建SVR模型
        
        Returns:
            SVR实例
        """
        model = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            max_iter=self.max_iter
        )
        
        return model
    
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            sample_weight: Optional[np.ndarray] = None) -> 'SVRPredictor':
        """
        训练SVR模型
        
        Args:
            X: 特征数据
            y: 目标变量
            sample_weight: 样本权重
            
        Returns:
            训练后的模型实例
        """
        if self.verbose:
            self.logger.info("Training SVR model...")
        
        # 数据预处理（SVR需要标准化）
        X_processed, y_processed = self._prepare_data(X, y, fit_scalers=True)
        
        # 训练模型
        self.model.fit(X_processed, y_processed, sample_weight=sample_weight)
        self.is_fitted = True
        
        # 计算残差标准差
        y_pred = self.model.predict(X_processed)
        residuals = y_processed - y_pred
        self.residual_std_ = np.std(residuals)
        
        if self.verbose:
            self.logger.info(f"SVR training completed. Support vectors: {self.model.n_support_}")
        
        return self
    
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
        
        # 数据预处理
        X_processed, _ = self._prepare_data(X)
        
        # 预测
        predictions = self.model.predict(X_processed)
        
        # 逆变换
        predictions = self._inverse_transform_target(predictions)
        
        return predictions
    
    def predict_with_uncertainty(self,
                                X: Union[np.ndarray, pd.DataFrame],
                                method: str = 'residual',
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带不确定性的预测
        
        Args:
            X: 特征数据
            method: 不确定性估计方法 ('residual', 'bootstrap')
            confidence_level: 置信水平
            
        Returns:
            预测值、下界、上界
        """
        from scipy import stats
        
        predictions = self.predict(X)
        
        if method == 'residual':
            # 使用残差标准差估计不确定性
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            margin = z_score * self.residual_std_
            
            lower_bound = predictions - margin
            upper_bound = predictions + margin
            
        elif method == 'bootstrap':
            # Bootstrap不确定性估计
            from sklearn.utils import resample
            
            X_processed, _ = self._prepare_data(X)
            n_bootstrap = 100
            bootstrap_preds = []
            
            # 获取训练数据
            if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
                X_train, y_train = self.X_train, self.y_train
            else:
                raise ValueError("Training data not available for bootstrap")
            
            for _ in range(n_bootstrap):
                # Bootstrap采样
                X_boot, y_boot = resample(X_train, y_train, random_state=self.random_state)
                
                # 训练模型
                boot_model = SVR(**self.model.get_params())
                boot_model.fit(X_boot, y_boot)
                
                # 预测
                boot_pred = boot_model.predict(X_processed)
                bootstrap_preds.append(boot_pred)
            
            bootstrap_preds = np.array(bootstrap_preds)
            
            # 计算置信区间
            alpha = 1 - confidence_level
            lower_percentile = (alpha/2) * 100
            upper_percentile = (1 - alpha/2) * 100
            
            lower_bound = np.percentile(bootstrap_preds, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_preds, upper_percentile, axis=0)
            
            # 逆变换
            lower_bound = self._inverse_transform_target(lower_bound)
            upper_bound = self._inverse_transform_target(upper_bound)
        
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
        
        return predictions, lower_bound, upper_bound
    
    def hyperparameter_tuning(self,
                             X: Union[np.ndarray, pd.DataFrame],
                             y: Union[np.ndarray, pd.Series],
                             param_grid: Optional[Dict[str, List]] = None,
                             cv: int = 5,
                             scoring: str = 'neg_mean_squared_error',
                             n_jobs: int = -1) -> 'SVRPredictor':
        """
        超参数调优
        
        Args:
            X: 特征数据
            y: 目标变量
            param_grid: 参数网格
            cv: 交叉验证折数
            scoring: 评分方法
            n_jobs: 并行作业数
            
        Returns:
            调优后的模型
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2, 0.5],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        
        if self.verbose:
            self.logger.info("Starting hyperparameter tuning...")
        
        # 数据预处理
        X_processed, y_processed = self._prepare_data(X, y, fit_scalers=True)
        
        # 网格搜索
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1 if self.verbose else 0
        )
        
        grid_search.fit(X_processed, y_processed)
        
        # 更新模型参数
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        if self.verbose:
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best score: {grid_search.best_score_}")
        
        return self
    
    def get_support_vectors_info(self) -> Dict[str, Any]:
        """
        获取支持向量信息
        
        Returns:
            支持向量相关信息
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting support vector info")
        
        info = {
            'n_support_vectors': self.model.n_support_,
            'support_vector_indices': self.model.support_,
            'dual_coef': self.model.dual_coef_,
            'intercept': self.model.intercept_
        }
        
        # 如果是RBF核，还可以获取gamma值
        if self.kernel == 'rbf':
            info['gamma'] = self.model.gamma
        
        return info
    
    def analyze_kernel_performance(self,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.Series],
                                 kernels: List[str] = None,
                                 cv: int = 5) -> Dict[str, Dict[str, float]]:
        """
        分析不同核函数的性能
        
        Args:
            X: 特征数据
            y: 目标变量
            kernels: 要测试的核函数列表
            cv: 交叉验证折数
            
        Returns:
            各核函数的性能评估结果
        """
        from sklearn.model_selection import cross_val_score
        
        if kernels is None:
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        
        X_processed, y_processed = self._prepare_data(X, y, fit_scalers=True)
        
        results = {}
        
        for kernel in kernels:
            if self.verbose:
                self.logger.info(f"Testing kernel: {kernel}")
            
            # 创建模型
            model = SVR(
                kernel=kernel,
                C=self.C,
                epsilon=self.epsilon,
                cache_size=self.cache_size
            )
            
            # 交叉验证
            scores = cross_val_score(
                model, X_processed, y_processed,
                cv=cv, scoring='neg_mean_squared_error'
            )
            
            results[kernel] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'rmse_mean': np.sqrt(-np.mean(scores)),
                'rmse_std': np.sqrt(np.std(scores))
            }
        
        return results
    
    def plot_learning_curve(self,
                           X: Union[np.ndarray, pd.DataFrame],
                           y: Union[np.ndarray, pd.Series],
                           train_sizes: np.ndarray = None,
                           cv: int = 5,
                           figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        绘制学习曲线
        
        Args:
            X: 特征数据
            y: 目标变量
            train_sizes: 训练集大小
            cv: 交叉验证折数
            figsize: 图像大小
        """
        try:
            import matplotlib.pyplot as plt
            from sklearn.model_selection import learning_curve
        except ImportError:
            self.logger.error("matplotlib is required for plotting")
            return
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        X_processed, y_processed = self._prepare_data(X, y, fit_scalers=True)
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_processed, y_processed,
            train_sizes=train_sizes, cv=cv,
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        # 转换为RMSE
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)
        
        train_rmse_mean = np.mean(train_rmse, axis=1)
        train_rmse_std = np.std(train_rmse, axis=1)
        val_rmse_mean = np.mean(val_rmse, axis=1)
        val_rmse_std = np.std(val_rmse, axis=1)
        
        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_rmse_mean, 'o-', color='blue', label='Training RMSE')
        plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std,
                        train_rmse_mean + train_rmse_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_rmse_mean, 'o-', color='red', label='Validation RMSE')
        plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std,
                        val_rmse_mean + val_rmse_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('RMSE')
        plt.title('SVR Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        获取模型摘要信息
        
        Returns:
            模型摘要字典
        """
        summary = super().get_model_summary()
        
        if self.is_fitted:
            support_info = self.get_support_vectors_info()
            summary.update({
                'kernel': self.kernel,
                'n_support_vectors': support_info['n_support_vectors'],
                'C': self.C,
                'epsilon': self.epsilon,
                'gamma': self.gamma
            })
        
        return summary