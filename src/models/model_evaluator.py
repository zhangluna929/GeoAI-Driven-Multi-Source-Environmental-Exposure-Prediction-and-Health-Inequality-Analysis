"""
模型评估器

提供全面的模型性能评估和比较功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, 
                 results_dir: Union[str, Path] = "results/evaluation",
                 verbose: bool = True):
        """
        初始化模型评估器
        
        Args:
            results_dir: 结果保存目录
            verbose: 是否输出详细信息
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
    
    def evaluate_single_model(self,
                             model,
                             X_test: Union[np.ndarray, pd.DataFrame],
                             y_test: Union[np.ndarray, pd.Series],
                             metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        评估单个模型
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            metrics: 评估指标列表
            
        Returns:
            评估结果字典
        """
        if metrics is None:
            metrics = ['rmse', 'mae', 'r2', 'mape']
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 确保形状一致
        if len(y_pred.shape) > 1:
            y_pred = y_pred.ravel()
        
        y_test_array = np.array(y_test)
        if len(y_test_array.shape) > 1:
            y_test_array = y_test_array.ravel()
        
        # 计算指标
        results = {}
        
        for metric in metrics:
            try:
                if metric == 'rmse':
                    results['rmse'] = np.sqrt(mean_squared_error(y_test_array, y_pred))
                elif metric == 'mae':
                    results['mae'] = mean_absolute_error(y_test_array, y_pred)
                elif metric == 'r2':
                    results['r2'] = r2_score(y_test_array, y_pred)
                elif metric == 'mape':
                    # 避免除零错误
                    mask = y_test_array != 0
                    if np.any(mask):
                        results['mape'] = np.mean(
                            np.abs((y_test_array[mask] - y_pred[mask]) / y_test_array[mask])
                        ) * 100
                    else:
                        results['mape'] = np.inf
                elif metric == 'mse':
                    results['mse'] = mean_squared_error(y_test_array, y_pred)
                elif metric == 'nrmse':
                    # 标准化RMSE
                    rmse = np.sqrt(mean_squared_error(y_test_array, y_pred))
                    y_range = np.max(y_test_array) - np.min(y_test_array)
                    results['nrmse'] = rmse / y_range if y_range > 0 else np.inf
                elif metric == 'bias':
                    results['bias'] = np.mean(y_pred - y_test_array)
                elif metric == 'correlation':
                    results['correlation'] = np.corrcoef(y_test_array, y_pred)[0, 1]
                else:
                    self.logger.warning(f"Unknown metric: {metric}")
            except Exception as e:
                self.logger.error(f"Error calculating {metric}: {e}")
                results[metric] = np.nan
        
        return results
    
    def compare_models(self,
                      models: Dict[str, Any],
                      X_test: Union[np.ndarray, pd.DataFrame],
                      y_test: Union[np.ndarray, pd.Series],
                      metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        比较多个模型
        
        Args:
            models: 模型字典 {model_name: model}
            X_test: 测试特征
            y_test: 测试目标
            metrics: 评估指标列表
            
        Returns:
            比较结果DataFrame
        """
        if self.verbose:
            self.logger.info("Comparing models...")
        
        comparison_results = []
        
        for model_name, model in models.items():
            if self.verbose:
                self.logger.info(f"Evaluating {model_name}...")
            
            try:
                results = self.evaluate_single_model(model, X_test, y_test, metrics)
                results['model'] = model_name
                comparison_results.append(results)
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if len(comparison_df) > 0:
            # 设置model为索引
            comparison_df.set_index('model', inplace=True)
            
            # 排序（按R²降序，RMSE升序）
            if 'r2' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('r2', ascending=False)
            elif 'rmse' in comparison_df.columns:
                comparison_df = comparison_df.sort_values('rmse', ascending=True)
        
        return comparison_df
    
    def cross_validation_evaluation(self,
                                   models: Dict[str, Any],
                                   X: Union[np.ndarray, pd.DataFrame],
                                   y: Union[np.ndarray, pd.Series],
                                   cv: int = 5,
                                   scoring: str = 'neg_mean_squared_error',
                                   time_series: bool = False) -> pd.DataFrame:
        """
        交叉验证评估
        
        Args:
            models: 模型字典
            X: 特征数据
            y: 目标变量
            cv: 交叉验证折数
            scoring: 评分方法
            time_series: 是否为时间序列数据
            
        Returns:
            交叉验证结果DataFrame
        """
        if self.verbose:
            self.logger.info("Performing cross-validation evaluation...")
        
        # 选择交叉验证策略
        if time_series:
            cv_strategy = TimeSeriesSplit(n_splits=cv)
        else:
            cv_strategy = cv
        
        cv_results = []
        
        for model_name, model in models.items():
            if self.verbose:
                self.logger.info(f"Cross-validating {model_name}...")
            
            try:
                scores = cross_val_score(
                    model, X, y, 
                    cv=cv_strategy,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                # 转换为正值（如果是负值评分）
                if scoring.startswith('neg_'):
                    scores = -scores
                
                cv_results.append({
                    'model': model_name,
                    'cv_mean': np.mean(scores),
                    'cv_std': np.std(scores),
                    'cv_min': np.min(scores),
                    'cv_max': np.max(scores)
                })
                
            except Exception as e:
                self.logger.error(f"Error in cross-validation for {model_name}: {e}")
                continue
        
        cv_df = pd.DataFrame(cv_results)
        if len(cv_df) > 0:
            cv_df.set_index('model', inplace=True)
            cv_df = cv_df.sort_values('cv_mean', ascending=False)
        
        return cv_df
    
    def residual_analysis(self,
                         model,
                         X_test: Union[np.ndarray, pd.DataFrame],
                         y_test: Union[np.ndarray, pd.Series],
                         model_name: str = "Model") -> Dict[str, Any]:
        """
        残差分析
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            y_test: 测试目标
            model_name: 模型名称
            
        Returns:
            残差分析结果
        """
        y_pred = model.predict(X_test)
        
        # 确保形状一致
        if len(y_pred.shape) > 1:
            y_pred = y_pred.ravel()
        y_test_array = np.array(y_test).ravel()
        
        # 计算残差
        residuals = y_test_array - y_pred
        
        # 残差统计
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'q25': np.percentile(residuals, 25),
            'q50': np.percentile(residuals, 50),
            'q75': np.percentile(residuals, 75)
        }
        
        # 正态性检验
        try:
            from scipy import stats
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # 限制样本数量
            ks_stat, ks_p = stats.kstest(residuals, 'norm')
            
            normality_tests = {
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'ks_stat': ks_stat,
                'ks_p': ks_p
            }
        except ImportError:
            normality_tests = {}
        
        # 自相关检验（如果数据按时间排序）
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_stat, lb_p = acorr_ljungbox(residuals, lags=10, return_df=False)
            autocorr_test = {
                'ljung_box_stat': lb_stat,
                'ljung_box_p': lb_p
            }
        except ImportError:
            autocorr_test = {}
        
        return {
            'residuals': residuals,
            'predictions': y_pred,
            'true_values': y_test_array,
            'statistics': residual_stats,
            'normality_tests': normality_tests,
            'autocorr_test': autocorr_test
        }
    
    def plot_model_comparison(self,
                             comparison_df: pd.DataFrame,
                             figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        绘制模型比较图
        
        Args:
            comparison_df: 模型比较结果
            figsize: 图像大小
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("matplotlib and seaborn are required for plotting")
            return
        
        metrics = [col for col in comparison_df.columns if col != 'model']
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            self.logger.error("No metrics to plot")
            return
        
        # 计算子图布局
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # 条形图
            bars = ax.bar(comparison_df.index, comparison_df[metric])
            ax.set_title(f'{metric.upper()}')
            ax.set_ylabel(metric.upper())
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # 添加数值标签
            for bar, value in zip(bars, comparison_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # 删除多余的子图
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self,
                      residual_analysis: Dict[str, Any],
                      model_name: str = "Model",
                      figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        绘制残差分析图
        
        Args:
            residual_analysis: 残差分析结果
            model_name: 模型名称
            figsize: 图像大小
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("matplotlib and seaborn are required for plotting")
            return
        
        residuals = residual_analysis['residuals']
        predictions = residual_analysis['predictions']
        true_values = residual_analysis['true_values']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
        
        # 1. 预测值vs真实值
        axes[0, 0].scatter(true_values, predictions, alpha=0.6)
        axes[0, 0].plot([true_values.min(), true_values.max()], 
                       [true_values.min(), true_values.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Predictions')
        axes[0, 0].set_title('Predictions vs True Values')
        
        # 2. 残差vs预测值
        axes[0, 1].scatter(predictions, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predictions')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals vs Predictions')
        
        # 3. 残差直方图
        axes[0, 2].hist(residuals, bins=50, density=True, alpha=0.7)
        axes[0, 2].set_xlabel('Residuals')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Residual Distribution')
        
        # 4. Q-Q图
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
        except ImportError:
            axes[1, 0].text(0.5, 0.5, 'SciPy required for Q-Q plot', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 5. 残差时间序列（如果有时间信息）
        axes[1, 1].plot(residuals)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Time Series')
        
        # 6. 残差箱线图
        axes[1, 2].boxplot(residuals)
        axes[1, 2].set_ylabel('Residuals')
        axes[1, 2].set_title('Residual Box Plot')
        
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self,
                                  models: Dict[str, Any],
                                  X_test: Union[np.ndarray, pd.DataFrame],
                                  y_test: Union[np.ndarray, pd.Series],
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        生成综合评估报告
        
        Args:
            models: 模型字典
            X_test: 测试特征
            y_test: 测试目标
            save_path: 报告保存路径
            
        Returns:
            评估报告字典
        """
        if self.verbose:
            self.logger.info("Generating evaluation report...")
        
        report = {
            'model_comparison': self.compare_models(models, X_test, y_test),
            'residual_analyses': {},
            'feature_importances': {},
            'model_summaries': {}
        }
        
        # 详细分析每个模型
        for model_name, model in models.items():
            if self.verbose:
                self.logger.info(f"Detailed analysis for {model_name}...")
            
            # 残差分析
            try:
                report['residual_analyses'][model_name] = self.residual_analysis(
                    model, X_test, y_test, model_name
                )
            except Exception as e:
                self.logger.error(f"Error in residual analysis for {model_name}: {e}")
            
            # 特征重要性
            try:
                if hasattr(model, 'get_feature_importance'):
                    report['feature_importances'][model_name] = model.get_feature_importance()
                elif hasattr(model, 'feature_importances_'):
                    report['feature_importances'][model_name] = model.feature_importances_
            except Exception as e:
                self.logger.error(f"Error getting feature importance for {model_name}: {e}")
            
            # 模型摘要
            try:
                if hasattr(model, 'get_model_summary'):
                    report['model_summaries'][model_name] = model.get_model_summary()
            except Exception as e:
                self.logger.error(f"Error getting model summary for {model_name}: {e}")
        
        # 保存报告
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _save_report(self, 
                    report: Dict[str, Any], 
                    save_path: str) -> None:
        """
        保存评估报告
        
        Args:
            report: 评估报告
            save_path: 保存路径
        """
        import json
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换不可序列化的对象
        serializable_report = {}
        
        for key, value in report.items():
            if key == 'model_comparison' and isinstance(value, pd.DataFrame):
                serializable_report[key] = value.to_dict()
            elif key == 'residual_analyses':
                serializable_report[key] = {}
                for model_name, analysis in value.items():
                    serializable_report[key][model_name] = {
                        'statistics': analysis.get('statistics', {}),
                        'normality_tests': analysis.get('normality_tests', {}),
                        'autocorr_test': analysis.get('autocorr_test', {})
                    }
            else:
                serializable_report[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        if self.verbose:
            self.logger.info(f"Report saved to {save_path}")