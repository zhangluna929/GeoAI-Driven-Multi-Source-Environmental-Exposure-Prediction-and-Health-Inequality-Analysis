"""
模型选择器

提供自动化的模型选择和超参数优化功能。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from pathlib import Path
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    ParameterGrid, ParameterSampler
)
from sklearn.metrics import make_scorer, mean_squared_error
import warnings
import time


class ModelSelector:
    """模型选择器"""
    
    def __init__(self,
                 scoring: str = 'neg_mean_squared_error',
                 cv: int = 5,
                 n_jobs: int = -1,
                 verbose: bool = True,
                 random_state: int = 42):
        """
        初始化模型选择器
        
        Args:
            scoring: 评分方法
            cv: 交叉验证折数
            n_jobs: 并行作业数
            verbose: 是否输出详细信息
            random_state: 随机种子
        """
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # 存储结果
        self.selection_results = {}
        self.best_models = {}
    
    def select_best_model(self,
                         models: Dict[str, Any],
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         param_grids: Optional[Dict[str, Dict]] = None,
                         search_method: str = 'grid',
                         n_iter: int = 50) -> Tuple[str, Any, Dict[str, float]]:
        """
        选择最佳模型
        
        Args:
            models: 候选模型字典 {model_name: model}
            X: 特征数据
            y: 目标变量
            param_grids: 参数网格字典 {model_name: param_grid}
            search_method: 搜索方法 ('grid', 'random', 'none')
            n_iter: 随机搜索迭代次数
            
        Returns:
            最佳模型名称、最佳模型、评估结果
        """
        if self.verbose:
            self.logger.info("Starting model selection...")
        
        model_scores = {}
        
        for model_name, model in models.items():
            if self.verbose:
                self.logger.info(f"Evaluating {model_name}...")
            
            start_time = time.time()
            
            try:
                # 参数优化
                if param_grids and model_name in param_grids and search_method != 'none':
                    best_model, best_score = self._optimize_hyperparameters(
                        model, X, y, param_grids[model_name], search_method, n_iter
                    )
                else:
                    # 直接交叉验证
                    scores = cross_val_score(
                        model, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs
                    )
                    best_model = model
                    best_score = np.mean(scores)
                
                # 转换为正值（如果是负值评分）
                if self.scoring.startswith('neg_'):
                    display_score = -best_score
                else:
                    display_score = best_score
                
                model_scores[model_name] = {
                    'model': best_model,
                    'score': best_score,
                    'display_score': display_score,
                    'time': time.time() - start_time
                }
                
                if self.verbose:
                    self.logger.info(f"{model_name} - Score: {display_score:.4f}, Time: {model_scores[model_name]['time']:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        if not model_scores:
            raise ValueError("No models could be evaluated successfully")
        
        # 选择最佳模型
        if self.scoring.startswith('neg_'):
            best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['score'])
        else:
            best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['score'])
        
        best_model = model_scores[best_model_name]['model']
        
        # 存储结果
        self.selection_results = model_scores
        self.best_models[best_model_name] = best_model
        
        if self.verbose:
            self.logger.info(f"Best model: {best_model_name} (Score: {model_scores[best_model_name]['display_score']:.4f})")
        
        return best_model_name, best_model, model_scores
    
    def _optimize_hyperparameters(self,
                                 model: Any,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.Series],
                                 param_grid: Dict,
                                 search_method: str,
                                 n_iter: int) -> Tuple[Any, float]:
        """
        优化超参数
        
        Args:
            model: 模型
            X: 特征数据
            y: 目标变量
            param_grid: 参数网格
            search_method: 搜索方法
            n_iter: 迭代次数
            
        Returns:
            最佳模型和最佳分数
        """
        if search_method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=self.cv, scoring=self.scoring,
                n_jobs=self.n_jobs, verbose=0
            )
        elif search_method == 'random':
            search = RandomizedSearchCV(
                model, param_grid, n_iter=n_iter, cv=self.cv,
                scoring=self.scoring, n_jobs=self.n_jobs, verbose=0,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown search method: {search_method}")
        
        search.fit(X, y)
        
        return search.best_estimator_, search.best_score_
    
    def get_default_param_grids(self) -> Dict[str, Dict]:
        """
        获取默认参数网格
        
        Returns:
            默认参数网格字典
        """
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'SVR': {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            }
        }
        
        return param_grids
    
    def ensemble_selection(self,
                          models: Dict[str, Any],
                          X: Union[np.ndarray, pd.DataFrame],
                          y: Union[np.ndarray, pd.Series],
                          ensemble_methods: List[str] = ['voting', 'stacking'],
                          meta_model=None) -> Dict[str, Any]:
        """
        集成模型选择
        
        Args:
            models: 基模型字典
            X: 特征数据
            y: 目标变量
            ensemble_methods: 集成方法列表
            meta_model: 元学习器（用于stacking）
            
        Returns:
            集成模型字典
        """
        from sklearn.ensemble import VotingRegressor
        from sklearn.model_selection import cross_val_predict
        from sklearn.linear_model import LinearRegression
        
        ensemble_models = {}
        
        # 投票回归器
        if 'voting' in ensemble_methods:
            voting_estimators = [(name, model) for name, model in models.items()]
            voting_regressor = VotingRegressor(estimators=voting_estimators)
            ensemble_models['VotingRegressor'] = voting_regressor
        
        # Stacking
        if 'stacking' in ensemble_methods:
            if meta_model is None:
                meta_model = LinearRegression()
            
            # 生成基模型的交叉验证预测
            base_predictions = np.column_stack([
                cross_val_predict(model, X, y, cv=self.cv)
                for model in models.values()
            ])
            
            # 训练元模型
            meta_model.fit(base_predictions, y)
            
            # 创建stacking模型类
            class StackingRegressor:
                def __init__(self, base_models, meta_model):
                    self.base_models = base_models
                    self.meta_model = meta_model
                    self.is_fitted = False
                
                def fit(self, X, y):
                    # 训练基模型
                    for model in self.base_models.values():
                        model.fit(X, y)
                    
                    # 生成基模型预测
                    base_preds = np.column_stack([
                        model.predict(X) for model in self.base_models.values()
                    ])
                    
                    # 训练元模型
                    self.meta_model.fit(base_preds, y)
                    self.is_fitted = True
                    
                    return self
                
                def predict(self, X):
                    if not self.is_fitted:
                        raise ValueError("Model must be fitted before prediction")
                    
                    base_preds = np.column_stack([
                        model.predict(X) for model in self.base_models.values()
                    ])
                    
                    return self.meta_model.predict(base_preds)
            
            stacking_regressor = StackingRegressor(models, meta_model)
            ensemble_models['StackingRegressor'] = stacking_regressor
        
        return ensemble_models
    
    def feature_selection_with_models(self,
                                    models: Dict[str, Any],
                                    X: Union[np.ndarray, pd.DataFrame],
                                    y: Union[np.ndarray, pd.Series],
                                    feature_selection_methods: List[str] = ['rfe', 'importance'],
                                    n_features_to_select: Union[int, float] = 0.5) -> Dict[str, Dict]:
        """
        结合模型的特征选择
        
        Args:
            models: 模型字典
            X: 特征数据
            y: 目标变量
            feature_selection_methods: 特征选择方法
            n_features_to_select: 要选择的特征数量或比例
            
        Returns:
            特征选择结果字典
        """
        from sklearn.feature_selection import RFE, SelectFromModel
        
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_array = X
        
        # 确定特征数量
        if isinstance(n_features_to_select, float):
            n_features = int(n_features_to_select * X_array.shape[1])
        else:
            n_features = n_features_to_select
        
        feature_selection_results = {}
        
        for model_name, model in models.items():
            model_results = {}
            
            # RFE特征选择
            if 'rfe' in feature_selection_methods:
                try:
                    rfe = RFE(model, n_features_to_select=n_features)
                    rfe.fit(X_array, y)
                    
                    selected_features = [feature_names[i] for i, selected in enumerate(rfe.support_) if selected]
                    model_results['rfe'] = {
                        'selected_features': selected_features,
                        'feature_ranking': rfe.ranking_,
                        'n_features': len(selected_features)
                    }
                except Exception as e:
                    self.logger.error(f"RFE failed for {model_name}: {e}")
            
            # 基于重要性的特征选择
            if 'importance' in feature_selection_methods:
                try:
                    selector = SelectFromModel(model, max_features=n_features)
                    selector.fit(X_array, y)
                    
                    selected_features = [feature_names[i] for i, selected in enumerate(selector.get_support()) if selected]
                    model_results['importance'] = {
                        'selected_features': selected_features,
                        'n_features': len(selected_features)
                    }
                    
                    if hasattr(selector.estimator_, 'feature_importances_'):
                        importance_scores = selector.estimator_.feature_importances_
                        model_results['importance']['feature_importances'] = dict(zip(feature_names, importance_scores))
                        
                except Exception as e:
                    self.logger.error(f"Importance-based selection failed for {model_name}: {e}")
            
            feature_selection_results[model_name] = model_results
        
        return feature_selection_results
    
    def progressive_model_selection(self,
                                  models: Dict[str, Any],
                                  X: Union[np.ndarray, pd.DataFrame],
                                  y: Union[np.ndarray, pd.Series],
                                  stages: List[Dict] = None) -> Dict[str, Any]:
        """
        渐进式模型选择
        
        Args:
            models: 模型字典
            X: 特征数据
            y: 目标变量
            stages: 选择阶段配置列表
            
        Returns:
            渐进式选择结果
        """
        if stages is None:
            stages = [
                {'method': 'quick', 'top_k': 3, 'cv': 3},
                {'method': 'detailed', 'top_k': 2, 'cv': 5},
                {'method': 'final', 'top_k': 1, 'cv': 10}
            ]
        
        current_models = models.copy()
        stage_results = {}
        
        for i, stage in enumerate(stages):
            if self.verbose:
                self.logger.info(f"Stage {i+1}: {stage['method']} evaluation")
            
            stage_name = f"stage_{i+1}_{stage['method']}"
            
            # 调整交叉验证
            original_cv = self.cv
            self.cv = stage.get('cv', self.cv)
            
            # 快速评估
            if stage['method'] == 'quick':
                # 使用较少的数据进行快速评估
                sample_size = min(1000, len(X))
                if isinstance(X, pd.DataFrame):
                    X_sample = X.sample(n=sample_size, random_state=self.random_state)
                    y_sample = y.loc[X_sample.index]
                else:
                    indices = np.random.choice(len(X), sample_size, replace=False)
                    X_sample = X[indices]
                    y_sample = y[indices]
                
                _, _, scores = self.select_best_model(current_models, X_sample, y_sample, search_method='none')
            else:
                _, _, scores = self.select_best_model(current_models, X, y, search_method='none')
            
            # 恢复原始CV设置
            self.cv = original_cv
            
            # 选择Top K模型
            top_k = stage.get('top_k', len(current_models))
            if self.scoring.startswith('neg_'):
                sorted_models = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
            else:
                sorted_models = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
            
            top_models = dict(sorted_models[:top_k])
            
            stage_results[stage_name] = {
                'scores': scores,
                'top_models': list(top_models.keys()),
                'best_model': sorted_models[0][0] if sorted_models else None
            }
            
            # 更新候选模型
            current_models = {name: scores[name]['model'] for name in top_models.keys()}
            
            if self.verbose:
                self.logger.info(f"Stage {i+1} completed. Top models: {list(top_models.keys())}")
        
        return stage_results
    
    def get_selection_summary(self) -> pd.DataFrame:
        """
        获取模型选择摘要
        
        Returns:
            选择摘要DataFrame
        """
        if not self.selection_results:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, results in self.selection_results.items():
            summary_data.append({
                'model': model_name,
                'score': results['display_score'],
                'time': results['time'],
                'score_per_second': results['display_score'] / results['time'] if results['time'] > 0 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('score', ascending=False)
        
        return summary_df
    
    def save_selection_results(self, 
                              filepath: Union[str, Path]) -> None:
        """
        保存选择结果
        
        Args:
            filepath: 保存路径
        """
        import joblib
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'selection_results': self.selection_results,
            'best_models': self.best_models,
            'selection_config': {
                'scoring': self.scoring,
                'cv': self.cv,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(results, filepath)
        
        if self.verbose:
            self.logger.info(f"Selection results saved to {filepath}")
    
    def load_selection_results(self, 
                              filepath: Union[str, Path]) -> None:
        """
        加载选择结果
        
        Args:
            filepath: 结果文件路径
        """
        import joblib
        
        results = joblib.load(filepath)
        
        self.selection_results = results['selection_results']
        self.best_models = results['best_models']
        
        if self.verbose:
            self.logger.info(f"Selection results loaded from {filepath}")
    
    def create_model_pipeline(self,
                             best_model: Any,
                             preprocessing_steps: List[Tuple[str, Any]] = None) -> Any:
        """
        创建包含预处理的模型管道
        
        Args:
            best_model: 最佳模型
            preprocessing_steps: 预处理步骤列表
            
        Returns:
            模型管道
        """
        from sklearn.pipeline import Pipeline
        
        if preprocessing_steps is None:
            preprocessing_steps = []
        
        # 添加模型作为最后一步
        steps = preprocessing_steps + [('model', best_model)]
        
        pipeline = Pipeline(steps)
        
        return pipeline