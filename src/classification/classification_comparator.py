"""
分类系统比较器

比较不同城市/乡村分类系统的结果和影响。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings


class ClassificationComparator:
    """分类系统比较器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化分类比较器
        
        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        self.comparison_results = {}
    
    def compare_classifications(self,
                              data: gpd.GeoDataFrame,
                              classification_columns: Dict[str, str],
                              reference_column: Optional[str] = None) -> Dict[str, Any]:
        """
        比较多种分类系统
        
        Args:
            data: 包含多种分类结果的数据
            classification_columns: 分类列名字典 {system_name: column_name}
            reference_column: 参考分类列名
            
        Returns:
            比较结果
        """
        if self.verbose:
            self.logger.info("Comparing classification systems...")
        
        results = {
            'summary': {},
            'agreement_matrix': {},
            'consistency_metrics': {},
            'spatial_analysis': {},
            'environmental_impact': {}
        }
        
        # 1. 基本分布比较
        results['summary'] = self._compare_distributions(data, classification_columns)
        
        # 2. 一致性分析
        results['agreement_matrix'] = self._calculate_agreement_matrix(data, classification_columns)
        results['consistency_metrics'] = self._calculate_consistency_metrics(data, classification_columns)
        
        # 3. 空间模式分析
        results['spatial_analysis'] = self._analyze_spatial_patterns(data, classification_columns)
        
        # 4. 如果有参考分类，计算准确性
        if reference_column and reference_column in data.columns:
            results['accuracy_assessment'] = self._assess_accuracy(
                data, classification_columns, reference_column
            )
        
        self.comparison_results = results
        
        if self.verbose:
            self._log_comparison_summary(results)
        
        return results
    
    def _compare_distributions(self,
                             data: gpd.GeoDataFrame,
                             classification_columns: Dict[str, str]) -> Dict[str, Any]:
        """
        比较分类分布
        
        Args:
            data: 数据
            classification_columns: 分类列
            
        Returns:
            分布比较结果
        """
        distributions = {}
        
        for system_name, column_name in classification_columns.items():
            if column_name in data.columns:
                dist = data[column_name].value_counts(normalize=True) * 100
                distributions[system_name] = dist.to_dict()
        
        # 计算分布差异
        system_names = list(distributions.keys())
        distribution_differences = {}
        
        for i, system1 in enumerate(system_names):
            for j, system2 in enumerate(system_names[i+1:], i+1):
                # 获取共同类别
                categories1 = set(distributions[system1].keys())
                categories2 = set(distributions[system2].keys())
                
                # 计算KL散度或其他距离度量
                diff = self._calculate_distribution_difference(
                    distributions[system1], distributions[system2]
                )
                distribution_differences[f"{system1}_vs_{system2}"] = diff
        
        return {
            'distributions': distributions,
            'differences': distribution_differences
        }
    
    def _calculate_distribution_difference(self,
                                         dist1: Dict[str, float],
                                         dist2: Dict[str, float]) -> float:
        """
        计算分布差异
        
        Args:
            dist1: 分布1
            dist2: 分布2
            
        Returns:
            差异度量
        """
        # 获取所有类别
        all_categories = set(dist1.keys()) | set(dist2.keys())
        
        # 构建概率向量
        p1 = np.array([dist1.get(cat, 0) for cat in all_categories])
        p2 = np.array([dist2.get(cat, 0) for cat in all_categories])
        
        # 归一化
        p1 = p1 / p1.sum() if p1.sum() > 0 else p1
        p2 = p2 / p2.sum() if p2.sum() > 0 else p2
        
        # 计算JS散度
        m = (p1 + p2) / 2
        js_divergence = 0.5 * self._kl_divergence(p1, m) + 0.5 * self._kl_divergence(p2, m)
        
        return float(js_divergence)
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算KL散度"""
        # 避免log(0)
        epsilon = 1e-10
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
        
        return np.sum(p * np.log(p / q))
    
    def _calculate_agreement_matrix(self,
                                  data: gpd.GeoDataFrame,
                                  classification_columns: Dict[str, str]) -> pd.DataFrame:
        """
        计算分类一致性矩阵
        
        Args:
            data: 数据
            classification_columns: 分类列
            
        Returns:
            一致性矩阵
        """
        system_names = list(classification_columns.keys())
        n_systems = len(system_names)
        
        agreement_matrix = np.zeros((n_systems, n_systems))
        
        for i, system1 in enumerate(system_names):
            for j, system2 in enumerate(system_names):
                col1 = classification_columns[system1]
                col2 = classification_columns[system2]
                
                if col1 in data.columns and col2 in data.columns:
                    # 计算一致性（相同分类的比例）
                    if i == j:
                        agreement = 1.0
                    else:
                        # 需要处理不同分类系统的标签映射
                        agreement = self._calculate_pairwise_agreement(
                            data[col1], data[col2]
                        )
                else:
                    agreement = np.nan
                
                agreement_matrix[i, j] = agreement
        
        agreement_df = pd.DataFrame(
            agreement_matrix,
            index=system_names,
            columns=system_names
        )
        
        return agreement_df
    
    def _calculate_pairwise_agreement(self,
                                    class1: pd.Series,
                                    class2: pd.Series) -> float:
        """
        计算两个分类系统的一致性
        
        Args:
            class1: 分类系统1
            class2: 分类系统2
            
        Returns:
            一致性得分
        """
        # 使用调整兰德指数
        try:
            # 转换为数值编码
            from sklearn.preprocessing import LabelEncoder
            
            le1 = LabelEncoder()
            le2 = LabelEncoder()
            
            encoded1 = le1.fit_transform(class1.astype(str))
            encoded2 = le2.fit_transform(class2.astype(str))
            
            ari = adjusted_rand_score(encoded1, encoded2)
            return ari
        except Exception:
            # 回退到简单一致性
            return float((class1 == class2).mean())
    
    def _calculate_consistency_metrics(self,
                                     data: gpd.GeoDataFrame,
                                     classification_columns: Dict[str, str]) -> Dict[str, float]:
        """
        计算一致性指标
        
        Args:
            data: 数据
            classification_columns: 分类列
            
        Returns:
            一致性指标
        """
        metrics = {}
        
        # 计算平均一致性
        agreement_matrix = self._calculate_agreement_matrix(data, classification_columns)
        
        # 排除对角线后的平均值
        mask = ~np.eye(agreement_matrix.shape[0], dtype=bool)
        valid_agreements = agreement_matrix.values[mask]
        valid_agreements = valid_agreements[~np.isnan(valid_agreements)]
        
        if len(valid_agreements) > 0:
            metrics['mean_agreement'] = float(np.mean(valid_agreements))
            metrics['std_agreement'] = float(np.std(valid_agreements))
            metrics['min_agreement'] = float(np.min(valid_agreements))
            metrics['max_agreement'] = float(np.max(valid_agreements))
        
        # 计算整体一致性（所有系统同时一致的比例）
        if len(classification_columns) > 1:
            all_same = np.ones(len(data), dtype=bool)
            columns = [col for col in classification_columns.values() if col in data.columns]
            
            if len(columns) > 1:
                first_col = data[columns[0]]
                for col in columns[1:]:
                    all_same &= (data[col] == first_col)
                
                metrics['overall_consistency'] = float(all_same.mean())
        
        return metrics
    
    def _analyze_spatial_patterns(self,
                                data: gpd.GeoDataFrame,
                                classification_columns: Dict[str, str]) -> Dict[str, Any]:
        """
        分析空间模式
        
        Args:
            data: 数据
            classification_columns: 分类列
            
        Returns:
            空间分析结果
        """
        spatial_results = {}
        
        try:
            from libpysal.weights import Queen
            from esda import Moran
            
            # 计算空间权重
            w = Queen.from_dataframe(data)
            
            for system_name, column_name in classification_columns.items():
                if column_name in data.columns:
                    # 转换为数值编码
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(data[column_name].astype(str))
                    
                    # 计算Moran's I
                    moran = Moran(encoded_values, w)
                    
                    spatial_results[system_name] = {
                        'moran_i': moran.I,
                        'p_value': moran.p_norm,
                        'z_score': moran.z_norm,
                        'interpretation': 'clustered' if moran.I > 0 and moran.p_norm < 0.05 else 'random'
                    }
        
        except ImportError:
            self.logger.warning("libpysal not available for spatial autocorrelation analysis")
            spatial_results = {}
        
        return spatial_results
    
    def _assess_accuracy(self,
                        data: gpd.GeoDataFrame,
                        classification_columns: Dict[str, str],
                        reference_column: str) -> Dict[str, Any]:
        """
        评估分类准确性
        
        Args:
            data: 数据
            classification_columns: 分类列
            reference_column: 参考列
            
        Returns:
            准确性评估结果
        """
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy_results = {}
        
        reference = data[reference_column]
        
        for system_name, column_name in classification_columns.items():
            if column_name in data.columns:
                predicted = data[column_name]
                
                # 计算总体准确率
                accuracy = accuracy_score(reference, predicted)
                
                # 分类报告
                try:
                    report = classification_report(reference, predicted, output_dict=True)
                    
                    accuracy_results[system_name] = {
                        'overall_accuracy': accuracy,
                        'macro_avg_precision': report['macro avg']['precision'],
                        'macro_avg_recall': report['macro avg']['recall'],
                        'macro_avg_f1': report['macro avg']['f1-score'],
                        'weighted_avg_precision': report['weighted avg']['precision'],
                        'weighted_avg_recall': report['weighted avg']['recall'],
                        'weighted_avg_f1': report['weighted avg']['f1-score']
                    }
                except Exception as e:
                    self.logger.warning(f"Could not compute detailed metrics for {system_name}: {e}")
                    accuracy_results[system_name] = {'overall_accuracy': accuracy}
        
        return accuracy_results
    
    def analyze_environmental_impact(self,
                                   data: gpd.GeoDataFrame,
                                   classification_columns: Dict[str, str],
                                   environmental_columns: List[str]) -> Dict[str, Any]:
        """
        分析不同分类系统对环境暴露评估的影响
        
        Args:
            data: 数据
            classification_columns: 分类列
            environmental_columns: 环境变量列
            
        Returns:
            环境影响分析结果
        """
        if self.verbose:
            self.logger.info("Analyzing environmental impact of different classifications...")
        
        impact_results = {}
        
        for env_var in environmental_columns:
            if env_var not in data.columns:
                continue
            
            impact_results[env_var] = {}
            
            for system_name, column_name in classification_columns.items():
                if column_name not in data.columns:
                    continue
                
                # 按分类计算环境变量统计
                env_stats = data.groupby(column_name)[env_var].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ])
                
                # 计算分类间差异
                class_means = env_stats['mean']
                max_diff = class_means.max() - class_means.min()
                cv = class_means.std() / class_means.mean() if class_means.mean() != 0 else 0
                
                impact_results[env_var][system_name] = {
                    'class_statistics': env_stats.to_dict(),
                    'max_difference': max_diff,
                    'coefficient_of_variation': cv,
                    'range_ratio': class_means.max() / class_means.min() if class_means.min() > 0 else np.inf
                }
        
        return impact_results
    
    def create_comparison_report(self,
                               comparison_results: Optional[Dict[str, Any]] = None,
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        创建比较报告
        
        Args:
            comparison_results: 比较结果（可选，使用上次比较结果）
            save_path: 保存路径
            
        Returns:
            比较报告
        """
        if comparison_results is None:
            comparison_results = self.comparison_results
        
        if not comparison_results:
            raise ValueError("No comparison results available")
        
        report = {
            'executive_summary': self._create_executive_summary(comparison_results),
            'detailed_findings': comparison_results,
            'recommendations': self._generate_recommendations(comparison_results)
        }
        
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _create_executive_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建执行摘要
        
        Args:
            results: 比较结果
            
        Returns:
            执行摘要
        """
        summary = {}
        
        # 分类系统数量
        if 'summary' in results and 'distributions' in results['summary']:
            n_systems = len(results['summary']['distributions'])
            summary['n_classification_systems'] = n_systems
        
        # 平均一致性
        if 'consistency_metrics' in results:
            metrics = results['consistency_metrics']
            summary['mean_agreement'] = metrics.get('mean_agreement', 'N/A')
            summary['overall_consistency'] = metrics.get('overall_consistency', 'N/A')
        
        # 最相似的分类系统对
        if 'agreement_matrix' in results:
            agreement_matrix = results['agreement_matrix']
            if isinstance(agreement_matrix, pd.DataFrame):
                # 找到最高一致性（排除对角线）
                mask = ~np.eye(agreement_matrix.shape[0], dtype=bool)
                masked_matrix = agreement_matrix.values.copy()
                masked_matrix[~mask] = np.nan
                
                max_idx = np.nanargmax(masked_matrix)
                i, j = np.unravel_index(max_idx, masked_matrix.shape)
                
                summary['most_similar_pair'] = {
                    'systems': (agreement_matrix.index[i], agreement_matrix.columns[j]),
                    'agreement': masked_matrix[i, j]
                }
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        生成建议
        
        Args:
            results: 比较结果
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于一致性的建议
        if 'consistency_metrics' in results:
            mean_agreement = results['consistency_metrics'].get('mean_agreement', 0)
            
            if mean_agreement < 0.3:
                recommendations.append(
                    "分类系统间一致性较低，建议选择单一标准化分类系统进行研究"
                )
            elif mean_agreement < 0.7:
                recommendations.append(
                    "分类系统间存在中等程度差异，建议进行敏感性分析"
                )
            else:
                recommendations.append(
                    "分类系统间一致性较高，可以使用任一系统进行分析"
                )
        
        # 基于空间模式的建议
        if 'spatial_analysis' in results:
            spatial_results = results['spatial_analysis']
            clustered_systems = [
                name for name, stats in spatial_results.items()
                if stats.get('interpretation') == 'clustered'
            ]
            
            if clustered_systems:
                recommendations.append(
                    f"以下分类系统显示空间聚集模式，适用于空间分析：{', '.join(clustered_systems)}"
                )
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any], save_path: str) -> None:
        """
        保存报告
        
        Args:
            report: 报告内容
            save_path: 保存路径
        """
        import json
        from pathlib import Path
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换pandas对象为可序列化格式
        serializable_report = self._make_serializable(report)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            self.logger.info(f"Comparison report saved to {save_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        将对象转换为可序列化格式
        
        Args:
            obj: 输入对象
            
        Returns:
            可序列化对象
        """
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _log_comparison_summary(self, results: Dict[str, Any]) -> None:
        """
        记录比较摘要
        
        Args:
            results: 比较结果
        """
        self.logger.info("Classification Comparison Summary:")
        
        if 'consistency_metrics' in results:
            metrics = results['consistency_metrics']
            self.logger.info(f"  Mean agreement: {metrics.get('mean_agreement', 'N/A'):.3f}")
            self.logger.info(f"  Overall consistency: {metrics.get('overall_consistency', 'N/A'):.3f}")
        
        if 'summary' in results and 'distributions' in results['summary']:
            n_systems = len(results['summary']['distributions'])
            self.logger.info(f"  Number of classification systems: {n_systems}")
    
    def plot_comparison_results(self,
                              comparison_results: Optional[Dict[str, Any]] = None,
                              plot_types: List[str] = ['agreement_heatmap', 'distribution_comparison'],
                              figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        绘制比较结果
        
        Args:
            comparison_results: 比较结果
            plot_types: 绘图类型列表
            figsize: 图像大小
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("matplotlib and seaborn are required for plotting")
            return
        
        if comparison_results is None:
            comparison_results = self.comparison_results
        
        if not comparison_results:
            self.logger.error("No comparison results available for plotting")
            return
        
        n_plots = len(plot_types)
        cols = min(2, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # 一致性热图
        if 'agreement_heatmap' in plot_types and 'agreement_matrix' in comparison_results:
            ax = axes[plot_idx // cols, plot_idx % cols]
            agreement_matrix = comparison_results['agreement_matrix']
            
            if isinstance(agreement_matrix, pd.DataFrame):
                sns.heatmap(agreement_matrix, annot=True, cmap='viridis', ax=ax)
                ax.set_title('Classification Agreement Matrix')
            
            plot_idx += 1
        
        # 分布比较
        if 'distribution_comparison' in plot_types and 'summary' in comparison_results:
            if plot_idx < len(axes.flat):
                ax = axes[plot_idx // cols, plot_idx % cols]
                
                distributions = comparison_results['summary'].get('distributions', {})
                if distributions:
                    # 创建分布比较图
                    dist_df = pd.DataFrame(distributions).fillna(0)
                    dist_df.plot(kind='bar', ax=ax)
                    ax.set_title('Classification Distribution Comparison')
                    ax.set_ylabel('Percentage')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plot_idx += 1
        
        # 删除多余的子图
        for i in range(plot_idx, len(axes.flat)):
            fig.delaxes(axes.flat[i])
        
        plt.tight_layout()
        plt.show()