"""
Local Climate Zones (LCZ) 分类器

基于地方气候区划的城乡分类系统。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import warnings


class LCZClassifier:
    """地方气候区划(LCZ)分类器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化LCZ分类器
        
        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # LCZ分类定义
        self.lcz_definitions = {
            # 建成型LCZ (Built Types)
            1: {
                'name': 'Compact high-rise',
                'description': 'Dense mix of tall buildings to tens of stories. Few or no trees. Land cover mostly paved.',
                'type': 'Built',
                'urban_intensity': 'Very High'
            },
            2: {
                'name': 'Compact mid-rise', 
                'description': 'Dense mix of mid-rise buildings (3-9 stories). Few or no trees. Land cover mostly paved.',
                'type': 'Built',
                'urban_intensity': 'High'
            },
            3: {
                'name': 'Compact low-rise',
                'description': 'Dense mix of low-rise buildings (1-3 stories). Few or no trees. Land cover mostly paved.',
                'type': 'Built', 
                'urban_intensity': 'High'
            },
            4: {
                'name': 'Open high-rise',
                'description': 'Open arrangement of tall buildings to tens of stories. Abundance of pervious land cover.',
                'type': 'Built',
                'urban_intensity': 'Medium'
            },
            5: {
                'name': 'Open mid-rise',
                'description': 'Open arrangement of mid-rise buildings (3-9 stories). Abundance of pervious land cover.',
                'type': 'Built',
                'urban_intensity': 'Medium'
            },
            6: {
                'name': 'Open low-rise',
                'description': 'Open arrangement of low-rise buildings (1-3 stories). Abundance of pervious land cover.',
                'type': 'Built',
                'urban_intensity': 'Low'
            },
            7: {
                'name': 'Lightweight low-rise',
                'description': 'Dense mix of single-story buildings. Few or no trees. Land cover mostly hard-packed.',
                'type': 'Built',
                'urban_intensity': 'Medium'
            },
            8: {
                'name': 'Large low-rise',
                'description': 'Open arrangement of large low-rise buildings (1-3 stories). Few or no trees.',
                'type': 'Built',
                'urban_intensity': 'Low'
            },
            9: {
                'name': 'Sparsely built',
                'description': 'Sparse arrangement of small or medium-sized buildings in a natural setting.',
                'type': 'Built',
                'urban_intensity': 'Very Low'
            },
            10: {
                'name': 'Heavy industry',
                'description': 'Low-rise and mid-rise industrial structures. Few or no trees.',
                'type': 'Built',
                'urban_intensity': 'Medium'
            },
            
            # 自然型LCZ (Land Cover Types)
            11: {
                'name': 'Dense trees',
                'description': 'Heavily wooded landscape of deciduous and/or evergreen trees.',
                'type': 'Natural',
                'urban_intensity': 'None'
            },
            12: {
                'name': 'Scattered trees',
                'description': 'Lightly wooded landscape of deciduous and/or evergreen trees.',
                'type': 'Natural',
                'urban_intensity': 'None'
            },
            13: {
                'name': 'Bush, scrub',
                'description': 'Open arrangement of bushes, shrubs, and short, woody trees.',
                'type': 'Natural',
                'urban_intensity': 'None'
            },
            14: {
                'name': 'Low plants',
                'description': 'Featureless landscape of grass or herbaceous plants/crops.',
                'type': 'Natural',
                'urban_intensity': 'None'
            },
            15: {
                'name': 'Bare rock or paved',
                'description': 'Featureless landscape of rock or paved cover.',
                'type': 'Natural',
                'urban_intensity': 'None'
            },
            16: {
                'name': 'Bare soil or sand',
                'description': 'Featureless landscape of soil or sand cover.',
                'type': 'Natural',
                'urban_intensity': 'None'
            },
            17: {
                'name': 'Water',
                'description': 'Large, open water bodies such as seas and lakes.',
                'type': 'Natural',
                'urban_intensity': 'None'
            }
        }
        
        # LCZ参数特征（用于分类）
        self.lcz_parameters = {
            # 建筑相关参数
            'building_height': {  # 平均建筑高度(m)
                1: (25, 200), 2: (10, 25), 3: (3, 10), 4: (25, 200),
                5: (10, 25), 6: (3, 10), 7: (2, 4), 8: (3, 10),
                9: (3, 10), 10: (5, 15)
            },
            'building_density': {  # 建筑密度(%)
                1: (40, 80), 2: (40, 70), 3: (40, 70), 4: (20, 40),
                5: (20, 40), 6: (20, 40), 7: (60, 90), 8: (30, 50),
                9: (10, 20), 10: (20, 30)
            },
            'impervious_fraction': {  # 不透水表面比例(%)
                1: (40, 80), 2: (40, 80), 3: (40, 80), 4: (20, 40),
                5: (20, 40), 6: (20, 40), 7: (40, 80), 8: (40, 80),
                9: (0, 20), 10: (20, 50)
            }
        }
    
    def classify_from_features(self,
                             data: gpd.GeoDataFrame,
                             feature_mapping: Dict[str, str]) -> gpd.GeoDataFrame:
        """
        从特征数据进行LCZ分类
        
        Args:
            data: 包含分类特征的地理数据
            feature_mapping: 特征列名映射
            
        Returns:
            LCZ分类结果
        """
        if self.verbose:
            self.logger.info("Classifying LCZ from features...")
        
        result = data.copy()
        
        # 检查必需的特征
        required_features = ['building_height', 'building_density', 'impervious_fraction']
        for feature in required_features:
            if feature not in feature_mapping:
                raise ValueError(f"Missing feature mapping for: {feature}")
            if feature_mapping[feature] not in data.columns:
                raise ValueError(f"Column not found: {feature_mapping[feature]}")
        
        # 提取特征
        features = {}
        for feature, column in feature_mapping.items():
            features[feature] = result[column]
        
        # 分类逻辑
        lcz_codes = []
        
        for idx in result.index:
            # 提取当前样本的特征值
            sample_features = {f: features[f].loc[idx] for f in features}
            
            # 计算与每个LCZ的匹配度
            best_lcz = self._match_to_lcz(sample_features)
            lcz_codes.append(best_lcz)
        
        result['lcz_code'] = lcz_codes
        
        # 添加LCZ描述信息
        result['lcz_name'] = result['lcz_code'].map(
            lambda x: self.lcz_definitions[x]['name'] if x in self.lcz_definitions else 'Unknown'
        )
        result['lcz_type'] = result['lcz_code'].map(
            lambda x: self.lcz_definitions[x]['type'] if x in self.lcz_definitions else 'Unknown'
        )
        result['urban_intensity'] = result['lcz_code'].map(
            lambda x: self.lcz_definitions[x]['urban_intensity'] if x in self.lcz_definitions else 'Unknown'
        )
        
        if self.verbose:
            self._log_classification_summary(result)
        
        return result
    
    def _match_to_lcz(self, sample_features: Dict[str, float]) -> int:
        """
        将样本特征匹配到最合适的LCZ类别
        
        Args:
            sample_features: 样本特征值
            
        Returns:
            最匹配的LCZ代码
        """
        best_match = None
        best_score = -1
        
        # 只考虑建成型LCZ (1-10)
        for lcz_code in range(1, 11):
            score = 0
            total_weight = 0
            
            for param, value in sample_features.items():
                if param in self.lcz_parameters and lcz_code in self.lcz_parameters[param]:
                    param_range = self.lcz_parameters[param][lcz_code]
                    min_val, max_val = param_range
                    
                    # 计算匹配度
                    if min_val <= value <= max_val:
                        match_score = 1.0  # 完全匹配
                    else:
                        # 计算距离
                        if value < min_val:
                            distance = min_val - value
                            range_size = max_val - min_val
                        else:
                            distance = value - max_val
                            range_size = max_val - min_val
                        
                        # 距离越近，分数越高
                        match_score = max(0, 1 - distance / range_size)
                    
                    score += match_score
                    total_weight += 1
            
            if total_weight > 0:
                avg_score = score / total_weight
                if avg_score > best_score:
                    best_score = avg_score
                    best_match = lcz_code
        
        return best_match if best_match is not None else 9  # 默认为稀疏建设
    
    def classify_from_remote_sensing(self,
                                   rs_data: Dict[str, xr.DataArray],
                                   model_path: Optional[str] = None) -> xr.DataArray:
        """
        从遥感数据进行LCZ分类
        
        Args:
            rs_data: 遥感数据字典 {band_name: data}
            model_path: 预训练模型路径
            
        Returns:
            LCZ分类结果
        """
        if self.verbose:
            self.logger.info("Classifying LCZ from remote sensing data...")
        
        # 如果有预训练模型，加载并使用
        if model_path and Path(model_path).exists():
            return self._classify_with_pretrained_model(rs_data, model_path)
        else:
            # 使用简化的基于规则的分类
            return self._classify_with_rules(rs_data)
    
    def _classify_with_rules(self, rs_data: Dict[str, xr.DataArray]) -> xr.DataArray:
        """
        基于规则的LCZ分类
        
        Args:
            rs_data: 遥感数据
            
        Returns:
            分类结果
        """
        # 需要的波段
        required_bands = ['ndvi', 'ndbi', 'lst']  # NDVI, NDBI, 地表温度
        
        # 检查数据可用性
        available_bands = set(rs_data.keys())
        missing_bands = set(required_bands) - available_bands
        
        if missing_bands:
            self.logger.warning(f"Missing bands for classification: {missing_bands}")
        
        # 使用可用的数据进行简化分类
        if 'ndvi' in rs_data:
            ndvi = rs_data['ndvi']
            
            # 基于NDVI的简单分类
            classification = xr.where(
                ndvi > 0.5, 11,  # Dense trees
                xr.where(
                    ndvi > 0.3, 12,  # Scattered trees
                    xr.where(
                        ndvi > 0.1, 14,  # Low plants
                        xr.where(
                            ndvi > -0.1, 6,  # Open low-rise (假设)
                            15  # Bare rock or paved
                        )
                    )
                )
            )
        else:
            # 如果没有NDVI，创建默认分类
            template = list(rs_data.values())[0]
            classification = xr.full_like(template, 6, dtype=int)  # 默认为开放低层建筑
        
        # 如果有NDBI，进一步细化城市区域
        if 'ndbi' in rs_data:
            ndbi = rs_data['ndbi']
            
            # 高NDBI值表示城市建设区域
            urban_mask = ndbi > 0.1
            
            classification = xr.where(
                urban_mask & (ndbi > 0.3), 3,  # Compact low-rise
                xr.where(
                    urban_mask & (ndbi > 0.2), 6,  # Open low-rise
                    classification
                )
            )
        
        # 添加属性信息
        classification.attrs['lcz_labels'] = {
            code: self.lcz_definitions[code]['name'] 
            for code in np.unique(classification.values) if code in self.lcz_definitions
        }
        
        return classification
    
    def _classify_with_pretrained_model(self,
                                      rs_data: Dict[str, xr.DataArray],
                                      model_path: str) -> xr.DataArray:
        """
        使用预训练模型进行分类
        
        Args:
            rs_data: 遥感数据
            model_path: 模型路径
            
        Returns:
            分类结果
        """
        # 这里应该加载实际的预训练模型
        # 简化实现：返回基于规则的分类
        self.logger.warning("Pretrained model not implemented, using rule-based classification")
        return self._classify_with_rules(rs_data)
    
    def create_simplified_classification(self,
                                       lcz_data: Union[gpd.GeoDataFrame, xr.DataArray],
                                       scheme: str = 'urban_rural') -> Union[gpd.GeoDataFrame, xr.DataArray]:
        """
        创建简化的LCZ分类
        
        Args:
            lcz_data: LCZ分类数据
            scheme: 简化方案
            
        Returns:
            简化分类结果
        """
        if scheme == 'urban_rural':
            # 城市 vs 农村
            mapping = {
                1: 'Urban', 2: 'Urban', 3: 'Urban', 4: 'Urban', 5: 'Urban',
                6: 'Urban', 7: 'Urban', 8: 'Urban', 9: 'Rural', 10: 'Urban',
                11: 'Rural', 12: 'Rural', 13: 'Rural', 14: 'Rural',
                15: 'Rural', 16: 'Rural', 17: 'Water'
            }
        elif scheme == 'built_natural':
            # 建成 vs 自然
            mapping = {
                1: 'Built', 2: 'Built', 3: 'Built', 4: 'Built', 5: 'Built',
                6: 'Built', 7: 'Built', 8: 'Built', 9: 'Built', 10: 'Built',
                11: 'Natural', 12: 'Natural', 13: 'Natural', 14: 'Natural',
                15: 'Natural', 16: 'Natural', 17: 'Water'
            }
        elif scheme == 'density':
            # 密度分类
            mapping = {
                1: 'High Density', 2: 'High Density', 3: 'High Density',
                4: 'Medium Density', 5: 'Medium Density', 6: 'Low Density',
                7: 'High Density', 8: 'Low Density', 9: 'Very Low Density',
                10: 'Medium Density', 11: 'Natural', 12: 'Natural',
                13: 'Natural', 14: 'Natural', 15: 'Natural', 16: 'Natural',
                17: 'Water'
            }
        else:
            raise ValueError(f"Unknown simplification scheme: {scheme}")
        
        if isinstance(lcz_data, gpd.GeoDataFrame):
            result = lcz_data.copy()
            result['lcz_simplified'] = result['lcz_code'].map(mapping)
        else:  # xarray.DataArray
            result = lcz_data.copy()
            
            # 创建映射数组
            unique_codes = np.unique(lcz_data.values)
            for code in unique_codes:
                if code in mapping:
                    result = xr.where(lcz_data == code, mapping[code], result)
            
            result.attrs['simplified_scheme'] = scheme
            result.attrs['mapping'] = mapping
        
        return result
    
    def calculate_lcz_metrics(self,
                            lcz_data: gpd.GeoDataFrame,
                            environmental_data: Optional[gpd.GeoDataFrame] = None) -> Dict[str, Any]:
        """
        计算LCZ相关指标
        
        Args:
            lcz_data: LCZ分类数据
            environmental_data: 环境数据
            
        Returns:
            LCZ指标
        """
        metrics = {}
        
        # 基本分布
        lcz_counts = lcz_data['lcz_code'].value_counts()
        metrics['distribution'] = lcz_counts.to_dict()
        metrics['distribution_pct'] = (lcz_counts / len(lcz_data) * 100).to_dict()
        
        # 面积统计
        if 'area_km2' in lcz_data.columns:
            area_by_lcz = lcz_data.groupby('lcz_code')['area_km2'].sum()
            metrics['area_by_lcz'] = area_by_lcz.to_dict()
        
        # 城市化强度分析
        intensity_counts = lcz_data['urban_intensity'].value_counts()
        metrics['urban_intensity_distribution'] = intensity_counts.to_dict()
        
        # 建成型 vs 自然型比例
        type_counts = lcz_data['lcz_type'].value_counts()
        metrics['type_distribution'] = type_counts.to_dict()
        
        # 如果有环境数据，计算环境暴露指标
        if environmental_data is not None:
            # 空间连接
            joined_data = gpd.sjoin(lcz_data, environmental_data, how='inner', predicate='intersects')
            
            # 按LCZ计算环境指标统计
            env_cols = [col for col in environmental_data.columns 
                       if col not in ['geometry'] and environmental_data[col].dtype in ['float64', 'int64']]
            
            for col in env_cols:
                if col in joined_data.columns:
                    env_stats = joined_data.groupby('lcz_code')[col].agg(['mean', 'std', 'min', 'max'])
                    metrics[f'{col}_by_lcz'] = env_stats.to_dict()
        
        return metrics
    
    def validate_lcz_classification(self,
                                  classified_data: gpd.GeoDataFrame,
                                  reference_data: gpd.GeoDataFrame,
                                  reference_column: str = 'lcz_code') -> Dict[str, float]:
        """
        验证LCZ分类结果
        
        Args:
            classified_data: 分类结果
            reference_data: 参考数据
            reference_column: 参考列名
            
        Returns:
            验证指标
        """
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        if self.verbose:
            self.logger.info("Validating LCZ classification...")
        
        # 空间匹配
        matched_data = gpd.sjoin(classified_data, reference_data, how='inner', predicate='intersects')
        
        if len(matched_data) == 0:
            self.logger.error("No spatial overlap found between classified and reference data")
            return {}
        
        # 提取分类结果和参考值
        predicted = matched_data['lcz_code']
        reference = matched_data[reference_column]
        
        # 计算指标
        accuracy = accuracy_score(reference, predicted)
        
        # 混淆矩阵
        cm = confusion_matrix(reference, predicted)
        
        # 按类别准确率
        report = classification_report(reference, predicted, output_dict=True)
        
        validation_results = {
            'overall_accuracy': accuracy,
            'confusion_matrix': cm.tolist(),
            'class_accuracy': {str(k): v['precision'] for k, v in report.items() if k.isdigit()},
            'n_samples': len(matched_data)
        }
        
        if self.verbose:
            self.logger.info(f"Overall accuracy: {accuracy:.3f}")
            self.logger.info(f"Validation samples: {len(matched_data)}")
        
        return validation_results
    
    def _log_classification_summary(self, data: gpd.GeoDataFrame) -> None:
        """
        记录分类摘要
        
        Args:
            data: 分类数据
        """
        summary = data['lcz_name'].value_counts()
        self.logger.info("LCZ Classification Summary:")
        for category, count in summary.head(10).items():
            percentage = (count / len(data)) * 100
            self.logger.info(f"  {category}: {count} ({percentage:.1f}%)")
    
    def export_lcz_classification(self,
                                classified_data: Union[gpd.GeoDataFrame, xr.DataArray],
                                output_path: Union[str, Path],
                                format: str = 'shapefile') -> None:
        """
        导出LCZ分类结果
        
        Args:
            classified_data: 分类结果
            output_path: 输出路径
            format: 输出格式
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(classified_data, gpd.GeoDataFrame):
            if format == 'shapefile':
                classified_data.to_file(output_path)
            elif format == 'geojson':
                classified_data.to_file(output_path, driver='GeoJSON')
            elif format == 'csv':
                df = classified_data.drop('geometry', axis=1)
                df.to_csv(output_path, index=False)
        else:  # xarray.DataArray
            if format == 'netcdf':
                classified_data.to_netcdf(output_path)
            elif format == 'geotiff':
                classified_data.rio.to_raster(output_path)
        
        if self.verbose:
            self.logger.info(f"LCZ classification exported to {output_path}")