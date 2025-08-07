"""
RUCA (Rural-Urban Commuting Area) 分类器

基于通勤流量和人口密度的城乡分类系统。
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging


class RUCAClassifier:
    """RUCA分类器"""
    
    def __init__(self, verbose: bool = True):
        """
        初始化RUCA分类器
        
        Args:
            verbose: 是否输出详细信息
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        # RUCA分类代码定义
        self.ruca_definitions = {
            1: {'name': 'Metropolitan area core', 'type': 'Urban', 'description': 'Primary flow within an urbanized area (UA)'},
            2: {'name': 'Metropolitan area high commuting', 'type': 'Urban', 'description': 'Primary flow 30% or more to a UA'},
            3: {'name': 'Metropolitan area low commuting', 'type': 'Urban', 'description': 'Primary flow 10% to 30% to a UA'},
            4: {'name': 'Micropolitan area core', 'type': 'Large Rural', 'description': 'Primary flow within an urban cluster of 10,000 to 49,999'},
            5: {'name': 'Micropolitan high commuting', 'type': 'Large Rural', 'description': 'Primary flow 30% or more to an urban cluster of 10,000 to 49,999'},
            6: {'name': 'Micropolitan low commuting', 'type': 'Large Rural', 'description': 'Primary flow 10% to 30% to an urban cluster of 10,000 to 49,999'},
            7: {'name': 'Small town core', 'type': 'Small Rural', 'description': 'Primary flow within an urban cluster of 2,500 to 9,999'},
            8: {'name': 'Small town high commuting', 'type': 'Small Rural', 'description': 'Primary flow 30% or more to an urban cluster of 2,500 to 9,999'},
            9: {'name': 'Small town low commuting', 'type': 'Small Rural', 'description': 'Primary flow 10% to 30% to an urban cluster of 2,500 to 9,999'},
            10: {'name': 'Rural areas', 'type': 'Isolated Rural', 'description': 'Primary flow to a tract outside a UA or urban cluster'}
        }
        
        # 简化分类
        self.simplified_categories = {
            'Urban': [1, 2, 3],
            'Large Rural': [4, 5, 6],
            'Small Rural': [7, 8, 9],
            'Isolated Rural': [10]
        }
    
    def classify_by_population_and_commuting(self,
                                           census_data: gpd.GeoDataFrame,
                                           population_col: str = 'total_population',
                                           commuting_data: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """
        基于人口和通勤数据进行RUCA分类
        
        Args:
            census_data: 人口普查数据
            population_col: 人口列名
            commuting_data: 通勤数据
            
        Returns:
            包含RUCA分类的GeoDataFrame
        """
        if self.verbose:
            self.logger.info("Performing RUCA classification...")
        
        result = census_data.copy()
        
        # 计算人口密度 (如果没有的话)
        if 'population_density' not in result.columns:
            # 计算面积 (平方公里)
            if result.crs.is_geographic:
                # 转换到等面积投影计算面积
                area_gdf = result.to_crs('EPSG:3857')  # Web Mercator
                areas_m2 = area_gdf.geometry.area
                areas_km2 = areas_m2 / 1e6
            else:
                areas_km2 = result.geometry.area / 1e6
            
            result['area_km2'] = areas_km2
            result['population_density'] = result[population_col] / result['area_km2']
        
        # 识别城市化区域 (UA) 和城市群 (Urban Clusters)
        result['urban_area_type'] = self._identify_urban_areas(result, population_col)
        
        # 计算通勤流量（如果有通勤数据）
        if commuting_data is not None:
            result['commuting_flow'] = self._calculate_commuting_flow(result, commuting_data)
        else:
            # 使用人口密度作为代理变量
            result['commuting_flow'] = self._estimate_commuting_from_density(result)
        
        # 分配RUCA代码
        result['ruca_code'] = result.apply(self._assign_ruca_code, axis=1)
        
        # 添加RUCA描述
        result['ruca_name'] = result['ruca_code'].map(lambda x: self.ruca_definitions[x]['name'])
        result['ruca_type'] = result['ruca_code'].map(lambda x: self.ruca_definitions[x]['type'])
        result['ruca_description'] = result['ruca_code'].map(lambda x: self.ruca_definitions[x]['description'])
        
        if self.verbose:
            self._log_classification_summary(result)
        
        return result
    
    def _identify_urban_areas(self, 
                             data: gpd.GeoDataFrame,
                             population_col: str) -> pd.Series:
        """
        识别城市化区域类型
        
        Args:
            data: 地理数据
            population_col: 人口列名
            
        Returns:
            城市区域类型
        """
        urban_types = []
        
        for _, row in data.iterrows():
            pop = row[population_col]
            density = row.get('population_density', 0)
            
            if pop >= 50000 and density >= 1000:  # 城市化区域标准
                urban_types.append('UA')
            elif 10000 <= pop < 50000 and density >= 500:  # 城市群标准
                urban_types.append('UC_Large')
            elif 2500 <= pop < 10000 and density >= 300:  # 小城市群
                urban_types.append('UC_Small')
            else:
                urban_types.append('Rural')
        
        return pd.Series(urban_types, index=data.index)
    
    def _calculate_commuting_flow(self,
                                 data: gpd.GeoDataFrame,
                                 commuting_data: pd.DataFrame) -> pd.Series:
        """
        计算通勤流量百分比
        
        Args:
            data: 地理数据
            commuting_data: 通勤数据
            
        Returns:
            通勤流量百分比
        """
        # 这里应该实现实际的通勤流量计算
        # 简化实现：根据人口密度估算
        return self._estimate_commuting_from_density(data)
    
    def _estimate_commuting_from_density(self, data: gpd.GeoDataFrame) -> pd.Series:
        """
        从人口密度估算通勤流量
        
        Args:
            data: 地理数据
            
        Returns:
            估算的通勤流量
        """
        density = data['population_density']
        
        # 基于密度的通勤流量估算
        commuting_flow = np.where(
            density >= 1000, 0.8,  # 高密度区域，高通勤流量
            np.where(
                density >= 500, 0.5,  # 中密度区域
                np.where(
                    density >= 100, 0.2,  # 低密度区域
                    0.05  # 农村区域
                )
            )
        )
        
        return pd.Series(commuting_flow, index=data.index)
    
    def _assign_ruca_code(self, row: pd.Series) -> int:
        """
        分配RUCA代码
        
        Args:
            row: 数据行
            
        Returns:
            RUCA代码
        """
        urban_type = row['urban_area_type']
        commuting_flow = row['commuting_flow']
        
        if urban_type == 'UA':
            return 1  # Metropolitan area core
        elif urban_type == 'UC_Large':
            if commuting_flow >= 0.3:
                return 5  # Micropolitan high commuting
            elif commuting_flow >= 0.1:
                return 6  # Micropolitan low commuting
            else:
                return 4  # Micropolitan area core
        elif urban_type == 'UC_Small':
            if commuting_flow >= 0.3:
                return 8  # Small town high commuting
            elif commuting_flow >= 0.1:
                return 9  # Small town low commuting
            else:
                return 7  # Small town core
        else:
            # 进一步区分是否为通勤区域
            if commuting_flow >= 0.3:
                # 需要检查主要通勤目的地
                # 简化实现：假设高通勤流量到UA
                return 2  # Metropolitan area high commuting
            elif commuting_flow >= 0.1:
                return 3  # Metropolitan area low commuting
            else:
                return 10  # Rural areas
    
    def create_simplified_classification(self, 
                                       ruca_data: gpd.GeoDataFrame,
                                       categories: int = 4) -> gpd.GeoDataFrame:
        """
        创建简化的RUCA分类
        
        Args:
            ruca_data: 包含RUCA代码的数据
            categories: 分类数量 (2, 3, 4)
            
        Returns:
            简化分类的数据
        """
        result = ruca_data.copy()
        
        if categories == 2:
            # 城市 vs 农村
            result['ruca_simplified'] = result['ruca_code'].apply(
                lambda x: 'Urban' if x <= 3 else 'Rural'
            )
        elif categories == 3:
            # 城市、郊区、农村
            result['ruca_simplified'] = result['ruca_code'].apply(
                lambda x: 'Urban' if x <= 3 else ('Suburban' if x <= 6 else 'Rural')
            )
        elif categories == 4:
            # 使用标准的4类分类
            def categorize_4(code):
                for category, codes in self.simplified_categories.items():
                    if code in codes:
                        return category
                return 'Unknown'
            
            result['ruca_simplified'] = result['ruca_code'].apply(categorize_4)
        
        return result
    
    def calculate_connectivity_index(self, 
                                   ruca_data: gpd.GeoDataFrame,
                                   road_network: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """
        计算连通性指数
        
        Args:
            ruca_data: RUCA分类数据
            road_network: 道路网络数据
            
        Returns:
            包含连通性指数的数据
        """
        result = ruca_data.copy()
        
        if road_network is not None:
            # 计算实际的道路连通性
            result['connectivity_index'] = self._calculate_road_connectivity(result, road_network)
        else:
            # 基于RUCA代码估算连通性
            connectivity_map = {
                1: 1.0, 2: 0.9, 3: 0.7,  # Urban areas
                4: 0.6, 5: 0.5, 6: 0.4,  # Large rural
                7: 0.3, 8: 0.25, 9: 0.2,  # Small rural
                10: 0.1  # Isolated rural
            }
            
            result['connectivity_index'] = result['ruca_code'].map(connectivity_map)
        
        return result
    
    def _calculate_road_connectivity(self,
                                   data: gpd.GeoDataFrame,
                                   road_network: gpd.GeoDataFrame) -> pd.Series:
        """
        计算道路连通性
        
        Args:
            data: 地理数据
            road_network: 道路网络
            
        Returns:
            连通性指数
        """
        connectivity_scores = []
        
        for _, tract in data.iterrows():
            # 计算每个tract内的道路密度
            roads_in_tract = gpd.sjoin(road_network, gpd.GeoDataFrame([tract], crs=data.crs))
            
            if len(roads_in_tract) > 0:
                # 计算道路总长度
                if roads_in_tract.crs.is_geographic:
                    roads_projected = roads_in_tract.to_crs('EPSG:3857')
                    total_length = roads_projected.geometry.length.sum() / 1000  # km
                else:
                    total_length = roads_in_tract.geometry.length.sum() / 1000
                
                # 计算tract面积
                if data.crs.is_geographic:
                    tract_projected = gpd.GeoDataFrame([tract], crs=data.crs).to_crs('EPSG:3857')
                    area = tract_projected.geometry.area.iloc[0] / 1e6  # km²
                else:
                    area = tract['geometry'].area / 1e6
                
                # 道路密度 (km/km²)
                road_density = total_length / area if area > 0 else 0
                
                # 标准化到0-1范围
                connectivity_score = min(road_density / 10, 1.0)  # 假设10 km/km²为最大值
            else:
                connectivity_score = 0.0
            
            connectivity_scores.append(connectivity_score)
        
        return pd.Series(connectivity_scores, index=data.index)
    
    def _log_classification_summary(self, data: gpd.GeoDataFrame) -> None:
        """
        记录分类摘要
        
        Args:
            data: 分类结果数据
        """
        summary = data['ruca_type'].value_counts()
        self.logger.info("RUCA Classification Summary:")
        for category, count in summary.items():
            percentage = (count / len(data)) * 100
            self.logger.info(f"  {category}: {count} ({percentage:.1f}%)")
    
    def export_classification(self,
                            classified_data: gpd.GeoDataFrame,
                            output_path: Union[str, Path],
                            format: str = 'shapefile') -> None:
        """
        导出分类结果
        
        Args:
            classified_data: 分类结果
            output_path: 输出路径
            format: 输出格式
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'shapefile':
            classified_data.to_file(output_path)
        elif format == 'geojson':
            classified_data.to_file(output_path, driver='GeoJSON')
        elif format == 'csv':
            # 只保存属性数据
            df = classified_data.drop('geometry', axis=1)
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if self.verbose:
            self.logger.info(f"Classification results exported to {output_path}")
    
    def get_classification_statistics(self, 
                                    classified_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        获取分类统计信息
        
        Args:
            classified_data: 分类结果
            
        Returns:
            统计信息字典
        """
        stats = {}
        
        # 基本分布
        stats['distribution'] = classified_data['ruca_type'].value_counts().to_dict()
        stats['distribution_pct'] = (classified_data['ruca_type'].value_counts(normalize=True) * 100).to_dict()
        
        # 人口统计
        if 'total_population' in classified_data.columns:
            pop_by_type = classified_data.groupby('ruca_type')['total_population'].agg(['sum', 'mean', 'std'])
            stats['population_by_type'] = pop_by_type.to_dict()
        
        # 密度统计
        if 'population_density' in classified_data.columns:
            density_by_type = classified_data.groupby('ruca_type')['population_density'].agg(['mean', 'median', 'std'])
            stats['density_by_type'] = density_by_type.to_dict()
        
        # 连通性统计
        if 'connectivity_index' in classified_data.columns:
            conn_by_type = classified_data.groupby('ruca_type')['connectivity_index'].agg(['mean', 'std'])
            stats['connectivity_by_type'] = conn_by_type.to_dict()
        
        return stats