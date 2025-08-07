"""
神经网络模型

包含深度神经网络(DNN)和长短期记忆网络(LSTM)的实现。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from .base_model import BaseModel
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. Neural network models will not work.")


class DNNPredictor(BaseModel):
    """深度神经网络预测器"""
    
    def __init__(self,
                 hidden_layers: List[int] = [128, 64, 32],
                 activation: str = 'relu',
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 10,
                 reduce_lr_patience: int = 5,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        初始化DNN预测器
        
        Args:
            hidden_layers: 隐藏层神经元数量列表
            activation: 激活函数
            dropout_rate: Dropout比率
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            validation_split: 验证集比例
            early_stopping_patience: 早停耐心值
            reduce_lr_patience: 学习率衰减耐心值
            random_state: 随机种子
            verbose: 是否输出详细信息
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network models")
        
        super().__init__(
            model_name="DNN",
            random_state=random_state,
            verbose=verbose
        )
        
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        
        # 神经网络需要特征标准化
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # 设置随机种子
        tf.random.set_seed(random_state)
        
        self.history = None
        
    def build_model(self, input_dim: int) -> keras.Model:
        """
        构建DNN模型
        
        Args:
            input_dim: 输入维度
            
        Returns:
            Keras模型
        """
        model = models.Sequential()
        
        # 输入层
        model.add(layers.Input(shape=(input_dim,)))
        
        # 隐藏层
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units, 
                activation=self.activation,
                name=f'hidden_{i+1}'
            ))
            
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # 输出层
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            validation_data: Optional[Tuple] = None) -> 'DNNPredictor':
        """
        训练DNN模型
        
        Args:
            X: 特征数据
            y: 目标变量
            validation_data: 验证数据 (X_val, y_val)
            
        Returns:
            训练后的模型实例
        """
        if self.verbose:
            self.logger.info("Training DNN model...")
        
        # 数据预处理
        X_processed, y_processed = self._prepare_data(X, y, fit_scalers=True)
        
        # 构建模型
        input_dim = X_processed.shape[1]
        self.model = self.build_model(input_dim)
        
        if self.verbose:
            self.model.summary()
        
        # 准备回调函数
        callbacks_list = []
        
        # 早停
        if self.early_stopping_patience > 0:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            )
            callbacks_list.append(early_stopping)
        
        # 学习率衰减
        if self.reduce_lr_patience > 0:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1 if self.verbose else 0
            )
            callbacks_list.append(reduce_lr)
        
        # 准备验证数据
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_processed, y_val_processed = self._prepare_data(X_val, y_val)
            validation_data = (X_val_processed, y_val_processed)
        
        # 训练模型
        self.history = self.model.fit(
            X_processed, y_processed,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split if validation_data is None else 0,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=1 if self.verbose else 0
        )
        
        self.is_fitted = True
        
        # 计算残差标准差
        y_pred = self.model.predict(X_processed, verbose=0)
        residuals = y_processed - y_pred.ravel()
        self.residual_std_ = np.std(residuals)
        
        if self.verbose:
            self.logger.info("DNN training completed")
        
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
        predictions = self.model.predict(X_processed, verbose=0)
        predictions = predictions.ravel()
        
        # 逆变换
        predictions = self._inverse_transform_target(predictions)
        
        return predictions
    
    def predict_with_uncertainty(self,
                                X: Union[np.ndarray, pd.DataFrame],
                                method: str = 'dropout',
                                n_samples: int = 100,
                                confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        带不确定性的预测
        
        Args:
            X: 特征数据
            method: 不确定性估计方法 ('dropout', 'ensemble')
            n_samples: 采样次数
            confidence_level: 置信水平
            
        Returns:
            预测值、下界、上界
        """
        if method == 'dropout':
            # 使用Dropout进行蒙特卡洛估计
            predictions = self._predict_with_dropout(X, n_samples)
        elif method == 'ensemble':
            # 使用集成方法
            predictions = self._predict_with_ensemble(X, n_samples)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
        
        # 计算统计量
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # 计算置信区间
        from scipy import stats
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        margin = z_score * std_pred
        lower_bound = mean_pred - margin
        upper_bound = mean_pred + margin
        
        return mean_pred, lower_bound, upper_bound
    
    def _predict_with_dropout(self,
                             X: Union[np.ndarray, pd.DataFrame],
                             n_samples: int) -> np.ndarray:
        """
        使用Dropout进行蒙特卡洛预测
        
        Args:
            X: 特征数据
            n_samples: 采样次数
            
        Returns:
            预测样本数组
        """
        X_processed, _ = self._prepare_data(X)
        
        # 创建一个在预测时保持Dropout开启的函数
        def predict_with_dropout(x):
            return self.model(x, training=True)
        
        predictions = []
        for _ in range(n_samples):
            pred = predict_with_dropout(X_processed).numpy().ravel()
            pred = self._inverse_transform_target(pred)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def _predict_with_ensemble(self,
                              X: Union[np.ndarray, pd.DataFrame],
                              n_models: int) -> np.ndarray:
        """
        使用集成方法进行预测
        
        Args:
            X: 特征数据
            n_models: 模型数量
            
        Returns:
            预测样本数组
        """
        # 这里简化实现，实际应该训练多个模型
        X_processed, _ = self._prepare_data(X)
        
        predictions = []
        for _ in range(n_models):
            # 添加噪声模拟不同模型
            noise = np.random.normal(0, 0.01, X_processed.shape)
            X_noisy = X_processed + noise
            
            pred = self.model.predict(X_noisy, verbose=0).ravel()
            pred = self._inverse_transform_target(pred)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def plot_training_history(self,
                             figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        绘制训练历史
        
        Args:
            figsize: 图像大小
        """
        if self.history is None:
            self.logger.error("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("matplotlib is required for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 损失曲线
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE曲线
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
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
            summary.update({
                'hidden_layers': self.hidden_layers,
                'activation': self.activation,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.size(w).numpy() for w in self.model.trainable_weights])
            })
        
        return summary


class LSTMPredictor(BaseModel):
    """LSTM预测器，用于时间序列预测"""
    
    def __init__(self,
                 lstm_units: List[int] = [64, 32],
                 dense_units: List[int] = [32],
                 dropout_rate: float = 0.2,
                 recurrent_dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 sequence_length: int = 10,
                 validation_split: float = 0.2,
                 early_stopping_patience: int = 10,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        初始化LSTM预测器
        
        Args:
            lstm_units: LSTM层神经元数量列表
            dense_units: 全连接层神经元数量列表
            dropout_rate: Dropout比率
            recurrent_dropout: 循环Dropout比率
            learning_rate: 学习率
            batch_size: 批次大小
            epochs: 训练轮数
            sequence_length: 序列长度
            validation_split: 验证集比例
            early_stopping_patience: 早停耐心值
            random_state: 随机种子
            verbose: 是否输出详细信息
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
        
        super().__init__(
            model_name="LSTM",
            random_state=random_state,
            verbose=verbose
        )
        
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        
        # LSTM需要特征标准化
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        # 设置随机种子
        tf.random.set_seed(random_state)
        
        self.history = None
        
    def build_model(self, 
                   input_shape: Tuple[int, int]) -> keras.Model:
        """
        构建LSTM模型
        
        Args:
            input_shape: 输入形状 (sequence_length, n_features)
            
        Returns:
            Keras模型
        """
        model = models.Sequential()
        
        # LSTM层
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            
            if i == 0:
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    input_shape=input_shape,
                    name=f'lstm_{i+1}'
                ))
            else:
                model.add(layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    name=f'lstm_{i+1}'
                ))
        
        # 全连接层
        for i, units in enumerate(self.dense_units):
            model.add(layers.Dense(
                units,
                activation='relu',
                name=f'dense_{i+1}'
            ))
            
            if self.dropout_rate > 0:
                model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # 输出层
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_sequences(self,
                        data: np.ndarray,
                        targets: Optional[np.ndarray] = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        创建时间序列序列
        
        Args:
            data: 输入数据
            targets: 目标数据
            
        Returns:
            序列化的数据
        """
        n_samples = len(data) - self.sequence_length + 1
        n_features = data.shape[1] if len(data.shape) > 1 else 1
        
        # 创建序列
        X_seq = np.zeros((n_samples, self.sequence_length, n_features))
        
        for i in range(n_samples):
            X_seq[i] = data[i:i + self.sequence_length].reshape(self.sequence_length, n_features)
        
        if targets is not None:
            y_seq = targets[self.sequence_length - 1:]
            return X_seq, y_seq
        
        return X_seq
    
    def fit(self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            validation_data: Optional[Tuple] = None) -> 'LSTMPredictor':
        """
        训练LSTM模型
        
        Args:
            X: 特征数据
            y: 目标变量
            validation_data: 验证数据
            
        Returns:
            训练后的模型实例
        """
        if self.verbose:
            self.logger.info("Training LSTM model...")
        
        # 数据预处理
        X_processed, y_processed = self._prepare_data(X, y, fit_scalers=True)
        
        # 创建序列
        X_seq, y_seq = self.create_sequences(X_processed, y_processed)
        
        # 构建模型
        input_shape = (self.sequence_length, X_processed.shape[1])
        self.model = self.build_model(input_shape)
        
        if self.verbose:
            self.model.summary()
        
        # 准备回调函数
        callbacks_list = []
        
        if self.early_stopping_patience > 0:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=1 if self.verbose else 0
            )
            callbacks_list.append(early_stopping)
        
        # 处理验证数据
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_processed, y_val_processed = self._prepare_data(X_val, y_val)
            X_val_seq, y_val_seq = self.create_sequences(X_val_processed, y_val_processed)
            validation_data = (X_val_seq, y_val_seq)
        
        # 训练模型
        self.history = self.model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split if validation_data is None else 0,
            validation_data=validation_data,
            callbacks=callbacks_list,
            verbose=1 if self.verbose else 0
        )
        
        self.is_fitted = True
        
        if self.verbose:
            self.logger.info("LSTM training completed")
        
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
        
        # 创建序列
        X_seq = self.create_sequences(X_processed)
        
        # 预测
        predictions = self.model.predict(X_seq, verbose=0)
        predictions = predictions.ravel()
        
        # 逆变换
        predictions = self._inverse_transform_target(predictions)
        
        return predictions
    
    def predict_future(self,
                      X_last: Union[np.ndarray, pd.DataFrame],
                      n_steps: int = 1) -> np.ndarray:
        """
        未来预测
        
        Args:
            X_last: 最后的序列数据
            n_steps: 预测步数
            
        Returns:
            未来预测结果
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_processed, _ = self._prepare_data(X_last)
        
        # 确保有足够的数据创建序列
        if len(X_processed) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for prediction")
        
        # 使用最后的sequence_length个样本
        current_sequence = X_processed[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        predictions = []
        
        for _ in range(n_steps):
            # 预测下一步
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # 更新序列（这里简化处理，实际可能需要更复杂的特征工程）
            # 将预测值作为新的特征（如果合适的话）
            next_features = np.zeros((1, X_processed.shape[1]))
            next_features[0, 0] = next_pred  # 假设第一个特征是目标变量的滞后值
            
            # 滚动更新序列
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_features
        
        predictions = np.array(predictions)
        predictions = self._inverse_transform_target(predictions)
        
        return predictions
    
    def plot_training_history(self,
                             figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        绘制训练历史
        
        Args:
            figsize: 图像大小
        """
        if self.history is None:
            self.logger.error("No training history available")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("matplotlib is required for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 损失曲线
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('LSTM Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE曲线
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('LSTM Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()