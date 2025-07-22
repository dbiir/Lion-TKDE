import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    """自定义时间序列数据集"""
    def __init__(self, data, seq_length=10, forecast_horizon=1):
        """
        :param data: 标准化后的数据 (n_samples, n_features)
        :param seq_length: 输入序列长度
        :param forecast_horizon: 预测步长
        """
        self.data = data
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.stride = 1
        print(len(self.data) - self.seq_length - self.forecast_horizon + 1)
        
    def __len__(self):
        return (len(self.data) - self.seq_length - self.forecast_horizon)

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start:start+self.seq_length]
        y = self.data[start+self.seq_length:start+self.seq_length+self.forecast_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)

class RegionRNN(nn.Module):
    """多区域时间序列预测RNN模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, forecast_horizon):
        """
        :param input_size: 每个时间步的特征维度（region数量）
        :param hidden_size: RNN隐藏层维度
        :param num_layers: RNN层数
        :param output_size: 输出维度（预测的region数量）
        :param forecast_horizon: 预测步长
        """
        super(RegionRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        # 主RNN层
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # 输出层 - 预测多个时间步
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, output_size * forecast_horizon)
        )
        
    def forward(self, x):
        # x形状: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # RNN前向传播
        out, _ = self.rnn(x, h0)  # out形状: (batch_size, seq_length, hidden_size)
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]  # (batch_size, hidden_size)
        
        # 通过全连接层生成预测
        predictions = self.fc(out)  # (batch_size, output_size * forecast_horizon)
        
        # 重塑为 (batch_size, forecast_horizon, output_size)
        return predictions.view(batch_size, self.forecast_horizon, -1)

def prepare_data(file_path, test_size=0.2, seq_length=10, forecast_horizon=1, batch_size=16):
    """
    数据预处理流程
    :param file_path: 数据文件路径
    :param test_size: 测试集比例
    :param seq_length: 输入序列长度
    :param forecast_horizon: 预测步长
    :return: 训练和测试数据加载器，以及scaler对象
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 提取特征（所有region列）
    region_cols = [col for col in df.columns if col.startswith('region_id_')]
    data = df[region_cols].values
    
    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 10))
    region_data_scaled = scaler.fit_transform(data)
    # desired_variance = 8
    # region_data_scaled = region_data_scaled * np.sqrt(desired_variance)
    df_region_scaled = pd.DataFrame(region_data_scaled, columns=region_cols)

    # # 添加 OT 列（不标准化）
    # df_region_scaled["OT"] = df["OT"].values

    # 最终数据（包含标准化后的 region 列和原始 OT 列）
    data_scaled = df_region_scaled.values

    # 创建完整数据集
    full_dataset = TimeSeriesDataset(data_scaled, seq_length, forecast_horizon)
    
    # 计算划分点 - 保持时间序列顺序
    train_size = int((1 - test_size) * len(full_dataset))
    
    # 划分训练集和测试集
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, len(full_dataset) - train_size],
        generator=torch.Generator().manual_seed(42)  # 可重现性
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # region_cols.append("OT")
    return train_loader, test_loader, scaler, region_cols

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    """模型训练函数"""
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
    
    return train_losses

def evaluate_model(model, dataloader, criterion, device):
    """模型评估函数"""
    model.eval()
    running_mse = 0.0
    running_mae = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # sorted_outputs, _ = torch.sort(outputs, dim=-1)
            # sorted_targets, _ = torch.sort(targets, dim=-1)

            # 计算指标
            mse_loss = criterion(outputs, targets)
            mae_loss = torch.mean(torch.abs(outputs - targets))

            running_mse += mse_loss.item()
            running_mae += mae_loss.item()
            
            # 保存结果用于可视化
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.cpu().numpy())

    avg_mse = running_mse / len(dataloader)
    avg_mae = running_mae / len(dataloader)
    
    print("dataloader length: " + str(len(dataloader)))
    # 合并所有批次的预测结果
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    return avg_mse, avg_mae, predictions, actuals

def plot_results(actuals, predictions, region_idx=0, num_samples=50):
    """绘制预测结果与实际值对比图"""
    plt.figure(figsize=(15, 6))
    
    # 只显示指定region的结果
    actuals_region = actuals[:, 0, region_idx]  # 取第一个预测时间步
    preds_region = predictions[:, 0, region_idx]
    
    plt.plot(actuals_region[:num_samples], label='Actual', color='blue', alpha=0.7)
    plt.plot(preds_region[:num_samples], label='Predicted', color='red', alpha=0.7, linestyle='--')
    
    plt.title(f'Time Series Prediction - Region {region_idx+1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(horizon: int):
    # 参数设置
    forecast_horizon = horizon  # 预测步长
    seq_length = forecast_horizon       # 输入序列长度 (3倍于预测步长)
    batch_size = 32
    hidden_size = 64
    num_layers = 2
    learning_rate = 0.001
    num_epochs = 30
    test_size = 0.2
    
    # 设备设置
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载和预处理数据
    data_path = "../../dataset/ycsb/transaction_log_4.csv"  # 替换为您的文件路径
    train_loader, test_loader, scaler, region_cols = prepare_data(
        data_path, 
        test_size=test_size,
        seq_length=seq_length,
        forecast_horizon=forecast_horizon
    )
    print("train_loader length: " + str(len(train_loader)) + " test_loader length: " + str(len(test_loader)))
    
    # 模型参数
    input_size = len(region_cols)  # region数量
    output_size = input_size       # 预测所有region
    
    # 2. 初始化模型
    model = RegionRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        forecast_horizon=forecast_horizon
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. 训练模型
    print("Starting Training...")
    train_losses = train_model(model, train_loader, criterion, optimizer, device, num_epochs)
    
    # 绘制训练损失曲线
    plt.plot(train_losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.show()
    
    # 4. 评估模型
    print("\nEvaluating Model...")
    test_mse, test_mae, predictions, actuals = evaluate_model(model, test_loader, criterion, device)
    print(f"Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")
    
    # 5. 可视化结果
    plot_results(actuals, predictions, region_idx=0)  # 可视化第一个region的预测
    
    return model, test_loader, scaler, region_cols

if __name__ == '__main__':
    model, test_loader, scaler, region_cols = main(168)
    model, test_loader, scaler, region_cols = main(336)
    model, test_loader, scaler, region_cols = main(720)
    