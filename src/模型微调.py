#!/usr/bin/env python3
"""
DECIMER 模型微调脚本
基于你的 pubchem_smiles.csv 和对应图片进行微调
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from datetime import datetime

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "fine_tune_data_sets"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"

# 创建必要目录
LOGS_DIR.mkdir(exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f'finetuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DecipherDataset(Dataset):
    """DECIMER 数据集类"""
    
    def __init__(self, smiles_data, images_dir, transform=None, max_length=512):
        self.smiles_data = smiles_data
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.max_length = max_length
        
        # 构建字符到索引的映射
        self.char_to_idx = self._build_vocab()
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
    def _build_vocab(self):
        """构建SMILES字符词汇表"""
        # SMILES常用字符
        chars = set()
        for smiles in self.smiles_data['smiles']:
            chars.update(list(smiles))
        
        # 添加特殊字符
        chars.add('<PAD>')  # 填充
        chars.add('<SOS>')  # 开始
        chars.add('<EOS>')  # 结束
        chars.add('<UNK>')  # 未知
        
        char_to_idx = {char: idx for idx, char in enumerate(sorted(chars))}
        logger.info(f"词汇表大小: {len(char_to_idx)}")
        
        return char_to_idx
    
    def __len__(self):
        return len(self.smiles_data)
    
    def __getitem__(self, idx):
        row = self.smiles_data.iloc[idx]
        
        # 加载图片
        img_path = self.images_dir / f"{idx}.png"
        if not img_path.exists():
            logger.warning(f"图片不存在: {img_path}")
            # 创建空白图片作为占位符
            image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 处理SMILES字符串
        smiles = str(row['smiles'])
        smiles_tokens = ['<SOS>'] + list(smiles) + ['<EOS>']
        
        # 转换为索引
        smiles_indices = []
        for char in smiles_tokens:
            if char in self.char_to_idx:
                smiles_indices.append(self.char_to_idx[char])
            else:
                smiles_indices.append(self.char_to_idx['<UNK>'])
        
        # 填充或截断到固定长度
        if len(smiles_indices) < self.max_length:
            smiles_indices.extend([self.char_to_idx['<PAD>']] * (self.max_length - len(smiles_indices)))
        else:
            smiles_indices = smiles_indices[:self.max_length]
        
        return {
            'image': image,
            'smiles_indices': torch.tensor(smiles_indices, dtype=torch.long),
            'smiles_text': smiles
        }

class DecipherFineTuner:
    """DECIMER 微调器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 数据变换
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_data(self):
        """加载和预处理数据"""
        logger.info("加载数据...")
        
        # 读取SMILES数据
        csv_path = DATA_DIR / "pubchem_smiles.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到CSV文件: {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"CSV总共有 {len(df)} 个样本")
        
        # 检查必要的列
        if 'smiles' not in df.columns:
            logger.error("CSV文件中没有 'smiles' 列")
            raise ValueError("CSV文件必须包含 'smiles' 列")
        
        # 只取前50000行（对应你生成的图片）
        df = df.head(50000)
        logger.info(f"使用前50000行数据，对应生成的图片")
        
        # 过滤掉空的SMILES
        df = df.dropna(subset=['smiles'])
        df = df[df['smiles'].str.strip() != '']
        logger.info(f"过滤后剩余 {len(df)} 个有效样本")
        
        # 重置索引以匹配图片文件名 (0.png, 1.png, ...)
        df = df.reset_index(drop=True)
        
        # 验证图片存在性
        images_dir = DATA_DIR / "molecule_images"
        missing_images = 0
        for i in range(len(df)):
            img_path = images_dir / f"{i}.png"
            if not img_path.exists():
                missing_images += 1
        
        if missing_images > 0:
            logger.warning(f"有 {missing_images} 个图片文件缺失")
        
        logger.info(f"图片目录: {images_dir}")
        logger.info(f"预期图片范围: 0.png - {len(df)-1}.png")
        
        # 分割数据集
        train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        logger.info(f"数据分割: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
        
        # 创建数据集
        images_dir = DATA_DIR / "molecule_images"  # 使用你的实际图片目录
        
        self.train_dataset = DecipherDataset(train_data, images_dir, self.train_transform)
        self.val_dataset = DecipherDataset(val_data, images_dir, self.val_transform)
        self.test_dataset = DecipherDataset(test_data, images_dir, self.val_transform)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return self.train_dataset.char_to_idx
    
    def load_pretrained_model(self):
        """加载预训练的DECIMER模型"""
        logger.info("加载预训练模型...")
        
        # 这里需要根据实际的DECIMER模型结构来加载
        # 由于DECIMER可能使用TensorFlow，我们需要创建一个PyTorch版本
        # 或者使用转换工具
        
        # 临时创建一个简单的编码器-解码器结构作为示例
        from torchvision.models import resnet50
        
        # 图像编码器
        self.encoder = resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 512)
        
        # SMILES解码器
        vocab_size = len(self.train_dataset.char_to_idx)
        self.decoder = nn.LSTM(
            input_size=512, 
            hidden_size=512, 
            num_layers=2, 
            batch_first=True,
            dropout=0.2
        )
        self.output_projection = nn.Linear(512, vocab_size)
        
        # 移到设备
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.output_projection = self.output_projection.to(self.device)
        
        logger.info(f"模型加载完成，词汇表大小: {vocab_size}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            smiles_indices = batch['smiles_indices'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 图像编码
            image_features = self.encoder(images)  # [batch_size, 512]
            image_features = image_features.unsqueeze(1).repeat(1, smiles_indices.size(1), 1)  # [batch_size, seq_len, 512]
            
            # SMILES解码
            decoder_output, _ = self.decoder(image_features)  # [batch_size, seq_len, 512]
            predictions = self.output_projection(decoder_output)  # [batch_size, seq_len, vocab_size]
            
            # 计算损失
            loss = nn.CrossEntropyLoss(ignore_index=self.train_dataset.char_to_idx['<PAD>'])(
                predictions.view(-1, predictions.size(-1)),
                smiles_indices.view(-1)
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.output_projection.parameters()), 
                max_norm=1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # 记录训练步骤
            if batch_idx % self.config['log_steps'] == 0:
                logger.info(f"Epoch {epoch}, Step {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch} 平均训练损失: {avg_loss:.4f}")
        return avg_loss
    
    def validate(self, epoch):
        """验证模型"""
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                smiles_indices = batch['smiles_indices'].to(self.device)
                
                # 前向传播
                image_features = self.encoder(images)
                image_features = image_features.unsqueeze(1).repeat(1, smiles_indices.size(1), 1)
                
                decoder_output, _ = self.decoder(image_features)
                predictions = self.output_projection(decoder_output)
                
                # 计算损失
                loss = nn.CrossEntropyLoss(ignore_index=self.train_dataset.char_to_idx['<PAD>'])(
                    predictions.view(-1, predictions.size(-1)),
                    smiles_indices.view(-1)
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Epoch {epoch} 验证损失: {avg_loss:.4f}")
        return avg_loss
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练...")
        
        # 设置优化器
        params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.output_projection.parameters())
        self.optimizer = optim.AdamW(params, lr=self.config['learning_rate'], weight_decay=0.01)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate(epoch)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                logger.info(f"新的最佳模型！验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
                self.save_checkpoint(epoch, train_loss, val_loss, is_best=False)
            
            # 早停
            if patience_counter >= self.config['patience']:
                logger.info(f"早停触发，{patience_counter} epochs 无改进")
                break
        
        logger.info("训练完成！")
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """保存模型检查点"""
        checkpoint_dir = MODELS_DIR / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'output_projection_state_dict': self.output_projection.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab': self.train_dataset.char_to_idx
        }
        
        # 保存当前检查点
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已保存: {best_path}")

def main():
    """主函数"""
    
    # 训练配置
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'epochs': 30,
        'patience': 5,
        'log_steps': 100,
        'use_gpu': True
    }
    
    logger.info("=== DECIMER 模型微调开始 ===")
    logger.info(f"配置: {config}")
    
    try:
        # 创建微调器
        finetuner = DecipherFineTuner(config)
        
        # 加载数据
        vocab = finetuner.load_data()
        logger.info(f"数据加载完成，词汇表大小: {len(vocab)}")
        
        # 加载预训练模型
        finetuner.load_pretrained_model()
        
        # 开始训练
        finetuner.train()
        
        logger.info("=== 微调完成 ===")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main()