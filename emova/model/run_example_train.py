import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import logging
import random
from typing import List, Dict, Optional, Tuple, Union
import json

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# 设置随机种子以确保可复现性
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)


# 自定义数据集类
class MultimodalSegmentationDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            tokenizer,
            image_processor,
            max_length: int = 512,
            image_size: int = 448,
    ):
        """
        数据集初始化

        Args:
            data_path: 数据文件路径，包含图像路径、文本和分割标签路径
            tokenizer: 用于文本tokenization的分词器
            image_processor: 图像处理器
            max_length: 文本最大长度
            image_size: 图像大小
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

        # 读取数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 图像转换
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 分割掩码转换
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        self.image_size = image_size

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载图像
        image_path = item["image_path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        # 获取原始图像尺寸
        original_size = image.size  # (width, height)

        # 加载分割掩码
        mask_path = item["mask_path"]
        mask = Image.open(mask_path)
        mask_tensor = self.mask_transform(mask)
        # 转换为长整型并移除通道维度
        mask_tensor = mask_tensor.squeeze().long()

        # 处理文本
        text = item["text"]
        text_encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = text_encoding["input_ids"].squeeze(0)
        attention_mask = text_encoding["attention_mask"].squeeze(0)

        # 创建标签，复制input_ids并将padding token替换为-100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": image_tensor,
            "image_sizes": original_size,  # (width, height)
            "segmentation_masks": mask_tensor,
            "text": text,  # 保留原始文本以便调试
        }


# 数据整理函数，用于批处理
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    images = torch.stack([item["images"] for item in batch])

    # 收集图像尺寸
    image_sizes = [item["image_sizes"] for item in batch]

    # 收集分割掩码
    segmentation_masks = torch.stack([item["segmentation_masks"] for item in batch])

    # 收集原始文本
    texts = [item["text"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": images,
        "image_sizes": image_sizes,
        "segmentation_masks": segmentation_masks,
        "texts": texts,
    }


# 评估函数
def evaluate(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_lm_loss = 0
    total_steps = 0

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # 将数据移至设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)
            segmentation_masks = batch["segmentation_masks"].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images,
                image_sizes=batch["image_sizes"],
                segmentation_masks=segmentation_masks,
                generate_segmentation=True,
                return_dict=True,
            )

            # 计算损失
            loss = outputs.loss
            seg_loss = outputs.segmentation_loss if hasattr(outputs, "segmentation_loss") else torch.tensor(0.0)
            lm_loss = loss - (
                seg_loss * model.segmentation_loss_weight if hasattr(model, "segmentation_loss_weight") else 0.0)

            total_loss += loss.item()
            total_seg_loss += seg_loss.item() if isinstance(seg_loss, torch.Tensor) else 0.0
            total_lm_loss += lm_loss.item()
            total_steps += 1

    # 计算平均损失
    avg_loss = total_loss / total_steps
    avg_seg_loss = total_seg_loss / total_steps
    avg_lm_loss = total_lm_loss / total_steps

    return {
        "loss": avg_loss,
        "segmentation_loss": avg_seg_loss,
        "lm_loss": avg_lm_loss,
    }


# 计算分割指标
def compute_segmentation_metrics(predictions, targets, num_classes):
    """
    计算分割指标：像素准确率和IoU

    Args:
        predictions: 预测的分割结果，形状为 [B, H, W]
        targets: 真实的分割标签，形状为 [B, H, W]
        num_classes: 分割类别数量

    Returns:
        dict: 包含像素准确率和IoU的字典
    """
    # 计算像素准确率
    mask = targets != 255  # 忽略255（通常用作忽略索引）
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()
    pixel_acc = correct / total if total > 0 else 0

    # 计算IoU
    ious = []
    for cls in range(num_classes):
        pred_mask = predictions == cls
        target_mask = targets == cls
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        iou = intersection / union if union > 0 else 0
        ious.append(iou)

    mean_iou = sum(ious) / len(ious)

    return {
        "pixel_acc": pixel_acc,
        "mean_iou": mean_iou,
        "class_ious": ious,
    }


# 主训练函数
def train():
    # 配置参数
    config = {
        "model_name_or_path": "Qwen/Qwen2-7B-Instruct",  # 或者使用已微调的模型路径
        "train_data_path": "path/to/train_data.json",
        "eval_data_path": "path/to/eval_data.json",
        "output_dir": "path/to/output_dir",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "fp16": True,
        "logging_steps": 10,
        "save_steps": 500,
        "eval_steps": 500,
        "max_length": 512,
        "image_size": 448,
        "segmentation_loss_weight": 1.0,  # 分割损失权重
        "num_segmentation_classes": 150,  # 分割类别数量
    }

    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])

    # 确保分词器有padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    from emova.model.language_model.emova_qwen2 import EmovaQwen2ForCausalLM
    from transformers import AutoConfig

    # 加载配置
    model_config = AutoConfig.from_pretrained(config["model_name_or_path"])

    # 设置分割相关配置
    model_config.segmentation_loss_weight = config["segmentation_loss_weight"]

    # 如果需要，可以设置类别权重来处理类别不平衡
    # model_config.segmentation_class_weights = [1.0] * config["num_segmentation_classes"]

    # 初始化模型
    model = EmovaQwen2ForCausalLM.from_pretrained(
        config["model_name_or_path"],
        config=model_config,
    )

    # 添加分割损失函数
    import torch.nn.functional as F

    def segmentation_loss_fn(
            pred_logits,
            target_masks,
            num_classes: int,
            ignore_index: int = 255,
            weight: Optional[torch.Tensor] = None,
            class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算语义分割损失
        """
        if isinstance(pred_logits, list):
            # 如果是批次中的多个图像，计算每个图像的损失并平均
            losses = []
            for i, logits in enumerate(pred_logits):
                if i < len(target_masks):
                    mask = target_masks[i]
                    loss = F.cross_entropy(
                        logits.unsqueeze(0),  # 添加批次维度
                        mask.unsqueeze(0).long(),  # 添加批次维度并转为long类型
                        weight=class_weights,
                        ignore_index=ignore_index,
                        reduction='mean'
                    )
                    losses.append(loss)

            if losses:
                return torch.stack(losses).mean()
            else:
                # 如果没有有效的损失，返回零张量
                return torch.tensor(0.0, device=pred_logits[0].device if pred_logits else torch.device('cuda'))
        else:
            # 标准的批次输入
            return F.cross_entropy(
                pred_logits,
                target_masks.long(),
                weight=class_weights,
                ignore_index=ignore_index,
                reduction='mean'
            )

    # 设置分割损失函数
    model._segmentation_loss_fn = segmentation_loss_fn

    # 添加分割头
    from torch import nn

    # 创建简单的分割头
    class SegmentationHead(nn.Module):
        def __init__(self, hidden_size, num_classes):
            super().__init__()
            self.segmentation_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_classes)
            )

        def forward(self, hidden_states, image_sizes=None):
            # 假设hidden_states的形状为[batch_size, seq_len, hidden_size]
            # 我们需要提取与图像对应的隐藏状态
            # 这里简化处理，假设图像token位于序列的开始部分

            # 获取批次大小和隐藏层大小
            batch_size, seq_len, hidden_size = hidden_states.shape

            # 假设每个图像使用固定数量的token表示
            image_tokens = 576  # 例如，对于448x448的图像，使用24x24=576个token

            # 提取图像token的隐藏状态
            image_hidden_states = hidden_states[:, :image_tokens, :]

            # 重塑为[batch_size, height, width, hidden_size]
            height = width = int(image_tokens ** 0.5)  # 假设是正方形
            image_hidden_states = image_hidden_states.reshape(batch_size, height, width, hidden_size)

            # 应用分割投影
            # 首先将hidden_size维度移到最前面以便于线性层处理
            image_hidden_states = image_hidden_states.permute(0, 3, 1, 2)  # [B, hidden_size, H, W]

            # 应用线性投影到每个像素
            batch_size, hidden_size, height, width = image_hidden_states.shape
            image_hidden_states = image_hidden_states.reshape(batch_size, hidden_size, -1)  # [B, hidden_size, H*W]
            image_hidden_states = image_hidden_states.permute(0, 2, 1)  # [B, H*W, hidden_size]

            # 应用分割投影
            logits = self.segmentation_proj(image_hidden_states)  # [B, H*W, num_classes]

            # 重塑回[batch_size, num_classes, height, width]
            logits = logits.permute(0, 2, 1)  # [B, num_classes, H*W]
            logits = logits.reshape(batch_size, -1, height, width)  # [B, num_classes, H, W]

            # 如果提供了原始图像尺寸，将分割结果调整为原始尺寸
            if image_sizes is not None:
                resized_logits = []
                for i, size in enumerate(image_sizes):
                    # size是(width, height)，而F.interpolate需要(height, width)
                    h, w = size[1], size[0]
                    resized = F.interpolate(
                        logits[i:i + 1],
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    )
                    resized_logits.append(resized.squeeze(0))

                # 预测分割掩码
                predictions = [logit.argmax(dim=0) for logit in resized_logits]

                return {
                    "segmentation_logits": resized_logits,
                    "segmentation_masks": predictions
                }

            # 预测分割掩码
            predictions = logits.argmax(dim=1)  # [B, H, W]

            return {
                "segmentation_logits": logits,
                "segmentation_masks": predictions
            }

    # 创建分割头并添加到模型
    segmentation_head = SegmentationHead(
        hidden_size=model.config.hidden_size,
        num_classes=config["num_segmentation_classes"]
    )

    # 设置分割头
    model.set_segmentation_head(segmentation_head)

    # 将模型移至设备
    model = model.to(device)

    # 加载数据集
    # 这里假设有一个简单的图像处理器
    class SimpleImageProcessor:
        def __call__(self, images):
            return images

    image_processor = SimpleImageProcessor()

    train_dataset = MultimodalSegmentationDataset(
        data_path=config["train_data_path"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config["max_length"],
        image_size=config["image_size"],
    )

    eval_dataset = MultimodalSegmentationDataset(
        data_path=config["eval_data_path"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=config["max_length"],
        image_size=config["image_size"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["per_device_eval_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # 计算训练步数
    num_update_steps_per_epoch = len(train_dataloader) // config["gradient_accumulation_steps"]
    max_train_steps = config["num_train_epochs"] * num_update_steps_per_epoch

    # 准备优化器和学习率调度器
    # 只训练分割头和LLM的部分参数
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if "segmentation_head" in n and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": config["learning_rate"],
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "segmentation_head" in n and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": config["learning_rate"],
        },
        # 添加LLM的最后几层
        {
            "params": [p for n, p in model.named_parameters()
                       if "model.layers" in n and int(n.split(".")[2]) >= len(model.model.layers) - 2
                       and not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
            "lr": config["learning_rate"] / 10,  # 对LLM使用较小的学习率
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if "model.layers" in n and int(n.split(".")[2]) >= len(model.model.layers) - 2
                       and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": config["learning_rate"] / 10,  # 对LLM使用较小的学习率
        },
    ]

    optimizer = optim.AdamW(optimizer_grouped_parameters)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=max_train_steps,
    )

    # 设置混合精度训练
    scaler = torch.cuda.amp.GradScaler() if config["fp16"] else None

    # 训练循环
    global_step = 0
    best_eval_loss = float("inf")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['num_train_epochs']}")
    logger.info(f"  Batch size per device = {config['per_device_train_batch_size']}")
    logger.info(f"  Gradient Accumulation steps = {config['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps))

    for epoch in range(config["num_train_epochs"]):
        model.train()
        total_loss = 0
        total_seg_loss = 0
        total_lm_loss = 0

        for step, batch in enumerate(train_dataloader):
            # 将数据移至设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            images = batch["images"].to(device)
            segmentation_masks = batch["segmentation_masks"].to(device)

            # 混合精度训练
            with torch.cuda.amp.autocast() if config["fp16"] else nullcontext():
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images,
                    image_sizes=batch["image_sizes"],
                    segmentation_masks=segmentation_masks,
                    generate_segmentation=True,
                    return_dict=True,
                )

                # 获取损失
                loss = outputs.loss
                seg_loss = outputs.segmentation_loss if hasattr(outputs, "segmentation_loss") else torch.tensor(0.0)
                lm_loss = loss - (
                    seg_loss * model.segmentation_loss_weight if hasattr(model, "segmentation_loss_weight") else 0.0)

                # 梯度累积
                loss = loss / config["gradient_accumulation_steps"]

            # 反向传播
            if config["fp16"]:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 更新统计信息
            total_loss += loss.item() * config["gradient_accumulation_steps"]
            total_seg_loss += seg_loss.item() if isinstance(seg_loss, torch.Tensor) else 0.0
            total_lm_loss += lm_loss.item()

            # 梯度累积
            if (step + 1) % config["gradient_accumulation_steps"] == 0 or step == len(train_dataloader) - 1:
                # 梯度裁剪
                if config["fp16"]:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

                # 参数更新
                if config["fp16"]:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # 学习率调度
                lr_scheduler.step()

                # 梯度清零
                optimizer.zero_grad()

                # 更新进度条
                progress_bar.update(1)
                global_step += 1

                # 日志记录
                if global_step % config["logging_steps"] == 0:
                    avg_loss = total_loss / config["logging_steps"]
                    avg_seg_loss = total_seg_loss / config["logging_steps"]
                    avg_lm_loss = total_lm_loss / config["logging_steps"]

                    logger.info(
                        f"Epoch: {epoch}, Step: {global_step}, "
                        f"Loss: {avg_loss:.4f}, Seg Loss: {avg_seg_loss:.4f}, LM Loss: {avg_lm_loss:.4f}, "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.8f}"
                    )

                    total_loss = 0
                    total_seg_loss = 0
                    total_lm_loss = 0

                # 评估
                if global_step % config["eval_steps"] == 0:
                    logger.info("***** Running evaluation *****")
                    eval_results = evaluate(model, eval_dataloader, device)

                    logger.info(
                        f"Eval Loss: {eval_results['loss']:.4f}, "
                        f"Eval Seg Loss: {eval_results['segmentation_loss']:.4f}, "
                        f"Eval LM Loss: {eval_results['lm_loss']:.4f}"
                    )

                    # 保存最佳模型
                    if eval_results["loss"] < best_eval_loss:
                        best_eval_loss = eval_results["loss"]
                        logger.info(f"New best model with eval loss: {best_eval_loss:.4f}")

                        # 保存模型
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(os.path.join(config["output_dir"], "best_model"))
                        tokenizer.save_pretrained(os.path.join(config["output_dir"], "best_model"))

                # 保存模型
                if global_step % config["save_steps"] == 0:
                    # 保存模型
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(os.path.join(config["output_dir"], f"checkpoint-{global_step}"))
                    tokenizer.save_pretrained(os.path.join(config["output_dir"], f"checkpoint-{global_step}"))

    # 保存最终模型
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(os.path.join(config["output_dir"], "final_model"))
    tokenizer.save_pretrained(os.path.join(config["output_dir"], "final_model"))

    logger.info("Training completed!")
    return model, tokenizer


# 上下文管理器，用于混合精度训练
class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# 执行训练
if __name__ == "__main__":
    model, tokenizer = train()