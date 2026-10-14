# train.py
import argparse
import os
import time
import glob
import pickle
import math
import random
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from datetime import datetime
from lib.dataset import create_dataloaders

from baselines.GAT_ODE_irregular.GATODE_ir import GATODE_ir
from config.baoding import BaodingConfig
from config.zhuzhou import ZhuzhouConfig
from lib.utils import StandardScaler
from tqdm import tqdm
from lib.visualize import save_all_visualizations


# ----------------------------------------------------------
# 新增：F-AAE 计算（只用真实周期）
# ----------------------------------------------------------
def compute_f_aae(pred_flow, y_flow, y_period, eps=1e-6):
    # 计算每秒流量率 (veh/s)
    pred_flow_rate = pred_flow / (y_period + eps)  # 预测的每秒流量
    y_flow_rate = y_flow / (y_period + eps)  # 真实的每秒流量

    # 按周期加权的绝对误差
    weighted_error = torch.abs(pred_flow_rate - y_flow_rate) * y_period
    total_period = y_period.sum()

    # 转换为 veh/min
    f_aae = (weighted_error.sum() / total_period) * 60 if total_period > 0 else 0
    return f_aae.item()


# ----------------------------------------------------------
# evaluate
# ----------------------------------------------------------
def evaluate(model, dataloader, loss_fn, standard_scaler, cfg, logger, desc="Evaluating", epoch=None):
    model.eval()
    total_mae = 0.0
    total_rmse = 0.0
    total_f_aae = 0.0
    total_samples = 0
    standard_scaler_flow, standard_scaler_period = standard_scaler

    progress_bar = tqdm(total=len(dataloader), desc=desc, leave=False)

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(cfg.DEVICE)
            obs_mask = batch['obs_mask'].to(cfg.DEVICE)
            pred_mask = batch['pred_mask'].to(cfg.DEVICE)

            x = features[:, :, :cfg.OBS_PATCH_LEN, :, :]
            y = features[:, :, cfg.OBS_PATCH_LEN:, :, :]
            x_mask = obs_mask[:, :, :cfg.OBS_PATCH_LEN, :]
            y_mask = pred_mask[:, :, cfg.OBS_PATCH_LEN:, :]
            real_period = features[:, :, cfg.OBS_PATCH_LEN:, :, -2].clone()  # 保存原始周期
            # 标准化 → 前向 → 反标准化
            x[..., -1] = standard_scaler_flow.transform(x[..., -1])
            x[..., -2] = standard_scaler_period.transform(x[..., -2])
            y[..., -2] = standard_scaler_period.transform(y[..., -2])
            preds = model(x, y, x_mask, y_mask)
            preds = standard_scaler_flow.inverse_transform(preds)

            targets = y[..., -1].squeeze(-1)  # 真实流量
            valid_mask = y_mask.bool()
            valid_preds = preds[valid_mask]
            valid_targets = targets[valid_mask]

            mae_loss = loss_fn(valid_preds, valid_targets)
            mse_loss = torch.mean((valid_preds - valid_targets) ** 2)
            rmse_loss = torch.sqrt(mse_loss)

            # F-AAE（使用真实周期）
            f_aae = compute_f_aae(pred_flow=valid_preds,
                                  y_flow=valid_targets,
                                  y_period=real_period[valid_mask],
                                  )

            batch_samples = valid_preds.numel()
            total_mae += mae_loss.item() * batch_samples
            total_rmse += rmse_loss.item() * batch_samples
            total_f_aae += f_aae * batch_samples  # 加权累加
            total_samples += batch_samples
            progress_bar.update(1)

        progress_bar.close()

    avg_mae = total_mae / total_samples if total_samples > 0 else float('inf')
    avg_rmse = total_rmse / total_samples if total_samples > 0 else float('inf')
    avg_f_aae = total_f_aae / total_samples if total_samples > 0 else float('inf')

    return avg_mae, avg_rmse, avg_f_aae


# ----------------------------------------------------------
# train
# ----------------------------------------------------------
def train(model, train_loader, val_loader, test_loader,
          optimizer, standard_scaler, loss_fn, cfg, logger):
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.LR_DECAY_STEP, gamma=cfg.LR_DECAY_RATE)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    standard_scaler_flow, standard_scaler_period = standard_scaler

    logger.info("=" * 50)
    logger.info(f"Starting training with config:")
    config_dict = {k: v for k, v in cfg.__dict__.items() if k.isupper()}
    for key, value in config_dict.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 50)

    for epoch in range(cfg.EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        total_samples = 0

        batch_progress = tqdm(total=len(train_loader),
                              desc=f"Epoch {epoch + 1}/{cfg.EPOCHS}", leave=False)

        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(cfg.DEVICE)
            obs_mask = batch['obs_mask'].to(cfg.DEVICE)
            pred_mask = batch['pred_mask'].to(cfg.DEVICE)

            x = features[:, :, :cfg.OBS_PATCH_LEN, :, :]
            y = features[:, :, cfg.OBS_PATCH_LEN:, :, :]
            x_mask = obs_mask[:, :, :cfg.OBS_PATCH_LEN, :]
            y_mask = pred_mask[:, :, cfg.OBS_PATCH_LEN:, :]

            x[..., -1] = standard_scaler_flow.transform(x[..., -1])
            x[..., -2] = standard_scaler_period.transform(x[..., -2])
            y[..., -2] = standard_scaler_period.transform(y[..., -2])
            preds = model(x, y, x_mask, y_mask, train=True)
            # 处理训练时返回的字典（包含ortho_loss）
            if isinstance(preds, dict):
                ortho_loss = preds.get('ortho_loss', None)
                preds = preds['prediction']
            else:
                ortho_loss = None
            preds = standard_scaler_flow.inverse_transform(preds)

            targets = y[..., -1].squeeze(-1)
            valid_preds = preds[y_mask.bool()]
            valid_targets = targets[y_mask.bool()]

            loss = loss_fn(valid_preds, valid_targets)
            if ortho_loss is not None and cfg.ORTHO_WEIGHT is not None:
                loss = loss + cfg.ORTHO_WEIGHT * ortho_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(features)
            total_samples += len(features)
            batch_progress.update(1)

        batch_progress.close()
        avg_train_loss = train_loss / total_samples
        epoch_time = time.time() - start_time
        scheduler.step()

        # 每 5 个 epoch 或最后一轮验证
        if epoch % 1 == 0 or epoch == cfg.EPOCHS - 1:
            val_mae, val_rmse, val_f_aae = evaluate(
                model, val_loader, loss_fn, standard_scaler, cfg, logger, "Validation",
                epoch=epoch)  #
            test_mae, test_rmse, test_f_aae = evaluate(
                model, test_loader, loss_fn, standard_scaler, cfg, logger, "Testing")

            log_msg = (f"\n[{epoch + 1:3d}/{cfg.EPOCHS}] "
                       f"Train: {avg_train_loss:.4f}  "
                       f"Val: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, AAE={val_f_aae:.4f} veh/min  "
                       f"Test: MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, AAE={test_f_aae:.4f} veh/min  "
                       f"({epoch_time:.1f}s)\n")
            logger.info(log_msg)

            # 早停仍用 MAE
            if val_mae < best_val_loss - cfg.MIN_DELTA or \
                    math.isclose(val_mae, best_val_loss - cfg.MIN_DELTA):
                best_val_loss = val_mae
                epochs_no_improve = 0
                best_epoch = epoch
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': standard_scaler,
                    'cfg': cfg,
                    'val_mae': val_mae,
                    'val_rmse': val_rmse,
                    'val_f_aae': val_f_aae,
                }, os.path.join(cfg.SAVE_DIR, "best_model.pth"))
                logger.info(
                    f"  -> Saved best model (epoch {epoch + 1}, val_mae={val_mae:.4f}, val_aae={val_f_aae:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch + 1}, "
                                f"best val MAE {best_val_loss:.4f} (epoch {best_epoch + 1})")
                    break
        else:
            logger.info(f"[{epoch + 1:3d}/{cfg.EPOCHS}] Train: {avg_train_loss:.4f} ({epoch_time:.1f}s)")

    return best_val_loss


# ----------------------------------------------------------
# logger
# ----------------------------------------------------------
def setup_logger(cfg):
    log_dir = cfg.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger("GATODE_Training")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# ----------------------------------------------------------
# main
# ----------------------------------------------------------
def main(args):
    cfg = None
    if args.city == "baoding":
        cfg = BaodingConfig()
    elif args.city == "zhuzhou":
        cfg = ZhuzhouConfig()
    else:
        raise ValueError("Invalid city name!")

    seed = cfg.SEED
    logger = setup_logger(cfg)
    logger.info("=" * 50)
    logger.info("Initializing training process...")
    logger.info(f"Random seed: {seed}")

    cfg.DEVICE = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {cfg.DEVICE}")

    standard_scaler_flow = StandardScaler(mean=cfg.MEAN_FLOW, std=cfg.STD_FLOW, device=cfg.DEVICE)
    standard_scaler_period = StandardScaler(mean=cfg.MEAN_PERIOD, std=cfg.STD_PERIOD, device=cfg.DEVICE)
    standard_scaler = (standard_scaler_flow, standard_scaler_period)

    with open(cfg.DATA_DIR, 'rb') as f:
        data_list = pickle.load(f)
    logger.info(f"Data loaded from {cfg.DATA_DIR}")

    train_loader, val_loader, test_loader = create_dataloaders(data_list, cfg)
    logger.info(f"Data split: {len(train_loader.dataset)} train, "
                f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    model = GATODE_ir(**cfg.MODEL_PARAMS).to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = nn.L1Loss()

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Optimizer: {optimizer.__class__.__name__} (lr={cfg.LEARNING_RATE})")
    logger.info(f"Loss: {loss_fn.__class__.__name__}")
    logger.info("=" * 50)
    logger.info("Starting training ...")

    try:
        best_val_loss = train(model, train_loader, val_loader, test_loader,
                              optimizer, standard_scaler, loss_fn, cfg, logger)
        logger.info(f"Training done. Best val MAE: {best_val_loss:.4f}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

    best_path = os.path.join(cfg.SAVE_DIR, "best_model.pth")
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=cfg.DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        test_mae, test_rmse, test_f_aae = evaluate(
            model, test_loader, loss_fn, standard_scaler, cfg, logger, "Final Testing")
        logger.info(f"Best model Test → MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, AAE={test_f_aae:.4f} veh/min")

    final_path = os.path.join(cfg.SAVE_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")
    logger.info("=" * 50)
    logger.info("Training process completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='baoding')
    args = parser.parse_args()
    main(args)