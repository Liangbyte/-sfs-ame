## -*- coding: utf-8 -*-
import os, sys
sys.setrecursionlimit(15000)
import torch
import csv
import numpy as np
import random
import yaml
import json
import pprint
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import time
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import logging
from tqdm import tqdm
from dataset import FFPP_Dataset,TestDataset
import timm
from utils import *
from ViT_MoE import *
from models.DualStream import DualStreamVITLSNet



def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _extract_inputs_labels(batch):
    """从 DataLoader 返回的 batch 中提取 (inputs, labels)。
    支持 tuple/list 和 dict 两种常见格式。
    """
    # 列表或元组直接取前两个元素
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    # 字典：优先按键名匹配
    if isinstance(batch, dict):
        img_key = None
        lbl_key = None
        for k in batch.keys():
            kl = k.lower()
            if img_key is None and ('img' in kl or 'image' in kl or 'input' in kl):
                img_key = k
            if lbl_key is None and ('label' in kl or 'target' in kl):
                lbl_key = k
        if img_key is not None:
            inputs = batch[img_key]
        else:
            inputs = None
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    inputs = v
                    break
            if inputs is None:
                raise ValueError(f'Cannot find tensor inputs in batch dict keys: {list(batch.keys())}')
        if lbl_key is not None:
            labels = batch[lbl_key]
        else:
            tensor_vals = [v for v in batch.values() if isinstance(v, torch.Tensor)]
            if len(tensor_vals) >= 2:
                if tensor_vals[0] is inputs:
                    labels = tensor_vals[1]
                else:
                    labels = tensor_vals[0]
            else:
                raise ValueError(f'Cannot find labels in batch dict keys: {list(batch.keys())}')
        return inputs, labels
    raise ValueError('Unexpected batch type from DataLoader: {}'.format(type(batch)))


def move_to_device(obj, device):
    """Recursively move tensors in obj to device. Supports tensor, dict, list, tuple."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        moved = [move_to_device(v, device) for v in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    return obj


def train(args, model, optimizer, train_loader, valid_loader, scheduler, save_dir):
    max_accuracy = 0
    global_step = 0
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # ensure we know which device the model lives on
    device = next(model.parameters()).device

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # checkpoint
    if args.resume > -1:
        checkpoint = torch.load(os.path.join(save_dir, 'models_params_{}.tar'.format(args.resume)),
                                map_location='cuda:{}'.format(torch.cuda.current_device()))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))

        # Fix: Set initial_lr for each param_group (required by scheduler when resuming)
        for i, pg in enumerate(optimizer.param_groups):
            if 'initial_lr' not in pg:
                pg['initial_lr'] = pg['lr']

    for epoch in range(args.resume + 1, args.epochs):
        # train part
        print(f'\n=== Epoch {epoch+1}/{args.epochs} - Training ===')
        epoch_loss = 0.0
        total_num = 0
        correct_num = 0
        model.train()

        with torch.enable_grad():
            st_time = time.time()
            # track last seen loss components for logging
            last_ce_loss = 0.0
            last_moe_loss = 0.0
            last_xray_mse = 0.0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, labels = _extract_inputs_labels(batch)
                # Move inputs/labels to model device and normalize input shape for model: expect (B, C, H, W)
                inputs = move_to_device(inputs, device)
                labels = move_to_device(labels, device)

                if isinstance(inputs, torch.Tensor):
                    inp_dim = inputs.dim()
                    if inp_dim == 5:
                        # (B, T, C, H, W) -> merge B and T
                        B, T, C, H, W = inputs.shape
                        inputs_model = inputs.view(B * T, C, H, W)
                        labels_device = labels
                        outputs, moe_loss = model(inputs_model)
                        # outputs shape [B*T, num_classes] -> reshape to [B, T, num_classes]
                        outputs = outputs.view(B, T, -1)
                        outputs_mean = outputs.mean(dim=1)
                        ce_loss = criterion(outputs_mean, labels_device)
                        loss = ce_loss + 1 * moe_loss
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        total_num += B
                        correct_num += torch.sum(torch.argmax(outputs_mean, 1) == labels_device).item()
                        # update last loss components
                        last_ce_loss = float(ce_loss.item())
                        last_moe_loss = float(moe_loss.item())
                        last_xray_mse = 0.0
                    elif inp_dim == 4:
                        inputs_model = inputs
                        labels_device = labels
                        # if batch provides xray and boundary mask, compute xray_pred and include MSE loss
                        has_xray = (not getattr(args, 'disable_xray', False)) and isinstance(batch, dict) and (batch.get('xray') is not None) and ((batch.get('if_boundary') is not None) or (batch.get('if_boundaries') is not None))
                        # support wrapper: xray head may live at model.vit
                        has_mask_head = (getattr(model, 'xray_postprocess', None) is not None) or (getattr(getattr(model, 'vit', None), 'xray_postprocess', None) is not None)
                        if has_xray and has_mask_head:
                            if_bound = batch.get('if_boundary') if batch.get('if_boundary') is not None else batch.get('if_boundaries')
                            if_bound = move_to_device(if_bound, device)
                            xray_gt = move_to_device(batch.get('xray'), device)
                            outputs, moe_loss, xray_pred = model(inputs_model, if_boundaries=if_bound, return_xray=True)
                            ce_loss = criterion(outputs, labels_device)
                            try:
                                xray_mse = F.mse_loss(xray_pred.squeeze().float(), xray_gt.squeeze().float())
                            except Exception:
                                xray_mse = torch.zeros(1).to(device)
                            loss = ce_loss + 1 * moe_loss + args.xray_loss_weight * xray_mse
                            # xray loss logging removed (silent per-batch)
                            # update last loss components
                            last_ce_loss = float(ce_loss.item())
                            last_moe_loss = float(moe_loss.item())
                            last_xray_mse = float(xray_mse.item())
                        else:
                            outputs, moe_loss = model(inputs_model)
                            ce_loss = criterion(outputs, labels_device)
                            loss = ce_loss + 1 * moe_loss
                            # update last loss components
                            last_ce_loss = float(ce_loss.item())
                            last_moe_loss = float(moe_loss.item())
                            last_xray_mse = 0.0
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        total_num += inputs_model.size(0)
                        correct_num += torch.sum(torch.argmax(outputs, 1) == labels_device).item()
                    elif inp_dim == 3:
                        # single sample (C,H,W) -> add batch dim
                        inputs_model = inputs.unsqueeze(0)
                        labels_device = labels
                        has_xray = (not getattr(args, 'disable_xray', False)) and isinstance(batch, dict) and (batch.get('xray') is not None) and ((batch.get('if_boundary') is not None) or (batch.get('if_boundaries') is not None))
                        has_mask_head = (getattr(model, 'xray_postprocess', None) is not None) or (getattr(getattr(model, 'vit', None), 'xray_postprocess', None) is not None)
                        if has_xray and has_mask_head:
                            if_bound = batch.get('if_boundary') if batch.get('if_boundary') is not None else batch.get('if_boundaries')
                            if_bound = move_to_device(if_bound, device)
                            xray_gt = move_to_device(batch.get('xray'), device)
                            outputs, moe_loss, xray_pred = model(inputs_model, if_boundaries=if_bound, return_xray=True)
                            ce_loss = criterion(outputs, labels_device)
                            try:
                                xray_mse = F.mse_loss(xray_pred.squeeze().float(), xray_gt.squeeze().float())
                            except Exception:
                                xray_mse = torch.zeros(1).to(device)
                            loss = ce_loss + 1 * moe_loss + args.xray_loss_weight * xray_mse
                            # update last loss components
                            last_ce_loss = float(ce_loss.item())
                            last_moe_loss = float(moe_loss.item())
                            last_xray_mse = float(xray_mse.item())
                        else:
                            outputs, moe_loss = model(inputs_model)
                            ce_loss = criterion(outputs, labels_device)
                            loss = ce_loss + 1 * moe_loss
                            # update last loss components
                            last_ce_loss = float(ce_loss.item())
                            last_moe_loss = float(moe_loss.item())
                            last_xray_mse = 0.0
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        total_num += inputs_model.size(0)
                        correct_num += torch.sum(torch.argmax(outputs, 1) == labels_device).item()
                    else:
                        raise ValueError(f'Unsupported input tensor ndim: {inp_dim}')
                else:
                    raise ValueError(f'Unsupported input type: {type(inputs)}')
                global_step += 1

                # record train stat into tensorboardX
                if global_step % args.record_step == 0:
                    period = time.time() - st_time
                    # compute per-component last-batch loss values
                    total_last = last_ce_loss + last_moe_loss + args.xray_loss_weight * last_xray_mse
                    denom = float(args.lora_loss_weight) if hasattr(args, 'lora_loss_weight') and float(args.lora_loss_weight) != 0 else 1.0
                    lora_last = last_moe_loss / denom if last_moe_loss is not None else 0.0
                    log.info('Training state: Epoch [{:0>3}/{:0>3}], Iteration [{:0>3}/{:0>3}], '
                             'Loss(total): {:.4f} | CE: {:.4f} | MoE: {:.4f} | LoRA: {:.6f} | Xray(w): {:.4f} | Acc:{:.2%} time:{}m {}s'
                             .format(epoch + 1, args.epochs, i + 1, len(train_loader), epoch_loss / (i + 1),
                                     last_ce_loss, last_moe_loss, lora_last, args.xray_loss_weight * last_xray_mse,
                                     correct_num / total_num, int(period // 60), int(period % 60)))
                    st_time = time.time()
                    total_num = 0
                    correct_num = 0
        # eval part
        model.eval()
        video_predictions = []
        video_labels = []
        frame_predictions = []
        frame_labels = []

        with torch.no_grad():
            val_sample_count = 0
            val_start_time = time.time()
            for batch in tqdm(valid_loader, total=len(valid_loader), ncols=70, leave=False, unit='step'):
                inputs, labels = _extract_inputs_labels(batch)
                inputs = move_to_device(inputs, device)
                labels = move_to_device(labels, device)
                # remove outer batch dim if present (DataLoader with batch_size=1)
                inputs = inputs.squeeze(0)

                # ensure inputs has batch dim before feeding model
                if inputs.dim() == 3:
                    # (C, H, W) -> (1, C, H, W)
                    inputs_model = inputs.unsqueeze(0)
                    outputs, _ = model(inputs_model)
                    outputs = F.softmax(outputs, dim=-1)
                    frame = outputs.shape[0]
                    frame_predictions.extend(outputs[:, 1].cpu().tolist())
                    frame_labels.extend(labels.expand(frame).cpu().tolist())
                    pre = torch.mean(outputs[:, 1])
                    video_predictions.append(pre.cpu().item())
                    video_labels.append(labels.cpu().item())
                    try:
                        val_sample_count += labels.numel() if hasattr(labels, 'numel') else 1
                    except Exception:
                        val_sample_count += 1
                elif inputs.dim() == 4:
                    # (T, C, H, W) or (B, C, H, W)
                    # treat first dim as temporal frames
                    # if shape is (T, C, H, W) -> pass directly as batch of frames
                    inputs_model = inputs
                    outputs, _ = model(inputs_model)
                    outputs = F.softmax(outputs, dim=-1)
                    frame = outputs.shape[0]
                    frame_predictions.extend(outputs[:, 1].cpu().tolist())
                    frame_labels.extend(labels.expand(frame).cpu().tolist())
                    pre = torch.mean(outputs[:, 1])
                    video_predictions.append(pre.cpu().item())
                    video_labels.append(labels.cpu().item())
                    try:
                        val_sample_count += labels.numel() if hasattr(labels, 'numel') else 1
                    except Exception:
                        val_sample_count += 1
                else:
                    raise ValueError(f'Unsupported validation input ndim: {inputs.dim()}')

        val_time = time.time() - val_start_time
        frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
        video_results = cal_metrics(video_labels, video_predictions, threshold=0.5)

        log.info('valid result: Epoch [{:0>3}/{:0>3}], V_Acc: {:.2%}, V_Auc: {:.4} V_EER:{:.2%} F_Acc: {:.2%}, F_Auc: {:.4} F_EER:{:.2%}'
                 .format(epoch + 1, args.epochs, video_results.ACC, video_results.AUC, video_results.EER, frame_results.ACC, frame_results.AUC, frame_results.EER))
        
        # write per-epoch metrics to CSV under save_dir
        try:
            csv_path = os.path.join(save_dir, 'result.csv')
            file_exists = os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['epoch', 'V_Acc', 'V_Auc', 'V_EER', 'F_Acc', 'F_Auc', 'F_EER'])
                writer.writerow([
                    int(epoch + 1),
                    float(video_results.ACC), float(video_results.AUC), float(video_results.EER),
                    float(frame_results.ACC), float(frame_results.AUC), float(frame_results.EER)
                ])
        except Exception:
            pass

        # save model
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}
        # allow skipping disk writes for fast debug runs by setting environment var NO_SAVE=1
        if os.environ.get('NO_SAVE', '0') != '1':
            ckpt_path = os.path.join(save_dir, 'models_params_{}.tar'.format(epoch))
            torch.save(state, ckpt_path)
            
            # Verify CAS values in saved checkpoint (epoch 0 only for debugging)
            if epoch == 0:
                try:
                    saved = torch.load(ckpt_path, map_location='cpu')
                    saved_state = saved['model_state_dict']
                    cas_in_ckpt = {k: v.item() for k, v in saved_state.items() if 'cas_beta_v' in k and v.numel() == 1}
                    if len(cas_in_ckpt) > 0:
                        avg_val = sum(abs(v) for v in cas_in_ckpt.values()) / len(cas_in_ckpt)
                        print(f'[VERIFY] Saved checkpoint epoch={epoch}: {len(cas_in_ckpt)} CAS betas, avg={avg_val:.6f}')
                except Exception:
                    pass

        #  Loss, accuracy
        if video_results.AUC > max_accuracy:
            for m in os.listdir(save_dir):
                if m.startswith('model_params_best'):
                    curent_models = m
                    os.remove(os.path.join(save_dir, curent_models))
            max_accuracy = video_results.AUC
            # skip saving best model when NO_SAVE=1 (useful for fast debugging)
            if os.environ.get('NO_SAVE', '0') != '1':
                best_name = f"model_params_best_{video_results.ACC:.4f}auc{video_results.AUC:.4f}epoch{epoch+1:03d}.pkl"
                torch.save(model.state_dict(), os.path.join(save_dir, best_name))

        # Verify CAS values at end of epoch
        if epoch == 0:
            try:
                cas_after_epoch = []
                for n, p in model.named_parameters():
                    if 'cas_beta_v' in n:
                        cas_after_epoch.append(p.data.item())
                if len(cas_after_epoch) > 0:
                    avg_cas = sum(abs(v) for v in cas_after_epoch) / len(cas_after_epoch)
                    print(f'[DEBUG] CAS values BEFORE saving epoch {epoch}: avg={avg_cas:.8f}, values={cas_after_epoch}')
            except Exception:
                pass

        # Print gating statistics at end of each epoch
        try:
            print(f'\n{"="*80}')
            print(f'LoRA GATING STATISTICS - Epoch {epoch+1}/{args.epochs}')
            print(f'{"="*80}')
            
            # Collect gating parameters
            flat_gates = []
            for name, param in model.named_parameters():
                if 'LoRA_MoE.w_gate' in name or 'LoRA_MoE.w_noise' in name:
                    flat_gates.append((name, param.data.cpu()))
            
            if len(flat_gates) > 0:
                all_flat_vals = torch.cat([p.flatten() for _, p in flat_gates])
                print(f'\n{"LoRA Gates (w_gate)":<40} | {"Statistics":<35}')
                print(f'{"-"*80}')
                print(f'{"Total Parameters":<40} | {all_flat_vals.numel():>10,}')
                print(f'{"Mean":<40} | {all_flat_vals.mean().item():>10.6f}')
                print(f'{"Std":<40} | {all_flat_vals.std().item():>10.6f}')
                print(f'{"Min":<40} | {all_flat_vals.min().item():>10.6f}')
                print(f'{"Max":<40} | {all_flat_vals.max().item():>10.6f}')
                print(f'{"Abs Mean (activation strength)":<40} | {all_flat_vals.abs().mean().item():>10.6f}')
                
                if epoch > 0:
                    nonzero_ratio = (all_flat_vals.abs() > 1e-6).float().mean().item() * 100
                    print(f'\n{"Gate Update Status":<40} | {"Value":<35}')
                    print(f'{"-"*80}')
                    print(f'{"Non-zero gate ratio":<40} | {nonzero_ratio:>9.2f}%')
                    if nonzero_ratio < 1.0:
                        print(f'{"Status":<40} | ⚠ Gates NOT updating!')
                    else:
                        print(f'{"Status":<40} | ✓ Gates are updating')
            
            print(f'{"="*80}\n')
        except Exception as e:
            print(f'[WARNING] Failed to print gating statistics: {e}')

        scheduler.step()
    return max_accuracy


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device','-dv', type=int, default=0, help="specify which GPU to use")
    parser.add_argument('--model_dir', '-md', type=str, default='/root/autodl-fs/models/train')
    parser.add_argument('--resume','-rs', type=int, default=-1, help="which epoch continue to train")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--record_step', type=int, default=100, help="the iteration number to record train state")

    parser.add_argument('--batch_size','-bs', type=int, default=32)
    parser.add_argument('--learning_rate','-lr', type=float, default=3e-5)
    parser.add_argument('--shared_lr_scale', type=float, default=1.0,
                        help='Shared LoRA 学习率缩放系数，相对基础 2e-5')
    parser.add_argument('--specific_lr_scale', type=float, default=1.0,
                        help='Specific LoRA 学习率缩放系数，相对基础 2e-5')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_frames', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--xray_loss_weight', type=float, default=200.0, help='weight for xray MSE loss')
    parser.add_argument('--lora_loss_weight', type=float, default=200.0, help='weight for LoRA loss inside moe_loss')
    # Optional: override ForensicsAdapter FF++ dataset with a specific json list (e.g., DF.json)
    parser.add_argument('--ffpp_json', type=str, default=None,
                        help='Optional FF++ json list for training/validation (when using ForensicsAdapter). '
                             'If provided, overrides train.yaml train_dataset/test_dataset to use only this json; '
                             'if omitted, keep original train.yaml datasets.')
    # CAS 控制：--cas_on 开关，--cas_layers 指定开启的层数（优先级更高）
    parser.add_argument('--cas_on', action='store_true', help='启用 CAS 旁路（默认前4层）')
    parser.add_argument('--cas_layers', type=int, default=None, help='启用 CAS 的层数，优先级高于 --cas_on，0 表示全关')
    # Shared + Specific LoRA MoE 控制
    parser.add_argument('--use_svd_init', action='store_true', help='使用SVD初始化Specific Experts（推荐开启以获得更好的收敛）')
    parser.add_argument('--svd_strategy', type=str, default='cumulative', choices=['cumulative', 'partitioned'], 
                        help='SVD分配策略: cumulative(累积,推荐) or partitioned(分段,正交)')
    parser.add_argument('--shared_rank', type=int, default=64, help='Shared Expert的rank大小（默认64）')
    parser.add_argument('--lora_specific_only', action='store_true',
                        help='只训练 specific LoRA 专家: 冻结 shared_lora 参数，并在前向中仅使用 specific experts')
    parser.add_argument('--lora_shared_only', action='store_true',
                        help='只训练 shared LoRA 专家: 冻结 specific LoRA 及其 gating，并在前向中仅使用 shared expert')
    # 频域分支：RGB逐通道DWT开关（默认False=灰度DWT）
    parser.add_argument('--use_rgb_dwt', action='store_true', help='启用RGB逐通道DWT，拼接高频子带后用1x1卷积降维')
    # 双向cross-attention强度系数
    parser.add_argument('--cross_alpha_v', type=float, default=0.2)
    parser.add_argument('--cross_alpha_f', type=float, default=0.2)
    # Ablation flags
    parser.add_argument('--disable_xray', action='store_true', help='关闭mask/xray分支的监督损失（不使用xray MSE）')
    parser.add_argument('--disable_dwt_cnn', action='store_true', help='关闭灰度DWT频域CNN分支（使用单流ViT）')
    parser.add_argument('--disable_cafm', action='store_true', help='关闭CAFM双向交叉注意力（cross_alpha_v/f 强制为 0）')
    parser.add_argument('--disable_ss_lora', action='store_true', help='关闭Shared+Specific LoRA MoE（lora_topk=0）')
    # Stage-2 training: initialize from a shared-only model and freeze shared LoRA
    parser.add_argument('--init_from_shared', type=str, default=None,
                        help='路径指向已经训练好的 shared-only 模型权重，用于第二阶段只训练 specific 专家（会自动冻结 shared_lora 参数）')
    args = parser.parse_args()



    start_time = time.time()
    setup_seed(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.device)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    # create a run-specific timestamped subdirectory under the provided model_dir
    # When resuming, use the provided model_dir directly (don't create new timestamp)
    model_root = args.model_dir
    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)
    
    if args.resume > -1:
        # Resume mode: use model_dir as save_dir directly
        save_dir = model_root
        print(f'Resume mode: using existing directory {save_dir}')
    else:
        # New training: create timestamped subdirectory
        run_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        save_dir = os.path.join(model_root, run_stamp)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    # logging
    # When NO_SAVE=1: skip writing a log file (useful for fast/debug runs)
    no_save = os.environ.get('NO_SAVE', '0') == '1'
    if no_save:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s'
        )
    else:
        # create a new timestamped log file for each run to avoid overwriting previous logs
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        log_filename = os.path.join(save_dir, f'train_{timestamp}.log')
        logging.basicConfig(
            filename=log_filename,
            filemode='w',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO)
        # also expose path for easier discovery
        print(f'Logging to: {log_filename}')
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    # ensure console output is always present
    if not any(isinstance(h, logging.StreamHandler) for h in log.handlers):
        handler = logging.StreamHandler()
        log.addHandler(handler)
    # always append runtime logs to train_run.log in the project folder
    try:
        run_log_path = os.path.join(os.path.dirname(__file__), 'train_run.log')
        # create FileHandler and add if not already present
        fh = logging.FileHandler(run_log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s'))
        # avoid duplicate file handler
        add_fh = True
        for h in log.handlers:
            if isinstance(h, logging.FileHandler):
                try:
                    if os.path.abspath(getattr(h, 'baseFilename', '')) == os.path.abspath(run_log_path):
                        add_fh = False
                        break
                except Exception:
                    pass
        if add_fh:
            log.addHandler(fh)
            print(f'Also appending runtime logs to: {run_log_path}')
    except Exception:
        # non-fatal if file handler cannot be created
        pass
    logging.info(args.model_dir)
    log.info('model dir:' + args.model_dir)

    # Log current ablation configuration for this run
    try:
        mask_on = not getattr(args, 'disable_xray', False)
        dwt_on = not getattr(args, 'disable_dwt_cnn', False)
        cafm_on = (not getattr(args, 'disable_cafm', False)) and dwt_on
        ss_lora_on = not getattr(args, 'disable_ss_lora', False)

        def _onoff(v: bool) -> str:
            return 'ON ' if v else 'OFF'

        # Map to the 5 predefined experiment groups
        exp_name = 'Custom'
        if (not mask_on) and (not dwt_on) and (not cafm_on) and (not ss_lora_on):
            exp_name = 'Exp1: None (baseline ViT)'
        elif mask_on and (not dwt_on) and (not cafm_on) and (not ss_lora_on):
            exp_name = 'Exp2: Lmask only'
        elif mask_on and dwt_on and (not cafm_on) and (not ss_lora_on):
            exp_name = 'Exp3: Lmask + DWT'
        elif mask_on and dwt_on and cafm_on and (not ss_lora_on):
            exp_name = 'Exp4: Lmask + DWT + CAFM'
        elif mask_on and dwt_on and cafm_on and ss_lora_on:
            exp_name = 'Exp5: Lmask + DWT + CAFM + SS-LoRA'

        header = '=' * 70
        lines = [
            header,
            'TRAIN EXPERIMENT CONFIG (ABLATION)'.center(70),
            header,
            f'  Scheme      : {exp_name}',
            f'  Lmask / Xray: {_onoff(mask_on)}',
            f'  DWT (CNN)   : {_onoff(dwt_on)}',
            f'  CAFM        : {_onoff(cafm_on)}',
            f'  SS-LoRA     : {_onoff(ss_lora_on)}',
            header,
        ]
        for l in lines:
            print(l)
            log.info(l)
    except Exception:
        pass

    # Locate ForensicsAdapter by searching upward from this file's directory so sibling placement is supported
    def find_forensics_adapter(start_dir=None, max_up=6):
        cur = os.path.abspath(start_dir or os.path.dirname(__file__))
        for _ in range(max_up):
            candidate = os.path.join(cur, 'ForensicsAdapter')
            if os.path.isdir(candidate):
                return candidate
            cur = os.path.dirname(cur)
        return None

    fa_root = find_forensics_adapter()
    if fa_root is not None:
        try:
            # To avoid name conflicts with local `dataset.py`, create a package mapping
            import types
            import importlib
            dataset_pkg_path = os.path.join(fa_root, 'dataset')
            if not os.path.isdir(dataset_pkg_path):
                raise RuntimeError(f'ForensicsAdapter dataset dir not found: {dataset_pkg_path}')
            # backup any existing 'dataset' module
            prev_dataset_mod = sys.modules.get('dataset')
            dataset_pkg = types.ModuleType('dataset')
            dataset_pkg.__path__ = [dataset_pkg_path]
            sys.modules['dataset'] = dataset_pkg
            try:
                fa_dataset_mod = importlib.import_module('dataset.abstract_dataset')
            finally:
                # leave the mapping in place so imported modules resolve correctly
                pass
            DeepfakeAbstractBaseDataset = fa_dataset_mod.DeepfakeAbstractBaseDataset
            with open(os.path.join(fa_root, 'config', 'train.yaml'), 'r') as f:
                fa_config = yaml.safe_load(f)

            # Optional override: if --ffpp_json is provided, restrict train/test datasets to this json
            ffpp_json = getattr(args, 'ffpp_json', None)
            if ffpp_json is not None:
                try:
                    json_dir = os.path.dirname(ffpp_json)
                    json_name = os.path.splitext(os.path.basename(ffpp_json))[0]
                    # ForensicsAdapter expects dataset_json_folder and train_dataset/test_dataset names
                    fa_config['dataset_json_folder'] = json_dir
                    fa_config['train_dataset'] = [json_name]
                    fa_config['test_dataset'] = [json_name]
                    print(f'Using FF++ json for training/validation: {ffpp_json}')
                except Exception as e:
                    print('Warning: failed to apply --ffpp_json override, falling back to train.yaml config:', e)

            train_dataset = DeepfakeAbstractBaseDataset(config=fa_config, mode='train')
            fa_config_test = dict(fa_config)
            if isinstance(fa_config_test.get('test_dataset'), list):
                fa_config_test['test_dataset'] = fa_config_test['test_dataset'][0]
            valid_dataset = DeepfakeAbstractBaseDataset(config=fa_config_test, mode='test')
            train_collate = getattr(train_dataset, 'collate_fn', None)
            valid_collate = getattr(valid_dataset, 'collate_fn', None)

            try:
                header = '=' * 70
                lines = [
                    header,
                    'TRAIN DATA AUGMENTATION SUMMARY'.center(70),
                    header,
                    '  Dataset Type : ForensicsAdapter DeepfakeAbstractBaseDataset',
                    '  Train Aug[1] : As defined in ForensicsAdapter config/train.yaml',
                    '  Train Aug[2] : Extra resize to 256x256 if needed (bilinear)',
                    header,
                ]
                for l in lines:
                    print(l)
                    log.info(l)
            except Exception:
                pass
            # --- debug subset handling: allow using a very small subset for fast runs ---
            debug_train = os.environ.get('DEBUG_SUBSET')
            debug_val = os.environ.get('DEBUG_VAL_SUBSET')
            if debug_train is not None:
                n_train = int(debug_train)
            elif os.environ.get('NO_SAVE', '0') == '1':
                n_train = 200
            else:
                n_train = None
            if debug_val is not None:
                n_val = int(debug_val)
            elif os.environ.get('NO_SAVE', '0') == '1':
                n_val = 50
            else:
                n_val = None
            if n_train is not None:
                n_train = min(n_train, len(train_dataset))
                train_dataset = torch.utils.data.Subset(train_dataset, list(range(n_train)))
            if n_val is not None:
                n_val = min(n_val, len(valid_dataset))
                valid_dataset = torch.utils.data.Subset(valid_dataset, list(range(n_val)))
            # Build a WeightedRandomSampler to mitigate class imbalance in training
            try:
                from torch.utils.data import WeightedRandomSampler
                # try to get labels quickly from dataset attribute to avoid expensive __getitem__ calls
                if hasattr(train_dataset, 'label_list') and len(getattr(train_dataset, 'label_list')) > 0:
                    labels = list(train_dataset.label_list)
                else:
                    labels = []
                    for i in range(len(train_dataset)):
                        s = train_dataset[i]
                        if isinstance(s, dict):
                            lbl = s.get('label')
                        elif isinstance(s, (list, tuple)):
                            lbl = s[1]
                        else:
                            continue
                        labels.append(int(lbl))

                if len(labels) > 0:
                    class_counts = np.bincount(labels)
                    # avoid zero division
                    class_counts = np.where(class_counts == 0, 1, class_counts)
                    sample_weights = [1.0 / float(class_counts[l]) for l in labels]
                    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=train_collate)
                else:
                    # fallback to simple shuffling
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_collate)
            except Exception:
                # if sampler fails for any reason, fall back to standard DataLoader
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
            valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=valid_collate)

            # If using ForensicsAdapter DeepfakeAbstractBaseDataset, wrap validation into video-level dataset
            try:
                # Build video-level dataset directly from the dataset JSON (more robust than parent-dir grouping)
                ds_name = fa_config_test.get('test_dataset') if isinstance(fa_config_test.get('test_dataset'), str) else (fa_config_test.get('test_dataset')[0] if isinstance(fa_config_test.get('test_dataset'), list) else None)
                json_path = os.path.join(fa_config.get('dataset_json_folder', ''), f'{ds_name}.json') if ds_name else None
                if json_path and os.path.exists(json_path):
                    with open(json_path, 'r') as jf:
                        ds_info = json.load(jf)

                    # replicate the path cleaning and frame selection logic from DeepfakeAbstractBaseDataset
                    frame_num_test = fa_config.get('frame_num', {}).get('test', fa_config.get('frame_num', {}).get('train', 1))
                    root_dir = "/root/autodl-tmp/FaceForensics++/"  # same default used in abstract_dataset

                    videos = []  # list of (video_frame_paths, label)
                    # handle special cp name normalization as in abstract_dataset
                    cfg_name = ds_name
                    cp = None
                    if cfg_name == 'FaceForensics++_c40':
                        cfg_name = 'FaceForensics++'
                        cp = 'c40'
                    elif cfg_name == 'FF-DF_c40':
                        cfg_name = 'FF-DF'
                        cp = 'c40'
                    elif cfg_name == 'FF-F2F_c40':
                        cfg_name = 'FF-F2F'
                        cp = 'c40'
                    elif cfg_name == 'FF-FS_c40':
                        cfg_name = 'FF-FS'
                        cp = 'c40'
                    elif cfg_name == 'FF-NT_c40':
                        cfg_name = 'FF-NT'
                        cp = 'c40'

                    for label in ds_info[cfg_name]:
                        sub_info = ds_info[cfg_name][label][ 'test' if 'test' in ds_info[cfg_name][label] else list(ds_info[cfg_name][label].keys())[0] ]
                        if cp is None and cfg_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++', 'DeepFakeDetection', 'FaceShifter']:
                            sub_info = sub_info[fa_config.get('compression')]
                        elif cp == 'c40' and cfg_name in ['FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'FaceForensics++', 'DeepFakeDetection', 'FaceShifter']:
                            sub_info = sub_info['c40']

                        for video_name, video_info in sub_info.items():
                            if 'label' not in video_info:
                                continue
                            if video_info['label'] not in fa_config.get('label_dict', {}):
                                continue
                            label_val = fa_config['label_dict'][video_info['label']]
                            frame_paths = []
                            for path in video_info.get('frames', []):
                                cleaned_path = path.replace('FaceForensics++', '').lstrip('/\\').replace('\\', '/').replace('//', '/')
                                full_path = os.path.join(root_dir, cleaned_path)
                                frame_paths.append(full_path)
                            # select frames per video following same sampling logic
                            total_frames = len(frame_paths)
                            if frame_num_test < total_frames:
                                step = total_frames // frame_num_test if frame_num_test>0 else 1
                                selected = [frame_paths[i] for i in range(0, total_frames, step)][:frame_num_test]
                            else:
                                selected = frame_paths
                            videos.append((selected, int(label_val)))

                    class VideoValidDataset(torch.utils.data.Dataset):
                        def __init__(self, videos, valid_obj):
                            self.videos = videos
                            # valid_obj may be a Dataset or a Subset wrapper; get underlying dataset
                            base = getattr(valid_obj, 'dataset', valid_obj)
                            self.base = base

                        def __len__(self):
                            return len(self.videos)

                        def __getitem__(self, idx):
                            fps, lbl = self.videos[idx]
                            imgs = []
                            for p in fps:
                                # use underlying dataset's helper methods to load and preprocess
                                img = self.base.load_rgb(p)
                                arr = np.array(img)
                                t = self.base.to_tensor(arr)
                                t = self.base.normalize(t)
                                imgs.append(t.unsqueeze(0))
                            if len(imgs) == 0:
                                # fallback to first image in valid_dataset
                                first = self.base.__getitem__(0)[0]
                                return first.unsqueeze(0), 0
                            video_tensor = torch.cat(imgs, dim=0)
                            return video_tensor, lbl

                    video_valid = VideoValidDataset(videos, valid_dataset)
                    valid_loader = DataLoader(video_valid, batch_size=1, shuffle=False, num_workers=args.num_workers)
            except Exception:
                # if wrapping fails, continue using frame-level valid_loader
                pass
        except Exception as e:
            # Print full traceback for debugging and abort so user can see root cause
            import traceback
            traceback.print_exc()
            raise
    else:
        train_path = '/data3/law/data/FF++/c23/train'
        valid_path = '/data3/law/data/FF++/c23/valid'
        train_dataset = FFPP_Dataset(train_path, frame=20, phase='train')
        valid_dataset = TestDataset(valid_path, dataset='FFPP', frame=20)

        try:
            header = '=' * 70
            lines = [
                header,
                'TRAIN DATA AUGMENTATION SUMMARY'.center(70),
                header,
                '  Dataset Type : Local FFPP_Dataset (albumentations)',
                '  Train Aug[1] : Resize to 224x224 (alb.Resize)',
                '  Train Aug[2] : Normalize mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]',
                '  Train Aug[3] : ToTensorV2 (HWC -> CHW tensor)',
                header,
            ]
            for l in lines:
                print(l)
                log.info(l)
        except Exception:
            pass

        # For local fallback dataset, also try to use a WeightedRandomSampler
        try:
            from torch.utils.data import WeightedRandomSampler
            # try to use label_list attribute to avoid expensive per-sample loading
            if hasattr(train_dataset, 'label_list') and len(getattr(train_dataset, 'label_list')) > 0:
                labels = list(train_dataset.label_list)
            else:
                labels = []
                for i in range(len(train_dataset)):
                    s = train_dataset[i]
                    if isinstance(s, dict):
                        lbl = s.get('label')
                    elif isinstance(s, (list, tuple)):
                        lbl = s[1]
                    else:
                        continue
                    labels.append(int(lbl))
            if len(labels) > 0:
                class_counts = np.bincount(labels)
                class_counts = np.where(class_counts == 0, 1, class_counts)
                sample_weights = [1.0 / float(class_counts[l]) for l in labels]
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
            else:
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        except Exception:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


    # Dataset provides 256x256 images, set model img_size to match to avoid patch_embed assertion
    # Instantiate MoE-ViT without internal pretrained npz loading; we'll try to initialize
    # from timm's PyTorch in21k weights (keeps MoE/LoRA/adapter structure intact).
    # 解析 CAS 控制
    cas_layers = 0
    if getattr(args, 'cas_layers', None) is not None:
        try:
            cas_layers = max(0, int(args.cas_layers))
        except Exception:
            cas_layers = 0
    elif getattr(args, 'cas_on', False):
        cas_layers = 4

    # Create model with Shared+Specific LoRA MoE configuration
    model = vit_base_patch16_224_in21k(
        pretrained=False, 
        num_classes=2, 
        img_size=256, 
        cas_layers=cas_layers,
        shared_rank=getattr(args, 'shared_rank', 64),
        use_svd_init=getattr(args, 'use_svd_init', False),
        lora_topk=0 if getattr(args, 'disable_ss_lora', False) else 1,
    )
    # pass lora loss weight into model so it can scale moe_loss
    try:
        model.lora_loss_weight = float(args.lora_loss_weight)
    except Exception:
        model.lora_loss_weight = 200.0

    # Try to load matching pretrained weights from timm (preferred PyTorch weights)
    # But if user provided an init .pth, skip timm pretrain download to avoid network activity
    try:
        timm_model_name = 'vit_base_patch16_224_in21k'
        # user-provided init path (same as used later)
        user_provided_init = os.path.join(
            '/root',
            'autodl-tmp',
            'For_code',
            'main',
            'FaceforensicAdapt_code - 副本',
            'main',
            'MoE-FFD-main',
            'models',
            'train',
            'vit_base_patch16_224_in21k_moe_init.pth',
        )
        if os.path.exists(user_provided_init):
            try:
                print('User-provided init found before timm init, loading:', user_provided_init)
                sd = torch.load(user_provided_init, map_location='cpu')
                model.load_state_dict(sd, strict=False)
                print('Loaded user-provided init pth into model (strict=False). Skipping timm init and npz download.')
                loaded_init_from_pth = True
            except Exception as e:
                print('Failed to load user-provided init pth before timm init:', e)
                loaded_init_from_pth = False
        else:
            loaded_init_from_pth = False

        if not loaded_init_from_pth:
            if timm_model_name in timm.list_models():
                print(f'Initializing from timm pretrained: {timm_model_name}')
                # this call may download weights; we skip it when user init exists
                timm_model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
                timm_sd = timm_model.state_dict()
                moe_sd = model.state_dict()
                loaded_keys = []
                skipped_keys = []
                for k, v in timm_sd.items():
                    if k in moe_sd and moe_sd[k].shape == v.shape:
                        moe_sd[k].copy_(v)
                        loaded_keys.append(k)
                    else:
                        skipped_keys.append(k)
                model.load_state_dict(moe_sd, strict=False)
                print(f'Copied {len(loaded_keys)} tensors from timm; skipped {len(skipped_keys)} keys.')
            else:
                print(f'timm model {timm_model_name} not available in this timm installation; skipping timm init.')
    except Exception as e:
        print('Failed to initialize from timm pretrained weights:', e)

    # If timm init was not available or failed, try to load original .npz (Flax) weights
    try:
        # First, check if user-provided init .pth exists and load it to avoid any downloads
        user_provided_init = os.path.join(
            '/root',
            'autodl-tmp',
            'For_code',
            'main',
            'FaceforensicAdapt_code - 副本',
            'main',
            'MoE-FFD-main',
            'models',
            'train',
            'vit_base_patch16_224_in21k_moe_init.pth',
        )
        loaded_init_from_pth = False
        if os.path.exists(user_provided_init):
            try:
                print('User-provided init found, loading init pth into model:', user_provided_init)
                sd = torch.load(user_provided_init, map_location='cpu')
                model.load_state_dict(sd, strict=False)
                print('Loaded user-provided init pth into model (strict=False).')
                loaded_init_from_pth = True
            except Exception as e:
                print('Failed to load user-provided init pth:', e)
                loaded_init_from_pth = False

        npz_url = None
        # default_cfgs is defined in ViT_MoE.py
        cfg_key = 'vit_base_patch16_224_in21k'
        if cfg_key in default_cfgs:
            npz_url = default_cfgs[cfg_key].get('url', '')

        # Only attempt npz download/load if we did NOT already load the user-provided .pth
        if not loaded_init_from_pth and npz_url and npz_url.endswith('.npz'):
            try:
                filename = f'{cfg_key.replace("/","_")}_weights.npz'
                # prefer a local copy if available: run save_dir, user cache, or common root cache
                candidate_paths = [
                    os.path.join(save_dir, filename),
                    os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', filename),
                    os.path.join('/root', '.cache', 'torch', 'hub', 'checkpoints', filename),
                ]
                weight_path = candidate_paths[0]
                found = False
                for p in candidate_paths:
                    if os.path.exists(p):
                        weight_path = p
                        found = True
                        print(f'Using existing npz weights at {weight_path}')
                        break
                if not found:
                    # only download if no local copy exists
                    print(f'Downloading npz weights from {npz_url} to {weight_path}')
                    import urllib.request
                    urllib.request.urlretrieve(npz_url, weight_path)
                # load into model using provided loader
                if hasattr(model, 'load_pretrained'):
                    print('Loading .npz weights into model via model.load_pretrained(...)')
                    model.load_pretrained(weight_path)
                    print('Loaded .npz weights successfully.')
                else:
                    print('Model does not support load_pretrained; skipping .npz load.')
            except Exception as e:
                print('Failed to download or load .npz weights:', e)
        else:
            # no npz url present or already loaded user .pth
            pass
    except Exception:
        pass

    # Prefer user-provided init .pth at specific path if available (avoid remote downloads)
    # Build the path from components so spaces/Unicode are handled correctly in Python
    user_provided_init = os.path.join(
        '/root',
        'autodl-tmp',
        'For_code',
        'main',
        'FaceforensicAdapt_code - 副本',
        'main',
        'MoE-FFD-main',
        'models',
        'train',
        'vit_base_patch16_224_in21k_moe_init.pth',
    )
    init_pth = None
    if os.path.exists(user_provided_init):
        init_pth = user_provided_init
    else:
        # fallback to model_root copy
        model_root_init = os.path.join(model_root, 'vit_base_patch16_224_in21k_moe_init.pth')
        if os.path.exists(model_root_init):
            init_pth = model_root_init

    if init_pth is not None:
        try:
            print(f'\n{"="*70}')
            print(f'Loading checkpoint: {os.path.basename(init_pth)}')
            print(f'{"="*70}')
            sd = torch.load(init_pth, map_location='cpu')
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f'✗ Failed to load checkpoint: {e}')

    # Re-initialize CAS parameters after loading pretrained weights to ensure they start at 0.01
    # (pretrained weights may contain zero-valued CAS params that would overwrite our initialization)
    # ONLY do this when NOT resuming (args.resume == -1) 且未进行 stage-2 init_from_shared
    if cas_layers > 0 and args.resume == -1 and not getattr(args, 'init_from_shared', None):
        try:
            cas_reinit_count = 0
            for n, p in model.named_parameters():
                if 'cas_beta_v' in n:
                    p.data.fill_(0.01)
                    cas_reinit_count += 1
            if cas_reinit_count > 0:
                print(f'Re-initialized {cas_reinit_count} CAS beta parameters to 0.01 after loading pretrained weights.')
        except Exception as e:
            print('Failed to re-initialize CAS parameters:', e)
    
    # Initialize Specific Experts with SVD decomposition of pretrained QKV weights
    # ONLY do this when NOT resuming (args.resume == -1) and if enabled via --use_svd_init
    if args.resume == -1 and getattr(args, 'use_svd_init', False):
        try:
            from ViT_MoE import initialize_lora_moe_with_svd
            
            # Extract pretrained QKV weight from first attention block
            # Assumption: all blocks share similar QKV structure, use first block as template
            qkv_weight = None
            for name, param in model.named_parameters():
                if 'blocks.0.attn.qkv.weight' in name:
                    qkv_weight = param.data.clone()
                    break
            
            if qkv_weight is not None:
                svd_strategy = getattr(args, 'svd_strategy', 'cumulative')
                initialize_lora_moe_with_svd(model, qkv_weight, strategy=svd_strategy)
            else:
                print('⚠ Warning: Could not find QKV weight for SVD initialization. Skipping SVD init.')
        except Exception as e:
            print(f'⚠ Warning: SVD initialization failed: {e}')
            print('   Continuing with standard LoRA initialization...')

    # If requested, train only Specific LoRA experts: freeze shared_lora params
    # and set all Attention blocks to use lora_ablation_mode='specific'.
    if getattr(args, 'lora_specific_only', False):
        try:
            from ViT_MoE import Attention  # type: ignore
        except Exception:
            Attention = None
        # 1) Switch all Attention modules to specific-only mode
        try:
            for m in model.modules():
                if Attention is not None and isinstance(m, Attention):
                    m.lora_ablation_mode = 'specific'
        except Exception:
            pass

    # If requested, train only Shared LoRA expert: freeze specific experts and their gating
    # and set all Attention blocks to use lora_ablation_mode='shared'.
    if getattr(args, 'lora_shared_only', False):
        try:
            from ViT_MoE import convert_lora_to_shared_only  # type: ignore
        except Exception:
            convert_lora_to_shared_only = None
        if convert_lora_to_shared_only is not None:
            try:
                convert_lora_to_shared_only(model)
            except Exception:
                pass

    # Ensure classifier/head parameters are trainable (some weight loads may replace the head module)
    try:
        cls_module = None
        try:
            cls_module = model.get_classifier()
        except Exception:
            cls_module = None
        if cls_module is not None:
            modules = cls_module if isinstance(cls_module, (list, tuple)) else (cls_module,)
            # set requires_grad True for all params belonging to classifier modules
            cls_params = set()
            for m in modules:
                for p in m.parameters():
                    p.requires_grad = True
                    cls_params.add(id(p))
            # classifier parameters are set trainable (names suppressed for conciseness)
    except Exception as e:
        print('Failed to enforce classifier trainability:', e)

    # Ensure gating parameters (w_gate / w_noise) are trainable
    try:
        gating_changed = []
        gating_found = False
        for n, p in model.named_parameters():
            nl = n.lower()
            if 'w_gate' in nl or 'w_noise' in nl:
                gating_found = True
                if not p.requires_grad:
                    p.requires_grad = True
                    gating_changed.append((n, tuple(p.shape)))
                else:
                    # still record existing gating params
                    gating_changed.append((n, tuple(p.shape)))
        # count gating params (do not print full list)
        gating_count = 0
        if gating_found:
            for n, p in model.named_parameters():
                nl = n.lower()
                if 'w_gate' in nl or 'w_noise' in nl:
                    gating_count += p.numel() if p.requires_grad else 0
        else:
            gating_count = 0
    except Exception as e:
        print('Failed to ensure gating params trainable:', e)

    # Compute concise categorized counts and print as table
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        loRA_count = 0
        adapter_count = 0
        head_count = 0
        cas_count = 0
        mask_count = 0
        fusion_count = 0
        frozen_backbone_count = 0
        
        for n, p in model.named_parameters():
            nl = n.lower()
            if p.requires_grad:
                if 'lora' in nl:
                    loRA_count += p.numel()
                elif 'adapter' in nl:
                    adapter_count += p.numel()
                elif 'head' in nl:
                    head_count += p.numel()
                elif 'cas' in nl:
                    cas_count += p.numel()
                elif 'xray_postprocess' in nl or 'mask' in nl:
                    mask_count += p.numel()
                elif 'fusion_bottleneck' in nl:
                    fusion_count += p.numel()
            else:
                if not any(kw in nl for kw in ['lora', 'adapter', 'head', 'cas', 'mask', 'xray', 'fusion']):
                    frozen_backbone_count += p.numel()

        # Print table format
        print('\n' + '='*70)
        print('ViT BACKBONE PARAMETERS SUMMARY'.center(70))
        print('='*70)
        print(f"{'Module':<20} {'Total Params':>15} {'Trainable':>15} {'%':>8}")
        print('-'*70)
        
        def format_num(n):
            if n >= 1e6:
                return f'{n/1e6:.2f}M'
            elif n >= 1e3:
                return f'{n/1e3:.2f}K'
            else:
                return str(n)
        
        modules_data = [
            ('LoRA MoE', loRA_count, loRA_count),
            ('Adapter MoE', adapter_count, adapter_count),
            ('CAS', cas_count, cas_count),
            ('Gating (w_gate)', gating_count, gating_count),
            ('Fusion Bottleneck', fusion_count, fusion_count),
            ('Classifier Head', head_count, head_count),
            ('Xray/Mask', mask_count, mask_count),
            ('Frozen Backbone', frozen_backbone_count, 0),
        ]
        
        for name, total, trainable in modules_data:
            if total > 0:
                pct = (trainable / total * 100) if total > 0 else 0
                print(f"{name:<20} {format_num(total):>15} {format_num(trainable):>15} {pct:>7.1f}%")
        
        print('-'*70)
        print(f"{'TOTAL':<20} {format_num(total_params):>15} {format_num(trainable_params):>15} {trainable_params/total_params*100:>7.1f}%")
        # 额外输出一行总可训练参数，便于和测试阶段对比
        print(f"Total trainable params: {trainable_params/1e6:.2f} M")
        print('='*70)
    except Exception as e:
        print(f'Parameter summary failed: {e}')

    # Wrap the frozen ViT backbone with LSNet dual-stream fusion before moving to CUDA
    use_dualstream = not getattr(args, 'disable_dwt_cnn', False)
    eff_cross_alpha_v = 0.0 if getattr(args, 'disable_cafm', False) else args.cross_alpha_v
    eff_cross_alpha_f = 0.0 if getattr(args, 'disable_cafm', False) else args.cross_alpha_f

    if use_dualstream:
        try:
            model = DualStreamVITLSNet(
                model,
                num_classes=args.num_classes,
                img_size=256,
                gn_groups=32,
                cross_alpha_v=eff_cross_alpha_v,
                cross_alpha_f=eff_cross_alpha_f,
                use_rgb_dwt=args.use_rgb_dwt,
            )
        except Exception as e:
            print('Failed to initialize DualStreamVITLSNet:', e)
    else:
        print('Ablation: disable DWT CNN frequency branch, using single-stream ViT backbone only.')
        # In pure ViT baseline (no DualStream), use CLS-only feature for classification (no layer8 GAP fusion)
        try:
            if hasattr(model, 'use_fusion_bottleneck'):
                model.use_fusion_bottleneck = False
        except Exception:
            pass

    # 追加一行：当前“实际训练的完整模型”参数量统计（包含 DualStream CNN / fuse 等）
    try:
        full_total = sum(p.numel() for p in model.parameters())
        full_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct_full = full_trainable / full_total * 100.0 if full_total > 0 else 0.0
        print(f"Trainable params (FULL MODEL): {full_trainable:,} / {full_total:,} ({pct_full:.2f}%)")
    except Exception as e:
        print('Full-model parameter summary failed:', e)

    # Re-initialize CAS AGAIN after DualStream wrapping (to ensure it's not reset)
    # ONLY when NOT resuming (resume should keep loaded CAS values) 且未进行 stage-2 init_from_shared
    if cas_layers > 0 and args.resume == -1 and not getattr(args, 'init_from_shared', None):
        try:
            for n, p in model.named_parameters():
                if 'cas_beta_v' in n:
                    p.data.fill_(0.01)
        except Exception:
            pass

    # Optional: load a previously trained shared-only checkpoint for stage-2 training
    # This will copy shared LoRA (and other overlapping) weights (including DualStream/CAS, etc.),
    # then freeze shared_lora params so that only specific experts/gating/head 等继续更新。
    if args.resume == -1 and getattr(args, 'init_from_shared', None):
        init_shared_path = getattr(args, 'init_from_shared')
        try:
            print(f"\n{'='*70}")
            print('Stage-2 init: loading shared-only checkpoint for specific-expert training'.center(70))
            print(f"{'='*70}")
            print(f'Checkpoint path: {init_shared_path}')
            ckpt = torch.load(init_shared_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
            else:
                sd = ckpt
            missing, unexpected = model.load_state_dict(sd, strict=False)

            # 统计从 checkpoint 实际加载进模型的参数数量（按 tensor 数和元素数）
            try:
                model_sd_after = model.state_dict()
                total_elems_model = sum(p.numel() for p in model.parameters())
                loaded_names = [k for k in model_sd_after.keys() if k not in missing]
                loaded_name_set = set(loaded_names)
                loaded_elems = sum(model_sd_after[k].numel() for k in loaded_names)
                # 其中 shared_lora 的加载情况
                shared_loaded_names = [k for k in loaded_names if 'shared_lora' in k]
                shared_loaded_elems = sum(model_sd_after[k].numel() for k in shared_loaded_names) if len(shared_loaded_names) > 0 else 0
            except Exception:
                model_sd_after = None
                total_elems_model = sum(p.numel() for p in model.parameters())
                loaded_names, loaded_name_set, loaded_elems = [], set(), 0
                shared_loaded_names, shared_loaded_elems = [], 0

            print(f'Loaded shared-only weights with strict=False (missing={len(missing)}, unexpected={len(unexpected)}).')
            if loaded_elems > 0:
                print(f'Checkpoint-loaded params: {len(loaded_names)} tensors, {loaded_elems/1e6:.2f} M elements (of {total_elems_model/1e6:.2f} M total).')
                print(f'  - shared_lora subset: {len(shared_loaded_names)} tensors, {shared_loaded_elems/1e6:.2f} M elements.')
        except Exception as e:
            print(f'⚠ Failed to load shared-only checkpoint from {init_shared_path}: {e}')
        # After loading, freeze all parameters that came from the checkpoint (loaded_name_set),
        # EXCEPT specific_lora 和 gating (w_gate / w_noise), so that stage-2 training only updates
        # specific experts + their gates while keeping the rest of the network fixed.
        # 这样: 从 .tar 恢复的 backbone / shared / CAS / DualStream / head 等全部视为冻结特征提取器,
        # 而 specific_lora_* / w_gate / w_noise 无论是否存在于 checkpoint 中, 一律保持可训练。
        try:
            total_params = 0
            frozen_from_ckpt = 0
            trainable_specific_gate = 0
            for n, p in model.named_parameters():
                n_tot = p.numel()
                total_params += n_tot
                # specific_lora: 专家; w_gate/w_noise: 门控; head/fuse: 尾部分类与融合 MLP; xray_postprocess: 掩码/xray分支
                # 这些模块在 stage-2 中需要继续适配, 不应被冻结。
                is_specific_or_gate = (
                    ('specific_lora' in n) or
                    ('w_gate' in n) or ('w_noise' in n) or
                    ('head' in n) or ('fuse' in n) or
                    ('xray_postprocess' in n)
                )
                if n in loaded_name_set and not is_specific_or_gate:
                    # 参数来自 checkpoint 且不属于 specific/gating: 作为固定 backbone 冻结
                    if p.requires_grad:
                        p.requires_grad = False
                        frozen_from_ckpt += n_tot
                else:
                    # 对于 specific_lora / w_gate / w_noise, 以及 checkpoint 中不存在的新参数,
                    # 保持当前的 requires_grad (通常为 True), 作为第二阶段要训练的部分。
                    if p.requires_grad:
                        trainable_specific_gate += n_tot
            print(f'Total params in current model: {total_params/1e6:.2f} M.')
            print(f'  ├ Frozen params loaded from checkpoint: {frozen_from_ckpt/1e6:.2f} M.')
            print(f'  └ Trainable params (specific LoRA / gating / new): {trainable_specific_gate/1e6:.2f} M.')
        except Exception as e:
            print('⚠ Failed to freeze checkpoint-loaded params after init_from_shared:', e)

    model = model.cuda()

    # Compute Dual-Stream parameters (simplified)
    try:
        if hasattr(model, 'fuse'):
            dual_total = 0
            dual_trainable = 0
            ca_names = ['proj_freq2vit', 'proj_vit2freq_tokens', 'mha',
                       'mlp_v', 'mlp_f', 'norm_v_attn', 'norm_v_mlp', 'norm_f_attn', 'norm_f_mlp']
            
            for n, p in model.named_parameters():
                if any(k in n for k in ['c1', 'c2', 'c3', 'c4', 'fuse'] + ca_names):
                    dual_total += p.numel()
                    if p.requires_grad:
                        dual_trainable += p.numel()
        else:
            dual_total = 0
            dual_trainable = 0
    except Exception:
        dual_total = 0
        dual_trainable = 0

    # 额外打印: 整个模型(包含 ViT + CNN DualStream)的可训练参数总量
    # 同时拆分为 ViT 流和 CNN(频域/DualStream) 流的可训练参数
    try:
        full_trainable = 0
        full_total = 0
        vit_trainable = 0
        cnn_trainable = 0
        for name, p in model.named_parameters():
            n_tot = p.numel()
            full_total += n_tot
            if not p.requires_grad:
                continue
            full_trainable += n_tot
            # 在 DualStream 包裹下, ViT 参数名通常以 "vit." 开头
            if name.startswith('vit.'):
                vit_trainable += n_tot
            else:
                cnn_trainable += n_tot
        print(f"Full model trainable params (ViT + CNN): {full_trainable/1e6:.2f} M / Total: {full_total/1e6:.2f} M")
        # 当存在 DualStream 时, cnn_trainable 主要对应 CNN/频域+跨模态分支
        print(f"  ├ ViT-stream trainable params : {vit_trainable/1e6:.2f} M")
        print(f"  └ CNN-stream trainable params : {cnn_trainable/1e6:.2f} M")
        if dual_total > 0:
            print(f"      (Dual-Stream trainable params约: {dual_trainable/1e6:.2f} M (of {dual_total/1e6:.2f} M))")
    except Exception:
        pass

    # loss function and optimizer
    # compute class weights from training dataset to mitigate class imbalance
    from collections import Counter
    # Prefer using label_list attribute which is precomputed during dataset init
    if hasattr(train_dataset, 'label_list') and len(getattr(train_dataset, 'label_list')) > 0:
        train_counts = Counter([int(x) for x in train_dataset.label_list])
    # ... (rest of the code remains the same)
    else:
        train_counts = Counter()
        for i in range(len(train_dataset)):
            s = train_dataset[i]
            if isinstance(s, dict):
                lbl = s.get('label')
            elif isinstance(s, (list, tuple)):
                lbl = s[1]
            else:
                continue
            try:
                train_counts[int(lbl)] += 1
            except Exception:
                pass

    # Also compute validation dataset class counts for informational logging
    try:
        if hasattr(valid_dataset, 'label_list') and len(getattr(valid_dataset, 'label_list')) > 0:
            valid_counts = Counter([int(x) for x in valid_dataset.label_list])
        else:
            valid_counts = Counter()
            for i in range(len(valid_dataset)):
                s = valid_dataset[i]
                if isinstance(s, dict):
                    lbl = s.get('label')
                elif isinstance(s, (list, tuple)):
                    lbl = s[1]
                else:
                    continue
                try:
                    valid_counts[int(lbl)] += 1
                except Exception:
                    pass
    except Exception:
        valid_counts = Counter()

    # Print dataset summary in table format
    try:
        print('\n' + '='*70)
        print('DATASET SUMMARY'.center(70))
        print('='*70)
        print(f"{'Dataset':<30} {'Samples':>15} {'Neg(0)':>10} {'Pos(1)':>10}")
        print('-'*70)
        
        tr0 = train_counts.get(0, 0)
        tr1 = train_counts.get(1, 0)
        va0 = valid_counts.get(0, 0)
        va1 = valid_counts.get(1, 0)
        
        print(f"{'Training Set':<30} {len(train_dataset):>15} {tr0:>10} {tr1:>10}")
        print(f"{'Validation Set':<30} {len(valid_dataset):>15} {va0:>10} {va1:>10}")
        
        try:
            if isinstance(fa_config.get('test_dataset'), list):
                test_name = fa_config.get('test_dataset')[0]
            else:
                test_name = fa_config.get('test_dataset', 'N/A')
            print('-'*70)
            print(f"Test Dataset Config: {test_name}")
        except Exception:
            pass
        
        print('='*70)
    except Exception:
        pass

    total = float(sum(train_counts.values())) if sum(train_counts.values()) > 0 else 1.0
    # avoid division by zero by defaulting to 1
    cnt0 = train_counts.get(0, 1)
    cnt1 = train_counts.get(1, 1)
    w0 = total / (2.0 * cnt0)
    w1 = total / (2.0 * cnt1)
    weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # special defined optim with separate learning rates
    gating_param = []     # w_gate, w_noise (MoE gating)
    cas_param = []        # CAS parameters (need higher lr to grow from 0)
    shared_param = []     # Shared Expert (always active, needs moderate lr)
    specific_param = []   # Specific Experts (gated, needs higher lr with SVD init)
    other_param = []      # head, fusion, mask, etc.
    
    for name, param in model.named_parameters():
        if 'w_gate' in name or 'w_noise' in name:
            gating_param.append(param)
        elif 'cas' in name.lower():
            cas_param.append(param)
        elif 'shared_lora' in name:
            shared_param.append(param)
        elif 'specific_lora' in name:
            specific_param.append(param)
        else:
            other_param.append(param)

    # Correctly set per-parameter-group learning rates
    # Shared Expert: moderate lr (learns common patterns from all data)
    # Specific Experts: higher lr (learns specialized patterns, benefits from SVD init)
    # CAS: highest lr + NO weight_decay to prevent decay to zero
    try:
        shared_lr_scale = float(getattr(args, 'shared_lr_scale', 1.0))
    except Exception:
        shared_lr_scale = 1.0
    try:
        specific_lr_scale = float(getattr(args, 'specific_lr_scale', 1.0))
    except Exception:
        specific_lr_scale = 1.0

    base_shared_lr = 2e-5 * shared_lr_scale
    base_specific_lr = 2e-5 * specific_lr_scale

    optimizer = optim.Adam([
        {'params': gating_param, 'lr': 1e-4, 'weight_decay': 1e-5},
        {'params': cas_param, 'lr': 1e-3, 'weight_decay': 0.0},        # NO decay for CAS!
        {'params': shared_param, 'lr': base_shared_lr, 'weight_decay': 1e-5},    # Shared Expert (moderate, scalable)
        {'params': specific_param, 'lr': base_specific_lr, 'weight_decay': 1e-5},  # Specific Experts (scalable)
        {'params': other_param, 'lr': args.learning_rate, 'weight_decay': 1e-5}  # head, fusion, mask
    ], betas=(0.9, 0.999))
    
    # Set initial_lr for scheduler compatibility (required when using --resume)
    for pg in optimizer.param_groups:
        if 'initial_lr' not in pg:
            pg['initial_lr'] = pg['lr']
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=args.resume)

    # CRITICAL: Force set CAS beta to 0.01 after optimizer creation (ONLY for new training)
    # 注意: 当使用 --init_from_shared 进行第二阶段训练时, 不再重置 cas_beta_v, 以保持 checkpoint 中的 CAS 数值
    if cas_layers > 0 and args.resume == -1 and not getattr(args, 'init_from_shared', None):
        print('\n[CRITICAL] Final CAS initialization check (new training):')
        for n, p in model.named_parameters():
            if 'cas_beta_v' in n:
                current_val = p.data.item() if p.numel() == 1 else p.data.mean().item()
                p.data.fill_(0.01)
                new_val = p.data.item() if p.numel() == 1 else p.data.mean().item()
                print(f'  {n}: before={current_val:.8f} → after={new_val:.8f}')
    elif cas_layers > 0 and args.resume > -1:
        print('\n[RESUME] Keeping CAS values from checkpoint:')
        for n, p in model.named_parameters():
            if 'cas_beta_v' in n:
                val = p.data.item() if p.numel() == 1 else p.data.mean().item()
                print(f'  {n}: {val:.8f}')

    # Print training configuration in table format
    print('\n' + '='*70)
    print('TRAINING CONFIGURATION'.center(70))
    print('='*70)
    print(f"{'Parameter':<35} {'Value':<30}")
    print('-'*70)
    
    config_items = [
        ('Optimizer', 'Adam (β1=0.9, β2=0.999)'),
        ('Weight Decay', '1e-5 (CAS=0)'),
        ('Learning Rate (gating)', '1e-4'),
        ('Learning Rate (CAS)', '1e-3 (no decay)'),
        ('Learning Rate (Shared)', f'2e-5 * {getattr(args, "shared_lr_scale", 1.0)} = {base_shared_lr:.2e}'),
        ('Learning Rate (Specific)', f'2e-5 * {getattr(args, "specific_lr_scale", 1.0)} = {base_specific_lr:.2e}'),
        ('Learning Rate (other)', f'{args.learning_rate}'),
        ('Epochs', f'{args.epochs}'),
        ('Batch Size', f'{args.batch_size}'),
        ('Num Workers', f'{args.num_workers}'),
        ('Loss Function', 'CE + MoE + Xray(MSE)'),
        ('Xray Loss Weight', f'{args.xray_loss_weight}'),
        ('CAS Layers', f'{cas_layers}'),
        ('Shared Expert Rank', f'{getattr(args, "shared_rank", 64)}'),
        ('SVD Initialization', f'{"✓ " + args.svd_strategy if getattr(args, "use_svd_init", False) else "✗ Disabled"}'),
        ('Device', f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"),
    ]
    
    for param, value in config_items:
        print(f"{param:<35} {value:<30}")
    
    print('-'*70)
    print('Optimizer Param Groups:')
    group_names = ['gating (w_gate/noise)', 'CAS parameters', 'other (LoRA/head/mask/fusion)']
    for i, pg in enumerate(optimizer.param_groups):
        lr = pg.get('lr', 'N/A')
        params_list = pg.get('params', [])
        n_params = len(params_list)
        total_elements = sum(p.numel() for p in params_list)
        group_name = group_names[i] if i < len(group_names) else f'group_{i}'
        print(f"  Group {i} ({group_name:<30}): lr={lr:<10} count={n_params:>3} params={total_elements}")
    print('='*70)


    print('\n>>> Starting training...')
    best_auc = train(args, model,optimizer,train_loader,valid_loader,scheduler,save_dir)
    duration = time.time()-start_time
    desc = getattr(args, 'description', None)
    if desc:
        print(f'The task of {desc} is completed')
    else:
        print('The task is completed')
    if best_auc is not None:
        print('The best AUC is {:.2%}'.format(best_auc))
    print('The run time is {}h {}m'.format(int(duration//3600),int(duration%3600//60)))
