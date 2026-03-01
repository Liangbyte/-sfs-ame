## -*- coding: utf-8 -*-
import os, sys
sys.setrecursionlimit(15000)
import time
import torch
import numpy as np
import random
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import cv2

from ViT_MoE import vit_base_patch16_224_in21k
from utils import cal_metrics
from models.DualStream import DualStreamVITLSNet


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _get_last_block_attn(model):
    vit = getattr(model, 'vit', model)
    blocks = getattr(vit, 'blocks', None)
    if blocks is None or len(blocks) == 0:
        return None
    attn_mod = getattr(blocks[-1], 'attn', None)
    if attn_mod is None:
        return None
    return getattr(attn_mod, 'last_attn', None)


def _save_attention_heatmap(input_tensor, attn, save_path):
    if attn is None:
        return
    if not isinstance(input_tensor, torch.Tensor):
        return
    if input_tensor.dim() == 3:
        img = input_tensor.unsqueeze(0)
    else:
        img = input_tensor
    if img.size(0) > 1:
        img = img[:1]
    img = img.detach().cpu()[0]  # (C,H,W)

    # simple per-image min-max to [0,1]
    img_min = float(img.min())
    img_max = float(img.max())
    img_norm = (img - img_min) / (img_max - img_min + 1e-6)
    img_np = img_norm.permute(1, 2, 0).numpy()  # (H,W,C)
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)

    # attn: (B, heads, N, N)
    attn_b = attn[0].detach().cpu().numpy()
    attn_mean = attn_b.mean(axis=0)  # (N,N)
    cls_to_patches = attn_mean[0, 1:]  # (N_patches,)
    num_patches = cls_to_patches.shape[0]
    grid = int(np.sqrt(num_patches))
    if grid * grid != num_patches:
        return
    attn_map = cls_to_patches.reshape(grid, grid)
    attn_min = float(attn_map.min())
    attn_max = float(attn_map.max())
    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    H, W = img_np.shape[:2]
    attn_resized = cv2.resize(attn_map, (W, H))
    attn_gray = (attn_resized * 255.0).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attn_gray, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    except Exception:
        pass
    # save overlay
    cv2.imwrite(save_path, overlay)
    # also save the original image for side-by-side comparison
    root, ext = os.path.splitext(save_path)
    orig_path = root + '_orig' + (ext if ext != '' else '.png')
    cv2.imwrite(orig_path, img_bgr)


def _save_gradcam_heatmap(input_tensor, cam, save_path):
    if cam is None:
        return
    if not isinstance(input_tensor, torch.Tensor):
        return
    if input_tensor.dim() == 3:
        img = input_tensor.unsqueeze(0)
    else:
        img = input_tensor
    if img.size(0) > 1:
        img = img[:1]
    img = img.detach().cpu()[0]  # (C,H,W)

    img_min = float(img.min())
    img_max = float(img.max())
    img_norm = (img - img_min) / (img_max - img_min + 1e-6)
    img_np = img_norm.permute(1, 2, 0).numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)

    H, W = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (W, H))
    cam_min = float(cam_resized.min())
    cam_max = float(cam_resized.max())
    cam_norm = (cam_resized - cam_min) / (cam_max - cam_min + 1e-6)
    cam_gray = (cam_norm * 255.0).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_gray, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    except Exception:
        pass
    cv2.imwrite(save_path, overlay)
    root, ext = os.path.splitext(save_path)
    orig_path = root + '_orig' + (ext if ext != '' else '.png')
    cv2.imwrite(orig_path, img_bgr)


def _save_fused_cam_heatmap(input_tensor, cam_cnn, attn, save_path):
    """将 CNN Grad-CAM (cam_cnn) 与 ViT 最后一层 self-attn 的热图融合为一张图。

    - cam_cnn: 2D numpy array (h, w)，由 _run_gradcam_dualstream 计算得到
    - attn   : (B, heads, N, N) 的注意力权重，使用 CLS→patch 的平均注意力
    """
    if cam_cnn is None or attn is None:
        # 若任一为空，则退化为单独的 Grad-CAM 热图
        return _save_gradcam_heatmap(input_tensor, cam_cnn, save_path)

    if not isinstance(input_tensor, torch.Tensor):
        return
    if input_tensor.dim() == 3:
        img = input_tensor.unsqueeze(0)
    else:
        img = input_tensor
    if img.size(0) > 1:
        img = img[:1]
    img = img.detach().cpu()[0]  # (C,H,W)

    img_min = float(img.min())
    img_max = float(img.max())
    img_norm = (img - img_min) / (img_max - img_min + 1e-6)
    img_np = img_norm.permute(1, 2, 0).numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    H, W = img_np.shape[:2]

    # 1) CNN Grad-CAM: resize & normalize to [0,1]
    cam_cnn_resized = cv2.resize(cam_cnn, (W, H))
    cnn_min = float(cam_cnn_resized.min())
    cnn_max = float(cam_cnn_resized.max())
    cam_cnn_norm = (cam_cnn_resized - cnn_min) / (cnn_max - cnn_min + 1e-6)

    # 2) ViT CLS→patch attention map: 与 _save_attention_heatmap 相同的构造方式
    try:
        attn_b = attn[0].detach().cpu().numpy()        # (heads, N, N)
        attn_mean = attn_b.mean(axis=0)                # (N, N)
        cls_to_patches = attn_mean[0, 1:]              # (N_patches,)
        num_patches = cls_to_patches.shape[0]
        grid = int(np.sqrt(num_patches))
        if grid * grid != num_patches:
            attn_map_resized = None
        else:
            attn_map = cls_to_patches.reshape(grid, grid)
            a_min = float(attn_map.min())
            a_max = float(attn_map.max())
            attn_map = (attn_map - a_min) / (a_max - a_min + 1e-6)
            attn_map_resized = cv2.resize(attn_map, (W, H))
    except Exception:
        attn_map_resized = None

    if attn_map_resized is None:
        # 若 ViT 注意力构造失败，同样退化为单独 CNN Grad-CAM
        return _save_gradcam_heatmap(input_tensor, cam_cnn, save_path)

    # 3) 将两者简单平均融合
    fused = 0.5 * cam_cnn_norm + 0.5 * attn_map_resized
    fused_min = float(fused.min())
    fused_max = float(fused.max())
    fused_norm = (fused - fused_min) / (fused_max - fused_min + 1e-6)
    fused_gray = (fused_norm * 255.0).clip(0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(fused_gray, cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    except Exception:
        pass
    cv2.imwrite(save_path, overlay)
    root, ext = os.path.splitext(save_path)
    orig_path = root + '_orig' + (ext if ext != '' else '.png')
    cv2.imwrite(orig_path, img_bgr)


def _run_gradcam_dualstream(model, inputs, save_path, target_class=1):
    # Only support DualStreamVITLSNet with c4 conv branch
    if not hasattr(model, 'c4'):
        return

    target_module = model.c4
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations['value'] = out

    def bwd_hook(module, grad_in, grad_out):
        # grad_out is a tuple, first element is grad wrt out
        gradients['value'] = grad_out[0]

    handle_f = target_module.register_forward_hook(fwd_hook)
    # 使用新的 full backward hook，避免 PyTorch 的弃用警告
    handle_b = target_module.register_full_backward_hook(bwd_hook)

    # DualStream 中频域分支的 c4 包在 `with torch.set_grad_enabled(self.training)` 里，
    # eval() 模式下 self.training=False 会关闭该分支的梯度，导致 Grad-CAM 拿不到 grad。
    # 这里临时切换到 train 模式，仅用于这一小次前向/反向，之后再恢复原状态。
    was_training = model.training
    model.train()

    try:
        model.zero_grad(set_to_none=True)
        inputs = inputs.requires_grad_(True)
        outputs, _ = model(inputs)
        if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
            print('[Grad-CAM] Unexpected output shape, skip this sample')
            return
        if target_class is None:
            cls_idx = int(outputs.argmax(dim=-1)[0].item())
        else:
            cls_idx = int(target_class)
        score = outputs[0, cls_idx]
        score.backward(retain_graph=True)

        act = activations.get('value', None)
        grad = gradients.get('value', None)
        if act is None or grad is None:
            print('[Grad-CAM] Missing activations/gradients from hooks, skip this sample')
            return
        # act, grad: (B, C, H, W)
        act = act.detach()
        grad = grad.detach()
        B, C, H, W = act.shape
        if B == 0:
            print('[Grad-CAM] Empty activation batch, skip this sample')
            return
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (B,C,1,1)
        cam = (weights * act).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = torch.relu(cam)[0, 0].cpu().numpy()

        # 取 ViT 最后一层 self-attn，并与 CNN Grad-CAM 融合保存
        attn_last = _get_last_block_attn(model)
        if attn_last is not None:
            _save_fused_cam_heatmap(inputs, cam, attn_last, save_path)
        else:
            _save_gradcam_heatmap(inputs, cam, save_path)
    except Exception as e:
        print('[Grad-CAM] Failed for current sample:', e)
    finally:
        # 恢复原来的 train/eval 状态，并移除 hooks
        model.train(was_training)
        handle_f.remove()
        handle_b.remove()


def _run_gradcam_fused_head(model, inputs, save_path, target_class=None, grid_size: int = 16):
    """在 DualStreamVITLSNet 的融合向量 (fuse 输出) 上做一个“伪 Grad-CAM” 可视化。

    注意：这里的可视化没有严格的空间含义，仅作为直观参考。
    做法：
      - 在 model.fuse 上注册 forward/backward hook，拿到融合后的向量 feat ∈ R^{B×D} 及其梯度。
      - 对每个维度做 feat * grad 作为重要性分数，得到一个长度为 D 的向量。
      - 将该向量重排/下采样为 grid_size×grid_size（默认 16×16），当作 H×W 的“伪特征图”。
      - 再用 _save_gradcam_heatmap 把它插值到输入图大小后叠加到原图上。
    """
    if not hasattr(model, 'fuse'):
        print('[Grad-CAM-Fused] model has no attribute fuse; skip fused-head Grad-CAM')
        return

    target_module = model.fuse
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations['value'] = out  # (B, D)

    def bwd_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]  # (B, D)

    handle_f = target_module.register_forward_hook(fwd_hook)
    handle_b = target_module.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad(set_to_none=True)
        inputs = inputs.requires_grad_(True)
        outputs, _ = model(inputs)
        if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2:
            print('[Grad-CAM-Fused] Unexpected output shape, skip this sample')
            return
        if target_class is None:
            cls_idx = int(outputs.argmax(dim=-1)[0].item())
        else:
            cls_idx = int(target_class)
        score = outputs[0, cls_idx]
        score.backward(retain_graph=True)

        act = activations.get('value', None)
        grad = gradients.get('value', None)
        if act is None or grad is None:
            print('[Grad-CAM-Fused] Missing activations/gradients from hooks, skip this sample')
            return

        act = act.detach()   # (B, D)
        grad = grad.detach()  # (B, D)
        B, D = act.shape
        if B == 0 or D == 0:
            print('[Grad-CAM-Fused] Empty activation batch, skip this sample')
            return

        # 仅使用 batch 中第一张图的融合特征
        feat = act[0]   # (D,)
        g = grad[0]     # (D,)
        cam_vec = (feat * g).cpu()  # (D,)
        cam_vec = torch.relu(cam_vec)

        # 将一维向量重排为 grid_size×grid_size：
        # 默认使用 D 到 grid_size^2 的线性插值（若 D 恰好是 grid^2 的倍数，则做分组平均）。
        L = grid_size * grid_size
        cam_np = cam_vec.numpy()
        if D >= L and D % L == 0:
            cam_flat = cam_np.reshape(L, -1).mean(axis=1)
        else:
            # 使用一维插值到长度 L
            x_old = np.linspace(0.0, 1.0, num=D, endpoint=True)
            x_new = np.linspace(0.0, 1.0, num=L, endpoint=True)
            cam_flat = np.interp(x_new, x_old, cam_np)
        cam_2d = cam_flat.reshape(grid_size, grid_size)

        _save_gradcam_heatmap(inputs, cam_2d, save_path)
    except Exception as e:
        print('[Grad-CAM-Fused] Failed for current sample:', e)
    finally:
        handle_f.remove()
        handle_b.remove()


if __name__ == '__main__':
    import argparse, types, importlib
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-dv', type=int, default=0)
    parser.add_argument('--model_path', '-md', type=str, required=True, help='本地 checkpoint 路径 (.pkl/.pth/.tar)')
    parser.add_argument('--dfdcp_json', type=str, required=True, help='DFDCP 的 JSON 文件路径')
    parser.add_argument('--fa_root', type=str, default=None, help='ForensicsAdapter 根目录（可选）')
    parser.add_argument('--num_workers', type=int, default=4)
    # CAS 控制：--cas_on 开关，--cas_layers 指定开启的层数（优先级更高）
    parser.add_argument('--cas_on', action='store_true', help='启用 CAS 旁路（默认前4层）')
    parser.add_argument('--cas_layers', type=int, default=None, help='启用 CAS 的层数，优先级高于 --cas_on，0 表示全关')
    parser.add_argument('--test_one_tenth', action='store_true', help='只评估测试集的十分之一（每10个取1个）')
    parser.add_argument('--model', type=str, choices=['train','test'], default='test')
    # Ablation study for Shared+Specific LoRA MoE
    parser.add_argument('--ablation', type=str, default='both', choices=['both', 'shared', 'specific'],
                        help='消融实验模式: both(默认,Shared+Specific), shared(仅Shared), specific(仅Specific)')
    # Shared+Specific LoRA MoE configuration (must match training config)
    parser.add_argument('--shared_rank', type=int, default=64, help='Shared Expert的rank大小（必须与训练时一致！）')
    parser.add_argument('--use_svd_init', action='store_true', help='是否使用了SVD初始化（仅用于标记，不影响加载）')
    parser.add_argument('--lora_specific_only', action='store_true',
                        help='只用 specific LoRA 专家评估: 与训练时 --lora_specific_only 保持一致，以便统计可训练参数量')
    parser.add_argument('--lora_shared_only', action='store_true',
                        help='只用 shared LoRA 专家评估: 与训练时 --lora_shared_only 保持一致，以便统计可训练参数量')
    # 频域分支：RGB逐通道DWT开关（默认False=灰度DWT），需与训练一致
    parser.add_argument('--use_rgb_dwt', action='store_true', help='启用RGB逐通道DWT，拼接高频子带后用1x1卷积降维（测试需与训练一致）')
    # 双向cross-attention强度系数（需与训练时一致）
    parser.add_argument('--cross_alpha_v', type=float, default=0.2)
    parser.add_argument('--cross_alpha_f', type=float, default=0.2)
    # Ablation flags (需与训练阶段的设置保持一致)
    parser.add_argument('--disable_dwt_cnn', action='store_true', help='测试时关闭灰度DWT频域CNN分支（使用单流ViT）')
    parser.add_argument('--disable_cafm', action='store_true', help='测试时关闭CAFM双向交叉注意力（cross_alpha_v/f 强制为 0）')
    parser.add_argument('--disable_ss_lora', action='store_true', help='测试时关闭Shared+Specific LoRA MoE（lora_topk=0）')
    parser.add_argument('--vis_attn_dir', type=str, default=None, help='若非空，则在测试时保存最多4张注意力热图到该目录')
    parser.add_argument('--vis_gradcam_dir', type=str, default=None, help='若非空，则在测试时保存最多4张基于梯度的Grad-CAM热图到该目录')
    parser.add_argument('--tsne_out', type=str, default=None, help='若非空，则在测试集特征上做 t-SNE 可视化并保存到该目录')
    parser.add_argument('--tsne_runs', type=int, default=1, help='对同一批特征重复运行 t-SNE 的次数（>=1）')
    args = parser.parse_args()

    start_time = time.time()
    setup_seed(2024)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    # 定位 ForensicsAdapter
    def find_forensics_adapter(start_dir=None, max_up=6):
        cur = os.path.abspath(start_dir or os.path.dirname(__file__))
        for _ in range(max_up):
            candidate = os.path.join(cur, 'ForensicsAdapter')
            if os.path.isdir(candidate):
                return candidate
            cur = os.path.dirname(cur)
        return None

    fa_root = args.fa_root or find_forensics_adapter()
    if fa_root is None:
        raise RuntimeError('找不到 ForensicsAdapter，请通过 --fa_root 指定路径或把仓库放在上层目录')

    # 导入 ForensicsAdapter 的 dataset
    dataset_pkg_path = os.path.join(fa_root, 'dataset')
    if not os.path.isdir(dataset_pkg_path):
        raise RuntimeError('ForensicsAdapter 的 dataset 子目录不存在: ' + dataset_pkg_path)
    dataset_pkg = types.ModuleType('dataset')
    dataset_pkg.__path__ = [dataset_pkg_path]
    sys.modules['dataset'] = dataset_pkg
    fa_dataset_mod = importlib.import_module('dataset.abstract_dataset')
    DeepfakeAbstractBaseDataset = fa_dataset_mod.DeepfakeAbstractBaseDataset

    # 读取 FA 的 test.yaml 并覆盖 JSON 路径
    cfg_path = os.path.join(fa_root, 'config', 'test.yaml')
    if not os.path.exists(cfg_path):
        raise RuntimeError('ForensicsAdapter config/test.yaml 不存在: ' + cfg_path)
    fa_config = yaml.safe_load(open(cfg_path))
    fa_config['dataset_json_folder'] = os.path.dirname(args.dfdcp_json)
    fa_config['test_dataset'] = os.path.splitext(os.path.basename(args.dfdcp_json))[0]

    # 构造 dataset（完全复用 FA 的数据处理）
    print('正在构建 ForensicsAdapter 数据集...')
    test_dataset = DeepfakeAbstractBaseDataset(config=fa_config, mode='test')
    # try to import ForensicsAdapter metrics util
    fa_get_test_metrics = None
    try:
        # first try package import
        if fa_root not in sys.path:
            sys.path.insert(0, fa_root)
        fa_metrics_mod = importlib.import_module('trainer.metrics.utils')
        fa_get_test_metrics = getattr(fa_metrics_mod, 'get_test_metrics', None)
    except Exception:
        # fallback: load by file path
        try:
            metrics_file = os.path.join(fa_root, 'trainer', 'metrics', 'utils.py')
            if os.path.exists(metrics_file):
                from importlib.machinery import SourceFileLoader
                fa_metrics_mod = SourceFileLoader('fa_metrics_utils', metrics_file).load_module()
                fa_get_test_metrics = getattr(fa_metrics_mod, 'get_test_metrics', None)
            else:
                print('ForensicsAdapter metrics file not found at', metrics_file)
        except Exception as e:
            print('无法加载 ForensicsAdapter 指标模块:', e)

    # collate 函数
    collate_fn = getattr(test_dataset, 'collate_fn', None)
    if collate_fn is None:
        from torch.utils.data._utils.collate import default_collate
        def safe_collate(batch):
            batch = [b for b in batch if b is not None]
            if len(batch) == 0:
                return None
            return default_collate(batch)
        collate_fn = safe_collate

    # 可选：仅评估测试集的十分之一
    if args.test_one_tenth:
        try:
            from torch.utils.data import Subset
            total_len = len(test_dataset)
            if total_len > 0:
                indices = list(range(0, total_len, 10))
                test_dataset = Subset(test_dataset, indices)
                print(f'注意: 仅评估测试集的十分之一，共 {len(indices)} / {total_len} 样本')
        except Exception as e:
            print('构建 1/10 测试子集失败，改为全量评估:', e)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # 构建模型，test 模式下不让 timm 下载预训练权重
    use_timm_pretrained = True if args.model == 'train' else False
    # 解析 CAS 控制
    cas_layers = 0
    if getattr(args, 'cas_layers', None) is not None:
        try:
            cas_layers = max(0, int(args.cas_layers))
        except Exception:
            cas_layers = 0
    elif getattr(args, 'cas_on', False):
        cas_layers = 4

    vit = vit_base_patch16_224_in21k(
        pretrained=use_timm_pretrained, 
        num_classes=2, 
        img_size=256, 
        cas_layers=cas_layers,
        shared_rank=getattr(args, 'shared_rank', 64),
        use_svd_init=getattr(args, 'use_svd_init', False),
        lora_topk=0 if getattr(args, 'disable_ss_lora', False) else 1,
    )

    # Dual-stream & CAFM ablation (需与训练设置匹配)
    use_dualstream = not getattr(args, 'disable_dwt_cnn', False)
    eff_cross_alpha_v = 0.0 if getattr(args, 'disable_cafm', False) else getattr(args, 'cross_alpha_v', 0.2)
    eff_cross_alpha_f = 0.0 if getattr(args, 'disable_cafm', False) else getattr(args, 'cross_alpha_f', 0.2)

    if use_dualstream:
        try:
            model = DualStreamVITLSNet(
                vit,
                num_classes=2,
                img_size=256,
                gn_groups=32,
                cross_alpha_v=eff_cross_alpha_v,
                cross_alpha_f=eff_cross_alpha_f,
                use_rgb_dwt=getattr(args, 'use_rgb_dwt', False),
            )
            print('Eval: Using DualStreamVITLSNet (ViT + grayscale DWT LSNet-t).')
        except Exception as e:
            print('Eval: DualStream wrapper failed, falling back to single-stream ViT. Error:', e)
            model = vit
    else:
        print('Eval Ablation: disable DWT CNN frequency branch, using single-stream ViT backbone only.')
        # Single-stream ViT baseline at test time: use CLS-only feature for classification
        try:
            if hasattr(vit, 'use_fusion_bottleneck'):
                vit.use_fusion_bottleneck = False
        except Exception:
            pass
        model = vit

    model = model.cuda()
    print('模型总参数数: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # === 使 eval 阶段的 LoRA 结构与 train.py 完全一致（先做结构变换，再加载权重，避免无用 Missing key 警告） ===
    # 1) 若使用 only-specific 模式：冻结 shared_lora，并设置 Attention 为 specific-only
    try:
        if getattr(args, 'lora_specific_only', False):
            try:
                from ViT_MoE import Attention  # type: ignore
            except Exception:
                Attention = None
            try:
                for m in model.modules():
                    if Attention is not None and isinstance(m, Attention):
                        m.lora_ablation_mode = 'specific'
            except Exception:
                pass
    except Exception:
        pass

    # 2) 若使用 only-shared 模式：将 LoRA_MoElayer 转换为 SharedOnlyLoRA（与 train.py 中完全一致）
    try:
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
    except Exception:
        pass

    # 3) 确保分类头参数是可训练的（与 train.py 一致）
    try:
        cls_module = None
        try:
            cls_module = model.get_classifier()
        except Exception:
            cls_module = None
        if cls_module is not None:
            modules = cls_module if isinstance(cls_module, (list, tuple)) else (cls_module,)
            for m in modules:
                for p in m.parameters():
                    p.requires_grad = True
    except Exception as e:
        print('Eval: Failed to enforce classifier trainability:', e)

    # 4) 确保 gating 参数 (w_gate / w_noise) 是可训练的（与 train.py 一致）
    try:
        gating_found = False
        for n, p in model.named_parameters():
            nl = n.lower()
            if 'w_gate' in nl or 'w_noise' in nl:
                gating_found = True
                if not p.requires_grad:
                    p.requires_grad = True
        if gating_found:
            gating_count = 0
            for n, p in model.named_parameters():
                nl = n.lower()
                if 'w_gate' in nl or 'w_noise' in nl:
                    gating_count += p.numel() if p.requires_grad else 0
        else:
            gating_count = 0
    except Exception as e:
        print('Eval: Failed to ensure gating params trainable:', e)

    # 在 test 模式下加载本地 checkpoint（此时模型结构已与训练阶段一致）
    if args.model == 'test':
        ck = args.model_path
        if not os.path.exists(ck):
            raise RuntimeError('指定的 model_path 不存在: ' + ck)
        checkpoint = torch.load(ck, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state = checkpoint['state_dict']
            else:
                state = checkpoint
        else:
            state = checkpoint
        def try_load_current(mod, st):
            try:
                incompatible = mod.load_state_dict(st, strict=True)
                print('已严格加载本地 checkpoint 到当前模型')
                return True
            except Exception as e1:
                print('严格加载失败，尝试 strict=False；错误:', e1)
                try:
                    incompatible = mod.load_state_dict(st, strict=False)
                    print('已使用 strict=False 加载到当前模型。')
                    return True
                except Exception as e2:
                    print('strict=False 仍失败，错误:', e2)
                    return False
        loaded = try_load_current(model, state)
        if (not loaded) and hasattr(model, 'vit'):
            try:
                model.vit.load_state_dict(state, strict=False)
                print('已将 checkpoint 作为 ViT 子模块权重加载 (strict=False)。')
            except Exception as e3:
                print('加载到 vit 子模块失败：', e3)

    # 打印当前模型的可训练参数（名称、形状和数量），便于核对 eval 结构
    try:
        print('\n' + '=' * 70)
        print('TRAINABLE PARAMETER SUMMARY (EVAL)'.center(70))
        print('=' * 70)
        total_params = 0
        trainable_params = 0
        rows = []
        for name, p in model.named_parameters():
            n_params = int(p.numel())
            total_params += n_params
            if p.requires_grad:
                trainable_params += n_params
                rows.append((name, tuple(p.shape), n_params))
        # 逐行打印可训练参数
        for n, shape, cnt in rows:
            print(f'{n:<60} | {str(shape):<20} | {cnt:>8,}')
        ratio = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
        print('-' * 70)
        print(f'Trainable params: {trainable_params:,} / {total_params:,} ({ratio:.2f}%)')
        print('=' * 70 + '\n')
    except Exception as e:
        print('Eval: Failed to print trainable parameter summary:', e)

    # t-SNE: 如果指定了输出目录，注册一个 hook 捕获分类头前的特征
    tsne_out_dir = getattr(args, 'tsne_out', None)
    tsne_features = []
    tsne_labels = []
    tsne_handle = None
    if tsne_out_dir is not None:
        tsne_target = None
        if hasattr(model, 'vit') and hasattr(model.vit, 'head'):
            tsne_target = model.vit.head
        elif hasattr(model, 'head'):
            tsne_target = model.head
        if tsne_target is not None:
            def _tsne_hook(module, inp, out):
                # inp[0]: pre-logits特征 (N, D)
                feat = inp[0].detach().cpu()
                tsne_features.append(feat)
            tsne_handle = tsne_target.register_forward_hook(_tsne_hook)
            print('TSNE: hook registered on', tsne_target.__class__.__name__)
        else:
            print('TSNE: 未找到可用的 head 模块，关闭 t-SNE 特征收集')

    # Check CAS parameter values after loading checkpoint
    print('\n' + '='*70)
    print('CAS PARAMETER CHECK (AFTER LOADING CHECKPOINT)'.center(70))
    print('='*70)
    
    cas_params_found = False
    cas_beta_values = []
    
    for n, p in model.named_parameters():
        if 'cas_beta_v' in n:
            cas_params_found = True
            val = p.data.item() if p.numel() == 1 else p.data.abs().mean().item()
            cas_beta_values.append((n, val))
    
    if cas_params_found:
        print(f"{'Layer':<35} {'Beta Value':>15} {'Status':>15}")
        print('-'*70)
        
        all_zero = True
        
        for n, val in cas_beta_values:
            # Extract layer name more clearly
            if '.' in n:
                parts = n.split('.')
                # Find 'blocks.X' pattern
                layer_idx = 'unknown'
                for i, part in enumerate(parts):
                    if part == 'blocks' and i+1 < len(parts):
                        layer_idx = parts[i+1]
                        break
                layer_name = f"blocks.{layer_idx}.attn.cas_beta_v"
            else:
                layer_name = n
            
            status = '✓ ACTIVE' if abs(val) > 1e-8 else '✗ ZERO (no effect)'
            print(f"{layer_name:<35} {val:>15.8f} {status:>15}")
            
            if abs(val) > 1e-8:
                all_zero = False
        
        print('-'*70)
        print('Summary:')
        print(f"  Total CAS layers: {len(cas_beta_values)}")
        
        if all_zero:
            print('  ⚠️  All beta values are ZERO → CAS has NO effect!')
        else:
            active_count = sum(1 for _, v in cas_beta_values if abs(v) > 1e-8)
            print(f'  ✅ {active_count}/{len(cas_beta_values)} CAS layers are ACTIVE!')
    else:
        print('  No CAS parameters found (cas_layers=0)')
    
    print('='*70)
    
    # Check Shared Expert training status (very concise)
    print('\n' + '='*70)
    print('SHARED EXPERT STATUS CHECK'.center(70))
    print('='*70)
    
    shared_found = False
    shared_stats = []
    
    for n, p in model.named_parameters():
        if 'shared_lora' in n and ('weight' in n or 'bias' in n):
            shared_found = True
            param_data = p.data.cpu()
            abs_mean = param_data.abs().mean().item()
            std = param_data.std().item()
            
            # Determine status
            if not p.requires_grad:
                status = '✗ Frozen'
            elif abs_mean < 1e-7:
                status = '⚠ Zero'
            elif std < 1e-6:
                status = '⚠ Not Trained'
            else:
                status = '✓ Trained'
            
            shared_stats.append({
                'name': n.split('.')[-1],  # weight or bias
                'requires_grad': p.requires_grad,
                'abs_mean': abs_mean,
                'std': std,
                'status': status
            })
    
    if shared_found:
        print(f"{'Param':<15} {'Status':<15} {'Abs Mean':<12} {'Std':<12}")
        print('-'*70)
        for stat in shared_stats:
            print(f"{stat['name']:<15} {stat['status']:<15} {stat['abs_mean']:<12.6f} {stat['std']:<12.6f}")
        
        # Summary
        trained_count = sum(1 for s in shared_stats if '✓' in s['status'])
        print('-'*70)
        print(f"Summary: {trained_count}/{len(shared_stats)} params trained")
    else:
        print('  No Shared Expert parameters found (using old LoRA MoE)')
    
    print('='*70)
    
    # Ablation study: test with Shared-only, Specific-only, or Both
    ablation_mode = getattr(args, 'ablation', 'both')  # 'shared', 'specific', or 'both'
    
    # Wrapper to inject ablation_mode into LoRA_MoE forward calls
    if ablation_mode in ['shared', 'specific'] and shared_found:
        print('\n' + '='*70)
        print(f'⚠️  ABLATION MODE: {ablation_mode.upper()}-ONLY'.center(70))
        print('='*70)
        print(f'  Will evaluate using only {ablation_mode} expert outputs')
        print('='*70)
        
        # Monkey-patch all LoRA_MoE forward methods to use ablation_mode
        for name, module in model.named_modules():
            if type(module).__name__ == 'LoRA_MoElayer':
                original_forward = module.forward
                def make_ablation_forward(orig_fn, mode):
                    def ablation_forward(x, loss_coef=1, ablation_mode=mode):
                        return orig_fn(x, loss_coef, ablation_mode)
                    return ablation_forward
                module.forward = make_ablation_forward(original_forward, ablation_mode)
    
    # 在评估前重置所有 LoRA_MoE 层的 gate 使用统计，便于本次 run 统计
    try:
        for name, module in model.named_modules():
            if type(module).__name__ == 'LoRA_MoElayer' and hasattr(module, 'reset_gate_usage_stats'):
                module.reset_gate_usage_stats()
    except Exception as e:
        print('Eval: failed to reset LoRA_MoE gate usage stats:', e)

    # 决定 image/video 模式：以 FA 配置为准
    image_level = (int(fa_config.get('num_frames', fa_config.get('frame', 1))) == 1)
    print('\n评估模式:', 'image-level' if image_level else 'video-level')

    # 评估循环：严格使用 FA dataset 返回的格式（第0项 imgs，第1项 label）
    model.eval()
    frame_predictions = []
    frame_labels = []
    video_predictions = []
    video_labels = []
    # 统计纯前向推理耗时与样本数，用于计算 ms/image
    infer_time_total = 0.0
    infer_image_count = 0
    vis_dir = getattr(args, 'vis_attn_dir', None)
    vis_gc_dir = getattr(args, 'vis_gradcam_dir', None)
    # 最多随机可视化 40 个 batch（通常 batch_size=1 即 40 张图）
    vis_max = 40
    total_batches = len(test_loader)
    if (vis_dir is not None) or (vis_gc_dir is not None):
        import random as _rnd
        vis_indices = set(_rnd.sample(range(total_batches), k=min(vis_max, total_batches)))
    else:
        vis_indices = set()

    # Grad-CAM: 仅对测试集中“判断正确的负样本”(True Negative: label=0 且预测为 0) 可视化，
    # 并从中按负类置信度(1 - prob_fake) 选取得分最高的前 40 张
    gradcam_tn_max = 40
    gradcam_tn_candidates = []  # 每项: dict(conf=负类置信度, img=Tensor[N,C,H,W])
    gradcam_fake_max = 10
    gradcam_fake_candidates = {1: [], 2: [], 3: [], 4: []}
    sample_idx_for_paths = 0

    # 若指定了 Grad-CAM 输出目录，提前创建，避免因为某些样本失败而根本看不到文件夹
    if vis_gc_dir is not None:
        try:
            os.makedirs(vis_gc_dir, exist_ok=True)
        except Exception as e:
            print('Failed to create vis_gradcam_dir', vis_gc_dir, 'error:', e)

    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), ncols=80, desc='Eval', unit='batch')
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                pbar.update(1)
                continue
            # FA 返回 tuple/list，首两项是 (imgs, label)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs = batch[0]
                labels = batch[1]
            elif isinstance(batch, dict):
                # 逐键取非 None
                inputs = None
                for k in ('image','images','imgs'):
                    if batch.get(k) is not None:
                        inputs = batch.get(k); break
                labels = None
                for k in ('label','labels'):
                    if batch.get(k) is not None:
                        labels = batch.get(k); break
                if inputs is None or labels is None:
                    pbar.update(1); continue
            else:
                try:
                    inputs, labels = batch
                except Exception:
                    pbar.update(1); continue

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.cuda()
            if isinstance(labels, torch.Tensor):
                labels = labels.cuda()

            if image_level:
                if isinstance(inputs, torch.Tensor) and inputs.dim() == 3:
                    inputs_model = inputs.unsqueeze(0)
                else:
                    inputs_model = inputs

                # standard forward for metrics，同时统计前向推理耗时
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t0 = time.time()
                with torch.no_grad():
                    outputs, _ = model(inputs_model)
                    outputs = F.softmax(outputs, dim=-1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_time_total += (time.time() - _t0)
                infer_image_count += int(outputs.shape[0])
                probs = outputs[:,1].cpu().tolist()

                # t-SNE: 为当前样本记录标签（DataLoader batch_size=1）
                if tsne_handle is not None:
                    try:
                        if isinstance(labels, torch.Tensor):
                            if labels.dim() == 0:
                                lbl_tsne = int(labels.item())
                            else:
                                lbl_tsne = int(labels.view(-1)[0].item())
                        elif isinstance(labels, (list, tuple)) and len(labels) > 0:
                            lbl_tsne = int(labels[0])
                        else:
                            lbl_tsne = int(labels)
                        tsne_labels.append(lbl_tsne)
                    except Exception:
                        pass

                # attention heatmap visualization (CLS self-attn)
                if vis_dir is not None and batch_idx in vis_indices:
                    try:
                        attn = _get_last_block_attn(model)
                        if attn is not None:
                            save_path = os.path.join(vis_dir, f'attn_sample_{batch_idx+1}.png')
                            _save_attention_heatmap(inputs_model, attn, save_path)
                    except Exception:
                        pass

                # Grad-CAM candidates: 收集“判断正确的负样本”(True Negative: label=0 且 prob<0.5)，
                # 后续根据负类置信度从中选取 TOP-40 再统一生成 Grad-CAM
                if vis_gc_dir is not None:
                    # 提取当前 batch 的标注（兼容 tensor / list / 标量）
                    try:
                        if isinstance(labels, torch.Tensor):
                            if labels.dim() == 0:
                                lbl_val = int(labels.item())
                            else:
                                lbl_val = int(labels.view(-1)[0].item())
                        elif isinstance(labels, (list, tuple)) and len(labels) > 0:
                            lbl_val = int(labels[0])
                        else:
                            lbl_val = int(labels)
                    except Exception:
                        lbl_val = None

                    # 当前 batch 的第一张图片的假类概率（batch_size 通常为 1）
                    prob_val = float(probs[0]) if isinstance(probs, (list, tuple)) and len(probs) > 0 else None

                    # label=0 且 prob<0.5 → True Negative
                    if lbl_val == 0 and prob_val is not None and prob_val < 0.5:
                        # 负类置信度越大越好：conf_neg = 1 - prob_fake
                        conf_neg = 1.0 - prob_val
                        try:
                            img_cpu = inputs_model.detach().cpu()
                            gradcam_tn_candidates.append({'conf': conf_neg, 'img': img_cpu})
                        except Exception:
                            pass

                    # label=1 且 prob>=0.5 → Fake，按伪造方法类型分别收集候选样本
                    if lbl_val == 1 and prob_val is not None and prob_val >= 0.5:
                        method_id = None
                        try:
                            img_names = getattr(test_dataset, 'image_list', None)
                            if img_names is not None and sample_idx_for_paths < len(img_names):
                                p = str(img_names[sample_idx_for_paths])
                                p_low = p.lower()
                                if ('ff-real' in p_low) or ('original_sequences' in p_low) or ('/real/' in p_low):
                                    method_id = 0
                                elif ('deepfakes' in p_low) or ('ff-df' in p_low):
                                    method_id = 1
                                elif ('face2face' in p_low) or ('ff-f2f' in p_low):
                                    method_id = 2
                                elif ('faceswap' in p_low) or ('ff-fs' in p_low):
                                    method_id = 3
                                elif ('neuraltextures' in p_low) or ('ff-nt' in p_low):
                                    method_id = 4
                        except Exception:
                            method_id = None

                        if method_id in gradcam_fake_candidates and prob_val is not None:
                            try:
                                img_cpu = inputs_model.detach().cpu()
                                gradcam_fake_candidates[method_id].append({'conf': prob_val, 'img': img_cpu})
                            except Exception:
                                pass

                frame_predictions.extend(probs)
                if isinstance(labels, torch.Tensor) and labels.dim() == 0:
                    frame_labels.extend([int(labels.item())] * len(probs))
                elif isinstance(labels, (list,tuple)):
                    frame_labels.extend([int(l) for l in labels])
                else:
                    frame_labels.extend([int(labels)] * len(probs))
                sample_idx_for_paths += 1
            else:
                if isinstance(inputs, torch.Tensor) and inputs.dim() == 5 and inputs.size(0) == 1:
                    inputs_model = inputs.squeeze(0)
                else:
                    inputs_model = inputs
                # 视频级评估时，同样统计前向推理耗时
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t0 = time.time()
                with torch.no_grad():
                    outputs, _ = model(inputs_model)
                    outputs = F.softmax(outputs, dim=-1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_time_total += (time.time() - _t0)
                frame = outputs.shape[0]
                infer_image_count += int(frame)
                frame_predictions.extend(outputs[:,1].cpu().tolist())
                if isinstance(labels, torch.Tensor) and labels.dim() == 0:
                    frame_labels.extend(labels.expand(frame).cpu().tolist())
                else:
                    frame_labels.extend([int(labels)] * frame)
                pre = torch.mean(outputs[:,1])
                video_predictions.append(pre.cpu().item())
                video_labels.append(int(labels) if not isinstance(labels, torch.Tensor) else int(labels.item()))

            pbar.update(1)
        pbar.close()

    # === 打印 Shared+Specific LoRA MoE 中 specific 专家的 gate 使用统计 ===
    try:
        layer_idx = 0
        moe_layers_found = False
        print('\n' + '='*70)
        print('SPECIFIC EXPERT GATE USAGE (LoRA_MoE)'.center(70))
        print('='*70)
        for name, module in model.named_modules():
            if type(module).__name__ != 'LoRA_MoElayer':
                continue
            moe_layers_found = True
            # 只有在本次 eval 中真正被调用过的层才有意义
            bt = getattr(module, 'gate_batch_count', 0)
            tk = getattr(module, 'gate_token_count', 0)
            imp_sum = getattr(module, 'gate_importance_sum', None)
            load_sum = getattr(module, 'gate_load_sum', None)
            if bt == 0 or tk == 0 or imp_sum is None or load_sum is None:
                continue

            num_experts = int(imp_sum.numel())
            top_k = getattr(module, 'k', None)

            # 转成 numpy 便于后处理/打印
            imp_np = imp_sum.numpy().astype('float64')
            load_np = load_sum.numpy().astype('float64')
            token_cnt = float(tk)

            # usage_ratio: 每个 token 被路由到该 expert 的频率 (0~1)
            usage_ratio = load_np / max(token_cnt, 1.0)
            # importance_norm: 相对 gate 权重占比，总和约为 1
            total_imp = imp_np.sum() if imp_np.sum() > 0 else 1.0
            importance_norm = imp_np / total_imp

            print(f"Layer {layer_idx}: {name} | num_experts={num_experts}, top_k={top_k}, batches={bt}, tokens={tk}")
            print(f"{'Expert':<8}{'Usage%':>10}{'GateMass%':>14}{'Load':>10}")
            for eid in range(num_experts):
                u = usage_ratio[eid] * 100.0
                g = importance_norm[eid] * 100.0
                l = load_np[eid]
                print(f"{eid:<8}{u:>9.2f}%{g:>13.2f}%{l:>10.0f}")
            print('-'*70)
            layer_idx += 1

        if not moe_layers_found:
            print('No LoRA_MoElayer modules found in model; skip gate usage stats.')
        print('='*70)
    except Exception as e:
        print('Eval: failed to compute/print specific expert gate usage stats:', e)

    # 使用仓库内的 cal_metrics 输出结果（与 MoE-FFD-main 的评估流程一致）
    # MoE-FFD-main metrics (cal_metrics) - 帧级
    frame_results = cal_metrics(frame_labels, frame_predictions, threshold=0.5)
    print('MoE metrics (image-level): F_Acc: {:.2%}, F_Auc: {:.4}, F_EER:{:.2%}'.format(frame_results.ACC, frame_results.AUC, frame_results.EER))
    # 追加：EER 阈值下的准确率
    try:
        frame_results_eer = cal_metrics(frame_labels, frame_predictions, threshold='auto')
        print('MoE metrics (image-level @ EER-thr): ACC: {:.2%} (thr={:.4f})'.format(frame_results_eer.ACC, frame_results_eer.Thre))
    except Exception:
        pass

    # 另外，尝试基于 frame_predictions 做视频级聚合并输出 MoE 的视频级指标（如果可能）
    try:
        img_names = getattr(test_dataset, 'image_list', None)
        if img_names is not None and len(img_names) == len(frame_predictions):
            # 使用与 ForensicsAdapter 相同的聚合规则：按路径的上一级目录（parts[-2]）聚合每个视频的帧
            from collections import OrderedDict
            vid_to_preds = OrderedDict()
            vid_to_label = {}
            for nm, pred, lbl in zip(img_names, frame_predictions, frame_labels):
                s = nm
                # 支持 Windows 风格的路径分隔符
                if '\\' in s:
                    parts = s.split('\\')
                else:
                    parts = s.split('/')
                # 取上一级目录作为 video id
                if len(parts) >= 2:
                    vid = parts[-2]
                else:
                    vid = parts[0]
                if vid not in vid_to_preds:
                    vid_to_preds[vid] = []
                vid_to_preds[vid].append(pred)
                # 如果同一视频多次出现 label，保留最后一个（或相同则无影响）
                vid_to_label[vid] = int(lbl)
            # 按 FA 的做法，video_pred = frame_preds.mean(), video_label = int(mean(frame_labels))
            video_preds = []
            video_lbls = []
            for vid, preds in vid_to_preds.items():
                video_preds.append(float(np.mean(preds)))
                # 根据 FA 的实现，视频标签按帧标签平均后取 int
                # 这会导致只有当所有帧均为 1 时才为 1
                # 构造对应的帧标签列表以计算 mean
                # 尝试从 frame_labels 及 img_names 构建，但我们已在 vid_to_label 记录了最后的 label
                # 为最好兼容性，若需要更精确的 per-video label，请使用 ForensicsAdapter 返回的 video_label
                video_lbls.append(int(np.mean([int(l) for i,l in enumerate(frame_labels) if (img_names[i].split('/')[-2] if '/' in img_names[i] else img_names[i].split('\\')[-2]) == vid])))
            if len(video_preds) > 0:
                video_results = cal_metrics(video_lbls, video_preds, threshold=0.5)
                print('MoE metrics (video-level from frames): V_Acc: {:.2%}, V_Auc: {:.4} V_EER:{:.2%}'.format(video_results.ACC, video_results.AUC, video_results.EER))
                # 追加：视频级在 EER 阈值下的准确率
                try:
                    video_results_eer = cal_metrics(video_lbls, video_preds, threshold='auto')
                    print('MoE metrics (video-level @ EER-thr): ACC: {:.2%} (thr={:.4f})'.format(video_results_eer.ACC, video_results_eer.Thre))
                except Exception:
                    pass
        else:
            if img_names is None:
                print('注意: test_dataset 无 image_list，无法从帧聚合到视频以计算 MoE 视频级指标')
            else:
                print('注意: image_list 长度({})与 frame_predictions 长度({}) 不匹配；跳过 MoE 视频级指标'.format(len(img_names), len(frame_predictions)))
    except Exception as e:
        print('计算 MoE 视频级指标失败:', e)

    # ForensicsAdapter metrics (if available) - uses image paths list from dataset
    if fa_get_test_metrics is not None:
        try:
            img_names = getattr(test_dataset, 'image_list', None)
            if img_names is not None and len(img_names) == len(frame_predictions):
                fa_res = fa_get_test_metrics(np.array(frame_predictions), np.array(frame_labels), img_names)
                # pretty print FA metrics (frame-level)
                print('ForensicsAdapter metrics (frame-level):', {k: v for k, v in fa_res.items() if k in ('acc','auc','eer','ap')})

                # If FA returned video-level aggregates, prefer and print them
                v_auc = fa_res.get('video_auc', None)
                v_eer = fa_res.get('video_eer', None)
                v_acc = fa_res.get('video_acc', None)
                if v_auc is not None or v_eer is not None or v_acc is not None:
                    print('ForensicsAdapter metrics (video-level): V_Acc: {}, V_Auc: {}, V_EER: {}'.format(v_acc, v_auc, v_eer))
                else:
                    # If FA did not return video-level but did return per-video preds/labels, compute metrics
                    v_preds = fa_res.get('video_pred', None)
                    v_labels = fa_res.get('video_label', None)
                    if v_preds is not None and v_labels is not None:
                        try:
                            v_metrics = cal_metrics(v_labels, v_preds, threshold=0.5)
                            print('ForensicsAdapter metrics (video-level computed): V_Acc: {:.2%}, V_Auc: {:.4} V_EER:{:.2%}'.format(v_metrics.ACC, v_metrics.AUC, v_metrics.EER))
                        except Exception:
                            pass
                # If FA's video-level exists, prefer it when comparing with MoE's aggregation; otherwise keep existing MoE video aggregation
            else:
                # If lengths mismatch, try expanding img_names to match frame_predictions
                if img_names is not None and len(img_names) > 0:
                    print('注意: FA img list length does not match frame_predictions length; skipping FA metrics')
        except Exception as e:
            print('计算 ForensicsAdapter 指标失败:', e)

    # 追加打印：当前阈值(0.5)下的混淆矩阵（帧级、视频级）
    try:
        from sklearn.metrics import confusion_matrix
        import numpy as np
        # frame-level @ 0.5
        pred_frame_05 = (np.array(frame_predictions) > 0.5).astype(int)
        cm_f_05 = confusion_matrix(frame_labels, pred_frame_05, labels=[0,1])
        print('Confusion matrix at threshold 0.5 (frame-level):')
        print(cm_f_05)
        # video-level @ 0.5（如果有）
        if len(video_predictions) == len(video_labels) and len(video_predictions) > 0:
            pred_video_05 = (np.array(video_predictions) > 0.5).astype(int)
            cm_v_05 = confusion_matrix(video_labels, pred_video_05, labels=[0,1])
            print('Confusion matrix at threshold 0.5 (video-level):')
            print(cm_v_05)
    except Exception:
        pass

    duration = time.time() - start_time
    print('运行时长: {}h {}m'.format(int(duration//3600), int(duration%3600//60)))

    # === Inference Latency (ms/image) ===
    try:
        if infer_image_count > 0 and infer_time_total > 0:
            latency_ms = infer_time_total / float(infer_image_count) * 1000.0
            print('Inference Latency: {:.3f} ms/image (纯前向, 不含数据加载和后处理)'.format(latency_ms))
        else:
            print('Inference Latency: insufficient data to compute (infer_image_count={})'.format(infer_image_count))
    except Exception as e:
        print('Failed to compute inference latency:', e)

    # === GFLOPs (approximate) ===
    # 仅做一次前向 FLOPs 估计，使用 thop，如果环境未安装则跳过
    try:
        import torch
        try:
            from thop import profile
        except Exception:
            profile = None
        if profile is not None:
            # 根据模型参数自动推断 device，避免使用未定义的全局 device 变量
            try:
                model_device = next(model.parameters()).device
            except Exception:
                model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 构造一个与测试分辨率一致的虚拟输入
            img_size = int(fa_config.get('resolution', 256))
            dummy = torch.randn(1, 3, img_size, img_size, device=model_device)
            model.eval()
            with torch.no_grad():
                flops, params = profile(model, inputs=(dummy,), verbose=False)
            gflops = flops / 1e9
            print('Model Complexity: {:.3f} GFLOPs (thop 估计)'.format(gflops))
        else:
            print('Model Complexity (GFLOPs): 需要安装 thop 才能计算，示例: pip install thop')
    except Exception as e:
        print('Failed to compute GFLOPs (thop):', e)

    # 根据收集到的 True Negative 候选样本，按负类置信度选取 TOP-40 生成 Grad-CAM
    if vis_gc_dir is not None and len(gradcam_tn_candidates) > 0:
        try:
            os.makedirs(vis_gc_dir, exist_ok=True)
        except Exception:
            pass
        # 按 conf 从大到小排序，取前 gradcam_tn_max 个
        gradcam_tn_candidates.sort(key=lambda d: d.get('conf', 0.0), reverse=True)
        selected = gradcam_tn_candidates[:gradcam_tn_max]
        print('Grad-CAM-Fused: selected {} True Negative samples for visualization'.format(len(selected)))
        for idx, item in enumerate(selected, start=1):
            img = item['img']
            try:
                img_cuda = img.cuda()
                save_path_gc = os.path.join(vis_gc_dir, f'gradcam_tn_top{idx}.png')
                with torch.enable_grad():
                    # 使用融合后的向量做“伪 Grad-CAM”，target_class=None → 使用预测类别
                    _run_gradcam_fused_head(model, img_cuda.clone(), save_path_gc, target_class=None)
            except Exception as e:
                print(f'Grad-CAM-Fused: failed on selected sample {idx}:', e)

    # 额外：按伪造方法类型（DF/F2F/FS/NT）分别选取模型最自信的前 gradcam_fake_max 张生成 Grad-CAM
    if vis_gc_dir is not None:
        try:
            method_name_map = {1: 'DF', 2: 'F2F', 3: 'FS', 4: 'NT'}
            for m_id, items in gradcam_fake_candidates.items():
                if not items:
                    continue
                items_sorted = sorted(items, key=lambda d: d.get('conf', 0.0), reverse=True)
                selected_m = items_sorted[:gradcam_fake_max]
                print('Grad-CAM-Fused: selected {} fake samples for method {}'.format(len(selected_m), method_name_map.get(m_id, m_id)))
                for idx, item in enumerate(selected_m, start=1):
                    img = item['img']
                    try:
                        img_cuda = img.cuda()
                        save_path_gc = os.path.join(vis_gc_dir, f'gradcam_{method_name_map.get(m_id, m_id)}_top{idx}.png')
                        with torch.enable_grad():
                            _run_gradcam_fused_head(model, img_cuda.clone(), save_path_gc, target_class=1)
                    except Exception as e:
                        print('Grad-CAM-Fused: failed on fake sample method {}, idx {}:'.format(method_name_map.get(m_id, m_id), idx), e)
        except Exception as e:
            print('Grad-CAM-Fused: failed to generate per-method Grad-CAM:', e)

    # 若开启了 t-SNE，可视化并保存特征（仅生成图像，不落盘中间特征文件）
    if tsne_handle is not None:
        tsne_handle.remove()
    # TSNE debug: 打印收集到的特征和标签数量
    print(f"TSNE: collected features={len(tsne_features)}, labels={len(tsne_labels)}, out_dir={tsne_out_dir}")
    if tsne_out_dir is not None and len(tsne_features) > 0:
        try:
            import numpy as np
            # 收集原始特征
            feats = torch.cat(tsne_features, dim=0).cpu().numpy()  # (N, D)
            n_samples = feats.shape[0]

            # 默认使用原始 0/1 标签；若长度不匹配，则截断到最小长度
            if len(tsne_labels) >= n_samples:
                labels_np = np.array(tsne_labels[:n_samples], dtype=int)
            elif len(tsne_labels) > 0:
                labels_np = np.array(tsne_labels + [tsne_labels[-1]] * (n_samples - len(tsne_labels)), dtype=int)
            else:
                # 若未记录到任何标签，则退回到全 0 标签（只看特征分布）
                labels_np = np.zeros((n_samples,), dtype=int)

            # 尝试从图像路径中解析“伪造方法标签”（FF++ 的 Deepfakes / Face2Face / FaceSwap / NeuralTextures / Real）
            # 映射规则：0=Real, 1=DF, 2=F2F, 3=FS, 4=NT
            try:
                img_names = getattr(test_dataset, 'image_list', None)
            except Exception:
                img_names = None

            if img_names is not None and len(img_names) >= n_samples:
                method_labels = []
                for idx in range(n_samples):
                    p = str(img_names[idx])
                    p_low = p.lower()
                    lbl = 1  # default fake

                    # 1) Real: FF-real 或 FF++ 原始序列
                    if ('ff-real' in p_low) or ('original_sequences' in p_low) or ('/real/' in p_low):
                        lbl = 0

                    # 2) Deepfakes / Face2Face / FaceSwap / NeuralTextures (FF++ 真实子目录名)
                    elif 'deepfakes' in p_low:
                        lbl = 1  # DF
                    elif 'face2face' in p_low:
                        lbl = 2
                    elif 'faceswap' in p_low:
                        lbl = 3
                    elif 'neuraltextures' in p_low:
                        lbl = 4

                    # 3) 兼容旧式命名 FF-DF / FF-F2F / FF-FS / FF-NT
                    elif 'ff-df' in p_low:
                        lbl = 1
                    elif 'ff-f2f' in p_low:
                        lbl = 2
                    elif 'ff-fs' in p_low:
                        lbl = 3
                    elif 'ff-nt' in p_low:
                        lbl = 4
                    else:
                        # 无法解析方法时，退回到原来的 0/1 标签
                        if idx < len(labels_np):
                            lbl = int(labels_np[idx])
                    method_labels.append(lbl)
                labels_np = np.array(method_labels, dtype=int)

            if n_samples >= 5:
                os.makedirs(tsne_out_dir, exist_ok=True)
                # 直接调用顶层 tsne.py 中的 run_tsne_from_arrays，以原有 t-SNE 配置生成可视化图像
                try:
                    import sys as _sys
                    import importlib as _importlib
                    # 现在 tsne.py 与 eval.py 同级，直接使用当前文件所在目录
                    tsne_dir = os.path.dirname(os.path.abspath(__file__))
                    if tsne_dir not in _sys.path:
                        _sys.path.insert(0, tsne_dir)
                    tsne_mod = _importlib.import_module('tsne')
                    if hasattr(tsne_mod, 'run_tsne_from_arrays'):
                        tsne_runs = getattr(args, 'tsne_runs', 1)
                        if tsne_runs is None or tsne_runs < 1:
                            tsne_runs = 1
                        for run_idx in range(tsne_runs):
                            suffix = f'_run{run_idx+1}' if tsne_runs > 1 else ''
                            out_png = os.path.join(tsne_out_dir, f'tsne_moeffd{suffix}.png')
                            log_tag = f'moeffd{suffix}' if suffix else 'moeffd'
                            tsne_mod.run_tsne_from_arrays(feats, labels_np, out_png=out_png, log=log_tag)
                            print(f'TSNE: figure generated (run {run_idx+1}/{tsne_runs}) at', out_png)
                    else:
                        print('TSNE: tsne.py found but run_tsne_from_arrays not defined; skip auto plotting')
                except Exception as e_tsne:
                    print('TSNE: failed to import/use tsne.run_tsne_from_arrays, skip auto plotting:', e_tsne)
            else:
                print('TSNE: 样本数太少 ({}), 跳过 t-SNE'.format(n_samples))
        except Exception as e:
            print('TSNE: failed to compute t-SNE:', e)

