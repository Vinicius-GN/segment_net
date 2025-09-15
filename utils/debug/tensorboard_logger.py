import os
from typing import Dict, List, Optional, Tuple
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

def _lazy_import_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    return plt, np, sns

class TensorboardLogger:
    def __init__(
        self,
        enabled: bool,
        log_dir: str,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        palette_rgb: Optional[Dict[int, Tuple[int, int, int]]] = None,
        max_images: int = 4,
        log_every_steps: int = 50,
        denorm: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None,
        confmat_cmap: str = "YlOrRd",
        confmat_annot_threshold: float = 0.04,
        save_png_dir: Optional[str] = None,
    ):
        self.enabled = bool(enabled)
        self.num_classes = int(num_classes)
        self.max_images = int(max_images)
        self.log_every_steps = int(log_every_steps)
        self.denorm = denorm
        self.confmat_cmap = confmat_cmap
        self.confmat_annot_threshold = float(confmat_annot_threshold)
        self.save_png_dir = save_png_dir

        if not self.enabled:
            self.writer = None
            return

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

        if class_names is None:
            self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        else:
            self.class_names = list(class_names)[: self.num_classes]
            if len(self.class_names) < self.num_classes:
                self.class_names += [f"class_{i}" for i in range(len(self.class_names), self.num_classes)]

        if palette_rgb is None or len(palette_rgb) == 0:
            base = torch.tensor(
                [
                    [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70],
                    [102, 102, 156], [190, 153, 153], [153, 153, 153],
                    [250, 170, 30], [220, 220, 0], [107, 142, 35],
                    [152, 251, 152], [70, 130, 180], [220, 20, 60],
                    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                    [0, 80, 100], [0, 0, 230], [119, 11, 32]
                ],
                dtype=torch.uint8,
            )
            if base.size(0) < self.num_classes:
                reps = (self.num_classes + base.size(0) - 1) // base.size(0)
                base = base.repeat(reps, 1)
            self.palette = base[: self.num_classes]
        else:
            pal = torch.zeros((self.num_classes, 3), dtype=torch.uint8)
            for k, v in palette_rgb.items():
                k = int(k)
                if 0 <= k < self.num_classes:
                    pal[k] = torch.tensor(v, dtype=torch.uint8)
            self.palette = pal

        if self.save_png_dir is not None:
            os.makedirs(self.save_png_dir, exist_ok=True)

    @staticmethod
    def _to_cpu(x: Tensor) -> Tensor:
        return x.detach().to("cpu", non_blocking=True)

    def _maybe_denorm(self, x: Tensor) -> Tensor:
        if self.denorm is None:
            return x
        mean, std = self.denorm
        mean_t = torch.tensor(mean, dtype=x.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype).view(1, -1, 1, 1)
        return x * std_t + mean_t

    def _colorize_labels(self, labels: Tensor) -> Tensor:
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)

        t = labels
        # (N,H,W)
        if t.dim() == 2:
            t = t.unsqueeze(0)  # (1,H,W)
        elif t.dim() == 4:
            # (N,1,H,W)
            if t.size(1) == 1 and (t.size(-1) != self.num_classes):
                t = t.squeeze(1)
            # (N,H,W,1)
            elif t.size(-1) == 1 and (t.size(1) != self.num_classes):
                t = t.squeeze(-1)
            elif t.size(1) == self.num_classes:
                t = torch.argmax(t, dim=1)
            # (N,H,W,C)
            elif t.size(-1) == self.num_classes:
                t = torch.argmax(t, dim=-1)
            else:
                if t.size(1) > 1:
                    t = t[:, 0]
                else:
                    t = t.squeeze()
        elif t.dim() != 3:
            raise ValueError(f"labels: shape not suported {tuple(t.shape)} (expected (N,H,W))")

        lab = t.clamp(min=0, max=self.num_classes - 1).to(dtype=torch.long)
        pal = self.palette.to(lab.device)  # (C,3) uint8
        colored = pal[lab]                 # (N,H,W,3) uint8, indexação avançada
        colored = colored.permute(0, 3, 1, 2).float() / 255.0  # (N,3,H,W)
        return colored

    def _make_mosaic_triptych(self, images_nchw: Tensor, logits_nchw: Tensor,
                            targets: Tensor, max_images: int) -> Tensor:
        with torch.no_grad():
            N = min(int(images_nchw.size(0)), int(max_images))
            imgs  = self._to_cpu(self._maybe_denorm(images_nchw[:N])).clamp(0, 1)  # (N,3,H,W)
            preds = torch.argmax(logits_nchw[:N], dim=1)                            # (N,H,W)

            gts = targets[:N]
            gt_rgb   = self._colorize_labels(self._to_cpu(gts))
            pred_rgb = self._colorize_labels(self._to_cpu(preds))

            rows = [torch.cat([imgs[i], gt_rgb[i], pred_rgb[i]], dim=2) for i in range(N)]  # concat W
            grid = torch.cat(rows, dim=1)  # concat H  → (3, H*N, W*3)
            return grid

    def log_step_scalars(self, split: str, epoch: int, step: int, global_step: int,
                         loss: float = None, dice: float = None, acc: float = None, miou: float = None):
        if not self.enabled or (step % self.log_every_steps) != 0:
            return
        tag = split.strip().lower()
        if loss is not None: self.writer.add_scalar(f"{tag}/loss_step", float(loss), global_step)
        if dice is not None: self.writer.add_scalar(f"{tag}/dice_step", float(dice), global_step)
        if acc is not None: self.writer.add_scalar(f"{tag}/acc_step", float(acc), global_step)
        if miou is not None: self.writer.add_scalar(f"{tag}/miou_step", float(miou), global_step)

    def log_epoch_metrics(self, epoch: int,
                          train_loss: float, val_loss: float,
                          train_micro_acc: float, val_micro_acc: float,
                          train_micro_miou: float, val_micro_miou: float,
                          train_dice: float = None, val_dice: float = None):
        if not self.enabled: return
        e = int(epoch)
        self.writer.add_scalar("train/loss", float(train_loss), e)
        self.writer.add_scalar("val/loss", float(val_loss), e)
        self.writer.add_scalar("train/micro_acc", float(train_micro_acc), e)
        self.writer.add_scalar("val/micro_acc", float(val_micro_acc), e)
        self.writer.add_scalar("train/micro_miou", float(train_micro_miou), e)
        self.writer.add_scalar("val/micro_miou", float(val_micro_miou), e)
        if train_dice is not None: self.writer.add_scalar("train/dice_epoch", float(train_dice), e)
        if val_dice   is not None: self.writer.add_scalar("val/dice_epoch",   float(val_dice),   e)

    def log_confmat(self, epoch: int, confmat: "np.ndarray", split: str):
        if not self.enabled: return
        plt, np, sns = _lazy_import_plotting()
        cm = confmat.astype(np.float64)
        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
        annot = np.where(cm_norm >= self.confmat_annot_threshold, np.char.mod('%.3f', cm_norm), '')
        fig = plt.figure(figsize=(10, 8), dpi=110)
        ax = sns.heatmap(
            cm_norm, annot=annot, cmap=self.confmat_cmap,
            xticklabels=self.class_names, yticklabels=self.class_names,
            fmt="", cbar=True, vmin=0, vmax=1, annot_kws={"fontsize": 12}
        )
        title = ""
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=18, fontweight='bold')
        plt.ylabel('True', fontsize=18, fontweight='bold')
        plt.gca().xaxis.set_label_coords(0.5, 1.05)
        plt.gca().yaxis.set_label_coords(-0.07, 0.5)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=14, rotation=0)
        plt.tight_layout()
        self.writer.add_figure(f"{split}/confusion_matrix", fig, global_step=epoch, close=True)
        if self.save_png_dir is not None:
            png_path = os.path.join(self.save_png_dir, f"{split}_confmat_epoch_{epoch:04d}.png")
            fig.savefig(png_path)
            plt.close(fig)

    @torch.no_grad()
    def log_mosaic_from_batch(self, epoch: int, split: str,
                              batch: Tuple[Tensor, Tensor],
                              logits: Optional[Tensor] = None,
                              model: Optional[torch.nn.Module] = None,
                              device: Optional[torch.device] = None):
        if not self.enabled or self.max_images <= 0: return
        x_img, y_img = batch
        if logits is None:
            if model is None: return
            was_training = model.training
            model.eval()
            if device is not None: x_img = x_img.to(device, non_blocking=True)
            logits = model(x_img)
            if was_training: model.train()
        grid = self._make_mosaic_triptych(self._to_cpu(x_img), self._to_cpu(logits), self._to_cpu(y_img), self.max_images)
        self.writer.add_image(f"{split}/mosaic", grid, global_step=epoch)
    
    def log_image_file(self, epoch: int, split: str, image_path: str, tag: str = "mosaic"):

        if not self.enabled:
            return
        if not os.path.isfile(image_path):
            return
        try:
            from PIL import Image
            import numpy as np
        except Exception:
            plt, np, _ = _lazy_import_plotting()
            arr = plt.imread(image_path)          
        else:
            img = Image.open(image_path).convert("RGB")
            arr = np.asarray(img, dtype="float32") / 255.0  
        self.writer.add_image(f"{split}/{tag}", arr, global_step=epoch, dataformats="HWC")

    def close(self):
        if self.enabled and self.writer is not None:
            self.writer.flush()
            self.writer.close()