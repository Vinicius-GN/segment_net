import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

input_dir = "logs/goose_segformerb2_danet_segformer_h_focalCE_dice_20251113-172641/raw_inference"
output_dir = "logs/goose_segformerb2_danet_segformer_h_focalCE_dice_20251113-172641/raw_inference_heatmaps"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith("_attn.npy"):
        continue

    npy_path = os.path.join(input_dir, fname)
    attn = np.load(npy_path).astype(np.float32)  

    attn_min = attn.min()
    attn_max = attn.max()
    if attn_max > attn_min:
        attn_norm = (attn - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attn)  

    base_name = os.path.splitext(fname)[0]
    mosaic_name = base_name.replace("_attn", "_mosaic") + ".png"
    mosaic_path = os.path.join(input_dir, mosaic_name)

    if not os.path.exists(mosaic_path):
        continue

    rgb = np.array(Image.open(mosaic_path).convert("RGB"))

    fig, ax = plt.subplots()
    ax.imshow(rgb)

    im = ax.imshow(
        attn_norm,
        cmap="jet",
        alpha=0.7,
        vmin=0.0,
        vmax=1.0
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    ax.axis("off")
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
