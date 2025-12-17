#!/usr/bin/env python3
import json
from pathlib import Path

from PIL import Image
import numpy as np


def hex_to_rgb(hex_color: str):
    """Converte string '#rrggbb' para tupla (R, G, B)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Cor hex inválida: {hex_color}")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def build_luts(cfg: dict):
    """
    Constrói duas LUTs:

      1) fine_id_lut: id_fino (0..63) -> id_agregado (0..11)
      2) agg_rgb_lut: id_agregado (0..11) -> cor RGB (R, G, B)

    A partir de:
      - cfg["class_id_labels"]: nome_classe_fina -> id_fino (0..63)
      - cfg["class_id"]:        nome_classe_fina -> id_agregado (0..11)
      - cfg["target_classes"]:  nome_classe_agregada -> id_agregado (0..11)
      - cfg["target_color"]:    nome_classe_agregada -> hex_cor
    """
    class_id_labels = cfg["class_id_labels"]   # nome -> id fino
    fine_to_agg = cfg["class_id"]              # nome -> id agregado
    target_classes = cfg["target_classes"]     # nome agregado -> id agregado
    target_color = cfg["target_color"]         # nome agregado -> hex cor

    # ---- LUT 1: fine_id (0..63) -> agg_id (0..11) ----
    max_fine_id = max(class_id_labels.values())
    fine_id_lut = np.zeros(max_fine_id + 1, dtype=np.uint8)

    for name, fine_id in class_id_labels.items():
        if name not in fine_to_agg:
            raise ValueError(f"Classe '{name}' está em class_id_labels mas não em class_id")
        agg_id = fine_to_agg[name]
        fine_id_lut[fine_id] = agg_id

    # ---- LUT 2: agg_id (0..11) -> RGB ----
    max_agg_id = max(target_classes.values())
    agg_rgb_lut = np.zeros((max_agg_id + 1, 3), dtype=np.uint8)

    # id_agregado -> nome_classe_agregada
    agg_id_to_name = {v: k for k, v in target_classes.items()}

    for agg_id, agg_name in agg_id_to_name.items():
        if agg_name not in target_color:
            raise ValueError(f"Classe agregada '{agg_name}' não possui cor em target_color")
        rgb = hex_to_rgb(target_color[agg_name])
        agg_rgb_lut[agg_id, :] = rgb

    return fine_id_lut, agg_rgb_lut


def remap_image_grey_to_rgb(img: Image.Image,
                            fine_id_lut: np.ndarray,
                            agg_rgb_lut: np.ndarray) -> Image.Image:
    """
    Entrada: imagem em escala de cinza (L) com IDs finos (0..63).
    Saída: imagem RGB com cores das 12 classes agregadas.
    """
    # Garante escala de cinza 8 bits
    img = img.convert("L")
    arr = np.array(img)  # (H, W), uint8

    max_val = int(arr.max())
    if max_val >= fine_id_lut.shape[0]:
        raise ValueError(
            f"Encontrado id de classe {max_val}, maior/igual ao tamanho da LUT de fine_ids ({fine_id_lut.shape[0]})."
        )

    # 1) Mapeia id fino -> id agregado
    agg_ids = fine_id_lut[arr]         # (H, W) com valores 0..11

    # 2) Mapeia id agregado -> cor RGB
    rgb_arr = agg_rgb_lut[agg_ids]     # (H, W, 3), uint8

    return Image.fromarray(rgb_arr, mode="RGB")


def process_folder(cfg_path: Path, input_dir: Path, output_dir: Path):
    # Carrega config JSON
    with cfg_path.open("r") as f:
        cfg = json.load(f)

    fine_id_lut, agg_rgb_lut = build_luts(cfg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extensões de imagem comuns; ajuste se precisar
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    imgs = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]

    if not imgs:
        print(f"Nenhuma imagem encontrada em {input_dir}")
        return

    print(f"Encontradas {len(imgs)} imagens em {input_dir}")

    for img_path in imgs:
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path)
        remapped = remap_image_grey_to_rgb(img, fine_id_lut, agg_rgb_lut)
        remapped.save(out_path)

        print(f"[OK] {img_path} -> {out_path}")


def main():
    # Aqui já deixei caminhos fixos; se quiser depois coloco argparse
    cfg_path = Path("/home/lrm/workspace/segment_net/params/goose_category_3.json")
    input_dir = Path("/home/lrm/Infereces_Goose/Blacked_mask")       # grayscale 0..63
    output_dir = Path("/home/lrm/Infereces_Goose/Mask_rgb")  # RGB 12 classes

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config JSON não encontrado: {cfg_path}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Pasta de entrada inválida: {input_dir}")

    process_folder(cfg_path, input_dir, output_dir)


if __name__ == "__main__":
    main()
