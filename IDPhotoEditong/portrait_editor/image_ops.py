"""图片处理：人像分割、背景合成、格式转换和大小压缩。"""
from __future__ import annotations

import io
import os
from typing import Optional, Tuple

from PIL import Image, ImageOps


SUPPORTED_FORMATS = {
    "PNG": "PNG",
    "JPEG": "JPEG",
    "WebP": "WEBP",
    "BMP": "BMP",
    "TIFF": "TIFF",
}


def open_image(path: str) -> Image.Image:
    """读取图片并校正 EXIF 方向，返回独立 RGBA 图像。"""
    with Image.open(path) as im:
        fixed = ImageOps.exif_transpose(im)
        return fixed.convert("RGBA").copy()


def remove_background(image: Image.Image) -> Image.Image:
    """通过 rembg 移除背景，返回带透明通道的 RGBA 图像。"""
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError("未安装 rembg。请运行：pip install -r requirements.txt") from exc

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    result = remove(buf.getvalue())
    with Image.open(io.BytesIO(result)) as cutout:
        return cutout.convert("RGBA").copy()


def composite_background(
    foreground: Image.Image,
    background_path: Optional[str] = None,
    background_color: str = "#FFFFFF",
) -> Image.Image:
    """将透明人像放置到背景图片或纯色背景上，背景会缩放到人像画布尺寸。"""
    fg = foreground.convert("RGBA")
    width, height = fg.size

    if background_path:
        bg = open_image(background_path)
        bg = ImageOps.fit(bg, (width, height), method=Image.Resampling.LANCZOS)
    else:
        bg = Image.new("RGBA", (width, height), background_color)

    return Image.alpha_composite(bg.convert("RGBA"), fg)


def _to_output_mode(image: Image.Image, fmt: str) -> Image.Image:
    fmt = fmt.upper()
    if fmt == "JPEG":
        if image.mode == "RGBA":
            base = Image.new("RGB", image.size, "white")
            base.paste(image, mask=image.getchannel("A"))
            return base
        return image.convert("RGB")
    if fmt == "BMP":
        return image.convert("RGB")
    if fmt in ("PNG", "WEBP", "TIFF"):
        return image.convert("RGBA") if "A" in image.getbands() else image.convert("RGB")
    raise ValueError(f"不支持的输出格式：{fmt}")


def encode_image(image: Image.Image, fmt: str, quality: int = 90) -> bytes:
    """编码到内存。quality 主要对 JPEG/WebP 有效。"""
    fmt = fmt.upper()
    output = _to_output_mode(image, fmt)
    buf = io.BytesIO()
    if fmt == "JPEG":
        output.save(buf, format=fmt, quality=quality, optimize=True, progressive=True)
    elif fmt == "WEBP":
        output.save(buf, format=fmt, quality=quality, method=6)
    elif fmt == "PNG":
        output.save(buf, format=fmt, optimize=True, compress_level=9)
    elif fmt == "TIFF":
        output.save(buf, format=fmt, compression="tiff_adobe_deflate")
    else:
        output.save(buf, format=fmt)
    return buf.getvalue()


def _size_search(image: Image.Image, fmt: str, target_bytes: int) -> Tuple[bytes, bool]:
    """在 JPEG/WebP 的质量 5..95 中寻找不超过目标大小的最大质量结果。"""
    best = None
    best_quality = 0
    smallest = None
    for quality in range(5, 96, 5):
        data = encode_image(image, fmt, quality)
        if smallest is None or len(data) < len(smallest):
            smallest = data
        if len(data) <= target_bytes and quality > best_quality:
            best = data
            best_quality = quality
    if best is not None:
        return best, True
    return (smallest or encode_image(image, fmt, 5)), False


def compress_image(
    image: Image.Image,
    fmt: str,
    mode: str,
    target_bytes: Optional[int] = None,
    min_bytes: Optional[int] = None,
    max_bytes: Optional[int] = None,
    ratio_percent: Optional[float] = None,
) -> Tuple[bytes, str]:
    """
    返回 (编码后数据, 说明)。
    mode: none / exact / range / ratio
    """
    fmt = fmt.upper()
    if mode == "none":
        return encode_image(image, fmt, 90), "使用默认质量导出"

    if mode == "ratio":
        if ratio_percent is None or ratio_percent <= 0:
            raise ValueError("压缩比率必须大于 0%。")
        baseline = encode_image(image, fmt, 95)
        target_bytes = max(1, int(len(baseline) * ratio_percent / 100.0))
        mode = "exact"

    if mode == "exact":
        if not target_bytes or target_bytes <= 0:
            raise ValueError("目标大小必须大于 0。")
        if fmt in ("JPEG", "WEBP"):
            data, reached = _size_search(image, fmt, target_bytes)
            note = "已达到目标大小限制" if reached else "无法压到目标大小以下，已输出最小可用结果"
            return data, note
        data = encode_image(image, fmt)
        note = "已导出；此格式不保证精确目标大小" if len(data) > target_bytes else "文件大小不超过目标值"
        return data, note

    if mode == "range":
        if min_bytes is None or max_bytes is None or min_bytes <= 0 or max_bytes < min_bytes:
            raise ValueError("大小范围无效，请确保最大值不小于最小值，且最小值大于 0。")
        if fmt not in ("JPEG", "WEBP"):
            data = encode_image(image, fmt)
            return data, "此格式不支持按质量搜索大小范围，已按默认参数导出"

        candidates = [(len(encode_image(image, fmt, q)), q, encode_image(image, fmt, q))
                     for q in range(5, 96, 5)]
        in_range = [item for item in candidates if min_bytes <= item[0] <= max_bytes]
        if in_range:
            # 优先选范围内质量最高的结果
            _, _, data = max(in_range, key=lambda item: item[1])
            return data, "已找到范围内的结果"
        closest = min(candidates, key=lambda item: min(abs(item[0] - min_bytes), abs(item[0] - max_bytes)))
        return closest[2], "无法精确落入指定范围，已输出最接近的结果"

    raise ValueError("未知压缩模式。")


def save_bytes(path: str, data: bytes) -> None:
    """先写入临时文件再替换，避免中途失败留下半个文件。"""
    temp_path = path + ".tmp"
    with open(temp_path, "wb") as file:
        file.write(data)
    os.replace(temp_path, path)
