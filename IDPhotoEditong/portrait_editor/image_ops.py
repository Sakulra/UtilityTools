
"""图片处理：AI 人像分割、蒙版精修、背景合成、压缩与保存。"""
from __future__ import annotations

import io
import os
from typing import Optional, Tuple, Dict

from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFilter

SUPPORTED_FORMATS = {
    "PNG": "PNG",
    "JPEG": "JPEG",
    "WebP": "WEBP",
    "BMP": "BMP",
    "TIFF": "TIFF",
}

# 模型名称必须是当前安装的 rembg 版本实际支持的名称。
# birefnet 系列可能需要升级 rembg；若模型不受支持，程序会提示错误。
SUPPORTED_MODELS = {
    "u2net": "u2net",
    "u2netp": "u2netp",
    "u2net_human_seg": "u2net_human_seg",
    "isnet-general-use": "isnet-general-use",
    "birefnet-general": "birefnet-general",
    "birefnet-portrait": "birefnet-portrait",
}

# 缓存已创建的模型会话，避免反复加载模型。
_SESSION_CACHE: Dict[str, object] = {}


def open_image(path: str) -> Image.Image:
    """读取图片并校正 EXIF 方向，统一转换为 RGBA。"""
    with Image.open(path) as im:
        fixed = ImageOps.exif_transpose(im)
        return fixed.convert("RGBA").copy()


def get_model_session(model_name: str = "u2net"):
    """获取或创建 rembg 模型会话。"""
    model_name = model_name.strip()

    if not model_name:
        raise ValueError("请先选择抠图模型。")

    if model_name not in _SESSION_CACHE:
        try:
            from rembg import new_session
        except ImportError as exc:
            raise RuntimeError(
                "未安装 rembg，请运行：pip install -r requirements.txt"
            ) from exc

        try:
            _SESSION_CACHE[model_name] = new_session(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"无法加载模型 {model_name!r}。\n"
                "请检查 rembg 版本是否支持该模型，以及模型文件是否下载成功。\n"
                f"原始错误：{exc}"
            ) from exc

    return _SESSION_CACHE[model_name]


def remove_background(
    image: Image.Image,
    model_name: str = "u2net",
) -> Image.Image:
    """使用指定模型去除背景，返回 RGBA 透明人像。"""
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError(
            "未安装 rembg，请运行：pip install -r requirements.txt"
        ) from exc

    session = get_model_session(model_name)

    buf = io.BytesIO()
    image.convert("RGBA").save(buf, format="PNG")

    try:
        result = remove(buf.getvalue(), session=session)
        with Image.open(io.BytesIO(result)) as cutout:
            return cutout.convert("RGBA").copy()
    except Exception as exc:
        raise RuntimeError(f"模型 {model_name} 抠图失败：{exc}") from exc


def extract_alpha_mask(image: Image.Image) -> Image.Image:
    """
    提取透明度蒙版。
    白色代表不透明，黑色代表透明，灰色代表半透明。
    """
    return image.convert("RGBA").getchannel("A").copy()


def apply_alpha_mask(
    image: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """将精修后的蒙版应用到人像图层。"""
    foreground = image.convert("RGBA")
    alpha = mask.convert("L")

    if alpha.size != foreground.size:
        alpha = alpha.resize(
            foreground.size,
            Image.Resampling.LANCZOS,
        )

    foreground.putalpha(alpha)
    return foreground


def paint_mask(
    mask: Image.Image,
    points,
    radius: int = 15,
    value: int = 0,
    softness: float = 0,
) -> Image.Image:
    """
    在蒙版上绘制连续笔画，返回新的 L 模式蒙版。

    points：图片坐标序列，例如 [(100, 120), (110, 125)]。
    radius：笔刷半径，单位为原图像素。
    value：0 表示擦除，255 表示完全保留。
    softness：边缘模糊半径；0 表示硬边缘。
    """
    if radius < 1:
        raise ValueError("笔刷半径必须大于或等于 1。")

    result = mask.convert("L").copy()
    points = list(points)

    if not points:
        return result

    overlay = Image.new("L", result.size, 0)
    draw = ImageDraw.Draw(overlay)

    if len(points) == 1:
        x, y = points[0]
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=255,
        )
    else:
        draw.line(points, fill=255, width=radius * 2)
        for x, y in (points[0], points[-1]):
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=255,
            )

    if softness > 0:
        overlay = overlay.filter(
            ImageFilter.GaussianBlur(radius=softness)
        )

    # 擦除时降低 Alpha；恢复时可以将蒙版向白色方向提升。
    # 对恢复操作，建议由 GUI 使用原始 AI 蒙版恢复对应区域，
    # 以免把原本半透明的头发边缘直接变成完全不透明。
    if value <= 0:
        result = ImageChops.subtract(result, overlay)
    elif value >= 255:
        result = ImageChops.lighter(result, overlay)
    else:
        paint_layer = Image.new("L", result.size, value)
        result.paste(paint_layer, mask=overlay)

    return result


def feather_mask(
    mask: Image.Image,
    radius: float = 1.0,
) -> Image.Image:
    """轻微羽化蒙版边缘。radius=0 时不处理。"""
    if radius <= 0:
        return mask.convert("L").copy()
    return mask.convert("L").filter(
        ImageFilter.GaussianBlur(radius=radius)
    )


def composite_background(
    foreground: Image.Image,
    background_path: Optional[str] = None,
    background_color: str = "#FFFFFF",
) -> Image.Image:
    """将透明人像放置到背景图片或纯色背景上。"""
    fg = foreground.convert("RGBA")
    width, height = fg.size

    if background_path:
        bg = open_image(background_path)
        bg = ImageOps.fit(
            bg,
            (width, height),
            method=Image.Resampling.LANCZOS,
        )
    else:
        bg = Image.new(
            "RGBA",
            (width, height),
            background_color,
        )

    return Image.alpha_composite(bg.convert("RGBA"), fg)


def _to_output_mode(
    image: Image.Image,
    fmt: str,
) -> Image.Image:
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
        if "A" in image.getbands():
            return image.convert("RGBA")
        return image.convert("RGB")

    raise ValueError(f"不支持的输出格式：{fmt}")


def encode_image(
    image: Image.Image,
    fmt: str,
    quality: int = 90,
) -> bytes:
    """编码到内存。quality 主要对 JPEG/WebP 有效。"""
    fmt = fmt.upper()
    output = _to_output_mode(image, fmt)
    buf = io.BytesIO()

    if fmt == "JPEG":
        output.save(
            buf,
            format=fmt,
            quality=quality,
            optimize=True,
            progressive=True,
        )
    elif fmt == "WEBP":
        output.save(
            buf,
            format=fmt,
            quality=quality,
            method=6,
        )
    elif fmt == "PNG":
        output.save(
            buf,
            format=fmt,
            optimize=True,
            compress_level=9,
        )
    elif fmt == "TIFF":
        output.save(
            buf,
            format=fmt,
            compression="tiff_adobe_deflate",
        )
    else:
        output.save(buf, format=fmt)

    return buf.getvalue()


def _size_search(
    image: Image.Image,
    fmt: str,
    target_bytes: int,
) -> Tuple[bytes, bool]:
    """寻找不超过目标文件大小的最高可用编码质量。"""
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

    return (
        smallest or encode_image(image, fmt, 5),
        False,
    )


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
        target_bytes = max(
            1,
            int(len(baseline) * ratio_percent / 100.0),
        )
        mode = "exact"

    if mode == "exact":
        if target_bytes is None or target_bytes <= 0:
            raise ValueError("目标大小必须大于 0。")

        if fmt in ("JPEG", "WEBP"):
            data, reached = _size_search(
                image,
                fmt,
                target_bytes,
            )
            note = (
                "已达到目标大小限制"
                if reached
                else "无法压到目标大小以下，已输出最小可用结果"
            )
            return data, note

        data = encode_image(image, fmt)
        note = (
            "已导出；此格式不保证精确目标大小"
            if len(data) > target_bytes
            else "文件大小不超过目标值"
        )
        return data, note

    if mode == "range":
        if (
            min_bytes is None
            or max_bytes is None
            or min_bytes <= 0
            or max_bytes < min_bytes
        ):
            raise ValueError(
                "大小范围无效，请确保最大值不小于最小值，"
                "且最小值大于 0。"
            )

        if fmt not in ("JPEG", "WEBP"):
            data = encode_image(image, fmt)
            return (
                data,
                "此格式不支持按质量搜索大小范围，已按默认参数导出",
            )

        candidates = []
        for quality in range(5, 96, 5):
            data = encode_image(image, fmt, quality)
            candidates.append((len(data), quality, data))

        in_range = [
            item
            for item in candidates
            if min_bytes <= item[0] <= max_bytes
        ]

        if in_range:
            _, _, data = max(in_range, key=lambda item: item[1])
            return data, "已找到范围内的结果"

        closest = min(
            candidates,
            key=lambda item: min(
                abs(item[0] - min_bytes),
                abs(item[0] - max_bytes),
            ),
        )
        return (
            closest[2],
            "无法精确落入指定范围，已输出最接近的结果",
        )

    raise ValueError("未知压缩模式。")


def save_bytes(path: str, data: bytes) -> None:
    """先写入临时文件再替换，避免中途失败留下半个文件。"""
    temp_path = path + ".tmp"

    try:
        with open(temp_path, "wb") as file:
            file.write(data)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
