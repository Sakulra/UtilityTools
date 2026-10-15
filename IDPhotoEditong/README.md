# 本地人像抠图与图片压缩工具

适用环境：Python 3.9.25（建议 64 位 Windows；macOS/Linux 也可运行，需安装 Tk 支持）。

## 功能
- 弹窗选择任意像素尺寸的输入图片。
- 使用 rembg/ONNX 模型自动去除背景，保留透明人像。
- 可选择纯色背景，或选择一张图片作为背景。
- 支持按目标文件大小、目标大小范围或原图大小百分比压缩。
- 支持 PNG、JPEG、WebP、BMP、TIFF 格式保存。

## 安装与运行
建议在虚拟环境中运行：

```bash
python3.9 -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

首次执行抠图时，rembg 可能需要下载模型文件，请保持网络畅通。之后通常可离线使用。
如果在 Linux 上提示缺少 tkinter，请通过系统包管理器安装 python3-tk。

## 压缩说明
- **指定大小**：输入目标大小（KB/MB）。JPEG/WebP 会搜索合适的质量参数，尽量接近且不超过目标值；PNG 等格式不保证能达到精确目标大小。
- **大小范围**：输入最小和最大大小，程序尝试找到落入范围内的 JPEG/WebP 质量；无法达到时会提示。
- **压缩比率**：目标文件大小 = 当前编辑结果大小 × 百分比。例如 50% 表示目标约为当前文件的一半。
- JPEG 不支持透明通道；若输出 JPEG，透明区域会以白色填充。
- 图片尺寸不会被自动缩小，只调整编码质量/压缩参数。若目标文件大小过小，可能无法达到。

## 项目结构
- `main.py`：程序入口
- `portrait_editor/gui.py`：Tkinter 界面和操作流程
- `portrait_editor/image_ops.py`：抠图、背景合成、格式转换与压缩

替换说明
1. 将 portrait_editor/gui.py 复制到你项目的 portrait_editor/gui.py，覆盖旧文件。
2. 此 GUI 依赖更新版 image_ops.py 中的名称：
   SUPPORTED_MODELS、extract_alpha_mask、apply_alpha_mask
   并要求 remove_background(image, model_name="u2net") 支持 model_name 参数。
3. 运行：python main.py
4. 选择模型后点击“自动抠出人像”。抠图完成后可用“擦除背景/恢复人像”画笔编辑。
5. 画笔大小以原图像素计；预览缩放可放大局部，滚动条用于移动视图。
6. 恢复画笔恢复到 AI 初始蒙版在该区域的结果；“恢复透明人像”会重置整个蒙版。

