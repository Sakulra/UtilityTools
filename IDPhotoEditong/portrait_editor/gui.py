"""Tkinter 图形界面：模型选择、手动抠图精修、背景替换和图片压缩。"""
from __future__ import annotations
import os
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageChops
from .image_ops import (
    open_image, remove_background, composite_background, compress_image,
    save_bytes, SUPPORTED_FORMATS, SUPPORTED_MODELS,
    extract_alpha_mask, apply_alpha_mask,
)


class PortraitEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("本地人像抠图与图片压缩工具")
        self.root.geometry("1250x820")
        self.root.minsize(980, 680)

        self.source_path = None
        self.background_path = None
        self.original = None
        self.original_cutout = None
        self.original_mask = None
        self.current_mask = None
        self.cutout = None
        self.result = None
        self.bg_color = "#FFFFFF"
        self._use_color_background = False

        self.preview_photo = None
        self.display_scale = 1.0
        self.fit_scale = 1.0
        self.image_offset_x = 12
        self.image_offset_y = 12
        self.drawing = False
        self.stroke_points = []
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 30

        self.model_var = tk.StringVar(value="u2net")
        self.brush_size_var = tk.IntVar(value=24)
        self.brush_mode_var = tk.StringVar(value="erase")
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.format_var = tk.StringVar(value="PNG")
        self.compression_var = tk.StringVar(value="none")
        self.target_var = tk.StringVar(value="500")
        self.min_var = tk.StringVar(value="200")
        self.max_var = tk.StringVar(value="800")
        self.ratio_var = tk.StringVar(value="50")
        self.status_var = tk.StringVar(value="请选择一张图片开始。")

        self._build_ui()
        self._update_compression_fields()
        self._update_edit_buttons()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill="x")
        ttk.Button(top, text="选择待编辑图片", command=self.choose_source).pack(side="left", padx=3)
        self.cutout_button = ttk.Button(top, text="自动抠出人像", command=self.start_cutout)
        self.cutout_button.pack(side="left", padx=3)
        ttk.Button(top, text="选择背景图片", command=self.choose_background).pack(side="left", padx=3)
        ttk.Button(top, text="选择纯色背景", command=self.choose_color).pack(side="left", padx=3)
        ttk.Button(top, text="恢复透明人像", command=self.restore_cutout).pack(side="left", padx=3)

        model_frame = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        model_frame.pack(fill="x")
        ttk.Label(model_frame, text="AI 抠图模型：").pack(side="left")
        model_values = list(SUPPORTED_MODELS.keys()) if isinstance(SUPPORTED_MODELS, dict) else list(SUPPORTED_MODELS)
        if "u2net" not in model_values:
            model_values.insert(0, "u2net")
        ttk.Combobox(
            model_frame, textvariable=self.model_var, values=model_values,
            state="readonly", width=24
        ).pack(side="left", padx=(4, 10))
        ttk.Label(
            model_frame, text="首次使用某个模型可能需要下载模型文件；不同模型效果和速度不同。"
        ).pack(side="left")

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)
        preview_frame = ttk.Frame(body)
        preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.preview_canvas = tk.Canvas(
            preview_frame, background="#D9D9D9",
            highlightthickness=1, highlightbackground="#888888", cursor="crosshair"
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        vbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_canvas.yview)
        hbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_canvas.xview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        self.preview_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview_canvas.bind("<ButtonPress-1>", self._on_stroke_start)
        self.preview_canvas.bind("<B1-Motion>", self._on_stroke_move)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_stroke_end)
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)

        side = ttk.Frame(body, width=305)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="抠图精修", font=("", 12, "bold")).pack(anchor="w", pady=(2, 6))

        mode_frame = ttk.Frame(side)
        mode_frame.pack(fill="x", pady=3)
        ttk.Radiobutton(
            mode_frame, text="擦除背景", variable=self.brush_mode_var, value="erase"
        ).pack(side="left")
        ttk.Radiobutton(
            mode_frame, text="恢复人像", variable=self.brush_mode_var, value="restore"
        ).pack(side="left", padx=(8, 0))

        ttk.Label(side, text="画笔大小（原图像素）").pack(anchor="w", pady=(7, 0))
        brush_row = ttk.Frame(side)
        brush_row.pack(fill="x")
        brush_scale = ttk.Scale(
            brush_row, from_=2, to=180, orient="horizontal", command=self._on_brush_scale
        )
        brush_scale.set(self.brush_size_var.get())
        brush_scale.pack(side="left", fill="x", expand=True)
        self.brush_size_label = ttk.Label(brush_row, text="24 px", width=7)
        self.brush_size_label.pack(side="right")

        history_row = ttk.Frame(side)
        history_row.pack(fill="x", pady=(6, 2))
        self.undo_button = ttk.Button(history_row, text="撤销", command=self.undo, state="disabled")
        self.undo_button.pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.redo_button = ttk.Button(history_row, text="重做", command=self.redo, state="disabled")
        self.redo_button.pack(side="left", fill="x", expand=True, padx=(3, 0))

        ttk.Label(side, text="预览缩放").pack(anchor="w", pady=(8, 0))
        zoom_row = ttk.Frame(side)
        zoom_row.pack(fill="x")
        self.zoom_scale = ttk.Scale(
            zoom_row, from_=0.25, to=4.0, orient="horizontal",
            variable=self.zoom_var, command=self._on_zoom_change
        )
        self.zoom_scale.pack(side="left", fill="x", expand=True)
        self.zoom_label = ttk.Label(zoom_row, text="100%", width=7)
        self.zoom_label.pack(side="right")
        ttk.Button(side, text="适应窗口", command=self.fit_preview).pack(fill="x", pady=(3, 6))
        ttk.Label(
            side,
            text="在图片上拖动鼠标涂抹。擦除会让区域透明；恢复会恢复 AI 初始蒙版对应区域。放大后可用滚动条查看局部。",
            wraplength=285,
        ).pack(anchor="w", pady=(0, 8))

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="导出设置", font=("", 12, "bold")).pack(anchor="w", pady=(0, 6))
        ttk.Label(side, text="输出格式").pack(anchor="w")
        ttk.Combobox(
            side, textvariable=self.format_var, values=list(SUPPORTED_FORMATS.keys()),
            state="readonly"
        ).pack(fill="x", pady=(2, 8))

        ttk.Label(side, text="压缩方式").pack(anchor="w")
        self.compression_combo = ttk.Combobox(
            side, textvariable=self.compression_var, state="readonly",
            values=("none", "exact", "range", "ratio")
        )
        self.compression_combo.pack(fill="x", pady=(2, 4))
        self.compression_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_compression_fields())
        ttk.Label(
            side, text="none=默认；exact=指定大小；range=大小范围；ratio=基准大小百分比",
            wraplength=285
        ).pack(anchor="w", pady=(0, 6))

        self.exact_frame = ttk.Frame(side)
        ttk.Label(self.exact_frame, text="目标大小").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.exact_frame, textvariable=self.target_var, width=10).grid(row=0, column=1, sticky="w")
        ttk.Label(self.exact_frame, text="KB").grid(row=0, column=2, sticky="w")

        self.range_frame = ttk.Frame(side)
        ttk.Label(self.range_frame, text="最小 KB").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.range_frame, textvariable=self.min_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(self.range_frame, text="最大 KB").grid(row=1, column=0, sticky="w")
        ttk.Entry(self.range_frame, textvariable=self.max_var, width=8).grid(row=1, column=1, padx=2)

        self.ratio_frame = ttk.Frame(side)
        ttk.Label(self.ratio_frame, text="目标大小比例").grid(row=0, column=0, sticky="w")
        ttk.Entry(self.ratio_frame, textvariable=self.ratio_var, width=8).grid(row=0, column=1, padx=2)
        ttk.Label(self.ratio_frame, text="%").grid(row=0, column=2, sticky="w")

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Button(side, text="另存为…", command=self.export_image).pack(fill="x", pady=4)
        ttk.Label(
            side, text="JPEG 不支持透明背景，会以白色填充透明区域。",
            wraplength=285
        ).pack(anchor="w", pady=5)
        ttk.Label(
            side, text="程序默认保留原图像素尺寸；压缩目标不一定总能精确达到。",
            wraplength=285
        ).pack(anchor="w")

        ttk.Label(
            self.root, textvariable=self.status_var, anchor="w",
            relief="sunken", padding=6
        ).pack(fill="x", side="bottom")

    def _update_compression_fields(self):
        for frame in (self.exact_frame, self.range_frame, self.ratio_frame):
            frame.pack_forget()
        mode = self.compression_var.get()
        if mode == "exact":
            self.exact_frame.pack(anchor="w", fill="x", pady=4)
        elif mode == "range":
            self.range_frame.pack(anchor="w", fill="x", pady=4)
        elif mode == "ratio":
            self.ratio_frame.pack(anchor="w", fill="x", pady=4)

    def choose_source(self):
        path = filedialog.askopenfilename(
            title="选择待编辑图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.original = open_image(path)
        except Exception as exc:
            messagebox.showerror("读取失败", f"无法读取图片：\n{exc}")
            return
        self.source_path = path
        self.background_path = None
        self._use_color_background = False
        self.original_cutout = self.original_mask = self.current_mask = self.cutout = None
        self.result = self.original.copy()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.status_var.set(
            f"已载入：{os.path.basename(path)}，尺寸 {self.original.width} × {self.original.height}"
        )
        self._update_edit_buttons()
        self.fit_preview()

    def start_cutout(self):
        if self.original is None:
            messagebox.showinfo("提示", "请先选择待编辑图片。")
            return
        model_name = self.model_var.get().strip()
        if not model_name:
            messagebox.showinfo("提示", "请先选择 AI 抠图模型。")
            return
        self.status_var.set(f"正在使用模型 {model_name} 抠图，请稍候……")
        self.cutout_button.configure(state="disabled")
        threading.Thread(
            target=self._cutout_worker, args=(self.original.copy(), model_name), daemon=True
        ).start()

    def _cutout_worker(self, source_image, model_name):
        try:
            result = remove_background(source_image, model_name=model_name)
            self.root.after(0, lambda: self._cutout_done(result, None, model_name))
        except Exception as exc:
            error_text = str(exc)
            self.root.after(0, lambda err=error_text: self._cutout_done(None, err, model_name))

    def _cutout_done(self, result, error, model_name):
        self.cutout_button.configure(state="normal")
        if error:
            self.status_var.set("抠图失败。")
            messagebox.showerror(
                "抠图失败",
                f"{error}\n\n请确认 rembg 版本支持所选模型，并检查模型下载情况。",
            )
            return
        self.original_cutout = result.convert("RGBA").copy()
        self.original_mask = extract_alpha_mask(self.original_cutout)
        self.current_mask = self.original_mask.copy()
        self.cutout = apply_alpha_mask(self.original_cutout, self.current_mask)
        self.background_path = None
        self._use_color_background = False
        self.result = self.cutout.copy()
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.status_var.set(
            f"抠图完成（模型：{model_name}）。可在预览图上擦除背景或恢复人像。"
        )
        self._update_edit_buttons()
        self.fit_preview()

    def _render_current_result(self):
        if self.cutout is None:
            return
        self.cutout = apply_alpha_mask(self.original_cutout, self.current_mask)
        if self.background_path:
            self.result = composite_background(self.cutout, background_path=self.background_path)
        elif self._use_color_background:
            self.result = composite_background(self.cutout, background_color=self.bg_color)
        else:
            self.result = self.cutout.copy()

    def choose_background(self):
        if self.cutout is None:
            messagebox.showinfo("提示", "请先点击“自动抠出人像”。")
            return
        path = filedialog.askopenfilename(
            title="选择背景图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")],
        )
        if not path:
            return
        try:
            open_image(path)
            self.background_path = path
            self._use_color_background = False
            self._render_current_result()
            self.status_var.set(f"背景已替换：{os.path.basename(path)}")
            self._redraw_preview()
        except Exception as exc:
            messagebox.showerror("背景处理失败", str(exc))

    def choose_color(self):
        if self.cutout is None:
            messagebox.showinfo("提示", "请先点击“自动抠出人像”。")
            return
        color = colorchooser.askcolor(color=self.bg_color, title="选择背景颜色")
        if not color or not color[1]:
            return
        self.bg_color = color[1]
        self.background_path = None
        self._use_color_background = True
        self._render_current_result()
        self.status_var.set(f"背景颜色已设置为 {self.bg_color}")
        self._redraw_preview()

    def restore_cutout(self):
        if self.original_cutout is None:
            messagebox.showinfo("提示", "请先完成自动抠图。")
            return
        self._push_undo()
        self.current_mask = self.original_mask.copy()
        self.background_path = None
        self._use_color_background = False
        self._render_current_result()
        self.status_var.set("已恢复为 AI 初始透明人像。")
        self._update_edit_buttons()
        self._redraw_preview()

    def _on_brush_scale(self, value):
        size = max(2, int(float(value)))
        self.brush_size_var.set(size)
        self.brush_size_label.configure(text=f"{size} px")

    def _on_zoom_change(self, _value=None):
        self.zoom_label.configure(text=f"{int(self.zoom_var.get() * 100)}%")
        self._redraw_preview()

    def fit_preview(self):
        if self.result is None:
            return
        self.zoom_var.set(1.0)
        self._redraw_preview(fit=True)

    def _on_canvas_resize(self, _event=None):
        if self.result is not None:
            self._redraw_preview(fit=True)

    def _redraw_preview(self, fit=False):
        image = self.result
        if image is None:
            self.preview_canvas.delete("all")
            return
        canvas_w = max(100, self.preview_canvas.winfo_width())
        canvas_h = max(100, self.preview_canvas.winfo_height())
        if fit or self.fit_scale <= 0:
            self.fit_scale = min(
                (canvas_w - 30) / image.width,
                (canvas_h - 30) / image.height,
                1.0,
            )
            self.fit_scale = max(self.fit_scale, 0.01)
        self.display_scale = max(0.01, self.fit_scale * self.zoom_var.get())
        display_w = max(1, int(image.width * self.display_scale))
        display_h = max(1, int(image.height * self.display_scale))
        preview = image.copy().resize((display_w, display_h), Image.Resampling.LANCZOS)

        if preview.mode == "RGBA":
            checker = Image.new("RGB", preview.size, "#EEEEEE")
            draw = ImageDraw.Draw(checker)
            tile = max(8, int(16 * self.zoom_var.get()))
            for y in range(0, display_h, tile):
                for x in range(0, display_w, tile):
                    if ((x // tile) + (y // tile)) % 2:
                        draw.rectangle(
                            (x, y, min(x + tile - 1, display_w - 1),
                             min(y + tile - 1, display_h - 1)),
                            fill="#D5D5D5",
                        )
            checker.paste(preview, mask=preview.getchannel("A"))
            preview = checker

        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_canvas.delete("all")
        self.image_offset_x = self.image_offset_y = 12
        self.preview_canvas.create_image(
            self.image_offset_x, self.image_offset_y, image=self.preview_photo, anchor="nw"
        )
        self.preview_canvas.configure(scrollregion=(0, 0, display_w + 24, display_h + 24))

    def _canvas_to_image(self, event):
        x = self.preview_canvas.canvasx(event.x) - self.image_offset_x
        y = self.preview_canvas.canvasy(event.y) - self.image_offset_y
        if self.result is None or self.display_scale <= 0:
            return None
        ix, iy = int(x / self.display_scale), int(y / self.display_scale)
        if 0 <= ix < self.result.width and 0 <= iy < self.result.height:
            return ix, iy
        return None

    def _on_stroke_start(self, event):
        if self.current_mask is None or self.cutout is None:
            return
        point = self._canvas_to_image(event)
        if point is None:
            return
        self.drawing = True
        self.stroke_points = [point]
        self._push_undo()
        self._paint_stroke(self.stroke_points)

    def _on_stroke_move(self, event):
        if not self.drawing or self.current_mask is None:
            return
        point = self._canvas_to_image(event)
        if point is not None:
            self.stroke_points.append(point)
            self._paint_stroke(self.stroke_points)

    def _on_stroke_end(self, _event):
        if self.drawing:
            self.drawing = False
            self.stroke_points = []
            self._update_edit_buttons()

    def _paint_stroke(self, points):
        if not points or self.current_mask is None:
            return
        radius = max(1, int(self.brush_size_var.get() / 2))
        overlay = Image.new("L", self.current_mask.size, 0)
        draw = ImageDraw.Draw(overlay)
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
        else:
            draw.line(points, fill=255, width=radius * 2)
            for x, y in (points[0], points[-1]):
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(0.5, radius * 0.12)))
        if self.brush_mode_var.get() == "erase":
            self.current_mask = ImageChops.subtract(self.current_mask, overlay)
        else:
            # 使用原始 AI 蒙版恢复，可保留头发等边缘的半透明像素。
            self.current_mask = Image.composite(self.original_mask, self.current_mask, overlay)
        self._render_current_result()
        self._redraw_preview()

    def _push_undo(self):
        if self.current_mask is None:
            return
        self.undo_stack.append(self.current_mask.copy())
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._update_edit_buttons()

    def undo(self):
        if not self.undo_stack or self.current_mask is None:
            return
        self.redo_stack.append(self.current_mask.copy())
        self.current_mask = self.undo_stack.pop()
        self._render_current_result()
        self._redraw_preview()
        self._update_edit_buttons()
        self.status_var.set("已撤销上一步蒙版修改。")

    def redo(self):
        if not self.redo_stack or self.current_mask is None:
            return
        self.undo_stack.append(self.current_mask.copy())
        self.current_mask = self.redo_stack.pop()
        self._render_current_result()
        self._redraw_preview()
        self._update_edit_buttons()
        self.status_var.set("已重做蒙版修改。")

    def _update_edit_buttons(self):
        enabled = self.current_mask is not None
        self.undo_button.configure(
            state="normal" if enabled and self.undo_stack else "disabled"
        )
        self.redo_button.configure(
            state="normal" if enabled and self.redo_stack else "disabled"
        )

    def export_image(self):
        if self.result is None:
            messagebox.showinfo("提示", "请先选择并编辑图片。")
            return
        fmt = self.format_var.get().upper()
        extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff"}
        path = filedialog.asksaveasfilename(
            title="保存编辑结果", defaultextension=extensions[fmt],
            initialfile="edited_image" + extensions[fmt],
            filetypes=[(fmt + " 图片", "*" + extensions[fmt]), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            mode = self.compression_var.get()
            kwargs = {}
            if mode == "exact":
                kwargs["target_bytes"] = int(float(self.target_var.get()) * 1024)
            elif mode == "range":
                kwargs["min_bytes"] = int(float(self.min_var.get()) * 1024)
                kwargs["max_bytes"] = int(float(self.max_var.get()) * 1024)
            elif mode == "ratio":
                kwargs["ratio_percent"] = float(self.ratio_var.get())
            data, note = compress_image(self.result, fmt, mode, **kwargs)
            save_bytes(path, data)
            self.status_var.set(f"保存完成：{os.path.basename(path)}，{len(data) / 1024:.1f} KB")
            messagebox.showinfo(
                "保存完成",
                f"文件已保存：\n{path}\n\n实际大小：{len(data) / 1024:.1f} KB\n{note}",
            )
        except Exception as exc:
            messagebox.showerror("导出失败", f"导出时发生错误：\n{exc}")


def run_app():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    PortraitEditorApp(root)
    root.mainloop()
