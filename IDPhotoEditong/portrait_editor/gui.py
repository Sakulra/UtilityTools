"""Tkinter 图形界面。"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from PIL import Image, ImageTk

from .image_ops import (
    open_image, remove_background, composite_background,
    compress_image, save_bytes, SUPPORTED_FORMATS,
)


class PortraitEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("本地人像抠图与图片压缩工具")
        self.root.geometry("1000x720")
        self.root.minsize(820, 600)

        self.source_path = None
        self.background_path = None
        self.original = None
        self.cutout = None
        self.result = None
        self.preview_photo = None
        self.bg_color = "#FFFFFF"

        self.format_var = tk.StringVar(value="PNG")
        self.compression_var = tk.StringVar(value="none")
        self.target_var = tk.StringVar(value="500")
        self.min_var = tk.StringVar(value="200")
        self.max_var = tk.StringVar(value="800")
        self.ratio_var = tk.StringVar(value="50")
        self.status_var = tk.StringVar(value="请选择一张图片开始。")

        self._build_ui()
        self._update_compression_fields()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Button(top, text="选择待编辑图片", command=self.choose_source).pack(side="left", padx=4)
        ttk.Button(top, text="自动抠出人像", command=self.start_cutout).pack(side="left", padx=4)
        ttk.Button(top, text="选择背景图片", command=self.choose_background).pack(side="left", padx=4)
        ttk.Button(top, text="选择纯色背景", command=self.choose_color).pack(side="left", padx=4)
        ttk.Button(top, text="恢复透明人像", command=self.restore_cutout).pack(side="left", padx=4)

        body = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)
        self.preview = ttk.Label(body, text="图片预览区域", anchor="center", relief="sunken")
        self.preview.pack(side="left", fill="both", expand=True, padx=(0, 10))

        side = ttk.Frame(body, width=300)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        ttk.Label(side, text="导出设置", font=("", 12, "bold")).pack(anchor="w", pady=(2, 8))
        ttk.Label(side, text="输出格式").pack(anchor="w")
        ttk.Combobox(side, textvariable=self.format_var, values=list(SUPPORTED_FORMATS.keys()),
                     state="readonly").pack(fill="x", pady=(2, 10))

        ttk.Label(side, text="压缩方式").pack(anchor="w")
        self.compression_combo = ttk.Combobox(
            side, textvariable=self.compression_var, state="readonly",
            values=("none", "exact", "range", "ratio")
        )
        self.compression_combo.pack(fill="x", pady=(2, 4))
        self.compression_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_compression_fields())
        ttk.Label(side, text="方式说明：none=默认；exact=指定大小；range=大小范围；ratio=原图百分比",
                  wraplength=280).pack(anchor="w", pady=(0, 8))

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

        ttk.Separator(side).pack(fill="x", pady=12)
        ttk.Button(side, text="另存为…", command=self.export_image).pack(fill="x", pady=4)
        ttk.Label(side, text="提示：首次抠图会下载模型，耗时可能较长。").pack(anchor="w", pady=8)
        ttk.Label(side, text="输入图像可为任意尺寸；程序默认不改变像素尺寸。",
                  wraplength=280).pack(anchor="w")

        ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken", padding=6).pack(fill="x", side="bottom")

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
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                       ("所有文件", "*.*")]
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
        self.cutout = None
        self.result = self.original.copy()
        self.status_var.set(f"已载入：{os.path.basename(path)}，尺寸 {self.original.width} × {self.original.height}")
        self._show_image(self.result)

    def start_cutout(self):
        if self.original is None:
            messagebox.showinfo("提示", "请先选择待编辑图片。")
            return
        self.status_var.set("正在抠图，请稍候……")
        self._set_busy(True)
        threading.Thread(target=self._cutout_worker, daemon=True).start()

    def _cutout_worker(self):
        try:
            result = remove_background(self.original)
            self.root.after(0, lambda: self._cutout_done(result, None))
        except Exception as exc:
            self.root.after(0, lambda err=str(exc): self._cutout_done(None, err))

    def _cutout_done(self, result, error):
        self._set_busy(False)
        if error:
            self.status_var.set("抠图失败。")
            messagebox.showerror("抠图失败", f"{error}\n\n请确认已安装 requirements.txt 中的依赖，并检查首次模型下载是否成功。")
            return
        self.cutout = result
        self.result = result.copy()
        self.background_path = None
        self.status_var.set("抠图完成。可选择背景图片或纯色背景。")
        self._show_image(self.result)

    def choose_background(self):
        if self.cutout is None:
            messagebox.showinfo("提示", "请先点击“自动抠出人像”。")
            return
        path = filedialog.askopenfilename(
            title="选择背景图片",
            filetypes=[("图片文件", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        try:
            self.background_path = path
            self.result = composite_background(self.cutout, background_path=path)
            self.status_var.set(f"背景已替换：{os.path.basename(path)}")
            self._show_image(self.result)
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
        self.result = composite_background(self.cutout, background_color=self.bg_color)
        self.status_var.set(f"背景颜色已设置为 {self.bg_color}")
        self._show_image(self.result)

    def restore_cutout(self):
        if self.cutout is None:
            return
        self.background_path = None
        self.result = self.cutout.copy()
        self.status_var.set("已恢复为透明背景的人像。")
        self._show_image(self.result)

    def _show_image(self, image):
        if image is None:
            return
        self.root.update_idletasks()
        max_w = max(300, self.preview.winfo_width() - 20)
        max_h = max(300, self.preview.winfo_height() - 20)
        thumb = image.copy()
        thumb.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        # 在预览中以浅灰底显示透明区域
        if thumb.mode == "RGBA":
            checker = Image.new("RGB", thumb.size, "#E8E8E8")
            checker.paste(thumb, mask=thumb.getchannel("A"))
            thumb = checker
        self.preview_photo = ImageTk.PhotoImage(thumb)
        self.preview.configure(image=self.preview_photo, text="")

    def _set_busy(self, busy):
        state = "disabled" if busy else "normal"
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Button) and widget.cget("text") == "自动抠出人像":
                        widget.configure(state=state)

    def export_image(self):
        if self.result is None:
            messagebox.showinfo("提示", "请先选择并编辑图片。")
            return
        fmt = self.format_var.get().upper()
        extensions = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp", "TIFF": ".tiff"}
        path = filedialog.asksaveasfilename(
            title="保存编辑结果",
            defaultextension=extensions[fmt],
            initialfile="edited_image" + extensions[fmt],
            filetypes=[(fmt + " 图片", "*" + extensions[fmt]), ("所有文件", "*.*")]
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
            messagebox.showinfo("保存完成", f"文件已保存：\n{path}\n\n实际大小：{len(data) / 1024:.1f} KB\n{note}")
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
