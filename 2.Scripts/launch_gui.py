import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from PIL import Image, ImageTk
from main import process_video


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
    """Extend Canvas to support rounded rectangle drawing."""
    points = [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]
    return self.create_polygon(points, **kwargs, smooth=True)


tk.Canvas.create_rounded_rectangle = _create_rounded_rectangle  # monkey‑patch


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


def launch_gui():
    # --------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------

    def browse_file(_evt=None):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv")]
        )
        if file_path:
            input_entry.delete(0, tk.END)
            input_entry.insert(0, file_path)
            filename = os.path.basename(file_path)
            file_display.config(text=f"📁 {filename}", fg="#1e293b")

    def start_upscaling():
        input_path = input_entry.get()
        resolution = resolution_var.get()
        codec = codec_var.get()
        core_usage = core_usage_var.get()
        model_type = model_var.get()

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("❌ Error", "Please select a valid video file.")
            return

        temp_dir = "temp_gui_output"
        use_all_cores = core_usage == "all"

        try:
            status_label.config(text="⏳ Processing… Please wait", fg="#2563eb")
            progress_bar.pack(pady=(10, 20))
            progress_bar.start()
            root.update()

            process_video(
                video_path=input_path,
                temp_dir=temp_dir,
                resolution=resolution,
                model_type=model_type,
                codec=codec,
                use_all_cores=use_all_cores,
            )

            progress_bar.stop()
            progress_bar.pack_forget()
            status_label.config(
                text="✅ Processing complete! Check the output folder.", fg="#16a34a"
            )

        except Exception as exc:
            progress_bar.stop()
            progress_bar.pack_forget()
            messagebox.showerror("❌ Error", f"Processing failed:\n{exc}")
            status_label.config(text="❌ Processing failed", fg="#dc2626")

    # --------------------------------------------------
    # ROOT WINDOW
    # --------------------------------------------------

    root = tk.Tk()
    root.title("Topaz Lite – CPU Video Upscaler")
    root.geometry("900x850")
    root.resizable(False, False)
    root.configure(bg="#f1f5f9")

    # Center window
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    x, y = (sw - 900) // 2, (sh - 850) // 2
    root.geometry(f"900x850+{x}+{y}")

    # --------------------------------------------------
    # SCROLLABLE FRAME
    # --------------------------------------------------

    canvas = tk.Canvas(root, bg="#f1f5f9", highlightthickness=0)
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
    scrollable = tk.Frame(canvas, bg="#f1f5f9")

    scrollable.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    window_id = canvas.create_window((0, 0), window=scrollable, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Make scroll wheel work anywhere
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Ensure canvas resizes with window
    def configure_canvas(event):
        canvas.itemconfig(window_id, width=event.width)

    canvas.bind('<Configure>', configure_canvas)

    # --------------------------------------------------
    # HERO SECTION – Fixed alignment and centering
    # --------------------------------------------------

    hero_width, hero_height = 820, 280

    # Container frame for proper centering
    hero_container = tk.Frame(scrollable, bg="#f1f5f9")
    hero_container.pack(fill="x", pady=(20, 40))

    hero_canvas = tk.Canvas(
        hero_container,
        width=hero_width,
        height=hero_height,
        bg="#ffffff",
        highlightthickness=0,
    )
    hero_canvas.pack(anchor="center")  # Center the canvas

    # Background elements (dotted curved path)
    hero_canvas.create_line(
        -50,
        hero_height // 1.3,
        hero_width + 50,
        -50,
        dash=(6, 8),
        width=2,
        fill="#cbd5e1",
    )

    # Big blue quarter circle bottom‑left
    hero_canvas.create_arc(
        -220,
        hero_height - 140,
        200,
        hero_height + 280,
        start=90,
        extent=90,
        style="pieslice",
        outline="",
        fill="#0037FF",
    )

    # Right yellow square behind phone placeholder
    hero_canvas.create_rectangle(
        hero_width - 190,
        50,
        hero_width - 60,
        180,
        fill="#FFE000",
        outline="",
    )

    # Load custom image (phone mock‑up) – convert to PNG if necessary
    try:
        CUSTOM_IMG_PATH = os.path.join(
            os.path.dirname(__file__),
            "Dazzling Examples of Mobile App UI Design to Inspire You in 2021 heroimg_2088_1252.webp",
        )
        img = Image.open(CUSTOM_IMG_PATH).resize((140, 260))
        phone_photo = ImageTk.PhotoImage(img)
        hero_canvas.create_image(hero_width - 130, 20, image=phone_photo, anchor="nw")
    except Exception:
        # if missing, draw placeholder rounded rectangle
        hero_canvas.create_rounded_rectangle(
            hero_width - 150,
            20,
            hero_width - 20,
            260,
            radius=25,
            outline="#e2e8f0",
            width=1,
            fill="#f8fafc",
        )

    # Blue decorative bars over phone
    for idx, y_offset in enumerate([40, 70, 100]):
        hero_canvas.create_rectangle(
            hero_width - 120,
            y_offset + 20,
            hero_width - 40,
            y_offset + 30,
            fill="#0037FF",
            outline="",
        )

    # Blue square on phone area to mimic mock‑up
    hero_canvas.create_rounded_rectangle(
        hero_width - 100,
        140,
        hero_width - 30,
        220,
        radius=15,
        fill="#0037FF",
        outline="",
    )

    # Clickable white circle for video selection
    circle_x, circle_y, circle_r = 300, hero_height - 80, 40

    circle = hero_canvas.create_oval(
        circle_x - circle_r,
        circle_y - circle_r,
        circle_x + circle_r,
        circle_y + circle_r,
        fill="#ffffff",
        outline="#e2e8f0",
        width=2,
    )

    plus = hero_canvas.create_text(
        circle_x, circle_y, text="+", font=("Segoe UI", 24, "bold"), fill="#1e293b"
    )

    hero_canvas.tag_bind(circle, "<Button-1>", browse_file)
    hero_canvas.tag_bind(plus, "<Button-1>", browse_file)

    # Checklist text above circle
    hero_canvas.create_text(
        circle_x,
        circle_y - 55,
        text="• No text in video\n• < 60 s & ≤ 1080p",
        font=("Segoe UI", 12),
        fill="#1e293b",
        anchor="s",
        justify="center",
    )

    # Right‑side equal rectangles inside hero big div (three small cards)
    right_start_x = hero_width - 350
    card_w, card_h, gap = 90, 40, 12

    for i in range(3):
        top = 25 + i * (card_h + gap)
        hero_canvas.create_rounded_rectangle(
            right_start_x,
            top,
            right_start_x + card_w,
            top + card_h,
            radius=10,
            fill="#ffffff",
            outline="#e2e8f0",
            width=1,
        )

    # Ideamotive‑like branding (use app name instead)
    hero_canvas.create_text(
        40,
        40,
        text="Topaz Lite",
        font=("Segoe UI", 24, "bold"),
        fill="#1e293b",
        anchor="nw",
    )

    # --------------------------------------------------
    # MAIN CONTENT – File selection & configuration
    # --------------------------------------------------

    main_frame = tk.Frame(scrollable, bg="#f1f5f9")
    main_frame.pack(fill="both", expand=True, padx=40)  # Added padding for centering

    # ------------ Input Card ------------ #

    input_canvas = tk.Canvas(main_frame, height=160, bg="#f1f5f9", highlightthickness=0)
    input_canvas.pack(fill="x", pady=(0, 25))

    input_canvas.create_rounded_rectangle(
        0, 0, 820, 160, radius=25, fill="#ffffff", outline="#e2e8f0", width=1
    )

    input_inner = tk.Frame(input_canvas, bg="#ffffff")
    input_canvas.create_window(410, 80, window=input_inner, width=800, height=140)

    input_title = tk.Label(
        input_inner,
        text="📁 Select video file",
        font=("Segoe UI", 16, "bold"),
        bg="#ffffff",
        fg="#1e293b",
    )
    input_title.pack(anchor="w", pady=(10, 15), padx=20)

    # File entry + browse button
    file_frame = tk.Frame(input_inner, bg="#ffffff")
    file_frame.pack(fill="x", padx=20, pady=(0, 8))

    input_entry = tk.Entry(
        file_frame,
        font=("Segoe UI", 12),
        relief="flat",
        bd=0,
        bg="#f8fafc",
        fg="#374151",
        highlightthickness=1,
        highlightcolor="#e2e8f0",
        highlightbackground="#e2e8f0",
    )
    input_entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 14))

    browse_btn = tk.Button(
        file_frame,
        text="Browse…",
        font=("Segoe UI", 12, "bold"),
        bg="#3b82f6",
        fg="#ffffff",
        activebackground="#2563eb",
        relief="flat",
        bd=0,
        padx=22,
        pady=8,
        cursor="hand2",
        command=browse_file,
    )
    browse_btn.pack(side="right")

    file_display = tk.Label(
        input_inner,
        text="No file selected",
        font=("Segoe UI", 10),
        bg="#ffffff",
        fg="#6b7280",
    )
    file_display.pack(anchor="w", padx=20)

    # ------------ Configuration Header ------------ #

    cfg_header_canvas = tk.Canvas(
        main_frame, height=110, bg="#f1f5f9", highlightthickness=0
    )
    cfg_header_canvas.pack(fill="x")

    cfg_header_canvas.create_rounded_rectangle(
        0, 0, 820, 110, radius=25, fill="#f59e0b", outline=""
    )

    cfg_inner = tk.Frame(cfg_header_canvas, bg="#f59e0b")
    cfg_header_canvas.create_window(410, 55, window=cfg_inner)

    cfg_title = tk.Label(
        cfg_inner,
        text="⚙️ Configuration settings",
        font=("Segoe UI", 22, "bold"),
        bg="#f59e0b",
        fg="#ffffff",
    )
    cfg_title.pack(pady=(4, 4))

    cfg_sub = tk.Label(
        cfg_inner,
        text="Customise your video processing parameters",
        font=("Segoe UI", 12),
        bg="#f59e0b",
        fg="#fef3c7",
    )
    cfg_sub.pack()

    # ------------ Dropdowns ------------ #

    dropdown_frame = tk.Frame(main_frame, bg="#f1f5f9")
    dropdown_frame.pack(fill="x", pady=(12, 0))

    # Variables
    resolution_var, codec_var, core_usage_var, model_var = (
        tk.StringVar(value="1080p"),
        tk.StringVar(value="h264"),
        tk.StringVar(value="all_but_one"),
        tk.StringVar(value="quality"),
    )

    style = ttk.Style()
    style.theme_use("clam")
    style.configure(
        "Custom.TCombobox",
        foreground="#1e293b",
        fieldbackground="#ffffff",
        background="#ffffff",
        bordercolor="#e2e8f0",
        arrowcolor="#64748b",
        padding=(14, 8),
        relief="flat",
    )

    def create_dropdown(frame, label_text, var, choices):
        card_canvas = tk.Canvas(
            frame, height=100, bg="#f1f5f9", highlightthickness=0, width=390
        )
        card_canvas.pack(side="left", expand=True, padx=10, pady=6)

        card_canvas.create_rounded_rectangle(
            0, 0, 390, 100, radius=20, fill="#ffffff", outline="#e2e8f0", width=1
        )

        inner = tk.Frame(card_canvas, bg="#ffffff")
        card_canvas.create_window(195, 50, window=inner, width=370, height=90)

        lbl = tk.Label(
            inner,
            text=label_text,
            font=("Segoe UI", 13, "bold"),
            bg="#ffffff",
            fg="#1e293b",
        )
        lbl.pack(anchor="w", pady=(8, 6), padx=12)

        cb = ttk.Combobox(
            inner,
            textvariable=var,
            values=choices,
            state="readonly",
            font=("Segoe UI", 12, "bold"),
            style="Custom.TCombobox",
            width=28,
        )
        cb.pack(anchor="w", padx=12)

    # Left column
    create_dropdown(
        dropdown_frame,
        "Output resolution",
        resolution_var,
        ["720p", "1080p"],
    )

    create_dropdown(
        dropdown_frame, "Video codec", codec_var, ["h264", "h265", "prores"]
    )

    # Right column
    dropdown_frame2 = tk.Frame(main_frame, bg="#f1f5f9")
    dropdown_frame2.pack(fill="x")

    create_dropdown(
        dropdown_frame2,
        "CPU core usage",
        core_usage_var,
        ["all", "all_but_one"],
    )

    create_dropdown(
        dropdown_frame2, "Processing model", model_var, ["quality", "fast"]
    )

    # ------------ Action Card ------------ #

    action_canvas = tk.Canvas(
        main_frame, height=190, bg="#f1f5f9", highlightthickness=0
    )
    action_canvas.pack(fill="x", pady=(25, 20))

    # Create the white rounded rectangle background for the action card
    action_canvas.create_rounded_rectangle(
        0, 0, 820, 190, radius=25, fill="#ffffff", outline="#e2e8f0", width=1
    )

    action_inner = tk.Frame(action_canvas, bg="#ffffff")
    action_canvas.create_window(410, 95, window=action_inner, width=800, height=170)

    # Start button - FIXED VERSION with proper styling
    start_btn = tk.Button(
        action_inner,
        text="🚀 Start video upscaling",
        font=("Segoe UI", 14, "bold"),
        bg="#16a34a",
        fg="#ffffff",
        activebackground="#15803d",
        activeforeground="#ffffff",
        relief="flat",
        bd=0,
        highlightthickness=0,  # Remove highlight border
        cursor="hand2",
        padx=40,
        pady=12,
        command=start_upscaling,
    )
    start_btn.pack(pady=(20, 10))

    # Progress bar (hidden initially)
    style.configure(
        "Custom.Horizontal.TProgressbar",
        thickness=10,
        troughcolor="#f1f5f9",
        bordercolor="#e2e8f0",
        background="#3b82f6",
        lightcolor="#3b82f6",
        darkcolor="#3b82f6",
    )

    progress_bar = ttk.Progressbar(
        action_inner,
        mode="indeterminate",
        length=420,
        style="Custom.Horizontal.TProgressbar",
    )

    # Status label
    status_label = tk.Label(
        action_inner,
        text="🎬 Ready to process your video",
        font=("Segoe UI", 12),
        bg="#ffffff",
        fg="#6b7280",
    )
    status_label.pack(pady=(15, 0))

    # --------------------------------------------------

    root.mainloop()


if __name__ == "__main__":
    launch_gui()