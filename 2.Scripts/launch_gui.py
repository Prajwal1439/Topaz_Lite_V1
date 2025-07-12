import tkinter as tk
from tkinter import filedialog, messagebox
import os
from main import process_video

def launch_gui():
    def browse_file():
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi")])
        if file_path:
            input_entry.delete(0, tk.END)
            input_entry.insert(0, file_path)

    def start_upscaling():
        input_path = input_entry.get()
        resolution = resolution_var.get()
        codec = codec_var.get()
        core_usage = core_usage_var.get()  # "all" or "all_but_one"

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid video file.")
            return

        temp_dir = "temp_gui_output"
        use_all_cores = (core_usage == "all")

        try:
            status_label.config(text="⏳ Processing... Please wait.")
            root.update()
            process_video(
                video_path=input_path,
                temp_dir=temp_dir,
                resolution=resolution,
                codec=codec,
                use_all_cores=use_all_cores
            )
            status_label.config(text="✅ Done! Check the output folder.")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
            status_label.config(text="❌ Failed.")

    root = tk.Tk()
    root.title("Topaz Lite - CPU Video Upscaler")
    root.geometry("500x350")  # Increased height for new option
    root.resizable(False, False)

    tk.Label(root, text="🎞️ Input Video:").pack(pady=5)
    input_entry = tk.Entry(root, width=50)
    input_entry.pack()
    tk.Button(root, text="Browse", command=browse_file).pack(pady=5)

    tk.Label(root, text="🖼 Output Resolution:").pack()
    resolution_var = tk.StringVar(value="720p")
    tk.OptionMenu(root, resolution_var, "720p", "1080p").pack()

    tk.Label(root, text="🎥 Output Codec:").pack()
    codec_var = tk.StringVar(value="h264")
    tk.OptionMenu(root, codec_var, "h264", "h265", "prores").pack()

    tk.Label(root, text="⚙️ CPU Core Usage:").pack()
    core_usage_var = tk.StringVar(value="all_but_one")
    tk.OptionMenu(root, core_usage_var, "all", "all_but_one").pack()

    tk.Button(root, text="🚀 Start Upscaling", command=start_upscaling, bg="#4CAF50", fg="white").pack(pady=10)

    status_label = tk.Label(root, text="", fg="blue")
    status_label.pack(pady=10)

    root.mainloop()

# Run if launched directly
if __name__ == "__main__":
    launch_gui()