"""
Professional GUI Application for Office Items Classification

Features:
- Dedicated screens for each mode
- Single image prediction
- Multiple images batch prediction
- Live camera feed with recording
- Video file processing
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import threading
import json
from datetime import datetime
from collections import Counter

from src.inference.predict import OfficeItemsPredictor


class ModernButton(tk.Button):
    """Custom styled button"""
    def __init__(self, parent, text="", command=None, bg='#2196F3', **kwargs):
        super().__init__(
            parent,
            text=text,
            command=command,
            font=('Segoe UI', 11, 'bold'),
            bg=bg,
            fg='white',
            activebackground='#1976D2',
            activeforeground='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10,
            **kwargs
        )
        self.default_bg = bg
        self.hover_bg = self._darken_color(bg)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    def _darken_color(self, hex_color):
        """Darken a hex color by 20%"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        darker = tuple(int(c * 0.8) for c in rgb)
        return f'#{darker[0]:02x}{darker[1]:02x}{darker[2]:02x}'

    def on_enter(self, e):
        self['background'] = self.hover_bg

    def on_leave(self, e):
        self['background'] = self.default_bg


class OfficeItemsClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Robotics - Office Items Classifier")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f5f5f5')

        # Variables
        self.predictor = None
        self.model_type = tk.StringVar(value='yolo12m')
        self.camera_active = False
        self.recording = False
        self.recording_start = None
        self.video_writer = None
        self.cap = None
        self.last_processed_frame = None
        self.current_mode = "home"  # home, single, multiple, camera, video

        # Prediction smoothing
        self.prediction_buffer = []

        # Recording folder
        self.recordings_folder = Path('results') / 'recorded_videos'
        self.recordings_folder.mkdir(parents=True, exist_ok=True)

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        # Title Bar
        title_frame = tk.Frame(self.root, bg='#1976D2', height=80)
        title_frame.pack(fill=tk.X)

        title_label = tk.Label(
            title_frame,
            text="🏢 Office Items Classifier",
            font=('Segoe UI', 24, 'bold'),
            bg='#1976D2',
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=30, pady=20)

        # Home button in title bar
        home_btn = tk.Button(
            title_frame,
            text="🏠 Home",
            command=self.show_home,
            font=('Segoe UI', 11, 'bold'),
            bg='#0D47A1',
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=20,
            pady=10
        )
        home_btn.pack(side=tk.RIGHT, padx=30, pady=15)

        # Main container
        self.main_container = tk.Frame(self.root, bg='#f5f5f5')
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Status Bar
        status_frame = tk.Frame(self.root, bg='#333', height=35)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready - Select a mode to begin",
            font=('Segoe UI', 10),
            bg='#333',
            fg='white',
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=15, pady=8)

        # Create all screens
        self.create_home_screen()
        self.create_single_image_screen()
        self.create_multiple_images_screen()
        self.create_camera_screen()
        self.create_video_screen()

        # Show home by default
        self.show_home()

    def create_home_screen(self):
        """Home screen with mode selection"""
        self.home_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        # Welcome section
        welcome_frame = tk.Frame(self.home_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        welcome_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=30)

        welcome_title = tk.Label(
            welcome_frame,
            text="👋 Welcome to Office Items Classifier",
            font=('Segoe UI', 28, 'bold'),
            bg='pink',
            fg='#1976D2'
        )
        welcome_title.pack(pady=(40, 20))

        welcome_subtitle = tk.Label(
            welcome_frame,
            text="Select a classification mode to get started",
            font=('Segoe UI', 14),
            bg='white',
            fg='#666'
        )
        welcome_subtitle.pack(pady=(0, 40))

        # Model Selection
        model_frame = tk.LabelFrame(
            welcome_frame,
            text="🤖 Current Model",
            font=('Segoe UI', 12, 'bold'),
            bg='white',
            fg='#1976D2',
            padx=30,
            pady=20
        )
        model_frame.pack(pady=20)

        models = [
            ('ResNet34', 'resnet34'),
            ('ResNet50', 'resnet50'),
            ('YOLO11m', 'yolo11m'),
            ('YOLO12m', 'yolo12m')
        ]

        model_buttons_frame = tk.Frame(model_frame, bg='white')
        model_buttons_frame.pack()

        for text, value in models:
            rb = tk.Radiobutton(
                model_buttons_frame,
                text=text,
                variable=self.model_type,
                value=value,
                font=('Segoe UI', 12),
                bg='white',
                activebackground='white',
                command=self.load_model,
                indicatoron=True
            )
            rb.pack(side=tk.LEFT, padx=15, pady=10)

        self.current_model_label = tk.Label(
            model_frame,
            text="✅ Loaded: YOLO12M",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#4CAF50'
        )
        self.current_model_label.pack(pady=(10, 0))

        # Mode buttons grid
        modes_frame = tk.Frame(welcome_frame, bg='white')
        modes_frame.pack(pady=30)

        # Row 1
        row1 = tk.Frame(modes_frame, bg='white')
        row1.pack(pady=10)

        btn_single = ModernButton(
            row1,
            text="📁 Single Image\n\nClassify one image",
            command=self.switch_to_single,
            bg='#4CAF50',
            width=20,
            height=4
        )
        btn_single.pack(side=tk.LEFT, padx=15)

        btn_multiple = ModernButton(
            row1,
            text="📂 Multiple Images\n\nBatch processing",
            command=self.switch_to_multiple,
            bg='#FF9800',
            width=20,
            height=4
        )
        btn_multiple.pack(side=tk.LEFT, padx=15)

        # Row 2
        row2 = tk.Frame(modes_frame, bg='white')
        row2.pack(pady=10)

        btn_camera = ModernButton(
            row2,
            text="📷 Live Camera\n\nReal-time detection",
            command=self.switch_to_camera,
            bg='#2196F3',
            width=20,
            height=4
        )
        btn_camera.pack(side=tk.LEFT, padx=15)

        btn_video = ModernButton(
            row2,
            text="🎥 Process Video\n\nVideo file analysis",
            command=self.switch_to_video,
            bg='#9C27B0',
            width=20,
            height=4
        )
        btn_video.pack(side=tk.LEFT, padx=15)

    def create_single_image_screen(self):
        """Single image classification screen"""
        self.single_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        content = tk.Frame(self.single_frame, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Left: Display
        left_panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        header = tk.Label(
            left_panel,
            text="📁 Single Image Classification",
            font=('Segoe UI', 16, 'bold'),
            bg='#f0f0f0',
            fg='#333',
            pady=15
        )
        header.pack(fill=tk.X)

        self.single_image_label = tk.Label(
            left_panel,
            bg='#000000',
            text="Click 'Select Image' to choose a file",
            font=('Segoe UI', 14),
            fg='#666'
        )
        self.single_image_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        btn_select = ModernButton(
            left_panel,
            text="📁 Select Image",
            command=self.predict_single_image,
            bg='#4CAF50'
        )
        btn_select.pack(pady=20)

        # Right: Results
        right_panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2, width=400)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 0))
        right_panel.pack_propagate(False)

        results_header = tk.Label(
            right_panel,
            text="📊 Prediction Results",
            font=('Segoe UI', 14, 'bold'),
            bg='#f0f0f0',
            fg='#333',
            pady=15
        )
        results_header.pack(fill=tk.X)

        self.single_results_text = tk.Text(
            right_panel,
            font=('Consolas', 10),
            bg='#f9f9f9',
            relief=tk.FLAT,
            padx=15,
            pady=15,
            wrap=tk.WORD
        )
        self.single_results_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        initial_text = "No image selected yet.\n\nClick 'Select Image' to begin."
        self.single_results_text.insert('1.0', initial_text)

    def create_multiple_images_screen(self):
        """Multiple images batch processing screen"""
        self.multiple_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        content = tk.Frame(self.multiple_frame, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Full width results
        panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2)
        panel.pack(fill=tk.BOTH, expand=True)

        header = tk.Label(
            panel,
            text="📂 Multiple Images - Batch Processing",
            font=('Segoe UI', 16, 'bold'),
            bg='#f0f0f0',
            fg='#333',
            pady=15
        )
        header.pack(fill=tk.X)

        scrollbar = tk.Scrollbar(panel)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.multiple_results_text = tk.Text(
            panel,
            font=('Consolas', 10),
            bg='#f9f9f9',
            relief=tk.FLAT,
            padx=20,
            pady=20,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set
        )
        self.multiple_results_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        scrollbar.config(command=self.multiple_results_text.yview)

        btn_select = ModernButton(
            panel,
            text="📂 Select Multiple Images",
            command=self.predict_multiple_images,
            bg='#FF9800'
        )
        btn_select.pack(pady=20)

        initial_text = "No images selected yet.\n\n"
        initial_text += "Click 'Select Multiple Images' to choose files for batch processing."
        self.multiple_results_text.insert('1.0', initial_text)

    def create_camera_screen(self):
        """Live camera screen with recording"""
        self.camera_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        content = tk.Frame(self.camera_frame, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Left: Camera feed
        left_panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        header = tk.Label(
            left_panel,
            text="📷 Live Camera Feed",
            font=('Segoe UI', 16, 'bold'),
            bg='#f0f0f0',
            fg='#333',
            pady=15
        )
        header.pack(fill=tk.X)

        self.camera_display_label = tk.Label(
            left_panel,
            bg='#000000',
            text="Click 'Start Camera' to begin",
            font=('Segoe UI', 14),
            fg='#666'
        )
        self.camera_display_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Camera controls
        controls_frame = tk.Frame(left_panel, bg='white')
        controls_frame.pack(pady=15)

        self.btn_camera_toggle = ModernButton(
            controls_frame,
            text="📷 Start Camera",
            command=self.toggle_camera,
            bg='#2196F3'
        )
        self.btn_camera_toggle.pack(side=tk.LEFT, padx=10)

        self.btn_camera_record = ModernButton(
            controls_frame,
            text="⏺ Start Recording",
            command=self.toggle_recording,
            bg='#F44336'
        )
        self.btn_camera_record.pack(side=tk.LEFT, padx=10)

        # Right: Recording info
        right_panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2, width=350)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 0))
        right_panel.pack_propagate(False)

        info_header = tk.Label(
            right_panel,
            text="📊 Camera Info",
            font=('Segoe UI', 14, 'bold'),
            bg='#f0f0f0',
            fg='#333',
            pady=15
        )
        info_header.pack(fill=tk.X)

        # Recording status
        self.recording_status_label = tk.Label(
            right_panel,
            text="⚪ Not Recording",
            font=('Segoe UI', 12, 'bold'),
            bg='white',
            fg='#666',
            pady=20
        )
        self.recording_status_label.pack()

        # Camera info text
        self.camera_info_text = tk.Text(
            right_panel,
            font=('Consolas', 10),
            bg='#f9f9f9',
            relief=tk.FLAT,
            padx=15,
            pady=15,
            wrap=tk.WORD,
            height=20
        )
        self.camera_info_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))

        btn_open_recordings = ModernButton(
            right_panel,
            text="📁 Open Recordings",
            command=self.open_recordings_folder,
            bg='#607D8B'
        )
        btn_open_recordings.pack(pady=(0, 20))

        initial_info = "Camera Status: Stopped\n\n"
        initial_info += "Click 'Start Camera' to begin\nlive predictions.\n\n"
        initial_info += "Recording:\n"
        initial_info += "• Start Recording: Save video\n"
        initial_info += "• Videos saved to model folder\n\n"
        initial_info += "Ready to start!"
        self.camera_info_text.insert('1.0', initial_info)

    def create_video_screen(self):
        """Video processing screen"""
        self.video_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        content = tk.Frame(self.video_frame, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2)
        panel.pack(fill=tk.BOTH, expand=True)

        header = tk.Label(
            panel,
            text="🎥 Video File Processing",
            font=('Segoe UI', 16, 'bold'),
            bg='#f0f0f0',
            fg='#333',
            pady=15
        )
        header.pack(fill=tk.X)

        info_text = tk.Label(
            panel,
            text="Process a video file and save it with predictions overlaid",
            font=('Segoe UI', 12),
            bg='white',
            fg='#666',
            pady=20
        )
        info_text.pack()

        self.video_results_text = tk.Text(
            panel,
            font=('Consolas', 10),
            bg='#f9f9f9',
            relief=tk.FLAT,
            padx=20,
            pady=20,
            wrap=tk.WORD,
            height=15
        )
        self.video_results_text.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)

        btn_process = ModernButton(
            panel,
            text="🎥 Select Video to Process",
            command=self.process_video,
            bg='#9C27B0'
        )
        btn_process.pack(pady=30)

        initial_text = "No video selected yet.\n\n"
        initial_text += "Steps:\n"
        initial_text += "1. Click 'Select Video to Process'\n"
        initial_text += "2. Choose input video file\n"
        initial_text += "3. Choose output location\n"
        initial_text += "4. Wait for processing (may take time)\n\n"
        initial_text += "Supported formats: MP4, AVI, MOV, MKV"
        self.video_results_text.insert('1.0', initial_text)

    def show_home(self):
        """Show home screen"""
        # Stop camera if active
        if self.camera_active:
            self.stop_camera()

        self.hide_all_screens()
        self.home_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "home"
        self.update_status("🏠 Home - Select a mode")

    def hide_all_screens(self):
        """Hide all screen frames"""
        self.home_frame.pack_forget()
        self.single_frame.pack_forget()
        self.multiple_frame.pack_forget()
        self.camera_frame.pack_forget()
        self.video_frame.pack_forget()

    def switch_to_single(self):
        """Switch to single image mode"""
        if self.camera_active:
            self.stop_camera()

        self.hide_all_screens()
        self.single_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "single"
        self.update_status("📁 Single Image Mode")

    def switch_to_multiple(self):
        """Switch to multiple images mode"""
        if self.camera_active:
            self.stop_camera()

        self.hide_all_screens()
        self.multiple_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "multiple"
        self.update_status("📂 Multiple Images Mode")

    def switch_to_camera(self):
        """Switch to camera mode"""
        self.hide_all_screens()
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "camera"
        self.update_status("📷 Camera Mode")

    def switch_to_video(self):
        """Switch to video processing mode"""
        if self.camera_active:
            self.stop_camera()

        self.hide_all_screens()
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "video"
        self.update_status("🎥 Video Processing Mode")

    def load_model(self):
        model_type = self.model_type.get()
        model_path = f'models/{model_type}_best.pt'

        if not Path(model_path).exists():
            self.current_model_label.config(
                text=f"❌ {model_type.upper()} not found",
                fg='#F44336'
            )
            messagebox.showerror(
                "Model Not Found",
                f"Model not found: {model_path}\n\nPlease train the model first."
            )
            return

        try:
            self.predictor = OfficeItemsPredictor(
                model_type=model_type,
                model_path=model_path
            )

            self.current_model_label.config(
                text=f"✅ Loaded: {model_type.upper()}",
                fg='#4CAF50'
            )

            self.update_status(f"✅ Model loaded: {model_type.upper()}")

        except Exception as e:
            self.current_model_label.config(
                text=f"❌ Error loading model",
                fg='#F44336'
            )
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def smooth_prediction(self, predicted_class, confidence, probs):
        """3-frame averaging"""
        self.prediction_buffer.append({
            'class': predicted_class,
            'confidence': confidence,
            'probs': probs
        })

        if len(self.prediction_buffer) > 3:
            self.prediction_buffer.pop(0)

        if len(self.prediction_buffer) < 2:
            return predicted_class, confidence, probs

        avg_probs = np.mean([p['probs'] for p in self.prediction_buffer], axis=0)
        avg_confidence = np.max(avg_probs)
        avg_class_idx = np.argmax(avg_probs)
        avg_class = self.predictor.class_names[avg_class_idx]

        return avg_class, avg_confidence, avg_probs

    def predict_single_image(self):
        if not self.predictor:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if not file_path:
            return

        try:
            self.update_status("⏳ Predicting...")

            predicted_class, confidence, probs = self.predictor.predict_image(file_path)

            # Display image
            img = Image.open(file_path)
            img.thumbnail((800, 600))
            photo = ImageTk.PhotoImage(img)
            self.single_image_label.config(image=photo, text="")
            self.single_image_label.image = photo

            # Display results
            result_text = f"📸 PREDICTION RESULTS\n"
            result_text += "=" * 40 + "\n\n"
            result_text += f"📁 File: {Path(file_path).name}\n"
            result_text += f"🤖 Model: {self.model_type.get().upper()}\n\n"
            result_text += f"🎯 Prediction:\n   {predicted_class.upper()}\n\n"
            result_text += f"📊 Confidence: {confidence*100:.2f}%\n"

            if confidence > 0.8:
                result_text += "   ✅ High Confidence\n\n"
            elif confidence > 0.6:
                result_text += "   ⚠️  Medium Confidence\n\n"
            else:
                result_text += "   ❌ Low Confidence\n\n"

            result_text += "Top 3 Predictions:\n"
            result_text += "-" * 40 + "\n"

            top_3 = np.argsort(probs)[-3:][::-1]
            for i, idx in enumerate(top_3, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                result_text += f"{emoji} {self.predictor.class_names[idx]:<12} {probs[idx]*100:.2f}%\n"

            self.single_results_text.delete('1.0', tk.END)
            self.single_results_text.insert('1.0', result_text)
            self.update_status("✅ Prediction complete")

        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.update_status("❌ Prediction failed")

    def predict_multiple_images(self):
        if not self.predictor:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        file_paths = filedialog.askopenfilenames(
            title="Select Multiple Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if not file_paths:
            return

        try:
            self.update_status(f"⏳ Processing {len(file_paths)} images...")

            results_text = f"📂 BATCH PREDICTION\n"
            results_text += "=" * 50 + "\n\n"
            results_text += f"🤖 Model: {self.model_type.get().upper()}\n"
            results_text += f"📊 Total: {len(file_paths)} images\n\n"

            for i, file_path in enumerate(file_paths, 1):
                predicted_class, confidence, _ = self.predictor.predict_image(file_path)

                emoji = "✅" if confidence > 0.7 else "⚠️"
                results_text += f"{emoji} {i}. {Path(file_path).name}\n"
                results_text += f"   → {predicted_class} ({confidence*100:.1f}%)\n\n"

                self.multiple_results_text.delete('1.0', tk.END)
                self.multiple_results_text.insert('1.0', results_text + f"\n⏳ Processing {i}/{len(file_paths)}...")
                self.root.update()

            results_text += f"\n✅ Complete!"
            self.multiple_results_text.delete('1.0', tk.END)
            self.multiple_results_text.insert('1.0', results_text)

            self.update_status(f"✅ Processed {len(file_paths)} images")
            messagebox.showinfo("Success", f"Processed {len(file_paths)} images!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")
            self.update_status("❌ Failed")

    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if not self.predictor:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        def init_camera():
            try:
                self.update_status("⏳ Starting camera...")

                self.cap = cv2.VideoCapture(0)

                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Cannot open camera")
                    self.update_status("❌ Camera failed")
                    return

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                self.camera_active = True
                self.btn_camera_toggle.config(text="⏹ Stop Camera", bg='#F44336')
                self.update_status("✅ Camera active")

                info_text = f"📷 CAMERA ACTIVE\n"
                info_text += "=" * 35 + "\n\n"
                info_text += f"🤖 Model: {self.model_type.get().upper()}\n"
                info_text += f"📺 Resolution: 640x480\n"
                info_text += f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}\n\n"
                info_text += "Live predictions running...\n\n"
                info_text += "Use recording button to\nsave video."

                self.camera_info_text.delete('1.0', tk.END)
                self.camera_info_text.insert('1.0', info_text)

                self.camera_thread = threading.Thread(target=self.update_camera, daemon=True)
                self.camera_thread.start()

            except Exception as e:
                messagebox.showerror("Error", f"Camera failed: {str(e)}")
                self.update_status("❌ Error")

        threading.Thread(target=init_camera, daemon=True).start()

    def stop_camera(self):
        self.camera_active = False
        self.prediction_buffer = []
        self.last_processed_frame = None

        if self.recording:
            self.stop_recording()

        if self.cap:
            self.cap.release()

        self.btn_camera_toggle.config(text="📷 Start Camera", bg='#2196F3')

        # Reset display to black with text
        self.camera_display_label.config(
            image='',
            text="Camera stopped\n\nClick 'Start Camera' to begin",
            fg='#666',
            bg='#000000'
        )

        self.update_status("⏹ Camera stopped")

        info_text = "Camera Status: Stopped\n\n"
        info_text += "Click 'Start Camera' to begin\nlive predictions.\n\n"
        info_text += "Recording:\n"
        info_text += "• Start Recording: Save video\n"
        info_text += "• Videos saved to model folder\n\n"
        info_text += "Ready to start!"

        self.camera_info_text.delete('1.0', tk.END)
        self.camera_info_text.insert('1.0', info_text)

    def update_camera(self):
        frame_count = 0

        while self.camera_active:
            ret, frame = self.cap.read()

            if not ret:
                break

            if frame_count % 2 == 0:
                display_frame = self.process_frame(frame)
                self.last_processed_frame = display_frame
            else:
                display_frame = self.last_processed_frame if self.last_processed_frame is not None else frame

            if self.recording and self.video_writer:
                self.video_writer.write(display_frame)

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((640, 480), Image.Resampling.NEAREST)
            photo = ImageTk.PhotoImage(image=img)

            self.camera_display_label.config(image=photo, text="")
            self.camera_display_label.image = photo

            self.root.update_idletasks()

            frame_count += 1

    def process_frame(self, frame):
        """Add predictions to frame"""
        temp_path = Path("temp_frame.jpg")
        cv2.imwrite(str(temp_path), frame)

        try:
            predicted_class, confidence, probs = self.predictor.predict_image(temp_path)
            predicted_class, confidence, probs = self.smooth_prediction(predicted_class, confidence, probs)

            height, width = frame.shape[:2]

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 270), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            if confidence > 0.8:
                color = (0, 255, 0)
                conf_text = "HIGH"
            elif confidence > 0.6:
                color = (0, 200, 255)
                conf_text = "MEDIUM"
            else:
                color = (0, 0, 255)
                conf_text = "LOW"

            cv2.putText(frame, predicted_class.upper(), (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            conf_display = f"Confidence: {confidence:.1%} ({conf_text})"
            cv2.putText(frame, conf_display, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            model_display = f"Model: {self.model_type.get().upper()}"
            cv2.putText(frame, model_display, (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, "Top 3 Predictions:", (20, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            top3_idx = np.argsort(probs)[-3:][::-1]
            y_offset = 175
            for i, idx in enumerate(top3_idx):
                cls = self.predictor.class_names[idx]
                prob = probs[idx]
                text_color = (0, 255, 0) if i == 0 else (255, 255, 255)
                text = f"{i+1}. {cls}: {prob:.1%}"
                cv2.putText(frame, text, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                y_offset += 30

            if self.recording:
                cv2.circle(frame, (width - 40, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (width - 100, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return frame

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if not self.camera_active:
            messagebox.showwarning("Warning", "Start camera first")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_type.get()

        model_recordings = self.recordings_folder / model_name
        model_recordings.mkdir(parents=True, exist_ok=True)

        output_path = model_recordings / f"recording_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            20.0,
            (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )

        self.recording = True
        self.recording_start = datetime.now()
        self.btn_camera_record.config(text="⏹ Stop Recording", bg='#4CAF50')
        self.recording_status_label.config(text="🔴 Recording...", fg='#F44336')
        self.update_status(f"🔴 Recording: {output_path.name}")

        info_text = f"🔴 RECORDING\n"
        info_text += "=" * 35 + "\n\n"
        info_text += f"📁 File:\n   {output_path.name}\n\n"
        info_text += f"📂 Location:\n   {model_recordings}\n\n"
        info_text += f"🕐 Started:\n   {self.recording_start.strftime('%H:%M:%S')}\n\n"
        info_text += "Recording in progress...\n\n"
        info_text += "Click 'Stop Recording'\nwhen done."

        self.camera_info_text.delete('1.0', tk.END)
        self.camera_info_text.insert('1.0', info_text)

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

        self.recording = False
        self.btn_camera_record.config(text="⏺ Start Recording", bg='#F44336')
        self.recording_status_label.config(text="⚪ Not Recording", fg='#666')

        duration = (datetime.now() - self.recording_start).total_seconds()
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        info_text = f"✅ RECORDING SAVED\n"
        info_text += "=" * 35 + "\n\n"
        info_text += f"⏱️  Duration:\n   {minutes}m {seconds}s\n\n"
        info_text += f"📂 Location:\n   {self.recordings_folder / self.model_type.get()}\n\n"
        info_text += "Recording saved!\n\n"
        info_text += "Click 'Open Recordings'\nto view videos."

        self.camera_info_text.delete('1.0', tk.END)
        self.camera_info_text.insert('1.0', info_text)

        self.update_status(f"✅ Recording saved ({minutes}m {seconds}s)")
        messagebox.showinfo("Saved", f"Duration: {minutes}m {seconds}s")

    def process_video(self):
        if not self.predictor:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        output_path = filedialog.asksaveasfilename(
            title="Save Processed Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4")]
        )

        if not output_path:
            return

        try:
            self.update_status("⏳ Processing video...")

            result_text = "⏳ PROCESSING VIDEO...\n"
            result_text += "=" * 45 + "\n\n"
            result_text += f"📹 Input: {Path(file_path).name}\n"
            result_text += f"💾 Output: {Path(output_path).name}\n\n"
            result_text += "This may take several minutes...\nPlease wait."

            self.video_results_text.delete('1.0', tk.END)
            self.video_results_text.insert('1.0', result_text)

            def process():
                self.predictor.predict_from_video(file_path, output_path, display=False)

                result_text = "✅ VIDEO PROCESSED\n"
                result_text += "=" * 45 + "\n\n"
                result_text += f"📹 Input: {Path(file_path).name}\n"
                result_text += f"💾 Output: {Path(output_path).name}\n"
                result_text += f"🤖 Model: {self.model_type.get().upper()}\n\n"
                result_text += "Processing complete!\nVideo saved successfully."

                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', result_text)
                self.update_status("✅ Video processed")

                messagebox.showinfo("Success", f"Video saved to:\n{output_path}")

            threading.Thread(target=process, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")
            self.update_status("❌ Failed")

    def open_recordings_folder(self):
        import os
        import platform

        folder = self.recordings_folder / self.model_type.get()
        folder.mkdir(parents=True, exist_ok=True)

        if platform.system() == 'Windows':
            os.startfile(folder)
        elif platform.system() == 'Darwin':
            os.system(f'open "{folder}"')
        else:
            os.system(f'xdg-open "{folder}"')

    def update_status(self, text):
        self.status_label.config(text=text)


def main():
    root = tk.Tk()
    app = OfficeItemsClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()