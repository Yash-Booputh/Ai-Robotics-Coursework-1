"""
GUI Application for Office Items Classification

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
    """Custom styled button with hover effects"""
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


class ScrollableFrame(tk.Frame):
    """A scrollable frame container"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self, bg=kwargs.get('bg', '#f5f5f5'), highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=kwargs.get('bg', '#f5f5f5'))

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self._update_scrollregion()
        )

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        # Bind canvas resize to update frame width
        self.canvas.bind('<Configure>', self._on_canvas_configure)

        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse wheel scrolling when hovering over the frame
        self.bind_mousewheel()

    def _update_scrollregion(self):
        """Update the scroll region to encompass all content"""
        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Update the width of the frame to match canvas"""
        self.canvas.itemconfig(self.canvas_frame, width=event.width)

    def bind_mousewheel(self):
        """Bind mousewheel events to this frame and its children"""
        def _on_mousewheel(event):
            if self.canvas.winfo_height() < self.scrollable_frame.winfo_height():
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        def _bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")

        self.canvas.bind('<Enter>', _bind_to_mousewheel)
        self.canvas.bind('<Leave>', _unbind_from_mousewheel)


class OfficeItemsClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IRIS Item Classifier")

        # Set minimum window size
        self.root.minsize(1000, 700)

        # Start maximized
        self.root.state('zoomed')  # Windows
        # For cross-platform compatibility
        try:
            self.root.attributes('-zoomed', True)  # Linux
        except:
            pass

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
        self.current_mode = "home"

        # Prediction smoothing
        self.prediction_buffer = []

        # Recording folder
        self.recordings_folder = Path('results') / 'recorded_videos'
        self.recordings_folder.mkdir(parents=True, exist_ok=True)

        # Camera capture variables
        self.capture_camera_active = False
        self.capture_cap = None
        self.captured_frame = None
        self.captures_folder = Path('results') / 'captured_images'
        self.captures_folder.mkdir(parents=True, exist_ok=True)

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        # Title Bar (Fixed height, responsive width)
        title_frame = tk.Frame(self.root, bg='#1976D2')
        title_frame.pack(fill=tk.X)

        title_container = tk.Frame(title_frame, bg='#1976D2')
        title_container.pack(side=tk.LEFT, padx=20, pady=15)

        title_label = tk.Label(
            title_container,
            text="IRIS Item Classifier",
            font=('Segoe UI', 20, 'bold'),
            bg='#1976D2',
            fg='white'
        )
        title_label.pack(anchor=tk.W)

        # Mode subtitle label
        self.mode_subtitle_label = tk.Label(
            title_container,
            text="Home",
            font=('Segoe UI', 10),
            bg='#1976D2',
            fg='#B3E5FC'
        )
        self.mode_subtitle_label.pack(anchor=tk.W)

        # Home button in title bar
        home_btn = tk.Button(
            title_frame,
            text="HOME",
            command=self.show_home,
            font=('Segoe UI', 10, 'bold'),
            bg='#0D47A1',
            fg='white',
            relief=tk.FLAT,
            cursor='hand2',
            padx=15,
            pady=8
        )
        home_btn.pack(side=tk.RIGHT, padx=20, pady=10)

        # Main container (Expands to fill space)
        self.main_container = tk.Frame(self.root, bg='#f5f5f5')
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Status Bar (Fixed height, responsive width)
        status_frame = tk.Frame(self.root, bg='#333')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = tk.Label(
            status_frame,
            text="[READY] Select a mode to begin",
            font=('Segoe UI', 9),
            bg='#333',
            fg='white',
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, padx=15, pady=6)

        # Create all screens
        self.create_home_screen()
        self.create_images_screen()
        self.create_camera_screen()
        self.create_capture_screen()  # New camera capture screen
        self.create_video_screen()

        # Show home by default
        self.mode_subtitle_label.config(text="Home")
        self.home_frame.pack(fill=tk.BOTH, expand=True)

    def create_home_screen(self):
        """Home screen with scrollable content"""
        # Use scrollable frame for home
        self.home_frame = ScrollableFrame(self.main_container, bg='#f5f5f5')

        # Content container
        content = tk.Frame(self.home_frame.scrollable_frame, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

        # Welcome section
        welcome_frame = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2)
        welcome_frame.pack(fill=tk.X, pady=(0, 15))

        welcome_title = tk.Label(
            welcome_frame,
            text="Welcome to IRIS Office Items Classification System ",
            font=('Segoe UI', 22, 'bold'),
            bg='white',
            fg='#1976D2'
        )
        welcome_title.pack(pady=(20, 8))

        welcome_subtitle = tk.Label(
            welcome_frame,
            text="Select a classification mode to get started",
            font=('Segoe UI', 11),
            bg='white',
            fg='#666'
        )
        welcome_subtitle.pack(pady=(0, 20))

        # Model Selection
        model_frame = tk.LabelFrame(
            content,
            text="Current Model",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#1976D2',
            padx=20,
            pady=12
        )
        model_frame.pack(fill=tk.X, pady=(0, 15))

        models = [
            ('ResNet34', 'resnet34'),
            ('ResNet50', 'resnet50'),
            ('YOLO11m', 'yolo11m'),
            ('YOLO12m', 'yolo12m')
        ]

        model_buttons_frame = tk.Frame(model_frame, bg='white')
        model_buttons_frame.pack(pady=8)

        for text, value in models:
            rb = tk.Radiobutton(
                model_buttons_frame,
                text=text,
                variable=self.model_type,
                value=value,
                font=('Segoe UI', 10),
                bg='white',
                activebackground='white',
                command=self.load_model,
                indicatoron=True
            )
            rb.pack(side=tk.LEFT, padx=10, pady=6)

        self.current_model_label = tk.Label(
            model_frame,
            text="[LOADED] YOLO12M",
            font=('Segoe UI', 10, 'bold'),
            bg='white',
            fg='#4CAF50'
        )
        self.current_model_label.pack(pady=(6, 0))

        # Mode buttons frame
        modes_container = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2)
        modes_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        modes_title = tk.Label(
            modes_container,
            text="Select Classification Mode",
            font=('Segoe UI', 13, 'bold'),
            bg='white',
            fg='#333'
        )
        modes_title.pack(pady=(15, 12))

        # Buttons container that responds to width
        buttons_container = tk.Frame(modes_container, bg='white')
        buttons_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Configure grid to be responsive (4 columns, 1 row)
        buttons_container.grid_columnconfigure(0, weight=1, minsize=200)
        buttons_container.grid_columnconfigure(1, weight=1, minsize=200)
        buttons_container.grid_columnconfigure(2, weight=1, minsize=200)
        buttons_container.grid_columnconfigure(3, weight=1, minsize=200)
        buttons_container.grid_rowconfigure(0, weight=1)

        # All buttons in one row with descriptions
        btn_images = ModernButton(
            buttons_container,
            text="CLASSIFY IMAGES\n\nUpload single or\nmultiple images",
            command=self.switch_to_images,
            bg='#4CAF50'
        )
        btn_images.grid(row=0, column=0, padx=5, pady=8, sticky='nsew', ipady=10)

        btn_capture = ModernButton(
            buttons_container,
            text="CAMERA CAPTURE\n\nCapture & classify\nsingle images",
            command=self.switch_to_capture,
            bg='#00BCD4'
        )
        btn_capture.grid(row=0, column=1, padx=5, pady=8, sticky='nsew', ipady=10)

        btn_camera = ModernButton(
            buttons_container,
            text="LIVE CAMERA\n\nReal-time\nclassification",
            command=self.switch_to_camera,
            bg='#FF9800'
        )
        btn_camera.grid(row=0, column=2, padx=5, pady=8, sticky='nsew', ipady=10)

        btn_video = ModernButton(
            buttons_container,
            text="PROCESS VIDEO\n\nClassify\nvideo files",
            command=self.switch_to_video,
            bg='#9C27B0'
        )
        btn_video.grid(row=0, column=3, padx=5, pady=8, sticky='nsew', ipady=10)

    def create_images_screen(self):
        """2-column layout with prediction info inside image preview on the right"""
        self.images_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        # Main container
        content = tk.Frame(self.images_frame, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT panel - Upload Controls (FIXED WIDTH - 320px)
        left_panel = tk.Frame(content, bg='white', width=320, relief=tk.RAISED, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)

        # Header
        header = tk.Label(
            left_panel,
            text="Upload Images",
            font=('Segoe UI', 14, 'bold'),
            bg='white',
            fg='#4CAF50'
        )
        header.pack(pady=10)

        # Upload buttons
        upload_frame = tk.Frame(left_panel, bg='white')
        upload_frame.pack(pady=6, padx=10, fill=tk.X)

        self.btn_select_images = ModernButton(
            upload_frame,
            text="SELECT IMAGES",
            command=self.select_images,
            bg='#2196F3'
        )
        self.btn_select_images.pack(pady=3, fill=tk.X)

        help_text = tk.Label(
            upload_frame,
            text="Select up to 10 images for classification",
            font=('Segoe UI', 8),
            bg='white',
            fg='#999'
        )
        help_text.pack(pady=3)

        self.btn_classify_images = ModernButton(
            upload_frame,
            text="CLASSIFY IMAGES",
            command=self.classify_images,
            bg='#4CAF50'
        )
        self.btn_classify_images.pack(pady=3, fill=tk.X)
        self.btn_classify_images.config(state=tk.DISABLED)

        self.btn_clear_images = ModernButton(
            upload_frame,
            text="CLEAR ALL",
            command=self.clear_images,
            bg='#F44336'
        )
        self.btn_clear_images.pack(pady=3, fill=tk.X)

        # File list
        list_label = tk.Label(
            left_panel,
            text="Selected Files (0)",
            font=('Segoe UI', 10, 'bold'),
            bg='white',
            fg='#333'
        )
        list_label.pack(pady=(10, 5))
        self.file_count_label = list_label

        list_frame = tk.Frame(left_panel, bg='white')
        list_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        list_scroll = tk.Scrollbar(list_frame)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.images_listbox = tk.Listbox(
            list_frame,
            font=('Segoe UI', 9),
            selectmode=tk.SINGLE,
            yscrollcommand=list_scroll.set,
            bg='#fafafa'
        )
        self.images_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.config(command=self.images_listbox.yview)
        self.images_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        # RIGHT panel - Image Preview with embedded prediction info (EXPANDS)
        right_panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=1)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Image display header
        display_header = tk.Frame(right_panel, bg='white')
        display_header.pack(pady=10)

        display_label = tk.Label(
            display_header,
            text="Image Preview",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg='#333'
        )
        display_label.pack()

        # Image info label (filename + counter)
        self.image_info_label = tk.Label(
            display_header,
            text="No image loaded",
            font=('Segoe UI', 9),
            bg='white',
            fg='#666'
        )
        self.image_info_label.pack(pady=(5, 0))

        # Navigation buttons
        nav_frame = tk.Frame(right_panel, bg='white')
        nav_frame.pack(pady=8)

        self.btn_prev_image = ModernButton(
            nav_frame,
            text="< PREVIOUS",
            command=self.prev_image,
            bg='#2196F3'
        )
        self.btn_prev_image.pack(side=tk.LEFT, padx=5)
        self.btn_prev_image.config(state=tk.DISABLED)

        self.btn_next_image = ModernButton(
            nav_frame,
            text="NEXT >",
            command=self.next_image,
            bg='#2196F3'
        )
        self.btn_next_image.pack(side=tk.LEFT, padx=5)
        self.btn_next_image.config(state=tk.DISABLED)

        # Main preview container with image on left and prediction info on right
        preview_container = tk.Frame(right_panel, bg='#e0e0e0', relief=tk.SUNKEN, borderwidth=2)
        preview_container.pack(padx=15, pady=(8, 15), fill=tk.BOTH, expand=True)

        # LEFT side - Image display (takes 70% width)
        img_container = tk.Frame(preview_container, bg='#f5f5f5')
        img_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 1), pady=2)

        self.images_display_label = tk.Label(
            img_container,
            text="No image selected\n\nClick 'Select Images' to begin",
            font=('Segoe UI', 12),
            bg='#f5f5f5',
            fg='#999'
        )
        self.images_display_label.pack(fill=tk.BOTH, expand=True)

        # RIGHT side - Prediction info panel (FIXED WIDTH - 280px)
        prediction_panel = tk.Frame(preview_container, bg='#f5f5f5', width=280)
        prediction_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(1, 2), pady=2)
        prediction_panel.pack_propagate(False)

        # Prediction header
        pred_header = tk.Label(
            prediction_panel,
            text="PREDICTION INFO",
            font=('Segoe UI', 10, 'bold'),
            bg='#4CAF50',  # ← Green to match classify button
            fg='white',
            pady=8
        )
        pred_header.pack(fill=tk.X)

        # Prediction content
        pred_content = tk.Frame(prediction_panel, bg='#f5f5f5')
        pred_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.prediction_info_text = tk.Text(
            pred_content,
            font=('Segoe UI', 9),
            bg='#f5f5f5',
            fg='#333',
            wrap=tk.WORD,
            relief=tk.FLAT,
            state=tk.DISABLED,
            borderwidth=0,
            highlightthickness=0
        )
        self.prediction_info_text.pack(fill=tk.BOTH, expand=True)

        # Configure text tags for colored text
        self.prediction_info_text.tag_configure("label", foreground='#666', font=('Segoe UI', 9, 'bold'))
        self.prediction_info_text.tag_configure("class", foreground='#4CAF50', font=('Segoe UI', 11, 'bold'))
        self.prediction_info_text.tag_configure("conf_high", foreground='#4CAF50', font=('Segoe UI', 9, 'bold'))
        self.prediction_info_text.tag_configure("conf_medium", foreground='#FF9800', font=('Segoe UI', 9, 'bold'))
        self.prediction_info_text.tag_configure("conf_low", foreground='#F44336', font=('Segoe UI', 9, 'bold'))
        self.prediction_info_text.tag_configure("top3", foreground='#666', font=('Segoe UI', 9))
        self.prediction_info_text.tag_configure("top1", foreground='#4CAF50', font=('Segoe UI', 9, 'bold'))

        # Set initial text
        self.update_prediction_info_box(None)

        # Storage
        self.selected_image_paths = []
        self.image_predictions = {}
        self.current_image_index = 0

    def update_prediction_info_box(self, image_path):
        """Update the prediction info box with current image's prediction"""
        self.prediction_info_text.config(state=tk.NORMAL)
        self.prediction_info_text.delete('1.0', tk.END)

        if image_path is None:
            # No image selected
            self.prediction_info_text.insert('1.0', "Upload image(s) to see results\n\n", "label")
            self.prediction_info_text.insert(tk.END, "Select images and click 'Classify Images' to begin")
        elif image_path not in self.image_predictions:
            # Image selected but not classified yet
            filename = Path(image_path).name
            # Truncate filename if too long
            if len(filename) > 35:
                display_name = filename[:32] + "..."
            else:
                display_name = filename

            self.prediction_info_text.insert('1.0', "Image:\n", "label")
            self.prediction_info_text.insert(tk.END, f"{display_name}\n\n")
            self.prediction_info_text.insert(tk.END, "Status: ", "label")
            self.prediction_info_text.insert(tk.END, "Not classified yet\n\n")
            self.prediction_info_text.insert(tk.END, "Click 'Classify Images' to get prediction")
        else:
            # Image has been classified - show prediction
            filename = Path(image_path).name
            # Truncate filename if too long
            if len(filename) > 35:
                display_name = filename[:32] + "..."
            else:
                display_name = filename

            prediction = self.image_predictions[image_path]
            predicted_class = prediction['class']
            confidence = prediction['confidence']
            probs = prediction['probs']

            # Determine confidence level
            if confidence > 0.8:
                conf_level = "HIGH"
                conf_tag = "conf_high"
            elif confidence > 0.6:
                conf_level = "MEDIUM"
                conf_tag = "conf_medium"
            else:
                conf_level = "LOW"
                conf_tag = "conf_low"

            # Display information
            self.prediction_info_text.insert('1.0', "Image:\n", "label")
            self.prediction_info_text.insert(tk.END, f"{display_name}\n\n")

            self.prediction_info_text.insert(tk.END, "Class:\n", "label")
            self.prediction_info_text.insert(tk.END, f"{predicted_class.upper()}\n\n", "class")

            self.prediction_info_text.insert(tk.END, "Confidence: ", "label")
            self.prediction_info_text.insert(tk.END, f"{confidence * 100:.1f}%", conf_tag)
            self.prediction_info_text.insert(tk.END, " | ", "label")
            self.prediction_info_text.insert(tk.END, "Level: ", "label")
            self.prediction_info_text.insert(tk.END, f"{conf_level}\n\n", conf_tag)

            self.prediction_info_text.insert(tk.END, "Top 3 Predictions:\n", "label")

            # Get top 3 predictions
            top3_idx = np.argsort(probs)[-3:][::-1]
            for i, idx in enumerate(top3_idx, 1):
                cls = self.predictor.class_names[idx]
                prob = probs[idx]
                if i == 1:
                    # Highlight the top prediction with green
                    self.prediction_info_text.insert(tk.END, f"  {i}. {cls} - {prob * 100:.1f}%\n", "top1")
                else:
                    # Other predictions in yellow
                    self.prediction_info_text.insert(tk.END, f"  {i}. {cls} - {prob * 100:.1f}%\n", "top3")

        self.prediction_info_text.config(state=tk.DISABLED)

    def create_camera_screen(self):
        """Responsive camera feed screen with LARGE feed display"""
        self.camera_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        # Header
        header = tk.Frame(self.camera_frame, bg='white')
        header.pack(fill=tk.X, padx=10, pady=10)

        title = tk.Label(
            header,
            text="Live Camera Classification",
            font=('Segoe UI', 18, 'bold'),
            bg='white',
            fg='#FF9800'
        )
        title.pack(pady=12)

        # Use PanedWindow but prioritize camera feed
        paned = tk.PanedWindow(self.camera_frame, orient=tk.HORIZONTAL, bg='#f5f5f5',
                               sashwidth=5, sashrelief=tk.RAISED)
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # LEFT panel - LARGE Camera display (75% of space)
        left_panel = tk.Frame(paned, bg='#1a1a1a')
        paned.add(left_panel, minsize=600, stretch="always")

        # Camera display that takes full space
        self.camera_display_label = tk.Label(
            left_panel,
            text="Camera Feed\n\nClick 'Start Camera' to begin",
            font=('Segoe UI', 14),
            bg='#1a1a1a',
            fg='white'
        )
        self.camera_display_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # RIGHT panel - Compact Controls (25% of space)
        right_panel = tk.Frame(paned, bg='white')
        paned.add(right_panel, minsize=280, stretch="never")

        # Scrollable controls
        right_scroll = ScrollableFrame(right_panel, bg='white')
        right_scroll.pack(fill=tk.BOTH, expand=True)

        controls_container = right_scroll.scrollable_frame

        controls_label = tk.Label(
            controls_container,
            text="Controls",
            font=('Segoe UI', 13, 'bold'),
            bg='white',
            fg='#333'
        )
        controls_label.pack(pady=(10, 8))

        # Camera controls - compact
        camera_controls = tk.Frame(controls_container, bg='white')
        camera_controls.pack(pady=5, padx=10, fill=tk.X)

        self.btn_camera_start = ModernButton(
            camera_controls,
            text="START",
            command=self.start_camera,
            bg='#4CAF50'
        )
        self.btn_camera_start.pack(pady=3, fill=tk.X)

        self.btn_camera_stop = ModernButton(
            camera_controls,
            text="STOP",
            command=self.stop_camera,
            bg='#F44336'
        )
        self.btn_camera_stop.pack(pady=3, fill=tk.X)
        self.btn_camera_stop.config(state=tk.DISABLED)

        # Recording section - compact
        tk.Label(
            controls_container,
            text="Recording",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#333'
        ).pack(pady=(10, 5))

        recording_controls = tk.Frame(controls_container, bg='white')
        recording_controls.pack(pady=5, padx=10, fill=tk.X)

        self.btn_camera_record = ModernButton(
            recording_controls,
            text="RECORD",
            command=self.toggle_recording,
            bg='#F44336'
        )
        self.btn_camera_record.pack(pady=3, fill=tk.X)

        self.btn_open_recordings = ModernButton(
            recording_controls,
            text="OPEN FOLDER",
            command=self.open_recordings_folder,
            bg='#9C27B0'
        )
        self.btn_open_recordings.pack(pady=3, fill=tk.X)

        self.recording_status_label = tk.Label(
            recording_controls,
            text="[NOT RECORDING]",
            font=('Segoe UI', 8),
            bg='white',
            fg='#666'
        )
        self.recording_status_label.pack(pady=5)

        # Info section - compact and collapsible
        tk.Label(
            controls_container,
            text="Information",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#333'
        ).pack(pady=(10, 5))

        info_frame = tk.Frame(controls_container, bg='white')
        info_frame.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

        info_scroll = tk.Scrollbar(info_frame)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.camera_info_text = tk.Text(
            info_frame,
            font=('Consolas', 9),
            wrap=tk.WORD,
            yscrollcommand=info_scroll.set,
            bg='#f9f9f9',
            height=12
        )
        self.camera_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.config(command=self.camera_info_text.yview)

        # Configure tags for styling
        self.camera_info_text.tag_configure("model_name", foreground='#4CAF50', font=('Consolas', 9, 'bold'))

        initial_info = f"Current Model: "

        self.camera_info_text.insert('1.0', initial_info)
        self.camera_info_text.insert(tk.END, f"{self.model_type.get().upper()}\n\n", "model_name")

        camera_tips = "Camera Tips:\n\n"
        camera_tips += "- Start camera to begin\n"
        camera_tips += "- Real-time predictions\n"
        camera_tips += "- Record with predictions\n"
        camera_tips += "- Stop when done"

        self.camera_info_text.insert(tk.END, camera_tips)

    def create_capture_screen(self):
        """Camera Capture screen - 3 column layout similar to images screen"""
        self.capture_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        # Main container
        content = tk.Frame(self.capture_frame, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT PANEL - Controls (FIXED WIDTH - 320px)
        left_panel = tk.Frame(content, bg='white', width=320, relief=tk.RAISED, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)

        # Header
        header = tk.Label(
            left_panel,
            text="Camera Controls",
            font=('Segoe UI', 14, 'bold'),
            bg='white',
            fg='#00BCD4'
        )
        header.pack(pady=10)

        # Status label
        self.capture_status_label = tk.Label(
            left_panel,
            text="Camera Ready",
            font=('Segoe UI', 10, 'bold'),
            bg='white',
            fg='#666'
        )
        self.capture_status_label.pack(pady=5)

        # Control buttons frame
        controls_frame = tk.Frame(left_panel, bg='white')
        controls_frame.pack(pady=6, padx=10, fill=tk.X)

        # Start/Stop Camera Button
        self.btn_start_capture_camera = ModernButton(
            controls_frame,
            text="START CAMERA",
            command=self.toggle_capture_camera,
            bg='#4CAF50'
        )
        self.btn_start_capture_camera.pack(pady=3, fill=tk.X)

        # Capture Button
        self.btn_capture = ModernButton(
            controls_frame,
            text="CAPTURE",
            command=self.capture_image,
            bg='#00BCD4'
        )
        self.btn_capture.pack(pady=3, fill=tk.X)
        self.btn_capture.config(state=tk.DISABLED)

        # Open folder button
        btn_open_folder = ModernButton(
            controls_frame,
            text="OPEN FOLDER",
            command=self.open_captures_folder,
            bg='#9E9E9E'
        )
        btn_open_folder.pack(pady=3, fill=tk.X)

        # Separator
        separator = tk.Frame(left_panel, bg='#ddd', height=1)
        separator.pack(fill=tk.X, padx=15, pady=10)

        # Instructions section (taking full space now)
        instructions_label = tk.Label(
            left_panel,
            text="Instructions",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#333'
        )
        instructions_label.pack(pady=(5, 8))

        instructions_frame = tk.Frame(left_panel, bg='#f5f5f5', relief=tk.SUNKEN, borderwidth=2)
        instructions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.capture_instructions_text = tk.Text(
            instructions_frame,
            font=('Segoe UI', 8),
            wrap=tk.WORD,
            bg='#f5f5f5',
            fg='#333',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.capture_instructions_text.pack(fill=tk.BOTH, expand=True)
        self.capture_instructions_text.tag_configure("model_name", foreground='#00BCD4', font=('Segoe UI', 10, 'bold'))

        # Initialize instructions
        self.update_capture_instructions()

        # MIDDLE PANEL - Camera Display (EXPANDS)
        middle_panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=1)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Camera display header
        display_header = tk.Frame(middle_panel, bg='white')
        display_header.pack(pady=10)

        display_label = tk.Label(
            display_header,
            text="Camera View",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg='#333'
        )
        display_label.pack()

        self.capture_info_label = tk.Label(
            display_header,
            text="Camera inactive - Click 'Start Camera' to begin",
            font=('Segoe UI', 9),
            bg='white',
            fg='#666'
        )
        self.capture_info_label.pack(pady=(5, 0))

        # Camera container - FIXED BORDER ON ALL SIDES
        camera_container = tk.Frame(middle_panel, bg='#e0e0e0', relief=tk.SUNKEN, borderwidth=2)
        camera_container.pack(padx=15, pady=(8, 15), fill=tk.BOTH, expand=True)

        self.capture_display = tk.Label(
            camera_container,
            text="Camera Inactive\n\nClick 'START CAMERA' to begin",
            font=('Segoe UI', 12),
            bg='#1a1a1a',
            fg='white'
        )
        self.capture_display.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # RIGHT PANEL - Classification Results (FIXED WIDTH - 320px)
        right_panel = tk.Frame(content, bg='white', width=320, relief=tk.RAISED, borderwidth=1)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)

        results_label = tk.Label(
            right_panel,
            text="Classification Result",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#333'
        )
        results_label.pack(pady=10)

        results_info = tk.Label(
            right_panel,
            text="Results will appear here after capture",
            font=('Segoe UI', 8),
            bg='white',
            fg='#999',
            wraplength=280
        )
        results_info.pack(pady=(0, 5))

        # Result display frame with Text widget for color formatting
        result_frame = tk.Frame(right_panel, bg='#f5f5f5', relief=tk.SUNKEN, borderwidth=2)
        result_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        self.capture_result_text = tk.Text(
            result_frame,
            font=('Segoe UI', 9),
            wrap=tk.WORD,
            bg='#f5f5f5',
            fg='#333',
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.capture_result_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for colored text
        self.capture_result_text.tag_configure("prediction", foreground='#1976D2', font=('Segoe UI', 10, 'bold'))
        self.capture_result_text.tag_configure("confidence", foreground='#2E7D32', font=('Segoe UI', 9, 'bold'))
        self.capture_result_text.tag_configure("level", foreground='#F57C00', font=('Segoe UI', 9, 'bold'))
        self.capture_result_text.tag_configure("model", foreground='#5E35B1', font=('Segoe UI', 9, 'bold'))
        self.capture_result_text.tag_configure("label", foreground='#333', font=('Segoe UI', 9, 'bold'))

        # Initial text
        initial_text = "No capture yet\n\n"
        initial_text += "Waiting for image capture...\n\n"
        initial_text += "Start the camera and click\n"
        initial_text += "the capture button to begin."
        self.capture_result_text.insert('1.0', initial_text)
        self.capture_result_text.config(state=tk.DISABLED)

        # Additional info section at bottom
        separator2 = tk.Frame(right_panel, bg='#ddd', height=1)
        separator2.pack(fill=tk.X, padx=15, pady=8)

        info_bottom_label = tk.Label(
            right_panel,
            text="Auto-Save Location",
            font=('Segoe UI', 9, 'bold'),
            bg='white',
            fg='#00BCD4'
        )
        info_bottom_label.pack(pady=(0, 5))

        self.capture_save_location_label = tk.Label(
            right_panel,
            text=f"results/captured_images/\n{self.model_type.get()}/",
            font=('Segoe UI', 8),
            bg='white',
            fg='#666',
            wraplength=280
        )
        self.capture_save_location_label.pack(pady=(0, 10))

    def create_video_screen(self):
        """Redesigned video processing screen with preview capabilities - SCROLLABLE"""
        # Use ScrollableFrame for responsive design
        scroll_frame = ScrollableFrame(self.main_container, bg='#f5f5f5')
        self.video_frame = scroll_frame

        # Content goes in scrollable_frame
        content_container = scroll_frame.scrollable_frame

        # Initialize video variables
        self.input_video_path = None
        self.output_video_path = None
        self.input_video_cap = None
        self.output_video_cap = None
        self.video_playing_input = False
        self.video_playing_output = False

        # Create output directory
        self.processed_videos_folder = Path('results') / 'processed_videos'
        self.processed_videos_folder.mkdir(parents=True, exist_ok=True)

        # Header
        header = tk.Frame(content_container, bg='white', relief=tk.RAISED, borderwidth=2)
        header.pack(fill=tk.X, padx=20, pady=(15, 8))  # Reduced padding

        title = tk.Label(
            header,
            text="Video Processing Classification",
            font=('Segoe UI', 16, 'bold'),  # Slightly smaller
            bg='white',
            fg='#9C27B0'
        )
        title.pack(pady=10)  # Reduced padding

        # Main content with split view
        content_frame = tk.Frame(content_container, bg='#f5f5f5')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        # LEFT PANEL - Input Video
        left_panel = tk.Frame(content_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Left header
        left_header = tk.Label(
            left_panel,
            text="INPUT VIDEO",
            font=('Segoe UI', 14, 'bold'),
            bg='#2196F3',
            fg='white'
        )
        left_header.pack(fill=tk.X, pady=(0, 10))

        # Video display area (input) - wrapped in fixed frame to prevent crushing
        input_display_frame = tk.Frame(left_panel, bg='#1a1a1a', height=300)  # Fixed pixel height
        input_display_frame.pack(padx=10, pady=8, fill=tk.X)
        input_display_frame.pack_propagate(False)  # Prevent frame from resizing to content

        self.input_video_display = tk.Label(
            input_display_frame,
            text="No Video Loaded\n\nClick 'Select Video' to upload",
            font=('Segoe UI', 11),
            bg='#1a1a1a',
            fg='white'
        )
        self.input_video_display.pack(fill=tk.BOTH, expand=True)

        # Input video info
        self.input_video_info = tk.Label(
            left_panel,
            text="No video selected",
            font=('Segoe UI', 8),  # Smaller font
            bg='white',
            fg='#666',
            justify=tk.LEFT
        )
        self.input_video_info.pack(padx=10, pady=(0, 8))  # Reduced padding

        # Input controls
        input_controls = tk.Frame(left_panel, bg='white')
        input_controls.pack(padx=10, pady=(0, 10), fill=tk.X)  # Reduced padding

        ModernButton(
            input_controls,
            text="SELECT VIDEO",
            command=self.select_input_video,
            bg='#2196F3'
        ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.btn_play_input = ModernButton(
            input_controls,
            text="▶ PLAY",
            command=self.toggle_input_video,
            bg='#4CAF50'
        )
        self.btn_play_input.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.btn_play_input.config(state=tk.DISABLED)

        # RIGHT PANEL - Output Video
        right_panel = tk.Frame(content_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Right header
        right_header = tk.Label(
            right_panel,
            text="OUTPUT VIDEO",
            font=('Segoe UI', 14, 'bold'),
            bg='#4CAF50',
            fg='white'
        )
        right_header.pack(fill=tk.X, pady=(0, 10))

        # Video display area (output) - wrapped in fixed frame to prevent crushing
        output_display_frame = tk.Frame(right_panel, bg='#1a1a1a', height=300)  # Fixed pixel height
        output_display_frame.pack(padx=10, pady=8, fill=tk.X)
        output_display_frame.pack_propagate(False)  # Prevent frame from resizing to content

        self.output_video_display = tk.Label(
            output_display_frame,
            text="No Processed Video\n\nProcess a video to see output",
            font=('Segoe UI', 11),
            bg='#1a1a1a',
            fg='white'
        )
        self.output_video_display.pack(fill=tk.BOTH, expand=True)

        # Output video info
        self.output_video_info = tk.Label(
            right_panel,
            text="No output yet",
            font=('Segoe UI', 8),  # Smaller font
            bg='white',
            fg='#666',
            justify=tk.LEFT
        )
        self.output_video_info.pack(padx=10, pady=(0, 8))  # Reduced padding

        # Output controls
        output_controls = tk.Frame(right_panel, bg='white')
        output_controls.pack(padx=10, pady=(0, 10), fill=tk.X)  # Reduced padding

        self.btn_process = ModernButton(
            output_controls,
            text="PROCESS VIDEO",
            command=self.process_video,
            bg='#9C27B0'
        )
        self.btn_process.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.btn_process.config(state=tk.DISABLED)

        self.btn_play_output = ModernButton(
            output_controls,
            text="▶ PLAY",
            command=self.toggle_output_video,
            bg='#4CAF50'
        )
        self.btn_play_output.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.btn_play_output.config(state=tk.DISABLED)

        # Bottom controls (in scrollable container)
        bottom_controls = tk.Frame(content_container, bg='white', relief=tk.RAISED, borderwidth=2)
        bottom_controls.pack(fill=tk.X, padx=20, pady=(0, 15))  # Reduced bottom padding

        ModernButton(
            bottom_controls,
            text="OPEN OUTPUT FOLDER",
            command=self.open_processed_videos_folder,
            bg='#FF9800'
        ).pack(side=tk.LEFT, padx=10, pady=10)

        ModernButton(
            bottom_controls,
            text="CLEAR ALL",
            command=self.clear_all_videos,
            bg='#F44336'
        ).pack(side=tk.LEFT, padx=10, pady=10)

        # Progress info
        self.video_progress_label = tk.Label(
            bottom_controls,
            text="Ready to process videos",
            font=('Segoe UI', 10),
            bg='white',
            fg='#666'
        )
        self.video_progress_label.pack(side=tk.LEFT, padx=20, pady=10)

    # Navigation methods
    def show_home(self):
        # Stop camera if it's running
        if self.camera_active:
            self.stop_camera()
        # Stop capture camera if it's running
        if self.capture_camera_active:
            self.stop_capture_camera()
        self.hide_all_frames()
        self.home_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "home"
        self.update_status("[HOME] Select a mode")

    def switch_to_images(self):
        # Stop cameras if running
        if self.camera_active:
            self.stop_camera()
        if self.capture_camera_active:
            self.stop_capture_camera()
        self.hide_all_frames()
        self.images_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "images"
        self.mode_subtitle_label.config(text="Image Classification Mode")
        self.update_status("[IMAGE CLASSIFICATION] Mode active")

    def switch_to_camera(self):
        self.hide_all_frames()
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        self.mode_subtitle_label.config(text="Live Camera Mode")
        self.update_status("[LIVE CAMERA] Mode active")

    def switch_to_capture(self):
        # Stop live camera if it's running
        if self.camera_active:
            self.stop_camera()
        # Stop capture camera if it's running
        if self.capture_camera_active:
            self.stop_capture_camera()
        self.hide_all_frames()
        self.capture_frame.pack(fill=tk.BOTH, expand=True)
        self.mode_subtitle_label.config(text="Camera Capture Mode")
        self.update_status("[CAMERA CAPTURE] Mode active")

    def switch_to_video(self):
        # Stop cameras if running
        if self.camera_active:
            self.stop_camera()
        if self.capture_camera_active:
            self.stop_capture_camera()
        self.hide_all_frames()
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.mode_subtitle_label.config(text="Video Processing Mode")
        self.update_status("[VIDEO PROCESSING] Mode active")

    def hide_all_frames(self):
        """Hide all mode frames"""
        for frame in [self.home_frame, self.images_frame, self.camera_frame, self.capture_frame, self.video_frame]:
            frame.pack_forget()

    def update_capture_instructions(self):
        """Update capture screen instructions with current model - COMPACT VERSION"""
        if not hasattr(self, 'capture_instructions_text'):
            return

        self.capture_instructions_text.config(state=tk.NORMAL)
        self.capture_instructions_text.delete('1.0', tk.END)

        # Much more compact instructions
        instructions_content = """MODEL: """
        self.capture_instructions_text.insert('1.0', instructions_content)
        self.capture_instructions_text.insert(tk.END, self.model_type.get().upper() + "\n\n", "model_name")

        # Compact workflow
        instructions_content2 = """How to use:
    1. START CAMERA - Activate camera
    3. CAPTURE - Click when ready (Place item close to camera)
    4. WAIT - Model processes image
    5. VIEW RESULT - See prediction popup
    6. AUTO-SAVED - Organized by model

    TIPS
    - Good lighting improves accuracy
    - Clear background works best
    - Keep item centered
    - Wait for camera focus

    FILES SAVED TO
    captured_images/[model]/[class]/
    """
        self.capture_instructions_text.insert(tk.END, instructions_content2)
        self.capture_instructions_text.config(state=tk.DISABLED)

    def load_model(self):
        """Load the selected model"""
        model_name = self.model_type.get()

        # Try both .pth and .pt extensions for ResNet models
        if 'resnet' in model_name:
            # Try .pth first (standard PyTorch format)
            model_path_pth = f'models/{model_name}_best.pth'
            model_path_pt = f'models/{model_name}_best.pt'

            if Path(model_path_pth).exists():
                model_path = model_path_pth
            elif Path(model_path_pt).exists():
                model_path = model_path_pt
            else:
                error_msg = f"Model not found!\n\nLooked for:\n• {model_path_pth}\n• {model_path_pt}\n\nPlease ensure the model file exists in the models folder."
                messagebox.showerror("Model Not Found", error_msg)
                return
        else:
            # YOLO models use .pt
            model_path = f'models/{model_name}_best.pt'
            if not Path(model_path).exists():
                error_msg = f"Model not found: {model_path}\n\nPlease ensure the model file exists in the models folder."
                messagebox.showerror("Model Not Found", error_msg)
                return

        try:
            self.update_status(f"[LOADING] {model_name.upper()}...")
            self.predictor = OfficeItemsPredictor(
                model_type=model_name,
                model_path=model_path
            )
            self.current_model_label.config(text=f"[LOADED] {model_name.upper()}")
            self.update_status(f"[READY] Model loaded: {model_name.upper()}")

            # Update capture screen if it exists
            self.update_capture_instructions()

            # Update save location label in capture screen
            if hasattr(self, 'capture_save_location_label'):
                self.capture_save_location_label.config(
                    text=f"results/captured_images/\n{self.model_type.get()}/"
                )

            # Update live camera info text if it exists
            if hasattr(self, 'camera_info_text'):
                self.camera_info_text.delete('1.0', tk.END)
                self.camera_info_text.tag_configure("model_name", foreground='#4CAF50', font=('Consolas', 9, 'bold'))

                initial_info = f"Current Model: "
                self.camera_info_text.insert('1.0', initial_info)
                self.camera_info_text.insert(tk.END, f"{self.model_type.get().upper()}\n\n", "model_name")

                camera_tips = "Camera Tips:\n\n"
                camera_tips += "- Start camera to begin\n"
                camera_tips += "- Real-time predictions\n"
                camera_tips += "- Record with predictions\n"
                camera_tips += "- Stop when done"
                self.camera_info_text.insert(tk.END, camera_tips)

        except FileNotFoundError as e:
            error_msg = f"Missing required files:\n\n{str(e)}\n\nMake sure you have:\n• Model weights file\n• config/config.yaml\n• models/class_names.json"
            messagebox.showerror("Missing Files", error_msg)
            self.update_status(f"[ERROR] Failed to load {model_name.upper()}")
        except Exception as e:
            error_msg = f"Failed to load model:\n\n{str(e)}\n\nCheck that:\n• Model file is not corrupted\n• All dependencies are installed\n• config.yaml is properly configured"
            messagebox.showerror("Loading Error", error_msg)
            self.update_status(f"[ERROR] Failed to load {model_name.upper()}")

    # Image classification methods
    def select_images(self):
        """Select single or multiple images (max 10)"""
        file_paths = filedialog.askopenfilenames(
            title="Select Image(s) - Single or Multiple (Max 10)",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not file_paths:
            return

        # Enforce 10 image limit
        if len(file_paths) > 10:
            messagebox.showwarning(
                "Too Many Images",
                f"You selected {len(file_paths)} images.\n\nMaximum allowed: 10 images\n\nPlease select fewer images."
            )
            return

        self.selected_image_paths = list(file_paths)
        self.image_predictions = {}
        self.current_image_index = 0

        # Update file count label
        count = len(self.selected_image_paths)
        self.file_count_label.config(text=f"Selected Files ({count})")

        # Update listbox
        self.images_listbox.delete(0, tk.END)
        for i, path in enumerate(self.selected_image_paths, 1):
            filename = Path(path).name
            self.images_listbox.insert(tk.END, f"{i}. {filename}")

        # Update UI
        self.btn_classify_images.config(state=tk.NORMAL)

        # Enable/disable navigation buttons
        if count > 1:
            self.btn_next_image.config(state=tk.NORMAL)
            self.btn_prev_image.config(state=tk.NORMAL)
        else:
            self.btn_next_image.config(state=tk.DISABLED)
            self.btn_prev_image.config(state=tk.DISABLED)

        if count == 1:
            self.update_status(f"[LOADED] 1 image ready for classification")
        else:
            self.update_status(f"[LOADED] {count} images ready for batch classification")

        # Auto-select and display first image
        self.images_listbox.selection_set(0)
        self.display_current_image()

    def display_current_image(self):
        """Display the current image based on current_image_index"""
        if not self.selected_image_paths or self.current_image_index >= len(self.selected_image_paths):
            return

        image_path = self.selected_image_paths[self.current_image_index]
        filename = Path(image_path).name

        try:
            img = Image.open(image_path)

            # Use FIXED size for all images - prevents resizing
            fixed_width = 600
            fixed_height = 450

            # Resize image to fit within fixed dimensions while maintaining aspect ratio
            img.thumbnail((fixed_width, fixed_height), Image.Resampling.LANCZOS)

            # Create a new image with fixed size and center the thumbnail
            fixed_img = Image.new('RGB', (fixed_width, fixed_height), color='#f5f5f5')

            # Calculate position to center the image
            img_width, img_height = img.size
            x = (fixed_width - img_width) // 2
            y = (fixed_height - img_height) // 2

            # Paste the thumbnail onto the fixed size canvas
            fixed_img.paste(img, (x, y))

            photo = ImageTk.PhotoImage(fixed_img)

            self.images_display_label.config(image=photo, text="")
            self.images_display_label.image = photo

            # Update info label with filename and counter
            total = len(self.selected_image_paths)
            current = self.current_image_index + 1
            info_text = f"Image {current} of {total}\n{filename}"
            self.image_info_label.config(text=info_text)

            # Update listbox selection to match
            self.images_listbox.selection_clear(0, tk.END)
            self.images_listbox.selection_set(self.current_image_index)
            self.images_listbox.see(self.current_image_index)

            # Update the prediction info box
            self.update_prediction_info_box(image_path)

        except Exception as e:
            self.images_display_label.config(text=f"Error loading image:\n{str(e)}")
            self.image_info_label.config(text="")
            self.update_prediction_info_box(None)

    def prev_image(self):
        """Navigate to previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()

    def next_image(self):
        """Navigate to next image"""
        if self.current_image_index < len(self.selected_image_paths) - 1:
            self.current_image_index += 1
            self.display_current_image()

    def on_image_select(self, event):
        """Display selected image from listbox"""
        if not self.images_listbox.curselection():
            return

        idx = self.images_listbox.curselection()[0]
        self.current_image_index = idx
        self.display_current_image()

    def classify_images(self):
        """Classify all selected images"""
        if not self.predictor:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        if not self.selected_image_paths:
            messagebox.showwarning("Warning", "No images selected")
            return

        self.update_status("[PROCESSING] Classifying images...")
        self.btn_classify_images.config(state=tk.DISABLED)

        def classify_thread():
            for idx, image_path in enumerate(self.selected_image_paths, 1):
                try:
                    predicted_class, confidence, probs = self.predictor.predict_image(image_path)

                    self.image_predictions[image_path] = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'probs': probs
                    }

                except Exception as e:
                    print(f"[ERROR] Image {idx}: {str(e)}")

            self.btn_classify_images.config(state=tk.NORMAL)
            self.update_status(f"[COMPLETE] Classified {len(self.selected_image_paths)} image(s)")

            # Refresh the current image display to show the prediction info
            self.display_current_image()

        threading.Thread(target=classify_thread, daemon=True).start()

    def clear_images(self):
        """Clear all selected images"""
        self.selected_image_paths = []
        self.image_predictions = {}
        self.current_image_index = 0

        # Reset file count label
        self.file_count_label.config(text="Selected Files (0)")

        self.images_listbox.delete(0, tk.END)
        self.images_display_label.config(
            image='',
            text="No image selected\n\nClick 'Select Images' to begin"
        )
        self.image_info_label.config(text="No image loaded")

        # Reset results with help text
        self.images_results_text.delete('1.0', tk.END)
        initial_text = "How to use:\n\n"
        initial_text += "1. Select up to 10 images\n"
        initial_text += "2. Preview them here\n"
        initial_text += "3. Click 'Classify Images'\n"
        initial_text += "4. View results here\n\n"
        initial_text += "Waiting for images..."
        self.images_results_text.insert('1.0', initial_text)

        self.btn_classify_images.config(state=tk.DISABLED)
        self.btn_prev_image.config(state=tk.DISABLED)
        self.btn_next_image.config(state=tk.DISABLED)
        self.update_status("[CLEARED] All images removed")

    # Camera methods
    def start_camera(self):
        """Start camera feed"""
        if not self.predictor:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        if self.camera_active:
            return

        # Show loading message
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Starting Camera")
        loading_window.resizable(False, False)
        loading_window.configure(bg='white')

        # Center the window on screen
        window_width = 400
        window_height = 220
        screen_width = loading_window.winfo_screenwidth()
        screen_height = loading_window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        loading_window.geometry(f'{window_width}x{window_height}+{x}+{y}')

        # Make it modal
        loading_window.transient(self.root)
        loading_window.grab_set()

        # Variable to track if user cancelled
        user_cancelled = [False]

        # Handle X button click - stop camera initialization
        def on_closing():
            user_cancelled[0] = True
            if self.cap:
                self.cap.release()
                self.cap = None
            self.camera_active = False
            loading_window.destroy()
            self.update_status("[CANCELLED] Camera start cancelled")

        loading_window.protocol("WM_DELETE_WINDOW", on_closing)

        # Loading content
        loading_frame = tk.Frame(loading_window, bg='white')
        loading_frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=30)

        loading_label = tk.Label(
            loading_frame,
            text="Starting Camera",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg='#FF9800'
        )
        loading_label.pack(pady=(0, 15))

        loading_msg = tk.Label(
            loading_frame,
            text="Please be patient...\nThis may take several seconds",
            font=('Segoe UI', 11),
            bg='white',
            fg='#666',
            justify=tk.CENTER
        )
        loading_msg.pack(pady=(0, 20))

        # Progress bar with custom style for visibility
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Loading.Horizontal.TProgressbar",
                       foreground='#FF9800',
                       background='#FF9800',
                       troughcolor='#e0e0e0',
                       bordercolor='#ccc',
                       lightcolor='#FF9800',
                       darkcolor='#FF9800')

        progress = ttk.Progressbar(
            loading_frame,
            style="Loading.Horizontal.TProgressbar",
            mode='indeterminate',
            length=320
        )
        progress.pack(pady=10)
        progress.start(8)

        # Status label
        status_label = tk.Label(
            loading_frame,
            text="Initializing camera...",
            font=('Segoe UI', 9),
            bg='white',
            fg='#999'
        )
        status_label.pack(pady=(10, 0))

        # Force update to show window immediately
        loading_window.update_idletasks()

        def start_camera_thread():
            success = False
            try:
                # Check if user cancelled
                if user_cancelled[0]:
                    return

                # Update status
                status_label.config(text="Opening camera device...")
                loading_window.update()

                if user_cancelled[0]:
                    return

                self.cap = cv2.VideoCapture(0)

                # Wait a moment for camera to initialize
                import time
                time.sleep(0.5)

                if user_cancelled[0]:
                    if self.cap:
                        self.cap.release()
                    return

                status_label.config(text="Checking camera connection...")
                loading_window.update()

                if not self.cap.isOpened():
                    raise Exception("Cannot open camera")

                if user_cancelled[0]:
                    if self.cap:
                        self.cap.release()
                    return

                # Try to read a test frame
                status_label.config(text="Testing camera feed...")
                loading_window.update()

                ret, test_frame = self.cap.read()
                if not ret:
                    raise Exception("Cannot read from camera")

                if user_cancelled[0]:
                    if self.cap:
                        self.cap.release()
                    return

                status_label.config(text="Starting video stream...")
                loading_window.update()

                self.camera_active = True
                self.btn_camera_start.config(state=tk.DISABLED)
                self.btn_camera_stop.config(state=tk.NORMAL)
                self.update_status("[ACTIVE] Camera running")

                self.camera_info_text.delete('1.0', tk.END)
                info = "CAMERA ACTIVE\n"
                info += "=" * 30 + "\n\n"
                info += "[ACTIVE] Live predictions enabled\n"
                info += "Point camera at objects\n"
                info += "Start recording to save\n\n"
                info += f"Model: {self.model_type.get().upper()}"
                self.camera_info_text.insert('1.0', info)

                success = True

                # Small delay before closing to show completion
                status_label.config(text="Camera ready!")
                loading_window.update()
                time.sleep(0.3)

            except Exception as e:
                if not user_cancelled[0]:
                    loading_window.destroy()
                    messagebox.showerror("Error", f"Camera error: {str(e)}\n\nPlease check:\n• Camera is connected\n• No other app is using it\n• Camera permissions are granted")
                self.camera_active = False
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return
            finally:
                # Close loading window if it still exists and user didn't cancel
                if not user_cancelled[0] and loading_window.winfo_exists():
                    loading_window.destroy()

            # Start the camera feed AFTER loading window is closed
            if success and not user_cancelled[0]:
                threading.Thread(target=self.update_camera_feed, daemon=True).start()

        # Start camera in a separate thread to not block UI
        threading.Thread(target=start_camera_thread, daemon=True).start()

    def stop_camera(self):
        """Stop camera feed"""
        if self.recording:
            self.stop_recording()

        # Set flag to stop the camera thread
        self.camera_active = False

        # Give the thread time to stop (important!)
        import time
        time.sleep(0.1)  # 100ms delay to let the feed loop exit

        # Now release the camera
        if self.cap:
            self.cap.release()
            self.cap = None

        # Clear the last processed frame
        self.last_processed_frame = None

        self.btn_camera_start.config(state=tk.NORMAL)
        self.btn_camera_stop.config(state=tk.DISABLED)

        # Force update to process any pending events
        self.root.update()

        # Now reset the display
        self.camera_display_label.configure(
            image='',
            text="Camera Feed\n\nClick 'Start' to begin",
            font=('Segoe UI', 14),
            bg='#1a1a1a',
            fg='white',
            compound='none'
        )

        # Delete the image reference completely
        if hasattr(self.camera_display_label, 'image'):
            delattr(self.camera_display_label, 'image')

        # Force another update
        self.root.update_idletasks()

        # Reset info text
        self.camera_info_text.delete('1.0', tk.END)

        # Configure tags for styling
        self.camera_info_text.tag_configure("model_name", foreground='#4CAF50', font=('Consolas', 9, 'bold'))

        initial_info = f"Current Model: "

        self.camera_info_text.insert('1.0', initial_info)
        self.camera_info_text.insert(tk.END, f"{self.model_type.get().upper()}\n\n", "model_name")

        camera_tips = "Camera Tips:\n\n"
        camera_tips += "- Start camera to begin\n"
        camera_tips += "- Real-time predictions\n"
        camera_tips += "- Record with predictions\n"
        camera_tips += "- Stop when done"

        self.camera_info_text.insert(tk.END, camera_tips)

        self.update_status("[STOPPED] Camera stopped")

    def update_camera_feed(self):
        """Update camera feed with predictions"""
        frame_count = 0

        while self.camera_active and self.cap and self.cap.isOpened():
            # Double-check camera is still active at loop start
            if not self.camera_active:
                break

            ret, frame = self.cap.read()

            if not ret:
                break

            # Process every 2nd frame for smooth performance
            if frame_count % 2 == 0:
                display_frame = self.process_frame(frame)
                self.last_processed_frame = display_frame
            else:
                display_frame = self.last_processed_frame if self.last_processed_frame is not None else frame

            # Write to video file if recording (frame is already mirrored from process_frame)
            if self.recording and self.video_writer and self.video_writer.isOpened():
                try:
                    self.video_writer.write(display_frame)
                except Exception as e:
                    print(f"Recording error: {e}")
                    # Stop recording on error to prevent crashes
                    self.stop_recording()

            # Check again before updating display
            if not self.camera_active:
                break

            # Get available display space dynamically (frame is already mirrored)
            try:
                label_width = self.camera_display_label.winfo_width()
                label_height = self.camera_display_label.winfo_height()

                # Only resize if we have valid dimensions
                if label_width > 1 and label_height > 1:
                    # Convert and display with responsive sizing
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)

                    # Calculate aspect ratio
                    frame_aspect = img.width / img.height
                    display_aspect = label_width / label_height

                    # Fit image to display while maintaining aspect ratio
                    if frame_aspect > display_aspect:
                        # Frame is wider - fit to width
                        new_width = label_width - 20  # Leave some padding
                        new_height = int(new_width / frame_aspect)
                    else:
                        # Frame is taller - fit to height
                        new_height = label_height - 20  # Leave some padding
                        new_width = int(new_height * frame_aspect)

                    # Ensure minimum size
                    new_width = max(320, new_width)
                    new_height = max(240, new_height)

                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image=img)
                else:
                    # Fallback to default size if dimensions not available
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img = img.resize((960, 720), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image=img)

            except Exception as e:
                print(f"Display error: {e}")
                # Fallback to default size on error
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((960, 720), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)

            # Only update if camera is still active
            if self.camera_active:
                self.camera_display_label.config(image=photo, text="")
                self.camera_display_label.image = photo

            self.root.update_idletasks()
            frame_count += 1

    def process_frame(self, frame):
        """Add predictions to frame with responsive overlay"""
        temp_path = Path("temp_frame.jpg")
        cv2.imwrite(str(temp_path), frame)

        try:
            predicted_class, confidence, probs = self.predictor.predict_image(temp_path)
            predicted_class, confidence, probs = self.smooth_prediction(predicted_class, confidence, probs)

            # Mirror the frame FIRST, then draw predictions on the mirrored frame
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # Calculate responsive scaling factors
            scale_x = width / 640.0  # Base width
            scale_y = height / 480.0  # Base height
            scale = min(scale_x, scale_y)  # Use minimum to ensure everything fits

            # Responsive dimensions
            overlay_height = int(205 * scale_y)
            overlay_width = int(295 * scale_x)
            margin_x = int(20 * scale_x)
            margin_y = int(35 * scale_y)

            # Font scales
            title_scale = 1.0 * scale  # Slightly reduced for cleaner look
            text_scale = 0.55 * scale

            # Thickness scales - thinner for sharp, clean text
            title_thickness = max(2, int(2 * scale))
            text_thickness = max(1, int(1 * scale))

            # Add semi-transparent overlay at top-left (not full width)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)
                conf_text = "HIGH"
            elif confidence > 0.6:
                color = (0, 200, 255)
                conf_text = "MEDIUM"
            else:
                color = (0, 0, 255)
                conf_text = "LOW"

            # Draw predictions with responsive positioning
            y_pos = margin_y
            cv2.putText(frame, predicted_class.upper(), (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, title_scale, color, title_thickness, cv2.LINE_AA)

            y_pos += int(28 * scale_y)
            conf_display = f"Confidence: {confidence:.1%} ({conf_text})"
            cv2.putText(frame, conf_display, (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, text_scale, color, text_thickness, cv2.LINE_AA)

            y_pos += int(25 * scale_y)
            model_display = f"Model: {self.model_type.get().upper()}"
            cv2.putText(frame, model_display, (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, text_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)

            y_pos += int(28 * scale_y)
            cv2.putText(frame, "Top 3 Predictions:", (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

            top3_idx = np.argsort(probs)[-3:][::-1]
            y_pos += int(25 * scale_y)
            for i, idx in enumerate(top3_idx):
                cls = self.predictor.class_names[idx]
                prob = probs[idx]
                text_color = (0, 255, 0) if i == 0 else (255, 255, 255)
                text = f"{i+1}. {cls}: {prob:.1%}"
                cv2.putText(frame, text, (margin_x, y_pos),
                           cv2.FONT_HERSHEY_TRIPLEX, text_scale, text_color, text_thickness, cv2.LINE_AA)
                y_pos += int(25 * scale_y)

            # Recording indicator (responsive positioning)
            if self.recording:
                rec_radius = int(12 * scale)
                rec_x = width - int(40 * scale_x)
                rec_y = int(30 * scale_y)
                cv2.circle(frame, (rec_x, rec_y), rec_radius, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (width - int(100 * scale_x), int(40 * scale_y)),
                           cv2.FONT_HERSHEY_TRIPLEX, 0.6 * scale, (0, 0, 255), text_thickness, cv2.LINE_AA)

            self.last_processed_frame = frame

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return frame

    def smooth_prediction(self, predicted_class, confidence, probs):
        """Smooth predictions to reduce flickering"""
        self.prediction_buffer.append({
            'class': predicted_class,
            'confidence': confidence,
            'probs': probs
        })

        if len(self.prediction_buffer) > 5:
            self.prediction_buffer.pop(0)

        # Use most common class
        classes = [p['class'] for p in self.prediction_buffer]
        most_common = Counter(classes).most_common(1)[0][0]

        # Average confidence and probs for most common class
        matching = [p for p in self.prediction_buffer if p['class'] == most_common]
        avg_confidence = np.mean([p['confidence'] for p in matching])
        avg_probs = np.mean([p['probs'] for p in matching], axis=0)

        return most_common, avg_confidence, avg_probs

    def toggle_recording(self):
        """Toggle recording on/off"""
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """Start recording camera feed"""
        if not self.camera_active:
            messagebox.showwarning("Warning", "Start camera first")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.model_type.get()

        model_recordings = self.recordings_folder / model_name
        model_recordings.mkdir(parents=True, exist_ok=True)

        output_path = model_recordings / f"recording_{timestamp}.mp4"

        # Get actual camera properties
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Use a more compatible codec and settings
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                20.0,
                (frame_width, frame_height),
                True
            )

            if not self.video_writer.isOpened():
                raise Exception("H264 codec failed")

        except:
            try:
                output_path = model_recordings / f"recording_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    20.0,
                    (frame_width, frame_height),
                    True
                )

                if not self.video_writer.isOpened():
                    raise Exception("XVID codec failed")

            except:
                output_path = model_recordings / f"recording_{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    str(output_path),
                    fourcc,
                    20.0,
                    (frame_width, frame_height),
                    True
                )

        # Check if video writer is working
        if not self.video_writer or not self.video_writer.isOpened():
            messagebox.showerror("Error", "Failed to initialize video recorder.\nTry a different codec or check disk space.")
            self.video_writer = None
            return

        self.recording = True
        self.recording_start = datetime.now()
        self.btn_camera_record.config(text="STOP", bg='#4CAF50')
        self.recording_status_label.config(text="[RECORDING]", fg='#F44336')
        self.update_status(f"[RECORDING] {output_path.name}")

        info_text = f"RECORDING\n"
        info_text += "=" * 30 + "\n\n"
        info_text += f"File:\n{output_path.name}\n\n"
        info_text += f"Location:\n{model_recordings.name}\n\n"
        info_text += f"Started:\n{self.recording_start.strftime('%H:%M:%S')}\n\n"
        info_text += "Recording...\n"
        info_text += "Click 'Stop' when done."

        self.camera_info_text.delete('1.0', tk.END)
        self.camera_info_text.insert('1.0', info_text)

    def stop_recording(self):
        """Stop recording"""
        if not self.recording:
            return

        self.recording = False

        # Safely release video writer
        if self.video_writer:
            try:
                self.video_writer.release()
            except Exception as e:
                print(f"Error releasing video writer: {e}")
            finally:
                self.video_writer = None

        self.btn_camera_record.config(text="RECORD", bg='#F44336')
        self.recording_status_label.config(text="[NOT RECORDING]", fg='#666')

        duration = (datetime.now() - self.recording_start).total_seconds()
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        info_text = f"RECORDING SAVED\n"
        info_text += "=" * 30 + "\n\n"
        info_text += f"Duration:\n{minutes}m {seconds}s\n\n"
        info_text += f"Location:\n{self.recordings_folder / self.model_type.get()}\n\n"
        info_text += "Recording saved!\n"
        info_text += "Click 'Open Folder'\nto view videos."

        self.camera_info_text.delete('1.0', tk.END)
        self.camera_info_text.insert('1.0', info_text)

        self.update_status(f"[SAVED] Recording saved ({minutes}m {seconds}s)")
        messagebox.showinfo("Saved", f"Recording saved successfully!\nDuration: {minutes}m {seconds}s")

    # Video processing methods
    def select_input_video(self):
        """Select input video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if not file_path:
            return

        # Stop any playing video
        self.stop_input_video()
        self.stop_output_video()

        self.input_video_path = file_path

        # Get video info
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            # Display first frame
            ret, frame = cap.read()
            if ret:
                self.display_video_frame(frame, self.input_video_display)

            cap.release()

            # Update info
            info = f"File: {Path(file_path).name}\n"
            info += f"Resolution: {width}x{height} | FPS: {fps}\n"
            info += f"Duration: {int(duration//60)}m {int(duration%60)}s | Frames: {frame_count}"
            self.input_video_info.config(text=info)

            # Enable buttons
            self.btn_play_input.config(state=tk.NORMAL)
            self.btn_process.config(state=tk.NORMAL)
            self.video_progress_label.config(text="Ready to process")
        else:
            messagebox.showerror("Error", "Failed to open video file")

    def display_video_frame(self, frame, label_widget):
        """Display a video frame on a label - fixed size to prevent crushing"""
        # Fixed display height to match frame wrapper (300px with some margin)
        display_height = 285  # Slightly less than 300px frame to account for any padding
        h, w = frame.shape[:2]
        aspect = w / h
        display_width = int(display_height * aspect)

        # Limit width if too wide
        max_width = 500  # Maximum width to prevent overflow
        if display_width > max_width:
            display_width = max_width
            display_height = int(max_width / aspect)

        frame_resized = cv2.resize(frame, (display_width, display_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=img)

        label_widget.config(image=photo, text="")
        label_widget.image = photo

    def toggle_input_video(self):
        """Toggle play/pause for input video"""
        if self.video_playing_input:
            self.stop_input_video()
        else:
            self.play_input_video()

    def play_input_video(self):
        """Play input video"""
        if not self.input_video_path:
            return

        # Validate file exists
        if not Path(self.input_video_path).exists():
            messagebox.showerror(
                "Video Not Found",
                f"The video file no longer exists:\n\n{Path(self.input_video_path).name}\n\nIt may have been deleted or moved."
            )
            self.clear_all_videos()
            return

        self.video_playing_input = True
        self.btn_play_input.config(text="⏸ PAUSE")

        def play_loop():
            self.input_video_cap = cv2.VideoCapture(self.input_video_path)

            if not self.input_video_cap.isOpened():
                messagebox.showerror("Error", "Failed to open video file")
                self.stop_input_video()
                return

            while self.video_playing_input and self.input_video_cap.isOpened():
                ret, frame = self.input_video_cap.read()

                if not ret:
                    # Loop video
                    self.input_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if self.video_playing_input:
                    self.display_video_frame(frame, self.input_video_display)
                    self.root.update_idletasks()

                    # Control playback speed (30 fps)
                    import time
                    time.sleep(0.033)

            if self.input_video_cap:
                self.input_video_cap.release()

        threading.Thread(target=play_loop, daemon=True).start()

    def stop_input_video(self):
        """Stop input video playback"""
        self.video_playing_input = False
        if self.input_video_cap:
            self.input_video_cap.release()
            self.input_video_cap = None
        self.btn_play_input.config(text="▶ PLAY")

    def toggle_output_video(self):
        """Toggle play/pause for output video"""
        if self.video_playing_output:
            self.stop_output_video()
        else:
            self.play_output_video()

    def play_output_video(self):
        """Play output video"""
        if not self.output_video_path:
            return

        # Validate file exists
        if not Path(self.output_video_path).exists():
            messagebox.showerror(
                "Video Not Found",
                f"The output video file no longer exists:\n\n{Path(self.output_video_path).name}\n\nIt may have been deleted or moved."
            )
            # Clear only output
            self.output_video_path = None
            self.output_video_display.config(
                image='',
                text="No Processed Video\n\nProcess a video to see output"
            )
            self.output_video_info.config(text="No output yet")
            self.btn_play_output.config(state=tk.DISABLED)
            return

        self.video_playing_output = True
        self.btn_play_output.config(text="⏸ PAUSE")

        def play_loop():
            self.output_video_cap = cv2.VideoCapture(self.output_video_path)

            if not self.output_video_cap.isOpened():
                messagebox.showerror("Error", "Failed to open output video file")
                self.stop_output_video()
                return

            while self.video_playing_output and self.output_video_cap.isOpened():
                ret, frame = self.output_video_cap.read()

                if not ret:
                    # Loop video
                    self.output_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if self.video_playing_output:
                    self.display_video_frame(frame, self.output_video_display)
                    self.root.update_idletasks()

                    # Control playback speed (30 fps)
                    import time
                    time.sleep(0.033)

            if self.output_video_cap:
                self.output_video_cap.release()

        threading.Thread(target=play_loop, daemon=True).start()

    def stop_output_video(self):
        """Stop output video playback"""
        self.video_playing_output = False
        if self.output_video_cap:
            self.output_video_cap.release()
            self.output_video_cap = None
        self.btn_play_output.config(text="▶ PLAY")

    def process_video(self):
        """Process video file with classification"""
        if not self.predictor:
            messagebox.showwarning("Warning", "Please load a model first")
            return

        if not self.input_video_path:
            messagebox.showwarning("Warning", "Please select a video first")
            return

        # Validate input file exists
        if not Path(self.input_video_path).exists():
            messagebox.showerror(
                "Video Not Found",
                f"The input video file no longer exists:\n\n{Path(self.input_video_path).name}\n\nIt may have been deleted or moved."
            )
            self.clear_all_videos()
            return

        # Stop any playing videos
        self.stop_input_video()
        self.stop_output_video()

        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = Path(self.input_video_path).stem
        output_filename = f"{input_name}_processed_{timestamp}.mp4"
        output_path = self.processed_videos_folder / output_filename

        # Create loading window (matching camera style exactly)
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Processing Video")
        loading_window.resizable(False, False)
        loading_window.configure(bg='white')

        # Center the window on screen (same as camera)
        window_width = 400
        window_height = 220
        screen_width = loading_window.winfo_screenwidth()
        screen_height = loading_window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        loading_window.geometry(f'{window_width}x{window_height}+{x}+{y}')

        # Make it modal (same as camera)
        loading_window.transient(self.root)
        loading_window.grab_set()

        # Loading content (same structure as camera)
        loading_frame = tk.Frame(loading_window, bg='white')
        loading_frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=30)

        loading_label = tk.Label(
            loading_frame,
            text="Processing Video",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg='#9C27B0'
        )
        loading_label.pack(pady=(0, 15))

        loading_msg = tk.Label(
            loading_frame,
            text="Please be patient...\nThis may take several minutes",
            font=('Segoe UI', 11),
            bg='white',
            fg='#666',
            justify=tk.CENTER
        )
        loading_msg.pack(pady=(0, 20))

        # Progress bar with custom style (same as camera)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("VideoProcessing.Horizontal.TProgressbar",
                       foreground='#9C27B0',
                       background='#9C27B0',
                       troughcolor='#e0e0e0',
                       bordercolor='#ccc',
                       lightcolor='#9C27B0',
                       darkcolor='#9C27B0')

        progress_bar = ttk.Progressbar(
            loading_frame,
            style="VideoProcessing.Horizontal.TProgressbar",
            mode='determinate',
            length=320,
            maximum=100
        )
        progress_bar.pack(pady=10)

        # Status label (same as camera)
        status_label = tk.Label(
            loading_frame,
            text="Initializing...",
            font=('Segoe UI', 9),
            bg='white',
            fg='#999'
        )
        status_label.pack(pady=(10, 0))

        # Force update to show window immediately (same as camera)
        loading_window.update_idletasks()

        try:
            self.update_status("[PROCESSING] Processing video...")
            self.btn_process.config(state=tk.DISABLED)

            def process():
                try:
                    # Custom video processing with our overlay
                    cap = cv2.VideoCapture(str(self.input_video_path))

                    if not cap.isOpened():
                        raise Exception("Cannot open input video")

                    # Get video properties
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Update loading window
                    status_label.config(text=f"Processing {total_frames} frames...")
                    loading_window.update()

                    # Create video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                    if not out.isOpened():
                        raise Exception("Cannot create output video writer")

                    frame_count = 0

                    # Process every frame
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Process every 3rd frame for performance (adjust as needed)
                        if frame_count % 3 == 0:
                            processed_frame = self.process_video_frame(frame)
                        else:
                            # Use last processed frame overlay
                            if frame_count > 0:
                                processed_frame = self.process_video_frame(frame)
                            else:
                                processed_frame = frame

                        out.write(processed_frame)
                        frame_count += 1

                        # Update progress in loading window (every 30 frames)
                        if frame_count % 30 == 0 or frame_count == total_frames:
                            progress = (frame_count / total_frames) * 100
                            progress_bar['value'] = progress
                            status_label.config(text=f"{frame_count}/{total_frames} frames ({progress:.0f}%)")
                            loading_window.update()

                    cap.release()
                    out.release()

                    # Update status before closing
                    status_label.config(text="Complete! Saving video...")
                    loading_window.update()

                    import time
                    time.sleep(0.3)  # Brief pause to show completion

                    # Close loading window
                    if loading_window.winfo_exists():
                        loading_window.destroy()

                    # Update UI after processing
                    self.output_video_path = str(output_path)

                    # Display first frame of output
                    cap = cv2.VideoCapture(str(output_path))
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            self.display_video_frame(frame, self.output_video_display)

                        # Get video info
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        duration = frame_count / fps if fps > 0 else 0

                        info = f"File: {output_filename}\n"
                        info += f"Resolution: {width}x{height} | FPS: {fps}\n"
                        info += f"Duration: {int(duration//60)}m {int(duration%60)}s"
                        self.output_video_info.config(text=info)

                        cap.release()

                    self.btn_play_output.config(state=tk.NORMAL)
                    self.video_progress_label.config(text="✓ Processing complete!", fg='#4CAF50')
                    self.update_status("[COMPLETE] Video processed successfully")
                    self.btn_process.config(state=tk.NORMAL)

                    messagebox.showinfo("Success", f"Video processed successfully!\n\nSaved to:\n{output_path.name}")

                except Exception as e:
                    # Close loading window on error
                    if loading_window.winfo_exists():
                        loading_window.destroy()

                    messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
                    self.video_progress_label.config(text="✗ Processing failed", fg='#F44336')
                    self.btn_process.config(state=tk.NORMAL)
                    self.update_status("[ERROR] Processing failed")

            threading.Thread(target=process, daemon=True).start()

        except Exception as e:
            # Close loading window on error
            if loading_window.winfo_exists():
                loading_window.destroy()

            messagebox.showerror("Error", f"Failed to start processing:\n{str(e)}")
            self.update_status("[ERROR] Failed")
            self.btn_process.config(state=tk.NORMAL)

    def open_processed_videos_folder(self):
        """Open processed videos folder"""
        import os
        import platform

        self.processed_videos_folder.mkdir(parents=True, exist_ok=True)

        if platform.system() == 'Windows':
            os.startfile(self.processed_videos_folder)
        elif platform.system() == 'Darwin':
            os.system(f'open "{self.processed_videos_folder}"')
        else:
            os.system(f'xdg-open "{self.processed_videos_folder}"')

    def clear_all_videos(self):
        """Clear all videos from the interface"""
        # Stop any playing videos
        self.stop_input_video()
        self.stop_output_video()

        # Reset variables
        self.input_video_path = None
        self.output_video_path = None

        # Reset displays
        self.input_video_display.config(
            image='',
            text="No Video Loaded\n\nClick 'Select Video' to upload"
        )
        self.output_video_display.config(
            image='',
            text="No Processed Video\n\nProcess a video to see output"
        )

        # Reset info labels
        self.input_video_info.config(text="No video selected")
        self.output_video_info.config(text="No output yet")

        # Reset buttons
        self.btn_play_input.config(state=tk.DISABLED)
        self.btn_play_output.config(state=tk.DISABLED)
        self.btn_process.config(state=tk.DISABLED)

        # Reset progress
        self.video_progress_label.config(text="Ready to process videos", fg='#666')

        # Clear image references
        if hasattr(self.input_video_display, 'image'):
            delattr(self.input_video_display, 'image')
        if hasattr(self.output_video_display, 'image'):
            delattr(self.output_video_display, 'image')

        self.update_status("[READY] Videos cleared")

    def process_video_frame(self, frame):
        """Process a single video frame with prediction overlay (same as camera)"""
        temp_path = Path("temp_video_frame.jpg")
        cv2.imwrite(str(temp_path), frame)

        try:
            predicted_class, confidence, probs = self.predictor.predict_image(temp_path)

            height, width = frame.shape[:2]

            # Calculate responsive scaling factors
            scale_x = width / 640.0
            scale_y = height / 480.0
            scale = min(scale_x, scale_y)

            # Responsive dimensions
            overlay_height = int(205 * scale_y)
            overlay_width = int(295 * scale_x)
            margin_x = int(20 * scale_x)
            margin_y = int(35 * scale_y)

            # Font scales
            title_scale = 1.0 * scale
            text_scale = 0.55 * scale

            # Thickness scales
            title_thickness = max(2, int(2 * scale))
            text_thickness = max(1, int(1 * scale))

            # Add semi-transparent overlay at top-left
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)
                conf_text = "HIGH"
            elif confidence > 0.6:
                color = (0, 200, 255)
                conf_text = "MEDIUM"
            else:
                color = (0, 0, 255)
                conf_text = "LOW"

            # Draw predictions with responsive positioning
            y_pos = margin_y
            cv2.putText(frame, predicted_class.upper(), (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, title_scale, color, title_thickness, cv2.LINE_AA)

            y_pos += int(28 * scale_y)
            conf_display = f"Confidence: {confidence:.1%} ({conf_text})"
            cv2.putText(frame, conf_display, (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, text_scale, color, text_thickness, cv2.LINE_AA)

            y_pos += int(25 * scale_y)
            model_display = f"Model: {self.model_type.get().upper()}"
            cv2.putText(frame, model_display, (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, text_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)

            y_pos += int(28 * scale_y)
            cv2.putText(frame, "Top 3 Predictions:", (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

            top3_idx = np.argsort(probs)[-3:][::-1]
            y_pos += int(25 * scale_y)
            for i, idx in enumerate(top3_idx):
                cls = self.predictor.class_names[idx]
                prob = probs[idx]
                text_color = (0, 255, 0) if i == 0 else (255, 255, 255)
                text = f"{i+1}. {cls}: {prob:.1%}"
                cv2.putText(frame, text, (margin_x, y_pos),
                           cv2.FONT_HERSHEY_TRIPLEX, text_scale, text_color, text_thickness, cv2.LINE_AA)
                y_pos += int(25 * scale_y)

        except Exception as e:
            print(f"Frame processing error: {e}")
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return frame


    def open_recordings_folder(self):
        """Open recordings folder"""
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

    # ========== CAMERA CAPTURE METHODS ==========

    def toggle_capture_camera(self):
        """Start or stop the capture camera"""
        if not self.capture_camera_active:
            self.start_capture_camera()
        else:
            self.stop_capture_camera()

    def start_capture_camera(self):
        """Start the capture camera (no predictions)"""
        # Create loading window
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Starting Camera")
        loading_window.geometry("400x250")
        loading_window.resizable(False, False)
        loading_window.configure(bg='white')
        loading_window.transient(self.root)
        loading_window.grab_set()

        # Center window
        loading_window.update_idletasks()
        x = (loading_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (loading_window.winfo_screenheight() // 2) - (250 // 2)
        loading_window.geometry(f'400x250+{x}+{y}')

        # Variable to track if user cancelled
        user_cancelled = [False]

        # Handle X button click - stop camera initialization
        def on_closing():
            user_cancelled[0] = True
            if self.capture_cap:
                self.capture_cap.release()
                self.capture_cap = None
            self.capture_camera_active = False
            loading_window.destroy()
            self.update_status("[CANCELLED] Camera start cancelled")
            self.capture_status_label.config(text="Camera Ready", fg='#666')

        loading_window.protocol("WM_DELETE_WINDOW", on_closing)

        # Loading content
        loading_frame = tk.Frame(loading_window, bg='white')
        loading_frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=30)

        loading_label = tk.Label(
            loading_frame,
            text="Starting Camera",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg='#00BCD4'
        )
        loading_label.pack(pady=(0, 15))

        loading_msg = tk.Label(
            loading_frame,
            text="Please be patient...\nThis may take several seconds",
            font=('Segoe UI', 11),
            bg='white',
            fg='#666',
            justify=tk.CENTER
        )
        loading_msg.pack(pady=(0, 20))

        # Progress bar with custom style for visibility
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("CaptureLoading.Horizontal.TProgressbar",
                       foreground='#00BCD4',
                       background='#00BCD4',
                       troughcolor='#e0e0e0',
                       bordercolor='#ccc',
                       lightcolor='#00BCD4',
                       darkcolor='#00BCD4')

        progress = ttk.Progressbar(
            loading_frame,
            style="CaptureLoading.Horizontal.TProgressbar",
            mode='indeterminate',
            length=320
        )
        progress.pack(pady=10)
        progress.start(8)

        # Status label
        status_label = tk.Label(
            loading_frame,
            text="Initializing camera...",
            font=('Segoe UI', 9),
            bg='white',
            fg='#999'
        )
        status_label.pack(pady=(10, 0))

        # Force update to show window immediately
        loading_window.update_idletasks()

        def start_cam():
            success = False
            try:
                # Check if user cancelled before starting
                if user_cancelled[0]:
                    return

                status_label.config(text="Opening camera device...")
                loading_window.update()

                # Check again after update
                if user_cancelled[0]:
                    return

                self.capture_cap = cv2.VideoCapture(0)

                # Wait a moment for camera to initialize
                import time
                time.sleep(0.5)

                # Check if user cancelled during initialization
                if user_cancelled[0]:
                    if self.capture_cap:
                        self.capture_cap.release()
                        self.capture_cap = None
                    return

                status_label.config(text="Checking camera connection...")
                loading_window.update()

                if not self.capture_cap.isOpened():
                    if not user_cancelled[0]:
                        loading_window.destroy()
                        messagebox.showerror("Error", "Could not open camera")
                    return

                # Check again before proceeding
                if user_cancelled[0]:
                    if self.capture_cap:
                        self.capture_cap.release()
                        self.capture_cap = None
                    return

                # Try to read a test frame
                status_label.config(text="Testing camera feed...")
                loading_window.update()

                ret, test_frame = self.capture_cap.read()
                if not ret:
                    if not user_cancelled[0]:
                        loading_window.destroy()
                        messagebox.showerror("Error", "Cannot read from camera")
                    if self.capture_cap:
                        self.capture_cap.release()
                        self.capture_cap = None
                    return

                # Final check before activating
                if user_cancelled[0]:
                    if self.capture_cap:
                        self.capture_cap.release()
                        self.capture_cap = None
                    return

                status_label.config(text="Starting video stream...")
                loading_window.update()

                self.capture_camera_active = True
                self.btn_start_capture_camera.config(text="STOP CAMERA", bg='#F44336')
                self.btn_capture.config(state=tk.NORMAL)
                self.capture_status_label.config(text="Camera Active", fg='#4CAF50')
                self.update_status("[CAPTURE] Camera started")

                success = True

                # Small delay before closing to show completion
                status_label.config(text="Camera ready!")
                loading_window.update()
                time.sleep(0.3)

            except Exception as e:
                if not user_cancelled[0]:
                    loading_window.destroy()
                    messagebox.showerror("Error", f"Failed to start camera:\n{str(e)}")
                    self.update_status("[ERROR] Camera failed")
                if self.capture_cap:
                    self.capture_cap.release()
                    self.capture_cap = None
                self.capture_camera_active = False
                return
            finally:
                # Close loading window if it still exists and user didn't cancel
                if not user_cancelled[0] and loading_window.winfo_exists():
                    loading_window.destroy()

            # Start the camera feed AFTER loading window is closed
            if success and not user_cancelled[0]:
                self.update_capture_camera_feed()

        # Start in thread
        threading.Thread(target=start_cam, daemon=True).start()

    def stop_capture_camera(self):
        """Stop the capture camera"""
        self.capture_camera_active = False

        if self.capture_cap:
            self.capture_cap.release()
            self.capture_cap = None

        self.btn_start_capture_camera.config(text="START CAMERA", bg='#4CAF50')
        self.btn_capture.config(state=tk.DISABLED)
        self.capture_status_label.config(text="Camera Stopped", fg='#666')

        # Reset display
        self.capture_display.config(
            image='',
            text="Camera Inactive\n\nClick 'START CAMERA' to begin"
        )

        # Clear image reference
        if hasattr(self.capture_display, 'image'):
            delattr(self.capture_display, 'image')

        self.update_status("[READY] Camera stopped")

    def update_capture_camera_feed(self):
        """Update camera feed without predictions"""
        if not self.capture_camera_active or not self.capture_cap:
            return

        ret, frame = self.capture_cap.read()

        if ret:
            # Store the current frame for capture
            self.captured_frame = frame.copy()

            # Convert and display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Resize to fit display while maintaining aspect ratio
            display_width = self.capture_display.winfo_width()
            display_height = self.capture_display.winfo_height()

            if display_width > 1 and display_height > 1:
                img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.capture_display.imgtk = imgtk
            self.capture_display.config(image=imgtk, text='')

        # Schedule next update
        if self.capture_camera_active:
            self.root.after(30, self.update_capture_camera_feed)

    def capture_image(self):
        """Capture current frame and process it"""
        if not self.capture_camera_active or self.captured_frame is None:
            messagebox.showwarning("Warning", "Camera is not active!")
            return

        # Freeze the current frame
        frozen_frame = self.captured_frame.copy()

        # Display frozen frame
        frame_rgb = cv2.cvtColor(frozen_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        display_width = self.capture_display.winfo_width()
        display_height = self.capture_display.winfo_height()

        if display_width > 1 and display_height > 1:
            img.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.capture_display.imgtk = imgtk
        self.capture_display.config(image=imgtk, text='')

        # Update status - Captured
        self.capture_status_label.config(text="Captured!", fg='#00BCD4')
        self.btn_capture.config(state=tk.DISABLED)
        self.update_status("[CAPTURED] Image captured, processing...")

        # Brief delay to show "Captured!" status
        self.root.after(800, lambda: self._continue_processing(frozen_frame))

    def _continue_processing(self, frozen_frame):
        """Continue with processing after showing captured status"""
        # Update status - Processing
        self.capture_status_label.config(text="Processing...", fg='#FF9800')
        self.update_status("[PROCESSING] Classifying image...")

        # Create loading window
        loading_window = tk.Toplevel(self.root)
        loading_window.title("Processing")
        loading_window.geometry("400x250")
        loading_window.resizable(False, False)
        loading_window.configure(bg='white')
        loading_window.transient(self.root)
        loading_window.grab_set()

        # Center window
        loading_window.update_idletasks()
        x = (loading_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (loading_window.winfo_screenheight() // 2) - (250 // 2)
        loading_window.geometry(f'400x250+{x}+{y}')

        # Loading content
        loading_frame = tk.Frame(loading_window, bg='white')
        loading_frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=30)

        loading_label = tk.Label(
            loading_frame,
            text="Processing Image",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg='#00BCD4'
        )
        loading_label.pack(pady=(0, 15))

        loading_msg = tk.Label(
            loading_frame,
            text="Classifying image...\nPlease wait...",
            font=('Segoe UI', 11),
            bg='white',
            fg='#666',
            justify=tk.CENTER
        )
        loading_msg.pack(pady=(0, 20))

        # Progress bar with custom style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("ProcessingCapture.Horizontal.TProgressbar",
                       foreground='#00BCD4',
                       background='#00BCD4',
                       troughcolor='#e0e0e0',
                       bordercolor='#ccc',
                       lightcolor='#00BCD4',
                       darkcolor='#00BCD4')

        progress = ttk.Progressbar(
            loading_frame,
            style="ProcessingCapture.Horizontal.TProgressbar",
            mode='indeterminate',
            length=320
        )
        progress.pack(pady=10)
        progress.start(8)

        # Status label
        status_label = tk.Label(
            loading_frame,
            text="Running AI model...",
            font=('Segoe UI', 9),
            bg='white',
            fg='#999'
        )
        status_label.pack(pady=(10, 0))

        # Force update to show window immediately
        loading_window.update_idletasks()

        # Process in thread
        def process_capture():
            try:
                # Save frame temporarily
                temp_path = Path("temp_capture.jpg")
                cv2.imwrite(str(temp_path), frozen_frame)

                # Predict
                predicted_class, confidence, probs = self.predictor.predict_image(temp_path)

                # Create filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_folder = self.captures_folder / self.model_type.get()
                model_folder.mkdir(parents=True, exist_ok=True)

                save_path = model_folder / f"{predicted_class}_{timestamp}.jpg"

                # Add prediction overlay to frame (same as live camera)
                processed_frame = self.add_prediction_overlay(frozen_frame.copy(), predicted_class, confidence, probs)

                # Save the captured image with overlay
                cv2.imwrite(str(save_path), processed_frame)

                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

                # Close loading window
                if loading_window.winfo_exists():
                    loading_window.destroy()

                # Update UI in main thread with processed frame
                self.root.after(0, lambda: self.display_capture_result(
                    processed_frame, predicted_class, confidence, probs, save_path
                ))

            except Exception as e:
                # Close loading window on error
                if loading_window.winfo_exists():
                    loading_window.destroy()
                self.root.after(0, lambda: self.handle_capture_error(str(e)))

        threading.Thread(target=process_capture, daemon=True).start()

    def add_prediction_overlay(self, frame, predicted_class, confidence, probs):
        """Add prediction overlay to frame (same style as live camera)"""
        height, width = frame.shape[:2]

        # Calculate responsive scaling factors
        scale_x = width / 640.0
        scale_y = height / 480.0
        scale = min(scale_x, scale_y)

        # Responsive dimensions
        overlay_height = int(205 * scale_y)
        overlay_width = int(295 * scale_x)
        margin_x = int(20 * scale_x)
        margin_y = int(35 * scale_y)

        # Font scales
        title_scale = 1.0 * scale
        text_scale = 0.55 * scale

        # Thickness scales
        title_thickness = max(2, int(2 * scale))
        text_thickness = max(1, int(1 * scale))

        # Add semi-transparent overlay at top-left
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Color based on confidence
        if confidence > 0.8:
            color = (0, 255, 0)
            conf_text = "HIGH"
        elif confidence > 0.6:
            color = (0, 200, 255)
            conf_text = "MEDIUM"
        else:
            color = (0, 0, 255)
            conf_text = "LOW"

        # Draw predictions with responsive positioning
        y_pos = margin_y
        cv2.putText(frame, predicted_class.upper(), (margin_x, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, title_scale, color, title_thickness, cv2.LINE_AA)

        y_pos += int(28 * scale_y)
        conf_display = f"Confidence: {confidence:.1%} ({conf_text})"
        cv2.putText(frame, conf_display, (margin_x, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, text_scale, color, text_thickness, cv2.LINE_AA)

        y_pos += int(25 * scale_y)
        model_display = f"Model: {self.model_type.get().upper()}"
        cv2.putText(frame, model_display, (margin_x, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, text_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)

        y_pos += int(28 * scale_y)
        cv2.putText(frame, "Top 3 Predictions:", (margin_x, y_pos),
                   cv2.FONT_HERSHEY_TRIPLEX, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

        top3_idx = np.argsort(probs)[-3:][::-1]
        y_pos += int(25 * scale_y)
        for i, idx in enumerate(top3_idx):
            cls = self.predictor.class_names[idx]
            prob = probs[idx]
            text_color = (0, 255, 0) if i == 0 else (255, 255, 255)
            text = f"{i+1}. {cls}: {prob:.1%}"
            cv2.putText(frame, text, (margin_x, y_pos),
                       cv2.FONT_HERSHEY_TRIPLEX, text_scale, text_color, text_thickness, cv2.LINE_AA)
            y_pos += int(25 * scale_y)

        return frame

    def display_capture_result(self, processed_frame, predicted_class, confidence, probs, save_path):
        """Display the capture result with overlay in a new window"""
        try:
            # Update status
            self.capture_status_label.config(text="Processed!", fg='#4CAF50')

            # Confidence level
            if confidence > 0.8:
                conf_level = "HIGH"
                level_color = "confidence"
            elif confidence > 0.6:
                conf_level = "MEDIUM"
                level_color = "level"
            else:
                conf_level = "LOW"
                level_color = "label"

            # Update result text with colors
            self.capture_result_text.config(state=tk.NORMAL)
            self.capture_result_text.delete('1.0', tk.END)

            self.capture_result_text.insert(tk.END, "Prediction:\n", "label")
            self.capture_result_text.insert(tk.END, f"{predicted_class.upper()}\n\n", "prediction")

            self.capture_result_text.insert(tk.END, "Confidence: ", "label")
            self.capture_result_text.insert(tk.END, f"{confidence:.1%}\n", "confidence")

            self.capture_result_text.insert(tk.END, "Level: ", "label")
            self.capture_result_text.insert(tk.END, f"{conf_level}\n\n", level_color)

            self.capture_result_text.insert(tk.END, "Top 3:\n", "label")

            # Get top 3 predictions
            top3_idx = np.argsort(probs)[-3:][::-1]
            for i, idx in enumerate(top3_idx):
                cls = self.predictor.class_names[idx]
                prob = probs[idx]
                self.capture_result_text.insert(tk.END, f"{i + 1}. {cls}: {prob:.1%}\n")

            self.capture_result_text.insert(tk.END, f"\nModel: ", "label")
            self.capture_result_text.insert(tk.END, f"{self.model_type.get().upper()}\n\n", "model")

            self.capture_result_text.insert(tk.END, f"Saved: {save_path.name}")

            self.capture_result_text.config(state=tk.DISABLED)

            self.update_status(f"[SUCCESS] Image captured and saved: {predicted_class}")

            # Create result window to show the processed image
            result_window = tk.Toplevel(self.root)
            result_window.title("Capture Result")
            result_window.configure(bg='white')
            result_window.transient(self.root)
            result_window.grab_set()

            # Create frame for content
            content_frame = tk.Frame(result_window, bg='white')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

            # Title
            title_label = tk.Label(
                content_frame,
                text="Capture Processed Successfully!",
                font=('Segoe UI', 14, 'bold'),
                bg='white',
                fg='#4CAF50'
            )
            title_label.pack(pady=(0, 15))

            # Convert processed frame to display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            # Resize image to fit nicely in window (max 800x600)
            max_width = 800
            max_height = 600
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)

            # Display image
            img_label = tk.Label(content_frame, image=imgtk, bg='white')
            img_label.image = imgtk  # Keep reference
            img_label.pack(pady=(0, 15))

            # Info text
            info_text = f"Class: {predicted_class.upper()} | Confidence: {confidence:.1%} ({conf_level})\n"
            info_text += f"Saved to: {save_path.name}"

            info_label = tk.Label(
                content_frame,
                text=info_text,
                font=('Segoe UI', 10),
                bg='white',
                fg='#666',
                justify=tk.CENTER
            )
            info_label.pack(pady=(0, 15))

            # Buttons frame
            buttons_frame = tk.Frame(content_frame, bg='white')
            buttons_frame.pack(pady=(0, 0))

            # Function to open file location
            def open_file_location():
                import os
                import platform

                # Get the folder containing the saved file
                folder = save_path.parent

                # Open folder in file explorer
                if platform.system() == 'Windows':
                    # Open folder and select the file
                    os.system(f'explorer /select,"{save_path}"')
                elif platform.system() == 'Darwin':  # macOS
                    # Open folder and select the file
                    os.system(f'open -R "{save_path}"')
                else:  # Linux
                    # Just open the folder (selecting specific file is harder on Linux)
                    os.system(f'xdg-open "{folder}"')

                self.update_status(f"[INFO] Opened file location: {folder.name}")

            # OK button
            def on_ok():
                result_window.destroy()
                # Re-enable capture button
                self.btn_capture.config(state=tk.NORMAL)
                self.capture_status_label.config(text="Camera Active", fg='#4CAF50')

            # Open Folder button
            open_folder_button = ModernButton(
                buttons_frame,
                text="OPEN FOLDER",
                command=open_file_location,
                bg='#FF9800'
            )
            open_folder_button.pack(side=tk.LEFT, padx=5, ipadx=20)

            # OK button
            ok_button = ModernButton(
                buttons_frame,
                text="OK",
                command=on_ok,
                bg='#4CAF50'
            )
            ok_button.pack(side=tk.LEFT, padx=5, ipadx=40)

            # Bind Enter key to OK
            result_window.bind('<Return>', lambda e: on_ok())

            # Set window size based on image
            result_window.update_idletasks()
            window_width = img.width + 40
            window_height = img.height + 180  # Extra space for title, info, button

            # Center window
            x = (result_window.winfo_screenwidth() // 2) - (window_width // 2)
            y = (result_window.winfo_screenheight() // 2) - (window_height // 2)
            result_window.geometry(f'{window_width}x{window_height}+{x}+{y}')

        except Exception as e:
            self.handle_capture_error(str(e))

    def handle_capture_error(self, error_msg):
        """Handle errors during capture"""
        self.capture_status_label.config(text="Error", fg='#F44336')
        self.btn_capture.config(state=tk.NORMAL)

        self.capture_result_text.config(state=tk.NORMAL)
        self.capture_result_text.delete('1.0', tk.END)
        self.capture_result_text.insert('1.0', f"Error during capture:\n{error_msg}")
        self.capture_result_text.config(state=tk.DISABLED)

        self.update_status("[ERROR] Capture failed")
        messagebox.showerror("Error", f"Failed to process capture:\n{error_msg}")

    def open_captures_folder(self):
        """Open the captures folder"""
        import os
        import platform

        folder = self.captures_folder / self.model_type.get()
        folder.mkdir(parents=True, exist_ok=True)

        if platform.system() == 'Windows':
            os.startfile(folder)
        elif platform.system() == 'Darwin':
            os.system(f'open "{folder}"')
        else:
            os.system(f'xdg-open "{folder}"')

        self.update_status(f"[INFO] Opened folder: {folder}")

    def update_status(self, text):
        """Update status bar"""
        self.status_label.config(text=text)


def main():
    root = tk.Tk()
    app = OfficeItemsClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()