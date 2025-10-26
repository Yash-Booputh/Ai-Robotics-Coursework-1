"""
Professional GUI Application for Office Items Classification

Fully Responsive Features:
- Dynamic layouts that adapt to window size
- Scrollable content areas
- Flexible grids and frames
- No hidden content on resize
- Minimum window size constraints
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
        self.root.title("AI Robotics - Office Items Classifier")

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

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        # Title Bar (Fixed height, responsive width)
        title_frame = tk.Frame(self.root, bg='#1976D2')
        title_frame.pack(fill=tk.X)

        title_label = tk.Label(
            title_frame,
            text="Office Items Classifier",
            font=('Segoe UI', 20, 'bold'),
            bg='#1976D2',
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)

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
        self.create_video_screen()

        # Show home by default
        self.show_home()

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
            text="Welcome to Office Items Classifier",
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

        # Configure grid to be responsive
        buttons_container.grid_columnconfigure(0, weight=1, minsize=250)
        buttons_container.grid_columnconfigure(1, weight=1, minsize=250)
        buttons_container.grid_columnconfigure(2, weight=1, minsize=250)

        # Buttons with consistent styling
        btn_images = ModernButton(
            buttons_container,
            text="CLASSIFY IMAGES\n\nUpload single or\nmultiple images",
            command=self.switch_to_images,
            bg='#4CAF50'
        )
        btn_images.grid(row=0, column=0, padx=8, pady=8, sticky='nsew', ipady=15)

        btn_camera = ModernButton(
            buttons_container,
            text="LIVE CAMERA\n\nReal-time\nclassification",
            command=self.switch_to_camera,
            bg='#FF9800'
        )
        btn_camera.grid(row=0, column=1, padx=8, pady=8, sticky='nsew', ipady=15)

        btn_video = ModernButton(
            buttons_container,
            text="PROCESS VIDEO\n\nClassify\nvideo files",
            command=self.switch_to_video,
            bg='#9C27B0'
        )
        btn_video.grid(row=0, column=2, padx=8, pady=8, sticky='nsew', ipady=15)

        # Make buttons expand vertically too
        buttons_container.grid_rowconfigure(0, weight=1)

    def create_images_screen(self):
        """Simple 3-column fixed layout"""
        self.images_frame = tk.Frame(self.main_container, bg='#f5f5f5')

        # Main container
        content = tk.Frame(self.images_frame, bg='#f5f5f5')
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT panel - Upload Controls (FIXED WIDTH - 320px)
        left_panel = tk.Frame(content, bg='white', width=320, relief=tk.RAISED, borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 5))
        left_panel.pack_propagate(False)  # Keep fixed width

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
        self.file_count_label = list_label  # Store reference to update count

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

        # MIDDLE panel - Image Preview (EXPANDS)
        middle_panel = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=1)
        middle_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Image display header
        display_header = tk.Frame(middle_panel, bg='white')
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
        nav_frame = tk.Frame(middle_panel, bg='white')
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

        # Image container - FIXED BORDER ON ALL SIDES
        img_container = tk.Frame(middle_panel, bg='#e0e0e0', relief=tk.SUNKEN, borderwidth=2)
        img_container.pack(padx=15, pady=(8, 15), fill=tk.BOTH, expand=True)

        self.images_display_label = tk.Label(
            img_container,
            text="No image selected\n\nClick 'Select Images' to begin",
            font=('Segoe UI', 12),
            bg='#f5f5f5',
            fg='#999'
        )
        self.images_display_label.pack(padx=2, pady=2, fill=tk.BOTH, expand=True)

        # RIGHT panel - Classification Results (FIXED WIDTH - 320px)
        right_panel = tk.Frame(content, bg='white', width=320, relief=tk.RAISED, borderwidth=1)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(5, 0))
        right_panel.pack_propagate(False)  # Keep fixed width

        results_label = tk.Label(
            right_panel,
            text="Classification Results",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg='#333'
        )
        results_label.pack(pady=10)

        results_info = tk.Label(
            right_panel,
            text="Results will appear here after classification",
            font=('Segoe UI', 8),
            bg='white',
            fg='#999',
            wraplength=280
        )
        results_info.pack(pady=(0, 5))

        results_frame = tk.Frame(right_panel, bg='white')
        results_frame.pack(padx=10, pady=(0, 10), fill=tk.BOTH, expand=True)

        results_scroll = tk.Scrollbar(results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.images_results_text = tk.Text(
            results_frame,
            font=('Consolas', 8),
            wrap=tk.WORD,
            yscrollcommand=results_scroll.set,
            bg='#fafafa',
            fg='#000000'  # Set default text color
        )
        self.images_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.images_results_text.yview)

        # Configure tags for bold text with colors
        self.images_results_text.tag_configure("bold", font=('Consolas', 8, 'bold'), foreground='#000000')
        self.images_results_text.tag_configure("class_name", font=('Consolas', 9, 'bold'), foreground='#1976D2')
        self.images_results_text.tag_configure("confidence", font=('Consolas', 9, 'bold'), foreground='#2E7D32')
        self.images_results_text.tag_configure("header", font=('Consolas', 9, 'bold'), foreground='#5E35B1')

        # Initial help text
        initial_text = "How to use:\n\n"
        initial_text += "1. Select up to 10 images\n"
        initial_text += "2. Preview them here\n"
        initial_text += "3. Click 'Classify Images'\n"
        initial_text += "4. View results here\n\n"
        initial_text += "Waiting for images..."
        self.images_results_text.insert('1.0', initial_text)

        # Storage
        self.selected_image_paths = []
        self.image_predictions = {}
        self.current_image_index = 0

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

    def create_video_screen(self):
        """Responsive video processing screen"""
        # Use scrollable frame
        scroll_frame = ScrollableFrame(self.main_container, bg='#f5f5f5')
        self.video_frame = scroll_frame

        content = scroll_frame.scrollable_frame

        # Header
        header = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2)
        header.pack(fill=tk.X, padx=20, pady=20)

        title = tk.Label(
            header,
            text="Process Video File",
            font=('Segoe UI', 18, 'bold'),
            bg='white',
            fg='#9C27B0'
        )
        title.pack(pady=15)

        # Content area
        main_content = tk.Frame(content, bg='white', relief=tk.RAISED, borderwidth=2)
        main_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Instructions
        instructions = tk.Label(
            main_content,
            text="Upload a video file to process with AI classification",
            font=('Segoe UI', 12),
            bg='white',
            fg='#666'
        )
        instructions.pack(pady=25)

        # Process button
        self.btn_process_video = ModernButton(
            main_content,
            text="SELECT & PROCESS VIDEO",
            command=self.process_video,
            bg='#9C27B0'
        )
        self.btn_process_video.pack(pady=15, padx=20, ipadx=20, ipady=5)

        # Info
        info_text = "Supported formats: MP4, AVI, MOV, MKV\n"
        info_text += "Processing adds predictions to each frame\n"
        info_text += "Output saved as MP4 file"

        info_label = tk.Label(
            main_content,
            text=info_text,
            font=('Segoe UI', 10),
            bg='white',
            fg='#666',
            justify=tk.LEFT
        )
        info_label.pack(pady=15)

        # Results area
        results_label = tk.Label(
            main_content,
            text="Processing Results",
            font=('Segoe UI', 14, 'bold'),
            bg='white',
            fg='#333'
        )
        results_label.pack(pady=(25, 10))

        results_frame = tk.Frame(main_content, bg='white')
        results_frame.pack(padx=30, pady=(0, 30), fill=tk.BOTH, expand=True)

        results_scroll = tk.Scrollbar(results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.video_results_text = tk.Text(
            results_frame,
            font=('Consolas', 10),
            wrap=tk.WORD,
            yscrollcommand=results_scroll.set,
            bg='#f9f9f9',
            height=12
        )
        self.video_results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.video_results_text.yview)

        initial_text = "Video Processing\n"
        initial_text += "=" * 50 + "\n\n"
        initial_text += "Waiting for video file...\n\n"
        initial_text += "Click 'Select & Process Video' to begin."

        self.video_results_text.insert('1.0', initial_text)

    # Navigation methods
    def show_home(self):
        self.hide_all_frames()
        self.home_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "home"
        self.update_status("[HOME] Select a mode")

    def switch_to_images(self):
        self.hide_all_frames()
        self.images_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "images"
        self.update_status("[IMAGE CLASSIFICATION] Mode active")

    def switch_to_camera(self):
        self.hide_all_frames()
        self.camera_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "camera"
        self.update_status("[LIVE CAMERA] Mode active")

    def switch_to_video(self):
        self.hide_all_frames()
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.current_mode = "video"
        self.update_status("[VIDEO PROCESSING] Mode active")

    def hide_all_frames(self):
        """Hide all mode frames"""
        for frame in [self.home_frame, self.images_frame, self.camera_frame, self.video_frame]:
            frame.pack_forget()

    # Model loading
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

        # Clear previous results
        self.images_results_text.delete('1.0', tk.END)
        result_text = f"[SUCCESS] {count} image(s) loaded successfully\n\n"
        result_text += "Instructions:\n"
        result_text += f"- {count} image(s) ready\n"
        result_text += "- Use < > to navigate\n" if count > 1 else ""
        result_text += "- Click 'Classify Images' to start\n\n"
        result_text += "Waiting for classification..."
        self.images_results_text.insert('1.0', result_text)

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

        except Exception as e:
            self.images_display_label.config(text=f"Error loading image:\n{str(e)}")
            self.image_info_label.config(text="")

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
            results_text = []  # Store as list of (text, tag) tuples

            # Header
            results_text.append(("CLASSIFICATION RESULTS\n", "bold"))
            results_text.append(("=" * 46 + "\n\n", None))
            results_text.append((f"Model: {self.model_type.get().upper()}\n", None))
            results_text.append((f"Images: {len(self.selected_image_paths)}\n\n", None))
            results_text.append(("=" * 46 + "\n\n", None))

            for idx, image_path in enumerate(self.selected_image_paths, 1):
                try:
                    filename = Path(image_path).name
                    predicted_class, confidence, probs = self.predictor.predict_image(image_path)

                    self.image_predictions[image_path] = {
                        'class': predicted_class,
                        'confidence': confidence,
                        'probs': probs
                    }

                    results_text.append((f"Image {idx}: {filename}\n", None))
                    results_text.append(("   Class: ", None))
                    results_text.append((f"{predicted_class}\n", "class_name"))
                    results_text.append(("   Confidence: ", None))
                    results_text.append((f"{confidence*100:.2f}%\n", "confidence"))

                    # Top 3 predictions
                    top3_idx = np.argsort(probs)[-3:][::-1]
                    results_text.append(("   Top 3:\n", "bold"))
                    for i, top_idx in enumerate(top3_idx, 1):
                        cls = self.predictor.class_names[top_idx]
                        prob = probs[top_idx]
                        results_text.append((f"      {i}. ", None))
                        results_text.append((f"{cls}", "bold"))
                        results_text.append((f": {prob*100:.2f}%\n", None))

                    results_text.append(("\n", None))

                except Exception as e:
                    results_text.append((f"[ERROR] Image {idx}: {str(e)}\n\n", None))

            # Summary statistics
            if self.image_predictions:
                results_text.append(("=" * 46 + "\n", None))
                results_text.append(("SUMMARY\n", "bold"))
                results_text.append(("=" * 46 + "\n\n", None))

                all_classes = [pred['class'] for pred in self.image_predictions.values()]
                class_counts = Counter(all_classes)

                results_text.append(("Class Distribution:\n", "bold"))
                for cls, count in class_counts.most_common():
                    percentage = (count / len(all_classes)) * 100
                    results_text.append((f"  - {cls}: ", None))
                    results_text.append((f"{count}", "bold"))
                    results_text.append((f" ({percentage:.1f}%)\n", None))

                avg_confidence = np.mean([pred['confidence'] for pred in self.image_predictions.values()])
                results_text.append(("\nAverage Confidence: ", None))
                results_text.append((f"{avg_confidence*100:.2f}%\n", "confidence"))

            # Clear and insert with formatting
            self.images_results_text.delete('1.0', tk.END)
            for text, tag in results_text:
                if tag:
                    self.images_results_text.insert(tk.END, text, tag)
                else:
                    self.images_results_text.insert(tk.END, text)

            self.btn_classify_images.config(state=tk.NORMAL)
            self.update_status(f"[COMPLETE] Classified {len(self.selected_image_paths)} image(s)")

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

            # Write to video file if recording
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

            # Get available display space dynamically
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

            height, width = frame.shape[:2]

            # Calculate responsive scaling factors
            scale_x = width / 640.0  # Base width
            scale_y = height / 480.0  # Base height
            scale = min(scale_x, scale_y)  # Use minimum to ensure everything fits

            # Responsive dimensions
            overlay_height = int(270 * scale_y)
            margin_x = int(20 * scale_x)
            margin_y = int(45 * scale_y)

            # Font scales
            title_scale = 1.2 * scale
            text_scale = 0.6 * scale

            # Thickness scales
            title_thickness = max(2, int(3 * scale))
            text_thickness = max(1, int(2 * scale))

            # Add semi-transparent overlay at top
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
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
                       cv2.FONT_HERSHEY_SIMPLEX, title_scale, color, title_thickness)

            y_pos += int(35 * scale_y)
            conf_display = f"Confidence: {confidence:.1%} ({conf_text})"
            cv2.putText(frame, conf_display, (margin_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, text_thickness)

            y_pos += int(30 * scale_y)
            model_display = f"Model: {self.model_type.get().upper()}"
            cv2.putText(frame, model_display, (margin_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), text_thickness)

            y_pos += int(35 * scale_y)
            cv2.putText(frame, "Top 3 Predictions:", (margin_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)

            top3_idx = np.argsort(probs)[-3:][::-1]
            y_pos += int(30 * scale_y)
            for i, idx in enumerate(top3_idx):
                cls = self.predictor.class_names[idx]
                prob = probs[idx]
                text_color = (0, 255, 0) if i == 0 else (255, 255, 255)
                text = f"{i+1}. {cls}: {prob:.1%}"
                cv2.putText(frame, text, (margin_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)
                y_pos += int(30 * scale_y)

            # Recording indicator (responsive positioning)
            if self.recording:
                rec_radius = int(12 * scale)
                rec_x = width - int(40 * scale_x)
                rec_y = int(30 * scale_y)
                cv2.circle(frame, (rec_x, rec_y), rec_radius, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (width - int(100 * scale_x), int(40 * scale_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 255), text_thickness)

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
    def process_video(self):
        """Process video file with classification"""
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
            self.update_status("[PROCESSING] Processing video...")

            result_text = "PROCESSING VIDEO...\n"
            result_text += "=" * 60 + "\n\n"
            result_text += f"Input: {Path(file_path).name}\n"
            result_text += f"Output: {Path(output_path).name}\n\n"
            result_text += "This may take several minutes...\nPlease wait."

            self.video_results_text.delete('1.0', tk.END)
            self.video_results_text.insert('1.0', result_text)

            def process():
                self.predictor.predict_from_video(file_path, output_path, display=False)

                result_text = "VIDEO PROCESSED\n"
                result_text += "=" * 60 + "\n\n"
                result_text += f"Input: {Path(file_path).name}\n"
                result_text += f"Output: {Path(output_path).name}\n"
                result_text += f"Model: {self.model_type.get().upper()}\n\n"
                result_text += "Processing complete!\nVideo saved successfully."

                self.video_results_text.delete('1.0', tk.END)
                self.video_results_text.insert('1.0', result_text)
                self.update_status("[COMPLETE] Video processed")

                messagebox.showinfo("Success", f"Video saved to:\n{output_path}")

            threading.Thread(target=process, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed: {str(e)}")
            self.update_status("[ERROR] Failed")

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

    def update_status(self, text):
        """Update status bar"""
        self.status_label.config(text=text)


def main():
    root = tk.Tk()
    app = OfficeItemsClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()