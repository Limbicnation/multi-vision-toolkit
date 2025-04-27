import argparse
import json
import os
import logging
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox
import shutil
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import torch
import re

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Theme constants
DARK_BG = "#1e1e1e"
DARK_FG = "#ffffff"
DARK_ACCENT = "#007acc"
DARK_SECONDARY = "#2d2d2d"
DARK_BUTTON = "#3c3c3c"
DARK_BUTTON_ACTIVE = "#505050"
DARK_APPROVE_BTN = "#2d9440"
DARK_REJECT_BTN = "#9e3a3a"

LIGHT_BG = "#f0f0f0"
LIGHT_FG = "#000000"
LIGHT_ACCENT = "#0078d7"
LIGHT_SECONDARY = "#e0e0e0"
LIGHT_BUTTON = "#dddddd"
LIGHT_BUTTON_ACTIVE = "#cccccc"
LIGHT_APPROVE_BTN = "#2ecc71"
LIGHT_REJECT_BTN = "#e74c3c"

def load_environment_from_dotenv():
    """Load environment variables from .env file if available"""
    try:
        env_path = Path('.env')
        if env_path.exists():
            logger.info(f"Loading environment from .env file: {env_path}")
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    key, value = line.split('=', 1)
                    os.environ[key] = value
            logger.info("Successfully loaded environment variables from .env file")
            return True
        return False
    except Exception as e:
        logger.warning(f"Error loading .env file: {str(e)}")
        return False

# Load HF token from environment at module level
load_environment_from_dotenv()
if "HF_TOKEN" in os.environ:
    logger.info("HuggingFace token found in environment variables")
    os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    try:
        # Try to set HF_HUB_TOKEN as well, which is sometimes used
        os.environ["HF_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    except:
        pass

# Only import models after setting up environment
try:
    from models.florence_model import Florence2Model
    from models.janus_model import JanusModel
    from models.qwen_model import QwenModel
except Exception as e:
    logger.error(f"Error importing models: {str(e)}")

@dataclass
class ImageAnalysisResult:
    """Data class to store image analysis results"""
    description: str
    clean_caption: Optional[str] = None

class ThemeManager:
    """Manages application themes (light/dark mode)"""
    
    def __init__(self, root):
        self.root = root
        self.theme = "light"  # Default theme
        self.load_theme_preference()
        
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        self.theme = "dark" if self.theme == "light" else "light"
        self.apply_theme()
        self.save_theme_preference()
        return self.theme
        
    def apply_theme(self):
        """Apply the current theme to the application"""
        style = ttk.Style()
        
        if self.theme == "dark":
            # Configure dark mode
            self.root.configure(bg=DARK_BG)
            style.configure("TFrame", background=DARK_BG)
            style.configure("TLabel", background=DARK_BG, foreground=DARK_FG)
            style.configure("TButton", foreground=DARK_FG)
            style.map("TButton", background=[("active", DARK_BUTTON_ACTIVE)])
            style.configure("TCombobox", fieldbackground=DARK_SECONDARY, foreground=DARK_FG)
            style.map("TCombobox", fieldbackground=[("readonly", DARK_SECONDARY)])
            style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=DARK_ACCENT, background=DARK_BG)
            style.configure("Caption.TLabel", foreground=DARK_FG, padding=10, background=DARK_SECONDARY)
            style.configure("Primary.TButton", foreground=DARK_FG)
            style.map("Primary.TButton", background=[("active", DARK_APPROVE_BTN)])
            style.configure("Reject.TButton", foreground=DARK_FG)
            style.map("Reject.TButton", background=[("active", DARK_REJECT_BTN)])
            style.configure("StatusBar.TFrame", background=DARK_SECONDARY)
            style.configure("StatusBar.TLabel", background=DARK_SECONDARY, foreground=DARK_FG)
            style.configure("TPanedwindow", background=DARK_BG)
            style.configure("TNotebook", background=DARK_BG)
            style.configure("TNotebook.Tab", background=DARK_SECONDARY, foreground=DARK_FG, padding=[10, 2])
            style.map("TNotebook.Tab", background=[("selected", DARK_ACCENT)], foreground=[("selected", DARK_FG)])
            style.configure("InfoFrame.TFrame", background=DARK_SECONDARY)
            
            # Set text widgets
            for text_widget in self._find_text_widgets(self.root):
                text_widget.config(bg=DARK_SECONDARY, fg=DARK_FG, insertbackground=DARK_FG)
                
                # Configure text tags
                text_widget.tag_configure("heading", foreground=DARK_ACCENT, font=("Segoe UI", 11, "bold"))
                text_widget.tag_configure("subheading", foreground="#00aaff", font=("Segoe UI", 10, "bold"))
                text_widget.tag_configure("important", foreground="#ffaa00", font=("Segoe UI", 10, "bold"))
                text_widget.tag_configure("object", foreground="#00ccaa", font=("Segoe UI", 10))
                text_widget.tag_configure("tag", foreground="#cc88ff", font=("Segoe UI", 10))
        else:
            # Configure light mode
            self.root.configure(bg=LIGHT_BG)
            style.configure("TFrame", background=LIGHT_BG)
            style.configure("TLabel", background=LIGHT_BG, foreground=LIGHT_FG)
            style.configure("TButton", foreground=LIGHT_FG)
            style.map("TButton", background=[("active", LIGHT_BUTTON_ACTIVE)])
            style.configure("TCombobox", fieldbackground="white", foreground=LIGHT_FG)
            style.map("TCombobox", fieldbackground=[("readonly", "white")])
            style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=LIGHT_ACCENT, background=LIGHT_BG)
            style.configure("Caption.TLabel", foreground=LIGHT_FG, padding=10, background=LIGHT_SECONDARY)
            style.configure("Primary.TButton", foreground="white")
            style.map("Primary.TButton", background=[("active", LIGHT_APPROVE_BTN)])
            style.configure("Reject.TButton", foreground="white")
            style.map("Reject.TButton", background=[("active", LIGHT_REJECT_BTN)])
            style.configure("StatusBar.TFrame", background=LIGHT_SECONDARY)
            style.configure("StatusBar.TLabel", background=LIGHT_SECONDARY, foreground=LIGHT_FG)
            style.configure("TPanedwindow", background=LIGHT_BG)
            style.configure("TNotebook", background=LIGHT_BG)
            style.configure("TNotebook.Tab", background=LIGHT_SECONDARY, foreground=LIGHT_FG, padding=[10, 2])
            style.map("TNotebook.Tab", background=[("selected", LIGHT_ACCENT)], foreground=[("selected", "white")])
            style.configure("InfoFrame.TFrame", background=LIGHT_SECONDARY)
            
            # Set text widgets
            for text_widget in self._find_text_widgets(self.root):
                text_widget.config(bg="white", fg=LIGHT_FG, insertbackground=LIGHT_FG)
                
                # Configure text tags
                text_widget.tag_configure("heading", foreground=LIGHT_ACCENT, font=("Segoe UI", 11, "bold"))
                text_widget.tag_configure("subheading", foreground="#0066cc", font=("Segoe UI", 10, "bold"))
                text_widget.tag_configure("important", foreground="#cc6600", font=("Segoe UI", 10, "bold"))
                text_widget.tag_configure("object", foreground="#008866", font=("Segoe UI", 10))
                text_widget.tag_configure("tag", foreground="#8844cc", font=("Segoe UI", 10))
    
    def _find_text_widgets(self, parent):
        """Find all Text widgets in the widget hierarchy"""
        result = []
        for widget in parent.winfo_children():
            if isinstance(widget, tk.Text):
                result.append(widget)
            result.extend(self._find_text_widgets(widget))
        return result
        
    def save_theme_preference(self):
        """Save theme preference to a settings file"""
        try:
            # Create settings directory if it doesn't exist
            settings_dir = Path("settings")
            settings_dir.mkdir(exist_ok=True)
            
            # Save the theme preference
            with open(settings_dir / "theme.json", "w") as f:
                json.dump({"theme": self.theme}, f)
        except Exception as e:
            logger.warning(f"Failed to save theme preference: {e}")
            
    def load_theme_preference(self):
        """Load theme preference from settings file"""
        try:
            theme_file = Path("settings/theme.json")
            if theme_file.exists():
                with open(theme_file, "r") as f:
                    data = json.load(f)
                    self.theme = data.get("theme", "light")
        except Exception as e:
            logger.warning(f"Failed to load theme preference: {e}")

class ModelManager:
    """Manages model initialization and switching"""
    def __init__(self):
        self.models: Dict[str, object] = {}
        self._current_model = None
        self._current_model_name = None
        
    def initialize_model(self, model_name: str) -> object:
        """Initialize a model with error handling and caching"""
        try:
            if model_name in self.models:
                logger.info(f"Using cached model: {model_name}")
                return self.models[model_name]
            
            logger.info(f"Initializing new model: {model_name}")
            if model_name.lower() == "florence2":
                model = Florence2Model()
            elif model_name.lower() == "janus":
                try:
                    model = JanusModel()
                except Exception as e:
                    logger.error(f"Failed to load Janus model: {str(e)}")
                    messagebox.showerror("Model Error", 
                        "Failed to load Janus model. Please update transformers:\n"
                        "pip install --upgrade transformers\n"
                        "or install from source:\n"
                        "pip install git+https://github.com/huggingface/transformers.git")
                    raise
            elif model_name.lower() == "qwen":
                try:
                    model = QwenModel()
                except Exception as e:
                    logger.error(f"Failed to load Qwen model: {str(e)}")
                    messagebox.showerror("Model Error", 
                        "Failed to load Qwen2.5-VL model. Please install requirements:\n"
                        "pip install --upgrade transformers accelerate\n"
                        "pip install qwen-vl-utils[decord]==0.0.8\n"
                        "or install from source:\n"
                        "pip install git+https://github.com/huggingface/transformers.git")
                    raise
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            self.models[model_name] = model
            return model
            
        except Exception as e:
            logger.error(f"Error initializing model {model_name}: {str(e)}")
            raise

    def get_model(self, model_name: str) -> object:
        """Get a model, initializing if necessary"""
        if self._current_model_name != model_name:
            self._current_model = self.initialize_model(model_name)
            self._current_model_name = model_name
        return self._current_model

class DatasetPreparator:
    """Handles dataset preparation and file operations"""
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        
    def is_supported_image(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.supported_formats
        
    def create_caption_file(self, image_path: str, caption: str) -> str:
        try:
            txt_path = Path(image_path).with_suffix('.txt')
            txt_path.write_text(caption, encoding='utf-8')
            return str(txt_path)
        except Exception as e:
            logger.error(f"Error creating caption file: {str(e)}")
            raise

class ReviewGUI:
    """Main GUI application for reviewing images"""

    def _update_caption_files(self, img_path, description, clean_caption):
        """Update JSON and TXT files with new caption"""
        try:
            # Get base name and JSON path
            base_name = img_path.stem
            json_path = img_path.parent / f"{base_name}_for_review.json"
            
            # Update JSON file
            data = {"results": {"caption": description}}
            json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            
            # Update TXT file
            if clean_caption:
                self.dataset_prep.create_caption_file(str(img_path), clean_caption)
                
            return True
        except Exception as e:
            logger.error(f"Error updating caption files: {str(e)}")
            return False

    def __init__(
        self, 
        review_dir: str, 
        approved_dir: str, 
        rejected_dir: str, 
        trigger_word: Optional[str] = None,
        model_name: str = "florence2"
    ):
        logger.info(f"Initializing ReviewGUI with model: {model_name}")
        self.review_dir = Path(review_dir)
        self.approved_dir = Path(approved_dir)
        self.rejected_dir = Path(rejected_dir)
        self.trigger_word = trigger_word
        
        self.model_name = model_name
        self.model_manager = ModelManager()
        try:
            self.model = self.model_manager.get_model(model_name)
        except Exception as e:
            logger.error(f"Failed to initialize model {model_name}: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize model {model_name}. Falling back to Florence2.")
            self.model_name = "florence2"
            self.model = self.model_manager.get_model("florence2")
        
        self.dataset_prep = DatasetPreparator()
        
        # Create directories
        for dir_path in [self.review_dir, self.approved_dir, self.rejected_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {dir_path}")
        
        self.root = tk.Tk()
        self.root.title("Multi-Vision Toolkit")
        self.root.geometry("1280x800")
        self.root.minsize(1024, 768)
        
        # Initialize theme manager
        self.theme_manager = ThemeManager(self.root)
        
        self.setup_gui()
        self.load_items()
        
        # Apply theme after GUI is set up
        self.theme_manager.apply_theme()

    def setup_gui(self):
        """Setup GUI components with error handling"""
        try:
            # Create main container
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create header
            self._setup_header()
            
            # Create content area with resizable panels
            self._setup_content_area()
            
            # Create status bar
            self._setup_status_bar()
            
            # Setup keyboard shortcuts
            self._setup_keyboard_shortcuts()
            
        except Exception as e:
            logger.error(f"Error setting up GUI: {str(e)}")
            raise
    
    def _setup_header(self):
        """Setup header with app title, model selector and theme toggle"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=10)
        
        # App title
        title_label = ttk.Label(
            header_frame, 
            text="Multi-Vision Toolkit", 
            style="Header.TLabel",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(side=tk.LEFT, padx=10)
        
        # Controls on the right
        controls_frame = ttk.Frame(header_frame)
        controls_frame.pack(side=tk.RIGHT)
        
        # Model selection
        ttk.Label(controls_frame, text="Model:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value=self.model_name)
        model_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.model_var,
            values=["florence2", "janus", "qwen"],
            state="readonly",
            width=10
        )
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        # Theme toggle
        theme_icon = "🌙" if self.theme_manager.theme == "light" else "☀️"
        self.theme_btn = ttk.Button(
            controls_frame,
            text=f"{theme_icon} Theme",
            command=self._toggle_theme
        )
        self.theme_btn.pack(side=tk.LEFT, padx=10)
    
    def _setup_content_area(self):
        """Setup the main content area with resizable panels"""
        # Create a PanedWindow for resizable panels
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.VERTICAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Image display area
        image_frame = ttk.Frame(self.paned_window)
        
        # Image container for centering
        self.image_container = ttk.Frame(image_frame)
        self.image_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.img_label = ttk.Label(self.image_container)
        self.img_label.pack(fill=tk.BOTH, expand=True)
        
        # Analysis area
        analysis_frame = ttk.Frame(self.paned_window)
        
        # Add frames to paned window
        self.paned_window.add(image_frame, weight=3)  # 75% of space
        self.paned_window.add(analysis_frame, weight=1)  # 25% of space
        
        # Analysis header
        ttk.Label(
            analysis_frame, 
            text="Image Analysis", 
            style="Header.TLabel",
            font=("Segoe UI", 12, "bold")
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        # Caption frame
        caption_frame = ttk.Frame(analysis_frame, style="InfoFrame.TFrame")
        caption_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Caption text widget
        self.caption_text = tk.Text(
            caption_frame, 
            wrap=tk.WORD, 
            height=5, 
            font=("Segoe UI", 10),
            padx=10,
            pady=10,
            relief=tk.FLAT,
            borderwidth=0
        )
        self.caption_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Make caption text editable (instead of DISABLED)
        self.caption_text.config(state=tk.NORMAL)
        
        # Add edit button to caption frame
        edit_button_frame = ttk.Frame(caption_frame)
        edit_button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.save_caption_btn = ttk.Button(
            edit_button_frame,
            text="Save Edited Caption",
            command=self._save_caption_edits
        )
        self.save_caption_btn.pack(side=tk.RIGHT, padx=5)
        
        # Metadata frame
        metadata_frame = ttk.Frame(analysis_frame, style="InfoFrame.TFrame")
        metadata_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File info grid
        info_grid = ttk.Frame(metadata_frame)
        info_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # Filename
        ttk.Label(info_grid, text="Filename:", width=12, anchor=tk.E).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.filename_label = ttk.Label(info_grid, text="")
        self.filename_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Dimensions
        ttk.Label(info_grid, text="Dimensions:", width=12, anchor=tk.E).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.dimensions_label = ttk.Label(info_grid, text="")
        self.dimensions_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Model info
        ttk.Label(info_grid, text="Model:", width=12, anchor=tk.E).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.model_label = ttk.Label(info_grid, text=self.model_name)
        self.model_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Control buttons
        controls_frame = ttk.Frame(analysis_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=10)
    
        # Add toolbar frame for additional controls
        toolbar_frame = ttk.Frame(controls_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    
        # Quality selection for captions
        quality_frame = ttk.Frame(toolbar_frame)
        quality_frame.pack(side=tk.LEFT, padx=10)
    
        ttk.Label(quality_frame, text="Caption Quality:").pack(side=tk.LEFT, padx=5)
        self.quality_var = tk.StringVar(value="standard")
        quality_combo = ttk.Combobox(
            quality_frame, 
            textvariable=self.quality_var,
            values=["standard", "detailed", "creative"],
            state="readonly",
            width=10
        )
        quality_combo.pack(side=tk.LEFT, padx=5)
        quality_combo.bind('<<ComboboxSelected>>', self._on_quality_change)
    
        # Action buttons
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.approve_btn = ttk.Button(
            action_frame,
            text="✓ Approve (A)",
            command=self.approve,
            style="Primary.TButton",
            width=15
        )
        self.approve_btn.pack(side=tk.LEFT, padx=5)
        
        self.reject_btn = ttk.Button(
            action_frame,
            text="✗ Reject (R)",
            command=self.reject,
            style="Reject.TButton",
            width=15
        )
        self.reject_btn.pack(side=tk.LEFT, padx=5)
        
        # Navigation buttons
        nav_frame = ttk.Frame(controls_frame)
        nav_frame.pack(side=tk.RIGHT, padx=5)
        
        self.prev_btn = ttk.Button(
            nav_frame,
            text="◀ Previous",
            command=self._prev_image,
            width=12
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = ttk.Button(
            nav_frame,
            text="Next ▶",
            command=self._next_image,
            width=12
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
    
    def _setup_status_bar(self):
        """Setup status bar at the bottom of the window"""
        status_frame = ttk.Frame(self.root, style="StatusBar.TFrame")
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(
            status_frame, 
            text="Ready", 
            anchor=tk.W,
            style="StatusBar.TLabel"
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        
        # Progress counter on the right
        self.progress_label = ttk.Label(
            status_frame, 
            text="0/0", 
            anchor=tk.E,
            style="StatusBar.TLabel"
        )
        self.progress_label.pack(side=tk.RIGHT, padx=10, pady=3)
    
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('a', lambda e: self.approve())
        self.root.bind('r', lambda e: self.reject())
        self.root.bind('<Left>', lambda e: self._prev_image())
        self.root.bind('<Right>', lambda e: self._next_image())
        self.root.bind('t', lambda e: self._toggle_theme())
        self.root.bind('<F5>', lambda e: self.load_items())
        self.root.bind('<F11>', lambda e: self._toggle_fullscreen())
    
    def _toggle_theme(self):
        """Toggle between light and dark theme"""
        new_theme = self.theme_manager.toggle_theme()
        theme_icon = "🌙" if new_theme == "light" else "☀️"
        
        # Update theme button
        if hasattr(self, 'theme_btn'):
            self.theme_btn.config(text=f"{theme_icon} Theme")
        
        # Update status
        self.status_label.config(text=f"Theme changed to {new_theme}")
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        is_fullscreen = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not is_fullscreen)
        
    def _prev_image(self):
        """Show the previous image"""
        if not hasattr(self, 'items') or not self.items:
            return
            
        if self.current > 0:
            self.current -= 1
            self.show_current()
            self.status_label.config(text="Previous image")
    
    def _next_image(self):
        """Show the next image"""
        if not hasattr(self, 'items') or not self.items:
            return
            
        if self.current < len(self.items) - 1:
            self.current += 1
            self.show_current()
            self.status_label.config(text="Next image")

    def _on_model_change(self, event):
        """Handle model switching with error handling"""
        try:
            new_model = self.model_var.get()
            if new_model != self.model_name:
                logger.info(f"Switching model from {self.model_name} to {new_model}")
                self.status_label.config(text=f"Switching to {new_model} model...")
                try:
                    self.model = self.model_manager.get_model(new_model)
                    self.model_name = new_model
                    if self.items:
                        self.show_current()
                    self.status_label.config(text=f"Switched to {new_model} model")
                except Exception as e:
                    logger.error(f"Failed to switch to model {new_model}: {str(e)}")
                    messagebox.showerror("Error", f"Failed to switch to {new_model}. Reverting to previous model.")
                    self.model_var.set(self.model_name)
                    self.status_label.config(text=f"Error switching model")
        except Exception as e:
            logger.error(f"Error in model change handler: {str(e)}")
            raise

    def _on_quality_change(self, event):
        """Handle quality selection change"""
        try:
            new_quality = self.quality_var.get()
            logger.info(f"Changed caption quality to: {new_quality}")
            
            # If there's an image loaded, offer to regenerate the caption
            if self.items:
                if messagebox.askyesno(
                    "Regenerate Caption",
                    f"Would you like to regenerate the caption with {new_quality} quality?"
                ):
                    self._regenerate_caption()
        except Exception as e:
            logger.error(f"Error in quality change handler: {str(e)}")

    def _regenerate_caption(self):
        """Regenerate caption for current image with current quality setting"""
        if not self.items:
            return
            
        try:
            self.status_label.config(text="Regenerating caption...")
            
            # Get current image path
            _, _, img_path = self.items[self.current]
            
            # Get current quality setting
            quality = self.quality_var.get()
            
            # Analyze image with current quality
            description, clean_caption = self.model.analyze_image(
                str(img_path), 
                quality=quality
            )
            
            # Update display
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, description)
            self.caption_text.config(state=tk.NORMAL)  # Keep editable
            
            # Apply trigger word if needed
            if self.trigger_word and clean_caption:
                clean_caption = f"{self.trigger_word}, {clean_caption}"
            
            # Update JSON and text files
            self._update_caption_files(img_path, description, clean_caption)
            
            self.status_label.config(text=f"Caption regenerated with {quality} quality")
        except Exception as e:
            logger.error(f"Error regenerating caption: {str(e)}")
            self.status_label.config(text="Error regenerating caption")

    def load_items(self):
        """Load image items with error handling"""
        self.items = []
        try:
            self.status_label.config(text="Loading images...")
            
            if not self.review_dir.exists():
                logger.warning(f"Review directory not found: {self.review_dir}")
                self.status_label.config(text=f"Review directory not found: {self.review_dir}")
                return
                
            for f in self.review_dir.iterdir():
                if self.dataset_prep.is_supported_image(f):
                    img_path = f
                    base_name = f.stem
                    json_path = f.parent / f"{base_name}_for_review.json"
                    
                    if not json_path.exists():
                        json_path.write_text(
                            json.dumps({"results": {"caption": ""}}, indent=2),
                            encoding='utf-8'
                        )
                    
                    self.items.append((base_name, json_path, img_path))
            
            self.current = 0
            self.progress_label.config(text=f"0/{len(self.items)}")
            
            if self.items:
                self.show_current()
                self.status_label.config(text=f"Loaded {len(self.items)} images")
            else:
                logger.info("No supported image files found for review.")
                messagebox.showinfo("Info", "No images found for review.")
                self.status_label.config(text="No images found for review")
                
                # Clear the display
                self.img_label.config(image="")
                self.caption_text.config(state=tk.NORMAL)
                self.caption_text.delete(1.0, tk.END)
                self.caption_text.insert(tk.END, "No images available for review.")
                self.caption_text.config(state=tk.DISABLED)
                self.filename_label.config(text="")
                self.dimensions_label.config(text="")
        except Exception as e:
            logger.error(f"Error loading items: {str(e)}")
            self.status_label.config(text=f"Error loading images: {str(e)}")
            raise

    def apply_text_highlighting(self, text_widget, content):
        """Apply syntax highlighting to text content"""
        text_widget.delete(1.0, tk.END)
        
        # First insert all content
        text_widget.insert(tk.END, content)
        
        # Apply highlighting for patterns
        self._highlight_pattern(text_widget, r"Description:", "heading")
        self._highlight_pattern(text_widget, r"Detected objects:", "subheading")
        self._highlight_pattern(text_widget, r"Keywords:", "subheading")
        self._highlight_pattern(text_widget, r"\b(person|people|man|woman|child|dog|cat|car|building)\b", "object")
        self._highlight_pattern(text_widget, r"\b([a-zA-Z0-9]+:[a-zA-Z0-9_]+)\b", "tag")  # Match patterns like "object:person"
    
    def _highlight_pattern(self, text_widget, pattern, tag, start="1.0", end="end"):
        """Apply a tag to all text that matches the pattern"""
        start = text_widget.index(start)
        end = text_widget.index(end)
        text_widget.mark_set("matchStart", start)
        text_widget.mark_set("matchEnd", start)
        text_widget.mark_set("searchLimit", end)

        count = tk.IntVar()
        while True:
            index = text_widget.search(
                pattern, "matchEnd", "searchLimit",
                count=count, regexp=True
            )
            if index == "" or count.get() == 0:
                break
            text_widget.mark_set("matchStart", index)
            text_widget.mark_set("matchEnd", f"{index}+{count.get()}c")
            text_widget.tag_add(tag, "matchStart", "matchEnd")

    def show_current(self):
        """Display current image with error handling"""
        if not self.items:
            return
            
        try:
            self.status_label.config(text="Loading image...")
            base_name, json_path, img_path = self.items[self.current]
            
            # Update progress
            self.progress_label.config(text=f"{self.current + 1}/{len(self.items)}")
            
            # Update filename
            self.filename_label.config(text=str(img_path.name))
            
            try:
                # Analyze the image
                self.status_label.config(text="Analyzing image...")
                description, clean_caption = self.model.analyze_image(str(img_path))
                self.status_label.config(text="Analysis complete")
            except Exception as e:
                logger.error(f"Error analyzing image: {str(e)}")
                description = "Error analyzing image"
                clean_caption = None
                self.status_label.config(text=f"Error analyzing image")
                messagebox.showwarning("Warning", f"Error analyzing image: {str(e)}")
            
            if self.trigger_word and clean_caption:
                clean_caption = f"{self.trigger_word}, {clean_caption}"
            
            # Load and display image
            img = Image.open(img_path)
            
            # Get dimensions
            img_width, img_height = img.size
            self.dimensions_label.config(text=f"{img_width} × {img_height}")
            
            # Calculate display size to maintain aspect ratio and center
            max_width, max_height = 800, 600
            
            # Scale down if needed, maintaining aspect ratio
            scale_w = max_width / img_width if img_width > max_width else 1
            scale_h = max_height / img_height if img_height > max_height else 1
            scale = min(scale_w, scale_h)
            
            if scale < 1:
                new_width, new_height = int(img_width * scale), int(img_height * scale)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update the image label
            self.img_label.configure(image=photo)
            self.img_label.image = photo  # Keep a reference
            
            # Center the image in the container
            self.img_label.place(
                relx=0.5, rely=0.5,
                anchor='center'
            )
            
            # Update caption with styled text and highlighting
            self.caption_text.config(state=tk.NORMAL)
            self.apply_text_highlighting(self.caption_text, description)
            self.caption_text.config(state=tk.NORMAL)  # Keep editable
            
            # Save analysis results
            data = {"results": {"caption": description}}
            json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            
            if clean_caption:
                self.dataset_prep.create_caption_file(str(img_path), clean_caption)
            
            # Update window title
            self.root.title(f"Multi-Vision Toolkit - {img_path.name} ({self.current + 1}/{len(self.items)})")
            
            # Update status
            self.status_label.config(text=f"Displaying image {self.current + 1} of {len(self.items)}")
            
        except Exception as e:
            logger.error(f"Error showing current item: {str(e)}")
            self.status_label.config(text=f"Error displaying image: {str(e)}")
            raise

    def move_item(self, dest_dir: Path):
        """Move current item to destination directory with error handling"""
        if not self.items:
            return
            
        try:
            action = "Approving" if dest_dir == self.approved_dir else "Rejecting"
            self.status_label.config(text=f"{action} image...")
            
            base_name, json_path, img_path = self.items[self.current]
            
            new_img_path = dest_dir / img_path.name
            txt_path = img_path.with_suffix('.txt')
            new_txt_path = dest_dir / txt_path.name
            
            shutil.move(str(img_path), str(new_img_path))
            if txt_path.exists():
                shutil.move(str(txt_path), str(new_txt_path))
            
            if json_path.exists():
                data = json.loads(json_path.read_text(encoding='utf-8'))
                data['review_status'] = 'approved' if dest_dir == self.approved_dir else 'rejected'
                data['timestamp'] = datetime.now().isoformat()
                
                new_json_path = dest_dir / f"{base_name}_reviewed.json"
                new_json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
                json_path.unlink()
                
            self.items.pop(self.current)
            
            # Update progress counter
            self.progress_label.config(text=f"{min(self.current + 1, len(self.items))}/{len(self.items)}")
            
            if self.items:
                if self.current >= len(self.items):
                    self.current = len(self.items) - 1
                self.show_current()
                self.status_label.config(text=f"Image {action.lower()} successfully")
            else:
                self.status_label.config(text="All images processed")
                messagebox.showinfo("Complete", "All images have been processed.")
                self.root.quit()
        except Exception as e:
            logger.error(f"Error moving files: {str(e)}")
            self.status_label.config(text=f"Error {action.lower()} image: {str(e)}")
            raise

    def approve(self):
        """Approve current item with error handling"""
        try:
            logger.info(f"Approving item {self.current + 1}/{len(self.items)}")
            self.move_item(self.approved_dir)
        except Exception as e:
            logger.error(f"Error approving item: {str(e)}")
            messagebox.showerror("Error", f"Failed to approve item: {str(e)}")
        
    def reject(self):
        """Reject current item with error handling"""
        try:
            logger.info(f"Rejecting item {self.current + 1}/{len(self.items)}")
            self.move_item(self.rejected_dir)
        except Exception as e:
            logger.error(f"Error rejecting item: {str(e)}")
            messagebox.showerror("Error", f"Failed to reject item: {str(e)}")

    def _save_caption_edits(self):
        """Save edited captions to both JSON and TXT files"""
        if not self.items:
            return
            
        try:
            # Get current edited text from the text widget
            edited_text = self.caption_text.get(1.0, tk.END).strip()
            
            # Get paths for current item
            base_name, json_path, img_path = self.items[self.current]
            
            # Extract clean caption (first line or description part)
            if "Description:" in edited_text:
                clean_caption = edited_text.split("Description:")[1].strip().split("\n")[0]
            else:
                clean_caption = edited_text.split("\n")[0]
            
            # Update JSON file
            try:
                data = json.loads(json_path.read_text(encoding='utf-8'))
                data["results"]["caption"] = edited_text
                json_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            except Exception as e:
                logger.warning(f"Could not update JSON file: {str(e)}")
            
            # Update TXT file (clean caption)
            caption_to_save = clean_caption
            if self.trigger_word:
                caption_to_save = f"{self.trigger_word}, {clean_caption}"
                
            txt_path = img_path.with_suffix('.txt')
            txt_path.write_text(caption_to_save, encoding='utf-8')
            
            self.status_label.config(text="Caption updated successfully")
            logger.info(f"Updated caption for {img_path.name}")
        except Exception as e:
            logger.error(f"Error saving caption: {str(e)}")
            self.status_label.config(text=f"Error saving caption")
            messagebox.showerror("Error", f"Failed to save caption: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='AI Training Dataset Preparation Tool')
    parser.add_argument('--review_dir', required=True, help='Review directory')
    parser.add_argument('--approved_dir', default='approved', help='Approved directory')
    parser.add_argument('--rejected_dir', default='rejected', help='Rejected directory')
    parser.add_argument('--trigger_word', help='Optional trigger word to add to captions')
    parser.add_argument('--model', default='florence2', choices=['florence2', 'janus', 'qwen'],
                      help='Vision model to use (default: florence2)')
    
    try:
        args = parser.parse_args()
        logger.info(f"Starting application with model: {args.model}")
        
        app = ReviewGUI(
            args.review_dir, 
            args.approved_dir, 
            args.rejected_dir,
            args.trigger_word,
            args.model
        )
        app.root.mainloop()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled application error: {str(e)}")
        print(f"Error: {e}")
        
        # If running in GUI mode, show error dialog
        try:
            import tkinter as tk
            from tkinter import messagebox
            if tk._default_root is not None:
                messagebox.showerror(
                    "Application Error",
                    f"An unexpected error occurred:\n\n{str(e)}\n\nCheck log file for details."
                )
        except:
            pass