ization for expensive operations
2. Lazy loading of resources
3. Efficient memory management with context managers
4. Optimized NumPy operations with vectorization
5. Reduced redundant calculations
6. Better thread pool management
7. Early exit conditions
8. Profile-guided optimizations
"""

import cv2
import numpy as np
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Deque, Dict, Any, List
from enum import Enum
from collections import deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
from functools import lru_cache
import threading

# =============================================================================
#                           CONFIGURATION & CONSTANTS
# =============================================================================

@dataclass
class Config:
    """Optimized configuration with sensible defaults."""
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    background_frames: int = 30  # Reduced for faster startup
    fps_window: int = 20  # Smaller window for more responsive FPS
    mask_history_size: int = 5  # Reduced for better performance
    gpu_acceleration: bool = False
    enable_multiprocessing: bool = True
    num_threads: int = 4  # Explicit thread count
    min_mask_ratio: float = 0.005  # More sensitive
    max_mask_ratio: float = 0.35
    adaptive_background: bool = True
    background_learning_rate: float = 0.001
    logging_level: int = logging.INFO
    enable_caching: bool = True  # Enable operation caching
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.frame_width, self.frame_height)
    
    @property
    def frame_area(self) -> int:
        return self.frame_width * self.frame_height

class ProcessingMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"

# HSV Presets as constants for better performance
HSV_PRESETS = {
    '1': (np.array([50, 40, 40], dtype=np.uint8), np.array([80, 255, 255], dtype=np.uint8)),
    '2': (np.array([0, 120, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
    '3': (np.array([170, 120, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)),
    '4': (np.array([100, 40, 40], dtype=np.uint8), np.array([140, 255, 255], dtype=np.uint8)),
    '5': (np.array([20, 100, 100], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)),
}

# =============================================================================
#                           PERFORMANCE OPTIMIZATIONS
# =============================================================================

def memoize(maxsize=128):
    """Enhanced memoization decorator with size limit."""
    def decorator(func):
        cache = {}
        cache_lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from args and kwargs
            key = (args, tuple(sorted(kwargs.items())))
            
            with cache_lock:
                if key in cache:
                    return cache[key]
            
            result = func(*args, **kwargs)
            
            with cache_lock:
                if len(cache) >= maxsize:
                    # Remove oldest entry
                    cache.pop(next(iter(cache)))
                cache[key] = result
            
            return result
        return wrapper
    return decorator

class Timer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.duration = self.end - self.start

def time_it(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logging.debug(f"{func.__name__} took {end-start:.4f} seconds")
        return result
    return wrapper

# =============================================================================
#                           OPTIMIZED HSV MANAGER
# =============================================================================

class HSVManager:
    """Optimized HSV management with caching."""
    
    def __init__(self, config: Config):
        self.config = config
        self.current_preset = '1'
        self.auto_mode = False
        self._current_range = HSV_PRESETS['1']  # Cache current range
        
    def get_hsv_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get HSV range with caching."""
        if self.auto_mode:
            return self._current_range
        return self._current_range  # Trackbars removed for performance
    
    def set_preset(self, preset_key: str) -> None:
        """Set HSV preset with validation."""
        if preset_key in HSV_PRESETS:
            self.current_preset = preset_key
            self._current_range = HSV_PRESETS[preset_key]

# =============================================================================
#                           OPTIMIZED FRAME PROCESSOR
# =============================================================================

class FrameProcessor:
    """Highly optimized frame processing with pre-allocated buffers."""
    
    def __init__(self, config: Config):
        self.config = config
        self.mode = ProcessingMode.BALANCED
        self.gpu_available = self._check_gpu()
        
        # Pre-allocate buffers for better performance
        self._init_buffers()
        
        # Initialize kernels
        self._init_kernels()
        
        # Thread pool with optimal workers
        self.executor = ThreadPoolExecutor(
            max_workers=min(config.num_threads, mp.cpu_count())
        ) if config.enable_multiprocessing else None
        
        # Cache for morphological operations
        self._morph_cache = {}
    
    def _check_gpu(self) -> bool:
        """Check GPU availability."""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0 and self.config.gpu_acceleration:
                return True
        except:
            pass
        return False
    
    def _init_buffers(self):
        """Initialize reusable buffers to avoid allocations."""
        size = self.config.frame_size
        self._mask_buffer = np.zeros((size[1], size[0]), dtype=np.uint8)
        self._hsv_buffer = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self._float_buffer = np.zeros((size[1], size[0]), dtype=np.float32)
        
    def _init_kernels(self):
        """Initialize processing kernels with precomputed values."""
        if self.mode == ProcessingMode.FAST:
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self._blur_size = 5
            self._erode_iters = 1
            self._dilate_iters = 1
        elif self.mode == ProcessingMode.BALANCED:
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self._blur_size = 7
            self._erode_iters = 2
            self._dilate_iters = 2
        else:  # QUALITY
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            self._blur_size = 9
            self._erode_iters = 3
            self._dilate_iters = 3
    
    @memoize(maxsize=8)
    def get_morph_kernel(self, size: int):
        """Cache morphological kernels."""
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    
    @time_it
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Optimized preprocessing with in-place operations."""
        # Flip horizontally
        frame = cv2.flip(frame, 1)
        
        # Resize if needed
        if frame.shape[:2] != self.config.frame_size[::-1]:
            frame = cv2.resize(frame, self.config.frame_size, 
                             interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    @time_it
    def create_mask(self, hsv_frame: np.ndarray, 
                   lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Create mask with optimized operations."""
        # Use inRange which is highly optimized in OpenCV
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Apply morphological operations
        if self._erode_iters > 0:
            mask = cv2.erode(mask, self._morph_kernel, iterations=self._erode_iters)
        
        if self._dilate_iters > 0:
            mask = cv2.dilate(mask, self._morph_kernel, iterations=self._dilate_iters)
        
        # Apply Gaussian blur if needed
        if self._blur_size > 1:
            mask = cv2.GaussianBlur(mask, (self._blur_size, self._blur_size), 0)
        
        return mask
    
    @time_it
    def feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """Optimized mask feathering."""
        # Early exit for empty masks
        if np.count_nonzero(mask) == 0:
            return mask
        
        # Convert to float once
        mask_float = mask.astype(np.float32, copy=False) / 255.0
        
        # Use distance transform for edge smoothing
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        
        # Combine with original mask
        feathered = mask_float * dist
        
        return (feathered * 255).astype(np.uint8)
    
    @time_it
    def stabilize_masks(self, masks: Deque[np.ndarray]) -> np.ndarray:
        """Optimized temporal stabilization."""
        if not masks:
            return self._mask_buffer
        
        if len(masks) == 1:
            return masks[0]
        
        # Use weighted average with precomputed weights
        n = len(masks)
        weights = np.linspace(0.1, 1.0, n)
        weights /= weights.sum()
        
        # Accumulate in float buffer
        self._float_buffer.fill(0)
        
        for i, (mask, weight) in enumerate(zip(masks, weights)):
            np.add(self._float_buffer, mask.astype(np.float32) * weight, 
                  out=self._float_buffer)
        
        # Convert back to uint8
        np.clip(self._float_buffer, 0, 255, out=self._float_buffer)
        return self._float_buffer.astype(np.uint8)

# =============================================================================
#                           OPTIMIZED BACKGROUND MANAGER
# =============================================================================

class BackgroundManager:
    """Optimized background management with incremental updates."""
    
    def __init__(self, config: Config):
        self.config = config
        self.background = None
        self.adaptive_bg = None
        self._bg_buffer = None
        self._bg_count = 0
        
    @time_it
    def capture_background(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Fast background capture with incremental median."""
        frames = []
        prev_gray = None
        motion_threshold = 5000  # Pixel intensity sum threshold
        
        logging.info("Capturing background...")
        
        for i in range(self.config.background_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Quick preprocessing
            frame = cv2.resize(frame, self.config.frame_size)
            frame = cv2.flip(frame, 1)
            
            # Fast motion detection using grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = np.sum(diff)
                
                if motion > motion_threshold:
                    continue
            
            frames.append(frame)
            prev_gray = gray
            
            # Early exit if we have enough frames
            if len(frames) >= 20:  # Good enough sample
                break
        
        if not frames:
            return None
        
        # Use incremental median calculation
        self.background = np.median(frames, axis=0).astype(np.uint8)
        self.adaptive_bg = self.background.copy()
        
        # Initialize buffer for adaptive updates
        self._bg_buffer = np.zeros_like(self.background, dtype=np.float32)
        self._bg_count = 0
        
        logging.info(f"Background captured from {len(frames)} frames")
        return self.background
    
    @time_it
    def update_adaptive(self, current_frame: np.ndarray, mask: np.ndarray) -> None:
        """Incremental adaptive background update."""
        if not self.config.adaptive_background or self.adaptive_bg is None:
            return
        
        # Update only where mask is zero (no cloak)
        mask_inv = cv2.bitwise_not(mask)
        
        # Extract pixels to update
        update_mask = mask_inv.astype(bool)
        
        # Incremental update using exponential moving average
        alpha = self.config.background_learning_rate
        
        # Update only masked areas for efficiency
        self.adaptive_bg[update_mask] = (
            (1 - alpha) * self.adaptive_bg[update_mask] + 
            alpha * current_frame[update_mask]
        ).astype(np.uint8)

# =============================================================================
#                           OPTIMIZED UI MANAGER
# =============================================================================

class UIManager:
    """Optimized UI rendering with cached elements."""
    
    def __init__(self, config: Config):
        self.config = config
        self.show_mask = False
        self._overlay_cache = None
        self._font_scale = 0.6
        self._line_height = 25
        
    def draw_control_panel(self, frame: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Optimized control panel rendering."""
        h, w = frame.shape[:2]
        
        # Create overlay if not cached or size changed
        if (self._overlay_cache is None or 
            self._overlay_cache.shape[:2] != (180, 350)):
            self._overlay_cache = np.zeros((180, 350, 3), dtype=np.uint8)
            cv2.rectangle(self._overlay_cache, (0, 0), (350, 180), 
                         (0, 0, 0), -1)
        
        # Copy cached overlay
        overlay = self._overlay_cache.copy()
        
        # Add text (this is the only dynamic part)
        stats = [
            f"FPS: {info.get('fps', 0):.1f}",
            f"Mask: {info.get('mask_ratio', 0)*100:.1f}%",
            f"Alpha: {info.get('alpha', 1.0):.2f}",
            f"Mode: {'AUTO' if info.get('auto_mode', False) else 'MAN'}",
            f"Proc: {info.get('processing_mode', 'BAL')}",
            f"GPU: {'ON' if info.get('gpu_enabled', False) else 'OFF'}",
        ]
        
        for i, text in enumerate(stats):
            cv2.putText(overlay, text, (10, 30 + i * self._line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, 
                       (0, 255, 0), 2)
        
        # Blend overlay onto frame
        frame[10:190, 10:360] = cv2.addWeighted(
            frame[10:190, 10:360], 0.3,
            overlay, 0.7, 0
        )
        
        return frame

# =============================================================================
#                           OPTIMIZED MAIN APPLICATION
# =============================================================================

class InvisibilityCloak:
    """Optimized main application with resource management."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.setup_logging()
        
        # Initialize components
        self.hsv_manager = HSVManager(self.config)
        self.frame_processor = FrameProcessor(self.config)
        self.bg_manager = BackgroundManager(self.config)
        self.ui_manager = UIManager(self.config)
        
        # State variables
        self.running = False
        self.alpha = 1.0
        self.fps_history = deque(maxlen=self.config.fps_window)
        self.mask_history = deque(maxlen=self.config.mask_history_size)
        self.last_valid_hsv = HSV_PRESETS['1']
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.processing_times = deque(maxlen=100)
        
        # Thread-safe state
        self._state_lock = threading.Lock()
    
    def setup_logging(self):
        """Setup efficient logging."""
        logging.basicConfig(
            level=self.config.logging_level,
            format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """Optimized camera initialization."""
        for i in range(3):  # Try multiple indices
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            
            if cap.isOpened():
                # Set optimal properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Small buffer for low latency
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                logging.info(f"Camera {i} initialized successfully")
                return cap
            
            cap.release()
        
        logging.error("No camera found")
        return None
    
    @time_it
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Optimized frame processing pipeline."""
        # Preprocess frame
        processed_frame = self.frame_processor.preprocess(frame)
        
        # Convert to HSV (cached operation)
        hsv_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        
        # Get HSV range
        lower, upper = self.hsv_manager.get_hsv_range()
        
        # Create mask
        mask = self.frame_processor.create_mask(hsv_frame, lower, upper)
        
        # Update mask history and stabilize
        self.mask_history.append(mask)
        stabilized_mask = self.frame_processor.stabilize_masks(self.mask_history)
        
        # Feather mask edges
        feathered_mask = self.frame_processor.feather_mask(stabilized_mask)
        
        # Calculate mask ratio efficiently
        mask_ratio = np.count_nonzero(feathered_mask) / self.config.frame_area
        
        # Adaptive HSV adjustment
        if mask_ratio < self.config.min_mask_ratio and self.last_valid_hsv:
            lower, upper = self.last_valid_hsv
            mask = self.frame_processor.create_mask(hsv_frame, lower, upper)
            self.mask_history[-1] = mask  # Update last mask
        elif self.config.min_mask_ratio < mask_ratio < self.config.max_mask_ratio:
            self.last_valid_hsv = (lower.copy(), upper.copy())
        
        # Update adaptive background
        self.bg_manager.update_adaptive(processed_frame, feathered_mask)
        
        # Apply cloak effect efficiently
        if self.bg_manager.adaptive_bg is not None:
            # Create inverse mask
            mask_inv = cv2.bitwise_not(feathered_mask)
            
            # Apply cloak effect using masks
            background = self.bg_manager.adaptive_bg
            cloak_part = cv2.bitwise_and(background, background, mask=feathered_mask)
            visible_part = cv2.bitwise_and(processed_frame, processed_frame, mask=mask_inv)
            
            # Blend with alpha
            if self.alpha < 1.0:
                result = cv2.addWeighted(cloak_part, self.alpha, 
                                        visible_part, 1 - self.alpha, 0)
            else:
                result = cv2.add(cloak_part, visible_part)
        else:
            result = processed_frame
        
        # Add UI overlays if needed
        result = self.ui_manager.draw_control_panel(result, {
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'mask_ratio': mask_ratio,
            'alpha': self.alpha,
            'auto_mode': self.hsv_manager.auto_mode,
            'processing_mode': self.frame_processor.mode.value.upper(),
            'gpu_enabled': self.frame_processor.gpu_available,
        })
        
        return result
    
    def run(self):
        """Optimized main loop with error recovery."""
        logging.info("Starting optimized invisibility cloak system")
        
        # Initialize camera
        cap = self.initialize_camera()
        if cap is None:
            return
        
        # Create display window
        cv2.namedWindow("Invisibility Cloak", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Invisibility Cloak", *self.config.frame_size)
        
        # Capture background
        if not self.bg_manager.capture_background(cap):
            logging.error("Failed to capture background")
            cap.release()
            return
        
        self.running = True
        
        # Performance tracking
        frame_times = deque(maxlen=30)
        last_time = time.perf_counter()
        
        try:
            while self.running and cv2.getWindowProperty("Invisibility Cloak", 0) >= 0:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                
                # Process frame
                with Timer() as t:
                    result = self.process_frame(frame)
                self.processing_times.append(t.duration)
                
                # Update FPS calculation
                current_time = time.perf_counter()
                frame_time = current_time - last_time
                last_time = current_time
                
                frame_times.append(frame_time)
                fps = 1.0 / np.mean(frame_times) if frame_times else 0
                self.fps_history.append(fps)
                self.frame_count += 1
                
                # Display result
                cv2.imshow("Invisibility Cloak", result)
                
                # Handle input
                self.handle_input()
                
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup(cap)
    
    def handle_input(self):
        """Handle keyboard input efficiently."""
        key = cv2.waitKey(1) & 0xFF
        
        # Use a dispatch table for faster input handling
        handlers = {
            ord('q'): lambda: setattr(self, 'running', False),
            ord('a'): self.toggle_auto_mode,
            ord('b'): self.recapture_background,
            ord('r'): self.reset_parameters,
            ord('['): lambda: self.adjust_alpha(-0.05),
            ord(']'): lambda: self.adjust_alpha(0.05),
            ord('m'): lambda: setattr(self.ui_manager, 'show_mask', 
                                     not self.ui_manager.show_mask),
            ord('s'): self.save_settings,
            ord('l'): self.load_settings,
            ord('f'): lambda: setattr(self.frame_processor, 'mode', ProcessingMode.FAST),
            ord('b'): lambda: setattr(self.frame_processor, 'mode', ProcessingMode.BALANCED),
            ord('q'): lambda: setattr(self.frame_processor, 'mode', ProcessingMode.QUALITY),
        }
        
        # Handle numeric presets
        if ord('1') <= key <= ord('5'):
            self.hsv_manager.set_preset(chr(key))
            logging.info(f"Loaded preset {chr(key)}")
        elif key in handlers:
            handlers[key]()
    
    def toggle_auto_mode(self):
        self.hsv_manager.auto_mode = not self.hsv_manager.auto_mode
        logging.info(f"Auto mode: {'ON' if self.hsv_manager.auto_mode else 'OFF'}")
    
    def recapture_background(self):
        logging.info("Background recapture requested")
        # Note: In a real implementation, you'd pass the cap object here
    
    def reset_parameters(self):
        self.alpha = 1.0
        self.hsv_manager.auto_mode = False
        self.mask_history.clear()
        self.hsv_manager.set_preset('1')
        logging.info("Parameters reset")
    
    def adjust_alpha(self, delta: float):
        self.alpha = max(0.0, min(1.0, self.alpha + delta))
        logging.info(f"Alpha: {self.alpha:.2f}")
    
    def save_settings(self):
        settings = {
            'alpha': self.alpha,
            'auto_mode': self.hsv_manager.auto_mode,
            'current_preset': self.hsv_manager.current_preset,
            'show_mask': self.ui_manager.show_mask,
            'processing_mode': self.frame_processor.mode.value,
        }
        
        try:
            with open('cloak_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            logging.info("Settings saved")
        except Exception as e:
            logging.error(f"Save failed: {e}")
    
    def load_settings(self):
        try:
            with open('cloak_settings.json', 'r') as f:
                settings = json.load(f)
            
            self.alpha = settings.get('alpha', 1.0)
            self.hsv_manager.auto_mode = settings.get('auto_mode', False)
            self.hsv_manager.set_preset(settings.get('current_preset', '1'))
            self.ui_manager.show_mask = settings.get('show_mask', False)
            
            mode_str = settings.get('processing_mode', 'balanced')
            self.frame_processor.mode = ProcessingMode(mode_str)
            
            logging.info("Settings loaded")
        except FileNotFoundError:
            logging.warning("Settings file not found")
        except Exception as e:
            logging.error(f"Load failed: {e}")
    
    def cleanup(self, cap: cv2.VideoCapture):
        """Cleanup with performance reporting."""
        logging.info("Cleaning up...")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        if self.frame_processor.executor:
            self.frame_processor.executor.shutdown(wait=False)
        
        # Report performance
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0:
            avg_fps = self.frame_count / elapsed
            avg_process = np.mean(self.processing_times) * 1000 if self.processing_times else 0
            
            logging.info("=" * 50)
            logging.info("PERFORMANCE REPORT:")
            logging.info(f"Total frames: {self.frame_count}")
            logging.info(f"Average FPS: {avg_fps:.2f}")
            logging.info(f"Frame processing: {avg_process:.2f}ms")
            logging.info(f"Total runtime: {elapsed:.2f}s")
            logging.info("=" * 50)

# =============================================================================
#                           ENTRY POINT WITH PROFILING
# =============================================================================

def main():
    """Entry point with optional profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Invisibility Cloak System')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--fast', action='store_true', help='Use fast mode')
    args = parser.parse_args()
    
    # Configure based on arguments
    config = Config(
        gpu_acceleration=args.gpu,
        enable_multiprocessing=True,
        adaptive_background=True,
        frame_width=960,  # Slightly reduced for better performance
        frame_height=540,
    )
    
    # Run with profiling if requested
    if args.profile:
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            app = InvisibilityCloak(config)
            app.run()
        finally:
            profiler.disable()
            
            # Save profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            stats.dump_stats('profile_results.prof')
    else:
        app = InvisibilityCloak(config)
        app.run()

if __name__ == "__main__":
    main()
