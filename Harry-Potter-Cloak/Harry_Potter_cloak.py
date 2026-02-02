
4. Background restoration with inpainting
5. Advanced temporal stabilization
6. Real-time parameter adjustment GUI

Key Features:
------------
1. Real-time cloak effect with multiple color support
2. AI-powered color range optimization
3. Temporal stability with optical flow
4. Background reconstruction
5. Performance profiling and optimization
6. Comprehensive GUI with parameter control
"""

import cv2
import numpy as np
import time
import json
import logging
import argparse
import threading
import queue
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any, Deque, Callable
from enum import Enum, auto
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import gc

# Optional imports for enhanced features
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cloak_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
#                           CONFIGURATION
# =============================================================================

@dataclass
class SystemConfig:
    """Enhanced system configuration."""
    # Camera settings
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 60
    
    # Processing settings
    processing_mode: str = "balanced"  # "fast", "balanced", "quality"
    enable_gpu: bool = CUDA_AVAILABLE
    enable_multithreading: bool = True
    num_threads: int = max(1, mp.cpu_count() - 1)
    
    # Color detection
    color_sensitivity: float = 0.8
    adaptive_threshold: bool = True
    use_multi_color: bool = True
    max_colors: int = 3
    
    # Background
    background_frames: int = 30
    background_update_rate: float = 0.01
    enable_background_restoration: bool = True
    background_model: str = "mog2"  # "static", "mog2", "knn"
    
    # Mask processing
    mask_smoothing: float = 0.7
    temporal_stability: int = 5
    feather_amount: float = 0.1
    min_mask_area: float = 0.001  # % of frame
    max_mask_area: float = 0.4    # % of frame
    
    # Performance
    enable_profiling: bool = False
    stats_window: int = 100
    cache_size: int = 10
    
    # UI
    show_controls: bool = True
    show_stats: bool = True
    show_debug: bool = False
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.frame_width, self.frame_height)
    
    @property
    def frame_area(self) -> int:
        return self.frame_width * self.frame_height


class ProcessingMode(Enum):
    FAST = auto()
    BALANCED = auto()
    QUALITY = auto()
    
    @classmethod
    def from_string(cls, mode_str: str):
        return {
            "fast": cls.FAST,
            "balanced": cls.BALANCED,
            "quality": cls.QUALITY
        }.get(mode_str.lower(), cls.BALANCED)


# =============================================================================
#                           PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """Comprehensive performance monitoring and profiling."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.timings: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.start_time = time.perf_counter()
        self.frame_count = 0
        
    def time_section(self, section_name: str) -> Callable:
        """Decorator to time a code section."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                self.timings[section_name].append(elapsed)
                return result
            return wrapper
        return decorator
    
    def increment_counter(self, counter_name: str, amount: int = 1):
        self.counters[counter_name] += amount
    
    def get_fps(self) -> float:
        total_time = time.perf_counter() - self.start_time
        if total_time == 0:
            return 0
        return self.frame_count / total_time
    
    def get_average_time(self, section_name: str) -> float:
        timings = self.timings.get(section_name, [])
        if not timings:
            return 0
        return np.mean(list(timings))
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            "fps": self.get_fps(),
            "frame_count": self.frame_count,
            "runtime": time.perf_counter() - self.start_time,
            "section_times": {},
            "counters": dict(self.counters)
        }
        
        for section, timings in self.timings.items():
            if timings:
                stats["section_times"][section] = {
                    "mean": np.mean(list(timings)),
                    "min": np.min(list(timings)),
                    "max": np.max(list(timings)),
                    "95th": np.percentile(list(timings), 95)
                }
        
        return stats
    
    def reset(self):
        self.timings.clear()
        self.counters.clear()
        self.start_time = time.perf_counter()
        self.frame_count = 0


# =============================================================================
#                           COLOR DETECTION SYSTEM
# =============================================================================

class ColorDetector:
    """Advanced color detection with adaptive learning."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Color presets with enhanced ranges
        self.color_presets = {
            'green': {
                'hsv_range': [(35, 40, 40), (85, 255, 255)],
                'lab_range': [(0, 0, 0), (255, 255, 255)],  # Will be updated
                'weight': 1.0
            },
            'red': {
                'hsv_range': [(0, 120, 70), (10, 255, 255)],
                'lab_range': [(0, 0, 0), (255, 255, 255)],
                'weight': 1.0
            },
            'blue': {
                'hsv_range': [(100, 40, 40), (140, 255, 255)],
                'lab_range': [(0, 0, 0), (255, 255, 255)],
                'weight': 1.0
            },
            'custom': {
                'hsv_range': [(0, 0, 0), (180, 255, 255)],
                'lab_range': [(0, 0, 0), (255, 255, 255)],
                'weight': 1.0
            }
        }
        
        self.current_color = 'green'
        self.adaptive_model = None
        self.color_history = deque(maxlen=100)
        
        if SKLEARN_AVAILABLE and config.adaptive_threshold:
            self._init_adaptive_model()
    
    def _init_adaptive_model(self):
        """Initialize adaptive color model."""
        try:
            self.adaptive_model = GaussianMixture(n_components=3, covariance_type='full')
        except:
            logger.warning("Failed to initialize adaptive color model")
            self.adaptive_model = None
    
    @monitor.time_section("color_detection")
    def detect_colors(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect colors in frame with multiple methods."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        masks = []
        confidence_scores = []
        
        # Use multiple color spaces for robust detection
        for color_name, preset in self.color_presets.items():
            if color_name == 'custom' and not self.config.use_multi_color:
                continue
            
            # HSV detection
            hsv_lower = np.array(preset['hsv_range'][0], dtype=np.uint8)
            hsv_upper = np.array(preset['hsv_range'][1], dtype=np.uint8)
            hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            
            # LAB detection for better color constancy
            lab_lower = np.array(preset['lab_range'][0], dtype=np.uint8)
            lab_upper = np.array(preset['lab_range'][1], dtype=np.uint8)
            lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
            
            # Combine masks
            combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
            
            if np.count_nonzero(combined_mask) > 0:
                masks.append(combined_mask)
                
                # Calculate confidence based on color purity and area
                mask_area = np.count_nonzero(combined_mask) / self.config.frame_area
                if 0.01 < mask_area < 0.5:  # Reasonable area range
                    confidence = preset['weight'] * (1.0 - abs(0.2 - mask_area) / 0.2)
                    confidence_scores.append(confidence)
                else:
                    confidence_scores.append(0)
        
        if not masks:
            return np.zeros(frame.shape[:2], dtype=np.uint8), {"confidence": 0}
        
        # Weighted combination of masks
        total_confidence = sum(confidence_scores)
        if total_confidence > 0:
            combined_mask = np.zeros_like(masks[0], dtype=np.float32)
            for mask, confidence in zip(masks, confidence_scores):
                combined_mask += mask.astype(np.float32) * (confidence / total_confidence)
            combined_mask = np.clip(combined_mask, 0, 255).astype(np.uint8)
        else:
            combined_mask = masks[0]
        
        # Adaptive learning
        if self.adaptive_model is not None and np.count_nonzero(combined_mask) > 100:
            self._update_adaptive_model(frame, combined_mask)
        
        stats = {
            "confidence": total_confidence / len(confidence_scores) if confidence_scores else 0,
            "num_colors": len(masks),
            "mask_area": np.count_nonzero(combined_mask) / self.config.frame_area
        }
        
        return combined_mask, stats
    
    def _update_adaptive_model(self, frame: np.ndarray, mask: np.ndarray):
        """Update adaptive color model with new samples."""
        try:
            # Extract masked pixels
            masked_pixels = frame[mask > 0]
            if len(masked_pixels) > 1000:
                # Sample for efficiency
                samples = masked_pixels[np.random.choice(len(masked_pixels), 1000, replace=False)]
                
                # Convert to HSV
                samples_hsv = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
                
                # Update model
                self.adaptive_model.partial_fit(samples_hsv)
                
                # Update color ranges based on model
                self._adjust_color_ranges()
        except Exception as e:
            logger.warning(f"Adaptive model update failed: {e}")
    
    def _adjust_color_ranges(self):
        """Adjust color ranges based on learned model."""
        if self.adaptive_model is None:
            return
        
        for color_name in self.color_presets:
            if color_name == 'custom':
                continue
            
            # Get mean and covariance from model
            means = self.adaptive_model.means_
            covariances = self.adaptive_model.covariances_
            
            # Find closest component to current color
            current_hsv = np.mean(self.color_presets[color_name]['hsv_range'], axis=0)
            distances = [np.linalg.norm(mean - current_hsv) for mean in means]
            closest_idx = np.argmin(distances)
            
            # Adjust range based on learned distribution
            mean = means[closest_idx]
            std = np.sqrt(np.diag(covariances[closest_idx]))
            
            new_lower = np.clip(mean - 2 * std, 0, 255).astype(np.uint8)
            new_upper = np.clip(mean + 2 * std, 0, 255).astype(np.uint8)
            
            # Update HSV range
            self.color_presets[color_name]['hsv_range'][0] = tuple(new_lower)
            self.color_presets[color_name]['hsv_range'][1] = tuple(new_upper)


# =============================================================================
#                           MASK PROCESSOR
# =============================================================================

class MaskProcessor:
    """Advanced mask processing with temporal stability and edge refinement."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Initialize processing kernels
        self._init_kernels()
        
        # Temporal buffers
        self.mask_history = deque(maxlen=config.temporal_stability)
        self.flow_history = deque(maxlen=3)
        
        # Optical flow for motion compensation
        self.prev_gray = None
        self.flow = None
        
    def _init_kernels(self):
        """Initialize processing kernels based on mode."""
        mode = ProcessingMode.from_string(self.config.processing_mode)
        
        if mode == ProcessingMode.FAST:
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.blur_kernel = (5, 5)
            self.iterations = 1
        elif mode == ProcessingMode.BALANCED:
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self.blur_kernel = (7, 7)
            self.iterations = 2
        else:  # QUALITY
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            self.blur_kernel = (9, 9)
            self.iterations = 3
    
    @monitor.time_section("mask_processing")
    def process_mask(self, mask: np.ndarray, frame: np.ndarray = None) -> np.ndarray:
        """Process mask with temporal stability and edge refinement."""
        if mask is None or mask.size == 0:
            return np.zeros(self.config.frame_size[::-1], dtype=np.uint8)
        
        processed_mask = mask.copy()
        
        # 1. Morphological operations
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # 2. Remove small noise
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.config.frame_area * self.config.min_mask_area
        
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.drawContours(processed_mask, [contour], 0, 0, -1)
        
        # 3. Apply Gaussian blur for smooth edges
        if self.blur_kernel[0] > 1:
            processed_mask = cv2.GaussianBlur(processed_mask, self.blur_kernel, 0)
        
        # 4. Temporal stabilization if frame is provided
        if frame is not None:
            processed_mask = self._apply_temporal_stabilization(processed_mask, frame)
        
        # 5. Feather edges
        if self.config.feather_amount > 0:
            processed_mask = self._feather_edges(processed_mask)
        
        # 6. Apply area constraints
        mask_area = np.count_nonzero(processed_mask) / self.config.frame_area
        if mask_area > self.config.max_mask_area:
            # Scale down mask
            scale_factor = self.config.max_mask_area / mask_area
            processed_mask = self._scale_mask(processed_mask, scale_factor)
        
        return processed_mask
    
    def _apply_temporal_stabilization(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Apply temporal stabilization using optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Warp previous mask using flow
            if self.mask_history:
                prev_mask = self.mask_history[-1]
                h, w = prev_mask.shape
                
                # Create coordinate grid
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x + flow[..., 0]).astype(np.float32)
                map_y = (y + flow[..., 1]).astype(np.float32)
                
                # Warp previous mask
                warped_mask = cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR)
                
                # Blend with current mask
                alpha = self.config.mask_smoothing
                mask = cv2.addWeighted(mask, 1 - alpha, warped_mask, alpha, 0)
            
            self.flow_history.append(flow)
        
        self.prev_gray = gray
        self.mask_history.append(mask.copy())
        
        return mask
    
    def _feather_edges(self, mask: np.ndarray) -> np.ndarray:
        """Feather mask edges for smooth transitions."""
        # Distance transform for soft edges
        mask_float = mask.astype(np.float32) / 255.0
        
        # Calculate distance from edges
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # Apply feathering
        feathered = mask_float * (1 - self.config.feather_amount) + \
                   dist_transform * self.config.feather_amount
        
        return np.clip(feathered * 255, 0, 255).astype(np.uint8)
    
    def _scale_mask(self, mask: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale mask area while preserving shape."""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        # Scale each contour
        scaled_mask = np.zeros_like(mask)
        for contour in contours:
            # Get contour moments
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            # Calculate center
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Scale contour from center
            scaled_contour = contour.astype(np.float32)
            scaled_contour = (scaled_contour - [cx, cy]) * np.sqrt(scale_factor) + [cx, cy]
            scaled_contour = scaled_contour.astype(np.int32)
            
            cv2.drawContours(scaled_mask, [scaled_contour], 0, 255, -1)
        
        return scaled_mask


# =============================================================================
#                           BACKGROUND MANAGER
# =============================================================================

class BackgroundManager:
    """Advanced background management with restoration capabilities."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        self.background = None
        self.background_model = None
        self.background_history = deque(maxlen=10)
        
        # Initialize background model
        self._init_background_model()
        
        # Inpainting for background restoration
        self.inpainting_radius = 3
        
    def _init_background_model(self):
        """Initialize background subtraction model."""
        if self.config.background_model == "mog2":
            self.background_model = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=False
            )
        elif self.config.background_model == "knn":
            self.background_model = cv2.createBackgroundSubtractorKNN(
                history=500, dist2Threshold=400, detectShadows=False
            )
    
    @monitor.time_section("background_capture")
    def capture_background(self, camera: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Capture initial background with motion detection."""
        frames = []
        motion_scores = []
        
        logger.info("Capturing background...")
        
        for i in range(self.config.background_frames):
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Preprocess frame
            frame = cv2.resize(frame, self.config.frame_size)
            frame = cv2.flip(frame, 1)
            
            # Calculate motion score
            if len(frames) > 0:
                prev_gray = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = cv2.absdiff(prev_gray, curr_gray)
                motion_score = np.mean(motion)
                motion_scores.append(motion_score)
                
                # Only keep frames with low motion
                if motion_score < 10:  # Threshold for motion
                    frames.append(frame)
            else:
                frames.append(frame)
            
            if len(frames) >= 20:  # Minimum samples
                break
        
        if not frames:
            logger.error("Failed to capture background")
            return None
        
        # Use median or average based on motion
        if np.mean(motion_scores) < 5:
            self.background = np.median(frames, axis=0).astype(np.uint8)
        else:
            self.background = np.mean(frames, axis=0).astype(np.uint8)
        
        # Initialize background model
        if self.background_model is not None:
            for frame in frames:
                self.background_model.apply(frame)
        
        logger.info(f"Background captured from {len(frames)} frames")
        return self.background
    
    @monitor.time_section("background_update")
    def update_background(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Update background model with new information."""
        if self.background is None:
            self.background = frame.copy()
            return
        
        if self.background_model is not None:
            # Update background model
            fg_mask = self.background_model.apply(frame)
            
            # Extract background from model
            if self.config.background_model == "mog2":
                bg_from_model = self.background_model.getBackgroundImage()
                if bg_from_model is not None:
                    # Blend with current background
                    alpha = self.config.background_update_rate
                    self.background = cv2.addWeighted(
                        self.background, 1 - alpha,
                        bg_from_model, alpha, 0
                    )
        
        # Adaptive background update in non-masked areas
        inverse_mask = cv2.bitwise_not(mask)
        update_region = inverse_mask > 0
        
        if np.any(update_region):
            alpha = self.config.background_update_rate * 0.1  # Slower update
            
            # Update only in non-cloak areas
            self.background[update_region] = cv2.addWeighted(
                self.background[update_region], 1 - alpha,
                frame[update_region], alpha, 0
            ).astype(np.uint8)
        
        # Store in history
        self.background_history.append(self.background.copy())
    
    @monitor.time_section("background_restoration")
    def restore_background(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Restore background in masked areas using inpainting."""
        if self.background is None or not self.config.enable_background_restoration:
            return frame.copy()
        
        # Create inpainted version
        if np.count_nonzero(mask) > 100:
            # Use Telea inpainting for small holes
            inpainted = cv2.inpaint(
                frame, mask,
                inpaintRadius=self.inpainting_radius,
                flags=cv2.INPAINT_TELEA
            )
            
            # Blend with learned background
            alpha = 0.3  # Weight for inpainted result
            restored = cv2.addWeighted(
                inpainted, alpha,
                self.background, 1 - alpha, 0
            )
            
            return restored
        
        return frame.copy()


# =============================================================================
#                           GPU ACCELERATOR
# =============================================================================

class GPUAccelerator:
    """GPU acceleration using OpenCV's UMat and optional CuPy."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.use_gpu = config.enable_gpu and cv2.ocl.haveOpenCL()
        self.use_cupy = CUDA_AVAILABLE and config.enable_gpu
        
        if self.use_gpu:
            cv2.ocl.setUseOpenCL(True)
            logger.info("OpenCL acceleration enabled")
        
        if self.use_cupy:
            logger.info("CuPy acceleration available")
        
        # Create GPU contexts
        self._init_gpu_contexts()
    
    def _init_gpu_contexts(self):
        """Initialize GPU contexts for different operations."""
        self.contexts = {}
        
        if self.use_gpu:
            # Create context for color conversion
            self.contexts['color'] = cv2.ocl.Context()
            
            # Create context for filtering
            self.contexts['filter'] = cv2.ocl.Context()
    
    def to_gpu(self, frame: np.ndarray) -> Any:
        """Convert frame to GPU memory."""
        if self.use_cupy:
            return cp.asarray(frame)
        elif self.use_gpu:
            return cv2.UMat(frame)
        else:
            return frame
    
    def from_gpu(self, gpu_frame: Any) -> np.ndarray:
        """Convert from GPU memory to CPU."""
        if isinstance(gpu_frame, cp.ndarray):
            return cp.asnumpy(gpu_frame)
        elif isinstance(gpu_frame, cv2.UMat):
            return gpu_frame.get()
        else:
            return gpu_frame
    
    @monitor.time_section("gpu_color_conversion")
    def cvt_color(self, frame: Any, code: int) -> Any:
        """Color conversion on GPU."""
        if self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.cvtColor(frame, code)
        elif self.use_cupy and isinstance(frame, cp.ndarray):
            # Implement custom CuPy color conversion if needed
            return cv2.cvtColor(self.from_gpu(frame), code)
        else:
            return cv2.cvtColor(frame, code)
    
    @monitor.time_section("gpu_filter")
    def gaussian_blur(self, frame: Any, ksize: Tuple[int, int], sigma: float = 0) -> Any:
        """Gaussian blur on GPU."""
        if self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.GaussianBlur(frame, ksize, sigma)
        elif self.use_cupy and isinstance(frame, cp.ndarray):
            # Use CuPy convolution
            import cupyx.scipy.ndimage
            sigma_px = sigma if sigma > 0 else 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8
            return cupyx.scipy.ndimage.gaussian_filter(frame, sigma_px)
        else:
            return cv2.GaussianBlur(frame, ksize, sigma)


# =============================================================================
#                           UI MANAGER
# =============================================================================

class UIManager:
    """Advanced UI manager with interactive controls."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # UI state
        self.show_controls = config.show_controls
        self.show_stats = config.show_stats
        self.show_debug = config.show_debug
        
        # Create named window
        self.window_name = "Invisibility Cloak 4.0"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *config.frame_size)
        
        # Create trackbars for real-time adjustment
        if self.show_controls:
            self._create_trackbars()
        
        # UI elements cache
        self.ui_cache = {}
    
    def _create_trackbars(self):
        """Create interactive trackbars."""
        cv2.createTrackbar('Color Sensitivity', self.window_name, 
                          int(self.config.color_sensitivity * 100), 100, 
                          self._on_sensitivity_change)
        cv2.createTrackbar('Mask Smoothing', self.window_name,
                          int(self.config.mask_smoothing * 100), 100,
                          self._on_smoothing_change)
        cv2.createTrackbar('Feather Amount', self.window_name,
                          int(self.config.feather_amount * 100), 100,
                          self._on_feather_change)
    
    def _on_sensitivity_change(self, value: int):
        self.config.color_sensitivity = value / 100.0
    
    def _on_smoothing_change(self, value: int):
        self.config.mask_smoothing = value / 100.0
    
    def _on_feather_change(self, value: int):
        self.config.feather_amount = value / 100.0
    
    @monitor.time_section("ui_render")
    def render(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Render UI elements on frame."""
        output = frame.copy()
        
        if self.show_stats:
            output = self._render_stats_panel(output, stats)
        
        if self.show_debug:
            output = self._render_debug_info(output, stats)
        
        return output
    
    def _render_stats_panel(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Render statistics panel."""
        panel_height = 200
        panel_width = 350
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), 
                     (0, 0, 0), -1)
        
        # Add stats text
        stats_text = [
            f"FPS: {stats.get('fps', 0):.1f}",
            f"Frame: {stats.get('frame_count', 0)}",
            f"Mask Area: {stats.get('mask_area', 0)*100:.1f}%",
            f"Confidence: {stats.get('confidence', 0):.2f}",
            f"Processing: {stats.get('processing_mode', 'N/A')}",
            f"GPU: {'ON' if stats.get('gpu_enabled', False) else 'OFF'}",
            f"Colors: {stats.get('num_colors', 0)}",
            f"Latency: {stats.get('pipeline_time', 0)*1000:.1f}ms",
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (20, 40 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 1)
        
        # Blend with original frame
        alpha = 0.7
        frame[10:panel_height, 10:panel_width] = cv2.addWeighted(
            frame[10:panel_height, 10:panel_width], 1 - alpha,
            overlay[10:panel_height, 10:panel_width], alpha, 0
        )
        
        return frame
    
    def _render_debug_info(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Render debug information."""
        debug_text = [
            f"Frame Size: {frame.shape[1]}x{frame.shape[0]}",
            f"Memory: {sys.getsizeof(frame) / 1024:.1f} KB",
            f"Time/Frame: {stats.get('frame_time', 0)*1000:.1f}ms",
        ]
        
        for i, text in enumerate(debug_text):
            cv2.putText(frame, text, (frame.shape[1] - 300, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 0), 1)
        
        return frame
    
    def create_debug_window(self, image: np.ndarray, title: str = "Debug"):
        """Create separate debug window."""
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image)


# =============================================================================
#                           MAIN PIPELINE
# =============================================================================

class InvisibilityCloakPipeline:
    """Main processing pipeline orchestrating all components."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Initialize components
        self.color_detector = ColorDetector(config)
        self.mask_processor = MaskProcessor(config)
        self.background_manager = BackgroundManager(config)
        self.gpu_accelerator = GPUAccelerator(config) if config.enable_gpu else None
        self.ui_manager = UIManager(config)
        
        # Pipeline state
        self.is_initialized = False
        self.current_frame = None
        self.current_mask = None
        
        # Frame buffer for async processing
        self.frame_buffer = deque(maxlen=3)
        
        # Thread pool for parallel processing
        if config.enable_multithreading:
            self.thread_pool = ThreadPoolExecutor(max_workers=config.num_threads)
        else:
            self.thread_pool = None
    
    @monitor.time_section("pipeline_initialize")
    def initialize(self, camera: cv2.VideoCapture) -> bool:
        """Initialize the pipeline."""
        # Capture background
        background = self.background_manager.capture_background(camera)
        if background is None:
            logger.error("Failed to initialize pipeline: No background")
            return False
        
        self.is_initialized = True
        logger.info("Pipeline initialized successfully")
        return True
    
    @monitor.time_section("pipeline_process")
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame through the pipeline."""
        if not self.is_initialized:
            return frame, {"error": "Pipeline not initialized"}
        
        pipeline_start = time.perf_counter()
        
        # Store current frame
        self.current_frame = frame.copy()
        
        # Step 1: Color detection
        raw_mask, color_stats = self.color_detector.detect_colors(frame)
        
        # Step 2: Mask processing
        processed_mask = self.mask_processor.process_mask(raw_mask, frame)
        self.current_mask = processed_mask
        
        # Step 3: Background restoration
        restored_background = self.background_manager.restore_background(
            frame, processed_mask
        )
        
        # Step 4: Apply cloak effect
        if self.background_manager.background is not None:
            output_frame = self._apply_cloak_effect(
                frame, processed_mask, restored_background
            )
        else:
            output_frame = frame
        
        # Step 5: Update background model
        self.background_manager.update_background(frame, processed_mask)
        
        # Step 6: Prepare statistics
        pipeline_time = time.perf_counter() - pipeline_start
        stats = {
            **color_stats,
            "pipeline_time": pipeline_time,
            "frame_time": 1.0 / self.monitor.get_fps() if self.monitor.get_fps() > 0 else 0,
            "fps": self.monitor.get_fps(),
            "frame_count": self.monitor.frame_count,
            "gpu_enabled": self.config.enable_gpu,
            "processing_mode": self.config.processing_mode,
        }
        
        # Update monitor
        self.monitor.frame_count += 1
        
        return output_frame, stats
    
    def _apply_cloak_effect(self, frame: np.ndarray, mask: np.ndarray, 
                           background: np.ndarray) -> np.ndarray:
        """Apply the invisibility cloak effect."""
        # Convert mask to 3 channels for blending
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Create alpha blend
        alpha = self.config.color_sensitivity
        
        # Blend frame with background based on mask
        foreground = frame.astype(np.float32)
        background = background.astype(np.float32)
        
        # Apply mask blending
        blended = foreground * (1 - mask_3ch * alpha) + \
                 background * (mask_3ch * alpha)
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def async_process(self, frame: np.ndarray) -> Any:
        """Process frame asynchronously if threading is enabled."""
        if self.thread_pool is None:
            return self.process_frame(frame)
        
        future = self.thread_pool.submit(self.process_frame, frame)
        return future
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        pipeline_stats = self.monitor.get_statistics()
        
        # Combine component stats
        component_stats = {
            "color_detector": self.color_detector.monitor.get_statistics(),
            "mask_processor": self.mask_processor.monitor.get_statistics(),
            "background_manager": self.background_manager.monitor.get_statistics(),
        }
        
        return {
            "pipeline": pipeline_stats,
            "components": component_stats,
            "system": {
                "threads": self.config.num_threads,
                "gpu": self.config.enable_gpu,
                "frame_size": self.config.frame_size,
            }
        }


# =============================================================================
#                           MAIN APPLICATION
# =============================================================================

class InvisibilityCloakSystem:
    """Main application class with enhanced features."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.pipeline = InvisibilityCloakPipeline(self.config)
        
        # Application state
        self.is_running = False
        self.camera = None
        
        # Performance tracking
        self.start_time = time.perf_counter()
        self.frame_counter = 0
        
        # Settings
        self.settings_file = Path("cloak_settings_v4.json")
    
    def setup_camera(self) -> bool:
        """Setup camera with optimal settings."""
        self.camera = cv2.VideoCapture(self.config.camera_index)
        
        if not self.camera.isOpened():
            logger.error(f"Cannot open camera at index {self.config.camera_index}")
            return False
        
        # Configure camera for optimal performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Verify camera settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        return True
    
    def load_settings(self) -> bool:
        """Load settings from file."""
        if not self.settings_file.exists():
            logger.warning("Settings file not found, using defaults")
            return False
        
        try:
            with open(self.settings_file, 'r') as f:
                saved_settings = json.load(f)
            
            # Update config with saved settings
            for key, value in saved_settings.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            logger.info("Settings loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False
    
    def save_settings(self) -> bool:
        """Save current settings to file."""
        try:
            settings_dict = asdict(self.config)
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings_dict, f, indent=2)
            
            logger.info("Settings saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def handle_keyboard(self, key: int) -> bool:
        """Handle keyboard input."""
        actions = {
            ord('q'): lambda: setattr(self, 'is_running', False),
            ord('s'): self.save_settings,
            ord('l'): self.load_settings,
            ord('d'): self.toggle_debug,
            ord('c'): self.toggle_controls,
            ord('r'): self.reset_background,
            ord('1'): lambda: self.change_color_preset('green'),
            ord('2'): lambda: self.change_color_preset('red'),
            ord('3'): lambda: self.change_color_preset('blue'),
            ord('4'): lambda: self.change_color_preset('custom'),
            ord('f'): lambda: self.change_processing_mode('fast'),
            ord('b'): lambda: self.change_processing_mode('balanced'),
            ord('q'): lambda: self.change_processing_mode('quality'),
            ord('+'): self.increase_sensitivity,
            ord('-'): self.decrease_sensitivity,
            ord(' '): self.capture_background_now,
            27: lambda: setattr(self, 'is_running', False),  # ESC
        }
        
        if key in actions:
            actions[key]()
            return True
        
        return False
    
    def toggle_debug(self):
        self.config.show_debug = not self.config.show_debug
        logger.info(f"Debug mode: {'ON' if self.config.show_debug else 'OFF'}")
    
    def toggle_controls(self):
        self.config.show_controls = not self.config.show_controls
        logger.info(f"Controls: {'SHOWN' if self.config.show_controls else 'HIDDEN'}")
    
    def reset_background(self):
        logger.info("Resetting background...")
        if self.camera is not None:
            self.pipeline.background_manager.capture_background(self.camera)
    
    def change_color_preset(self, preset: str):
        if preset in self.pipeline.color_detector.color_presets:
            self.pipeline.color_detector.current_color = preset
            logger.info(f"Color preset changed to: {preset}")
    
    def change_processing_mode(self, mode: str):
        self.config.processing_mode = mode
        self.pipeline.mask_processor._init_kernels()
        logger.info(f"Processing mode changed to: {mode}")
    
    def increase_sensitivity(self):
        self.config.color_sensitivity = min(1.0, self.config.color_sensitivity + 0.05)
        logger.info(f"Color sensitivity: {self.config.color_sensitivity:.2f}")
    
    def decrease_sensitivity(self):
        self.config.color_sensitivity = max(0.0, self.config.color_sensitivity - 0.05)
        logger.info(f"Color sensitivity: {self.config.color_sensitivity:.2f}")
    
    def capture_background_now(self):
        logger.info("Manual background capture triggered")
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.resize(frame, self.config.frame_size)
                frame = cv2.flip(frame, 1)
                self.pipeline.background_manager.background = frame.copy()
    
    def run(self):
        """Main application loop."""
        logger.info("Starting Invisibility Cloak System 4.0")
        
        # Setup camera
        if not self.setup_camera():
            return
        
        # Load settings
        self.load_settings()
        
        # Initialize pipeline
        if not self.pipeline.initialize(self.camera):
            self.camera.release()
            return
        
        # Main loop
        self.is_running = True
        frame_times = deque(maxlen=30)
        
        try:
            while self.is_running:
                loop_start = time.perf_counter()
                
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.001)
                    continue
                
                # Process frame
                output_frame, stats = self.pipeline.process_frame(frame)
                
                # Apply UI
                output_frame = self.pipeline.ui_manager.render(output_frame, stats)
                
                # Display
                cv2.imshow(self.pipeline.ui_manager.window_name, output_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                self.handle_keyboard(key)
                
                # Check window close
                if cv2.getWindowProperty(self.pipeline.ui_manager.window_name, 
                                        cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # Update performance tracking
                frame_time = time.perf_counter() - loop_start
                frame_times.append(frame_time)
                self.frame_counter += 1
                
                # Log performance periodically
                if self.frame_counter % 100 == 0:
                    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
                    logger.info(f"Performance: {avg_fps:.1f} FPS, "
                               f"Mask: {stats.get('mask_area', 0)*100:.1f}%, "
                               f"Confidence: {stats.get('confidence', 0):.2f}")
        
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")
        
        # Save settings
        self.save_settings()
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Shutdown thread pool
        if self.pipeline.thread_pool is not None:
            self.pipeline.thread_pool.shutdown(wait=True)
        
        # Print performance summary
        total_time = time.perf_counter() - self.start_time
        avg_fps = self.frame_counter / total_time if total_time > 0 else 0
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info(f"Total frames: {self.frame_counter}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Frame size: {self.config.frame_size}")
        
        # Detailed component stats
        perf_stats = self.pipeline.get_performance_stats()
        for component, stats in perf_stats["components"].items():
            if "section_times" in stats:
                logger.info(f"{component}:")
                for section, times in stats["section_times"].items():
                    logger.info(f"  {section}: {times['mean']*1000:.1f}ms "
                              f"(min: {times['min']*1000:.1f}ms, "
                              f"max: {times['max']*1000:.1f}ms)")
        
        logger.info("=" * 60)


# =============================================================================
#                           ENTRY POINT
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Invisibility Cloak System 4.0 - Advanced real-time background replacement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')
    parser.add_argument('--width', type=int, default=1280,
                       help='Frame width')
    parser.add_argument('--height', type=int, default=720,
                       help='Frame height')
    parser.add_argument('--fps', type=int, default=60,
                       help='Target FPS')
    parser.add_argument('--mode', choices=['fast', 'balanced', 'quality'],
                       default='balanced', help='Processing mode')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration')
    parser.add_argument('--no-gpu', dest='gpu', action='store_false',
                       help='Disable GPU acceleration')
    parser.set_defaults(gpu=CUDA_AVAILABLE)
    parser.add_argument('--threads', type=int, default=None,
                       help='Number of threads (default: CPU cores - 1)')
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-ui', dest='show_ui', action='store_false',
                       help='Disable UI controls')
    parser.set_defaults(show_ui=True)
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Create configuration
    config = SystemConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.fps,
        processing_mode=args.mode,
        enable_gpu=args.gpu,
        enable_multithreading=True,
        num_threads=args.threads or max(1, mp.cpu_count() - 1),
        enable_profiling=args.profile,
        show_controls=args.show_ui,
        show_debug=args.debug,
    )
    
    # Optional profiling
    if args.profile:
        import cProfile
        import pstats
        from pstats import SortKey
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            system = InvisibilityCloakSystem(config)
            system.run()
        finally:
            profiler.disable()
            
            # Save and display profiling data
            stats = pstats.Stats(profiler)
            stats.sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(20)
            
            # Save to file
            stats.dump_stats('performance_profile.prof')
            logger.info("Profiling data saved to 'performance_profile.prof'")
    else:
        # Run normally
        system = InvisibilityCloakSystem(config)
        system.run()


if __name__ == "__main__":
    main()
