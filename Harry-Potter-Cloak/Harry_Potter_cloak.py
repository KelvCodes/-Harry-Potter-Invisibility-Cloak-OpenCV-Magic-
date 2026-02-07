
import asyncio
import json
import logging
import multiprocessing as mp
import sys
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Optional imports for enhanced features
try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cloak_system.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
#                           CONFIGURATION ENHANCEMENTS
# ============================================================================

@dataclass
class ColorPreset:
    """Enhanced color preset with adaptive learning."""
    name: str
    hsv_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    lab_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]]
    weight: float = 1.0
    adaptive: bool = True
    history_size: int = 100
    confidence: float = 0.0
    samples: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=100))
    
    def update(self, sample: np.ndarray) -> None:
        """Update color model with new sample."""
        if not self.adaptive:
            return
        
        self.samples.append(sample)
        if len(self.samples) >= 10:
            self._adapt_from_samples()
    
    def _adapt_from_samples(self) -> None:
        """Adapt color ranges from collected samples."""
        if not self.samples:
            return
        
        samples_array = np.array(self.samples)
        mean_hsv = np.mean(samples_array, axis=0)
        std_hsv = np.std(samples_array, axis=0)
        
        # Adaptive widening based on variance
        std_factor = 1.5
        new_lower = np.clip(mean_hsv - std_factor * std_hsv, 0, 255).astype(int)
        new_upper = np.clip(mean_hsv + std_factor * std_hsv, 0, 255).astype(int)
        
        # Update ranges
        self.hsv_range = (tuple(new_lower), tuple(new_upper))
        
        # Update confidence based on sample consistency
        self.confidence = 1.0 - np.minimum(np.mean(std_hsv) / 50.0, 1.0)


@dataclass
class SystemConfig:
    """Enhanced system configuration with validation."""
    
    # Camera settings
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 60
    buffer_size: int = 1
    
    # Processing settings
    processing_mode: str = "balanced"  # fast, balanced, quality, dl
    enable_gpu: bool = CUDA_AVAILABLE
    enable_multithreading: bool = True
    num_threads: int = field(default_factory=lambda: max(1, mp.cpu_count() - 1))
    use_onnx: bool = ONNX_AVAILABLE
    
    # Color detection
    color_sensitivity: float = 0.85
    adaptive_threshold: bool = True
    use_multi_color: bool = True
    max_colors: int = 3
    color_learning_rate: float = 0.1
    use_deep_color: bool = True
    
    # Background management
    background_frames: int = 30
    background_update_rate: float = 0.05
    enable_background_restoration: bool = True
    background_model: str = "mog2"  # static, mog2, knn, u2net
    background_cache_size: int = 10
    
    # Mask processing
    mask_smoothing: float = 0.75
    temporal_stability: int = 7
    feather_amount: float = 0.15
    min_mask_area: float = 0.005
    max_mask_area: float = 0.35
    use_refinement_network: bool = False
    
    # Performance
    enable_profiling: bool = False
    stats_window: int = 100
    cache_size: int = 15
    use_pipeline_optimization: bool = True
    
    # UI settings
    show_controls: bool = True
    show_stats: bool = True
    show_debug: bool = False
    show_mask_overlay: bool = False
    theme: str = "dark"  # dark, light, matrix
    
    # Export settings
    save_video: bool = False
    video_format: str = "mp4v"
    save_frames: bool = False
    output_dir: str = "output"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
        
    def _validate(self):
        """Validate configuration parameters."""
        assert 0 < self.frame_width <= 7680, "Invalid frame width (max 8K)"
        assert 0 < self.frame_height <= 4320, "Invalid frame height (max 8K)"
        assert 1 <= self.target_fps <= 480, "Invalid FPS target"
        assert 0 <= self.color_sensitivity <= 1.5, "Invalid sensitivity"
        assert 0 <= self.mask_smoothing <= 1, "Invalid smoothing"
        assert 0 <= self.feather_amount <= 1, "Invalid feather amount"
        assert 0 <= self.min_mask_area < self.max_mask_area <= 1, "Invalid mask area bounds"
        
        if self.processing_mode not in ["fast", "balanced", "quality", "dl"]:
            logger.warning(f"Invalid processing mode: {self.processing_mode}, using balanced")
            self.processing_mode = "balanced"
        
        if self.background_model not in ["static", "mog2", "knn", "u2net"]:
            logger.warning(f"Invalid background model: {self.background_model}, using mog2")
            self.background_model = "mog2"
        
        if self.theme not in ["dark", "light", "matrix"]:
            self.theme = "dark"
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.frame_width, self.frame_height)
    
    @property
    def frame_area(self) -> int:
        return self.frame_width * self.frame_height
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        return cls(**data)


class ProcessingMode(Enum):
    """Enhanced processing modes."""
    FAST = auto()
    BALANCED = auto()
    QUALITY = auto()
    DEEP_LEARNING = auto()
    
    @classmethod
    def from_string(cls, mode_str: str) -> 'ProcessingMode':
        mapping = {
            "fast": cls.FAST,
            "balanced": cls.BALANCED,
            "quality": cls.QUALITY,
            "dl": cls.DEEP_LEARNING
        }
        return mapping.get(mode_str.lower(), cls.BALANCED)


# ============================================================================
#                           ENHANCED PERFORMANCE MONITOR
# ============================================================================

class EnhancedPerformanceMonitor:
    """High-performance timing and statistics with ML predictions."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.timings: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.counters: Dict[str, int] = defaultdict(int)
        self.predictions: Dict[str, float] = {}
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        
    def time_section(self, section_name: str) -> Callable:
        """Decorator for timing code sections."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                self.timings[section_name].append(elapsed)
                return result
            return wrapper
        return decorator
    
    def increment(self, counter_name: str, amount: int = 1) -> None:
        """Increment a counter."""
        self.counters[counter_name] += amount
    
    def get_fps(self) -> float:
        """Calculate current FPS."""
        if len(self.fps_history) > 0:
            return np.mean(self.fps_history)
        total_time = time.perf_counter() - self.start_time
        return self.frame_count / total_time if total_time > 0 else 0.0
    
    def update_fps(self, frame_time: float) -> None:
        """Update FPS calculation."""
        if frame_time > 0:
            self.fps_history.append(1.0 / frame_time)
    
    def get_average_time(self, section_name: str) -> float:
        """Get average time for a section."""
        timings = self.timings.get(section_name, [])
        return np.mean(timings) if timings else 0.0
    
    def predict_bottleneck(self) -> Dict[str, float]:
        """Predict performance bottlenecks."""
        if len(self.frame_count) < 10:
            return {}
        
        bottlenecks = {}
        for section, timings in self.timings.items():
            if len(timings) > 5:
                mean_time = np.mean(list(timings))
                std_time = np.std(list(timings))
                if std_time > mean_time * 0.5:  # High variance indicates bottleneck
                    bottlenecks[section] = std_time / mean_time
        
        return bottlenecks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics with predictions."""
        stats = {
            "fps": self.get_fps(),
            "frame_count": self.frame_count,
            "runtime": time.perf_counter() - self.start_time,
            "section_times": {},
            "counters": dict(self.counters),
            "bottlenecks": self.predict_bottleneck(),
            "predictions": self.predictions
        }
        
        for section, timings in self.timings.items():
            if timings:
                timings_list = list(timings)
                stats["section_times"][section] = {
                    "mean": np.mean(timings_list),
                    "min": np.min(timings_list),
                    "max": np.max(timings_list),
                    "std": np.std(timings_list),
                    "95th": np.percentile(timings_list, 95),
                    "median": np.median(timings_list)
                }
        
        return stats
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.timings.clear()
        self.counters.clear()
        self.predictions.clear()
        self.fps_history.clear()
        self.start_time = time.perf_counter()
        self.frame_count = 0


# ============================================================================
#                           ENHANCED COLOR DETECTOR
# ============================================================================

class EnhancedColorDetector:
    """Advanced color detection with deep learning and adaptive learning."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = EnhancedPerformanceMonitor()
        
        # Initialize color presets
        self.color_presets = self._initialize_color_presets()
        self.current_color = 'green'
        
        # Initialize deep learning model if available
        self.dl_model = self._init_dl_model() if config.use_deep_color else None
        self.color_history = deque(maxlen=100)
        
        # Adaptive model
        self.adaptive_model = self._init_adaptive_model()
        
        # Color space cache
        self.cache_enabled = True
        self.frame_cache = deque(maxlen=config.cache_size)
    
    def _initialize_color_presets(self) -> Dict[str, ColorPreset]:
        """Initialize enhanced color presets."""
        presets = {
            'green': ColorPreset(
                name='green',
                hsv_range=((35, 40, 40), (85, 255, 255)),
                lab_range=((0, 0, 0), (255, 255, 255)),
                weight=1.0
            ),
            'red': ColorPreset(
                name='red',
                hsv_range=((0, 120, 70), (10, 255, 255)),
                lab_range=((0, 0, 0), (255, 255, 255)),
                weight=1.0
            ),
            'blue': ColorPreset(
                name='blue',
                hsv_range=((100, 40, 40), (140, 255, 255)),
                lab_range=((0, 0, 0), (255, 255, 255)),
                weight=1.0
            ),
            'custom': ColorPreset(
                name='custom',
                hsv_range=((0, 0, 0), (180, 255, 255)),
                lab_range=((0, 0, 0), (255, 255, 255)),
                weight=0.5,
                adaptive=True
            )
        }
        return presets
    
    def _init_dl_model(self):
        """Initialize deep learning color segmentation model."""
        if not ONNX_AVAILABLE:
            return None
        
        try:
            # Load a lightweight segmentation model
            model_path = Path("models/color_segmentation.onnx")
            if model_path.exists():
                session = ort.InferenceSession(str(model_path))
                logger.info("Deep learning color model loaded")
                return session
        except Exception as e:
            logger.warning(f"Failed to load DL model: {e}")
        
        return None
    
    def _init_adaptive_model(self):
        """Initialize adaptive color learning model."""
        if SKLEARN_AVAILABLE and self.config.adaptive_threshold:
            try:
                # Use isolation forest for anomaly detection
                model = IsolationForest(
                    n_estimators=100,
                    contamination=0.1,
                    random_state=42
                )
                return model
            except Exception as e:
                logger.warning(f"Failed to init adaptive model: {e}")
        return None
    
    @monitor.time_section("enhanced_color_detection")
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced color detection with multiple strategies."""
        if self.dl_model is not None and self.config.use_deep_color:
            return self._detect_dl(frame)
        else:
            return self._detect_traditional(frame)
    
    def _detect_traditional(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Traditional color detection with enhancements."""
        # Convert to multiple color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        masks = []
        confidences = []
        color_stats = []
        
        for color_name, preset in self.color_presets.items():
            if color_name == 'custom' and not self.config.use_multi_color:
                continue
            
            # Enhanced HSV detection with adaptive ranges
            hsv_lower = np.array(preset.hsv_range[0], dtype=np.uint8)
            hsv_upper = np.array(preset.hsv_range[1], dtype=np.uint8)
            hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            
            # LAB detection for better color discrimination
            lab_lower = np.array(preset.lab_range[0], dtype=np.uint8)
            lab_upper = np.array(preset.lab_range[1], dtype=np.uint8)
            lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
            
            # Combine masks with morphological operations
            combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
            
            # Adaptive thresholding
            if self.config.adaptive_threshold:
                combined_mask = self._apply_adaptive_threshold(combined_mask)
            
            if np.count_nonzero(combined_mask) > 0:
                masks.append(combined_mask)
                mask_area = np.count_nonzero(combined_mask) / self.config.frame_area
                
                # Calculate enhanced confidence
                edge_density = self._calculate_edge_density(combined_mask)
                spatial_coherence = self._calculate_spatial_coherence(combined_mask)
                
                confidence = preset.weight * (
                    0.4 * (1.0 - abs(0.2 - mask_area) / 0.2) +
                    0.3 * edge_density +
                    0.3 * spatial_coherence
                )
                
                confidences.append(confidence)
                color_stats.append({
                    'name': color_name,
                    'area': mask_area,
                    'confidence': confidence
                })
        
        if not masks:
            return np.zeros(frame.shape[:2], dtype=np.uint8), {
                "confidence": 0,
                "colors_detected": 0
            }
        
        # Multi-color fusion
        combined_mask = self._fuse_masks(masks, confidences)
        
        # Update adaptive models
        if np.count_nonzero(combined_mask) > 100:
            self._update_color_models(frame, combined_mask, color_stats)
        
        stats = {
            "confidence": np.mean(confidences) if confidences else 0,
            "num_colors": len(masks),
            "mask_area": np.count_nonzero(combined_mask) / self.config.frame_area,
            "color_stats": color_stats
        }
        
        return combined_mask, stats
    
    def _detect_dl(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Deep learning based color detection."""
        try:
            # Preprocess frame for model
            input_data = self._preprocess_for_dl(frame)
            
            # Run inference
            input_name = self.dl_model.get_inputs()[0].name
            output_name = self.dl_model.get_outputs()[0].name
            output = self.dl_model.run([output_name], {input_name: input_data})[0]
            
            # Post-process output
            mask = self._postprocess_dl_output(output, frame.shape[:2])
            
            stats = {
                "confidence": 0.9,  # DL typically has high confidence
                "num_colors": 1,
                "mask_area": np.count_nonzero(mask) / self.config.frame_area,
                "method": "deep_learning"
            }
            
            return mask, stats
            
        except Exception as e:
            logger.warning(f"DL detection failed: {e}, falling back to traditional")
            return self._detect_traditional(frame)
    
    def _apply_adaptive_threshold(self, mask: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding to mask."""
        if np.count_nonzero(mask) < 100:
            return mask
        
        # Calculate local threshold
        mean_intensity = np.mean(mask[mask > 0])
        _, adaptive_mask = cv2.threshold(
            mask, 
            max(50, mean_intensity * 0.7), 
            255, 
            cv2.THRESH_BINARY
        )
        
        return adaptive_mask
    
    def _calculate_edge_density(self, mask: np.ndarray) -> float:
        """Calculate edge density for confidence estimation."""
        edges = cv2.Canny(mask, 100, 200)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        return min(edge_pixels / total_pixels * 10, 1.0)
    
    def _calculate_spatial_coherence(self, mask: np.ndarray) -> float:
        """Calculate spatial coherence of mask."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        # Calculate convexity defects
        largest_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_contour, returnPoints=False)
        
        try:
            defects = cv2.convexityDefects(largest_contour, hull)
            if defects is not None:
                defect_depth = np.mean(defects[:, 0, 3]) / 256.0
                coherence = 1.0 - min(defect_depth, 1.0)
            else:
                coherence = 1.0
        except:
            coherence = 0.8
        
        return coherence
    
    def _fuse_masks(self, masks: List[np.ndarray], 
                   confidences: List[float]) -> np.ndarray:
        """Intelligently fuse multiple masks."""
        if len(masks) == 1:
            return masks[0]
        
        # Create weighted combination
        total_confidence = sum(confidences)
        if total_confidence > 0:
            combined = np.zeros_like(masks[0], dtype=np.float32)
            for mask, confidence in zip(masks, confidences):
                combined += mask.astype(np.float32) * (confidence / total_confidence)
            
            # Apply non-linear fusion
            combined = np.clip(combined * 1.2, 0, 255).astype(np.uint8)
            
            # Use morphological reconstruction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        else:
            combined = masks[0]
        
        return combined
    
    def _update_color_models(self, frame: np.ndarray, mask: np.ndarray, 
                           color_stats: List[Dict[str, Any]]) -> None:
        """Update color models with new detections."""
        for color_stat in color_stats:
            color_name = color_stat['name']
            if color_name in self.color_presets:
                preset = self.color_presets[color_name]
                
                # Extract sample pixels
                if color_stat['confidence'] > 0.5:
                    color_mask = (mask > 0).astype(np.uint8)
                    masked_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
                    samples = masked_hsv[color_mask > 0]
                    
                    if len(samples) > 100:
                        # Randomly select samples
                        selected_samples = samples[
                            np.random.choice(len(samples), min(100, len(samples)), replace=False)
                        ]
                        for sample in selected_samples:
                            preset.update(sample)
    
    def _preprocess_for_dl(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for deep learning model."""
        # Resize to model input size
        resized = cv2.resize(frame, (256, 256))
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        # Add batch dimension and transpose for ONNX
        return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    
    def _postprocess_dl_output(self, output: np.ndarray, 
                              original_shape: Tuple[int, int]) -> np.ndarray:
        """Post-process DL output to mask."""
        # Get prediction
        pred = output[0, 0, :, :]  # Assuming single channel output
        # Resize to original size
        resized = cv2.resize(pred, (original_shape[1], original_shape[0]))
        # Threshold
        mask = (resized > 0.5).astype(np.uint8) * 255
        return mask


# ============================================================================
#                           ENHANCED MASK PROCESSOR
# ============================================================================

class EnhancedMaskProcessor:
    """Advanced mask processing with deep learning refinement."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = EnhancedPerformanceMonitor()
        
        self._init_kernels()
        self.mask_history = deque(maxlen=config.temporal_stability * 2)
        self.flow_history = deque(maxlen=3)
        self.quality_history = deque(maxlen=10)
        
        # Initialize refinement network if available
        self.refinement_model = self._init_refinement_model() if config.use_refinement_network else None
        
        # Optical flow parameters
        self.prev_gray = None
        self.prev_pyramid = None
        
    def _init_kernels(self) -> None:
        """Initialize processing kernels based on mode."""
        mode = ProcessingMode.from_string(self.config.processing_mode)
        
        if mode == ProcessingMode.FAST:
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.blur_kernel = (5, 5)
            self.iterations = 1
            self.flow_scale = 0.5
        elif mode == ProcessingMode.BALANCED:
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self.blur_kernel = (7, 7)
            self.iterations = 2
            self.flow_scale = 0.7
        elif mode == ProcessingMode.QUALITY:
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            self.blur_kernel = (9, 9)
            self.iterations = 3
            self.flow_scale = 0.9
        else:  # DEEP_LEARNING
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.blur_kernel = (3, 3)
            self.iterations = 1
            self.flow_scale = 1.0
    
    def _init_refinement_model(self):
        """Initialize mask refinement neural network."""
        if not ONNX_AVAILABLE:
            return None
        
        try:
            model_path = Path("models/mask_refinement.onnx")
            if model_path.exists():
                session = ort.InferenceSession(str(model_path))
                logger.info("Mask refinement model loaded")
                return session
        except Exception as e:
            logger.warning(f"Failed to load refinement model: {e}")
        
        return None
    
    @monitor.time_section("enhanced_mask_processing")
    def process(self, mask: np.ndarray, 
                frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Enhanced mask processing pipeline."""
        if mask is None or mask.size == 0:
            return np.zeros(self.config.frame_size[::-1], dtype=np.uint8)
        
        processed = mask.copy()
        
        # 1. Initial cleaning
        processed = self._clean_mask(processed)
        
        # 2. Temporal stabilization with optical flow
        if frame is not None and len(self.mask_history) > 0:
            processed = self._stabilize_with_flow(processed, frame)
        
        # 3. Quality-based refinement
        quality = self._calculate_mask_quality(processed)
        self.quality_history.append(quality)
        
        if quality < 0.7 and frame is not None:
            processed = self._refine_low_quality_mask(processed, frame)
        
        # 4. Deep learning refinement (if available)
        if (self.refinement_model is not None and 
            frame is not None and 
            self.config.use_refinement_network):
            processed = self._apply_refinement(processed, frame)
        
        # 5. Final smoothing and feathering
        processed = self._finalize_mask(processed)
        
        # Update history
        self.mask_history.append(processed.copy())
        
        return processed
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean mask with adaptive morphological operations."""
        # Adaptive kernel based on mask size
        mask_area = np.count_nonzero(mask)
        kernel_size = max(3, min(11, int(np.sqrt(mask_area) / 50)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Remove noise
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Remove small regions
        min_area = self.config.frame_area * self.config.min_mask_area
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        result = cleaned.copy()
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.drawContours(result, [contour], 0, 0, -1)
        
        return result
    
    def _stabilize_with_flow(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Temporal stabilization using enhanced optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None and self.mask_history:
            # Build image pyramid for better flow estimation
            prev_pyramid = self._build_pyramid(self.prev_gray)
            curr_pyramid = self._build_pyramid(gray)
            
            # Calculate multi-scale optical flow
            flow = self._calculate_multi_scale_flow(
                prev_pyramid, curr_pyramid
            )
            
            # Warp previous mask
            prev_mask = self.mask_history[-1]
            warped = self._warp_with_flow(prev_mask, flow)
            
            # Adaptive blending based on flow confidence
            flow_confidence = self._calculate_flow_confidence(flow)
            alpha = self.config.mask_smoothing * flow_confidence
            
            # Blend with current mask
            stabilized = cv2.addWeighted(mask, 1 - alpha, warped, alpha, 0)
            
            self.flow_history.append(flow)
        else:
            stabilized = mask
        
        self.prev_gray = gray
        return stabilized
    
    def _build_pyramid(self, image: np.ndarray, levels: int = 3) -> List[np.ndarray]:
        """Build Gaussian pyramid for multi-scale processing."""
        pyramid = [image]
        for i in range(1, levels):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid
    
    def _calculate_multi_scale_flow(self, prev_pyramid: List[np.ndarray],
                                   curr_pyramid: List[np.ndarray]) -> np.ndarray:
        """Calculate optical flow using multi-scale approach."""
        # Start from coarsest level
        flow = None
        for i in range(len(prev_pyramid) - 1, -1, -1):
            prev_level = prev_pyramid[i]
            curr_level = curr_pyramid[i]
            
            if flow is not None:
                # Upscale previous flow
                h, w = prev_level.shape
                flow = cv2.resize(flow, (w, h)) * 2
            
            # Calculate flow at this level
            level_flow = cv2.calcOpticalFlowFarneback(
                prev_level, curr_level, flow,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            if flow is None:
                flow = level_flow
            else:
                flow = level_flow + flow
        
        return flow
    
    def _warp_with_flow(self, image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp image using optical flow."""
        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        map_x = (x + flow[..., 0]).astype(np.float32)
        map_y = (y + flow[..., 1]).astype(np.float32)
        
        # Warp image
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return warped
    
    def _calculate_flow_confidence(self, flow: np.ndarray) -> float:
        """Calculate confidence in optical flow estimation."""
        # Calculate flow magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Calculate confidence based on magnitude consistency
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        
        if std_mag == 0:
            return 1.0
        
        # Higher confidence when flow is consistent
        consistency = 1.0 / (1.0 + std_mag / (mean_mag + 1e-6))
        
        return min(max(consistency, 0.1), 1.0)
    
    def _calculate_mask_quality(self, mask: np.ndarray) -> float:
        """Calculate mask quality score."""
        if np.count_nonzero(mask) == 0:
            return 0.0
        
        # 1. Edge sharpness
        edges = cv2.Canny(mask, 100, 200)
        edge_intensity = np.mean(edges[edges > 0]) if np.any(edges > 0) else 0
        
        # 2. Spatial coherence
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(largest)
            convexity = contour_area / hull_area if hull_area > 0 else 0
        else:
            convexity = 0
        
        # 3. Smoothness
        laplacian = cv2.Laplacian(mask.astype(np.float32), cv2.CV_32F)
        smoothness = 1.0 / (1.0 + np.std(laplacian))
        
        # Combine metrics
        quality = (
            0.4 * (edge_intensity / 255.0) +
            0.4 * convexity +
            0.2 * smoothness
        )
        
        return min(max(quality, 0.0), 1.0)
    
    def _refine_low_quality_mask(self, mask: np.ndarray, 
                                frame: np.ndarray) -> np.ndarray:
        """Refine low quality mask using additional techniques."""
        refined = mask.copy()
        
        # Use GrabCut for refinement
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Create mask for GrabCut
        mask_gc = np.where(mask > 128, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        
        # Run GrabCut (few iterations for speed)
        try:
            cv2.grabCut(frame, mask_gc, None, bgd_model, fgd_model, 
                      2, cv2.GC_INIT_WITH_MASK)
            
            # Convert result to binary mask
            refined = np.where((mask_gc == cv2.GC_FGD) | 
                             (mask_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except:
            pass
        
        return refined
    
    def _apply_refinement(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Apply neural network refinement to mask."""
        try:
            # Prepare input
            input_data = self._prepare_refinement_input(mask, frame)
            
            # Run inference
            input_name = self.refinement_model.get_inputs()[0].name
            output_name = self.refinement_model.get_outputs()[0].name
            output = self.refinement_model.run(
                [output_name], {input_name: input_data}
            )[0]
            
            # Post-process
            refined = self._postprocess_refinement(output, mask.shape)
            
            return refined
        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
            return mask
    
    def _prepare_refinement_input(self, mask: np.ndarray, 
                                 frame: np.ndarray) -> np.ndarray:
        """Prepare input for refinement network."""
        # Resize to network input size
        size = (256, 256)
        frame_resized = cv2.resize(frame, size)
        mask_resized = cv2.resize(mask, size)
        
        # Normalize
        frame_norm = frame_resized.astype(np.float32) / 255.0
        mask_norm = mask_resized.astype(np.float32) / 255.0
        
        # Concatenate
        combined = np.concatenate(
            [frame_norm, mask_norm[..., np.newaxis]], axis=-1
        )
        
        # Transpose for network
        return np.transpose(combined, (2, 0, 1))[np.newaxis, ...]
    
    def _postprocess_refinement(self, output: np.ndarray, 
                               original_shape: Tuple[int, int]) -> np.ndarray:
        """Post-process refinement output."""
        # Get mask
        pred = output[0, 0, :, :]
        
        # Resize to original
        resized = cv2.resize(pred, (original_shape[1], original_shape[0]))
        
        # Threshold
        refined = (resized > 0.5).astype(np.uint8) * 255
        
        return refined
    
    def _finalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Final mask processing with smoothing and feathering."""
        # Apply Gaussian blur
        if self.blur_kernel[0] > 1:
            mask = cv2.GaussianBlur(mask, self.blur_kernel, 0)
        
        # Feather edges with distance transform
        if self.config.feather_amount > 0:
            mask = self._feather_with_distance_transform(mask)
        
        # Apply area constraints
        mask = self._apply_adaptive_area_constraints(mask)
        
        return mask
    
    def _feather_with_distance_transform(self, mask: np.ndarray) -> np.ndarray:
        """Feather mask using distance transform for smooth edges."""
        mask_float = mask.astype(np.float32) / 255.0
        
        # Calculate distance to boundary
        dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist_outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        
        # Normalize distances
        max_dist = max(np.max(dist_inside), np.max(dist_outside))
        if max_dist > 0:
            dist_inside = dist_inside / max_dist
            dist_outside = dist_outside / max_dist
        
        # Create smooth transition
        feather_radius = self.config.feather_amount * 50
        feathered = mask_float.copy()
        
        if feather_radius > 0:
            # Smooth transition at boundaries
            transition = np.exp(-dist_outside**2 / (2 * (feather_radius / 10)**2))
            feathered = mask_float * (1 - transition) + transition * 0.5
        
        return np.clip(feathered * 255, 0, 255).astype(np.uint8)
    
    def _apply_adaptive_area_constraints(self, mask: np.ndarray) -> np.ndarray:
        """Apply area constraints adaptively."""
        mask_area = np.count_nonzero(mask) / self.config.frame_area
        
        if mask_area < self.config.min_mask_area:
            # If mask is too small, clear it
            return np.zeros_like(mask)
        
        if mask_area > self.config.max_mask_area:
            # If mask is too large, scale it down
            scale_factor = self.config.max_mask_area / mask_area
            
            # Find largest connected component
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Scale the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # Scale contour around center
                    scaled_contour = largest_contour.astype(np.float32)
                    scaled_contour = (scaled_contour - [cx, cy]) * np.sqrt(scale_factor) + [cx, cy]
                    scaled_contour = scaled_contour.astype(np.int32)
                    
                    # Create new mask
                    new_mask = np.zeros_like(mask)
                    cv2.drawContours(new_mask, [scaled_contour], 0, 255, -1)
                    return new_mask
        
        return mask


# ============================================================================
#                           ENHANCED BACKGROUND MANAGER
# ============================================================================

class EnhancedBackgroundManager:
    """Advanced background management with multiple strategies."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = EnhancedPerformanceMonitor()
        
        self.background = None
        self.background_model = None
        self.background_cache = deque(maxlen=config.background_cache_size)
        self.background_variations = []
        
        self._init_background_model()
        self._init_variation_detector()
        
    def _init_background_model(self) -> None:
        """Initialize enhanced background subtraction model."""
        if self.config.background_model == "mog2":
            self.background_model = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
            self.background_model.setShadowValue(0)
        elif self.config.background_model == "knn":
            self.background_model = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400,
                detectShadows=True
            )
            self.background_model.setShadowValue(0)
        elif self.config.background_model == "u2net" and ONNX_AVAILABLE:
            self._init_u2net_model()
    
    def _init_u2net_model(self) -> None:
        """Initialize U2Net for background segmentation."""
        try:
            model_path = Path("models/u2net.onnx")
            if model_path.exists():
                self.u2net_session = ort.InferenceSession(str(model_path))
                logger.info("U2Net background model loaded")
            else:
                logger.warning("U2Net model not found, using MOG2")
                self.config.background_model = "mog2"
                self._init_background_model()
        except Exception as e:
            logger.warning(f"Failed to init U2Net: {e}, using MOG2")
            self.config.background_model = "mog2"
            self._init_background_model()
    
    def _init_variation_detector(self) -> None:
        """Initialize background variation detector."""
        if SKLEARN_AVAILABLE:
            self.variation_model = MiniBatchKMeans(
                n_clusters=3,
                random_state=42,
                batch_size=100
            )
        else:
            self.variation_model = None
    
    @monitor.time_section("enhanced_background_capture")
    def capture(self, camera: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Enhanced background capture with multiple frames and validation."""
        frames = []
        motion_scores = []
        quality_scores = []
        
        logger.info("Capturing enhanced background...")
        
        for i in range(min(self.config.background_frames, 100)):
            ret, frame = camera.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, self.config.frame_size)
            frame = cv2.flip(frame, 1)
            
            # Calculate frame quality
            quality = self._calculate_frame_quality(frame)
            quality_scores.append(quality)
            
            if frames:
                # Calculate motion between consecutive frames
                prev_gray = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                motion = cv2.absdiff(prev_gray, curr_gray)
                motion_score = np.mean(motion)
                motion_scores.append(motion_score)
                
                # Only add frames with low motion and good quality
                if motion_score < 15 and quality > 0.7:
                    frames.append(frame)
            else:
                frames.append(frame)
            
            # Show progress
            if i % 10 == 0:
                logger.info(f"Capture progress: {i}/{self.config.background_frames}")
            
            if len(frames) >= 30:  # Collect at least 30 good frames
                break
        
        if len(frames) < 10:
            logger.warning(f"Insufficient good frames: {len(frames)}")
            if frames:
                self.background = frames[0].copy()
            else:
                return None
        else:
            # Use weighted median based on quality
            frames_array = np.array(frames)
            weights = np.array(quality_scores[:len(frames)])
            weights = weights / weights.sum()
            
            # Weighted median calculation
            sorted_indices = np.argsort(weights)
            cumulative_weight = np.cumsum(weights[sorted_indices])
            median_idx = sorted_indices[np.where(cumulative_weight >= 0.5)[0][0]]
            self.background = frames_array[median_idx].copy()
            
            # Initialize background model with all frames
            if self.background_model is not None:
                for frame in frames:
                    self.background_model.apply(frame)
        
        logger.info(f"Background captured from {len(frames)} frames")
        logger.info(f"Average quality: {np.mean(quality_scores):.2f}")
        logger.info(f"Average motion: {np.mean(motion_scores):.2f}")
        
        # Cache background
        self.background_cache.append(self.background.copy())
        
        return self.background
    
    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """Calculate frame quality score."""
        # 1. Sharpness (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # 2. Brightness (optimal range)
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        
        # 3. Contrast
        contrast = np.std(gray)
        contrast_score = min(contrast / 64, 1.0)
        
        # Combine scores
        quality = (
            0.5 * (sharpness / 1000) +  # Normalize sharpness
            0.3 * brightness_score +
            0.2 * contrast_score
        )
        
        return min(max(quality, 0.0), 1.0)
    
    @monitor.time_section("background_update")
    def update(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Update background with advanced strategies."""
        if self.background is None:
            self.background = frame.copy()
            return
        
        update_mask = cv2.bitwise_not(mask)
        
        if self.background_model is not None:
            # Update background subtraction model
            fg_mask = self.background_model.apply(frame)
            bg_image = self.background_model.getBackgroundImage()
            
            if bg_image is not None:
                # Adaptive learning rate based on mask quality
                mask_quality = np.count_nonzero(update_mask) / self.config.frame_area
                learning_rate = self.config.background_update_rate * mask_quality
                
                # Update background
                self.background = cv2.addWeighted(
                    self.background, 1 - learning_rate,
                    bg_image, learning_rate, 0
                )
        
        # Direct update in non-masked areas
        if np.count_nonzero(update_mask) > 100:
            # Calculate update weight based on area and motion
            update_weight = self.config.background_update_rate * 0.5
            
            # Update only in non-masked regions
            self.background[update_mask > 0] = cv2.addWeighted(
                self.background[update_mask > 0], 1 - update_weight,
                frame[update_mask > 0], update_weight, 0
            ).astype(np.uint8)
        
        # Update variation model
        if self.variation_model is not None and len(self.background_cache) > 10:
            self._update_variation_model(frame, mask)
        
        # Cache current background
        self.background_cache.append(self.background.copy())
    
    def _update_variation_model(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Update background variation model."""
        # Sample background pixels
        bg_pixels = frame[mask == 0]
        
        if len(bg_pixels) > 1000:
            # Random sample
            samples = bg_pixels[np.random.choice(
                len(bg_pixels), min(1000, len(bg_pixels)), replace=False
            )]
            
            # Update variation model
            try:
                self.variation_model.partial_fit(samples)
                
                # Store cluster centers as background variations
                self.background_variations = self.variation_model.cluster_centers_
            except:
                pass
    
    @monitor.time_section("background_restoration")
    def restore(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enhanced background restoration with multiple strategies."""
        if self.background is None or not self.config.enable_background_restoration:
            return frame.copy()
        
        mask_area = np.count_nonzero(mask)
        if mask_area < 100:
            return frame.copy()
        
        restored = frame.copy()
        
        # Choose restoration strategy based on mask size
        if mask_area < self.config.frame_area * 0.1:
            # Small holes: use fast inpainting
            restored = cv2.inpaint(
                frame, mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
        elif mask_area < self.config.frame_area * 0.3:
            # Medium holes: use background blending
            alpha = mask.astype(np.float32) / 255.0
            alpha = cv2.GaussianBlur(alpha, (21, 21), 0)
            
            restored = frame.astype(np.float32) * (1 - alpha[..., np.newaxis]) + \
                      self.background.astype(np.float32) * alpha[..., np.newaxis]
            restored = restored.astype(np.uint8)
        else:
            # Large holes: use advanced inpainting
            restored = self._advanced_inpainting(frame, mask)
        
        # Apply temporal smoothing from cache
        if len(self.background_cache) > 1:
            restored = self._apply_temporal_smoothing(restored, mask)
        
        return restored
    
    def _advanced_inpainting(self, frame: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
        """Advanced inpainting using multiple techniques."""
        # Method 1: Navier-Stokes inpainting (better for large areas)
        inpainted_ns = cv2.inpaint(
            frame, mask,
            inpaintRadius=7,
            flags=cv2.INPAINT_NS
        )
        
        # Method 2: Background blending
        alpha = mask.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (31, 31), 0)
        
        blended = frame.astype(np.float32) * (1 - alpha[..., np.newaxis]) + \
                 self.background.astype(np.float32) * alpha[..., np.newaxis]
        blended = blended.astype(np.uint8)
        
        # Combine methods
        # Use Navier-Stokes for texture, blending for color
        gray_inpainted = cv2.cvtColor(inpainted_ns, cv2.COLOR_BGR2GRAY)
        gray_blended = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
        
        # Edge-aware combination
        edges = cv2.Canny(mask, 50, 150)
        edge_mask = edges > 0
        
        result = inpainted_ns.copy()
        result[edge_mask] = blended[edge_mask]
        
        return result
    
    def _apply_temporal_smoothing(self, frame: np.ndarray, 
                                 mask: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing using background cache."""
        if len(self.background_cache) < 2:
            return frame
        
        # Create weighted average from cache
        weights = np.linspace(0.1, 1.0, len(self.background_cache))
        weights = weights / weights.sum()
        
        smoothed = np.zeros_like(frame, dtype=np.float32)
        for bg, weight in zip(self.background_cache, weights):
            smoothed += bg.astype(np.float32) * weight
        
        # Blend with current frame based on mask
        alpha = mask.astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
        
        result = frame.astype(np.float32) * (1 - alpha[..., np.newaxis]) + \
                smoothed.astype(np.float32) * alpha[..., np.newaxis]
        
        return result.astype(np.uint8)
    
    def get_background_variations(self) -> List[np.ndarray]:
        """Get background variations for dynamic scenes."""
        variations = []
        
        if self.background_variations:
            # Create variation images from cluster centers
            for center in self.background_variations:
                var_image = np.full_like(self.background, center)
                variations.append(var_image)
        
        return variations


# ============================================================================
#                           ENHANCED GPU ACCELERATOR
# ============================================================================

class EnhancedGPUAccelerator:
    """Enhanced GPU acceleration with multi-GPU support."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.use_gpu = config.enable_gpu and cv2.ocl.haveOpenCL()
        self.use_cupy = CUDA_AVAILABLE and config.enable_gpu
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False
        
        self.gpu_devices = []
        self._init_gpu_devices()
        
        if self.use_gpu:
            cv2.ocl.setUseOpenCL(True)
            logger.info(f"OpenCL acceleration enabled")
        
        if self.use_cupy:
            logger.info(f"CuPy acceleration available")
        
        if self.use_cuda:
            logger.info(f"CUDA acceleration available")
    
    def _init_gpu_devices(self) -> None:
        """Initialize available GPU devices."""
        if self.use_cuda:
            try:
                device_count = cv2.cuda.getCudaEnabledDeviceCount()
                for i in range(device_count):
                    device_info = cv2.cuda.getDevice(i)
                    self.gpu_devices.append({
                        'id': i,
                        'name': device_info.name(),
                        'memory': device_info.totalMemory()
                    })
                logger.info(f"Found {device_count} CUDA devices")
            except:
                self.use_cuda = False
    
    def get_best_device(self) -> Optional[int]:
        """Get the best available GPU device."""
        if not self.gpu_devices:
            return None
        
        # Select device with most memory
        return max(range(len(self.gpu_devices)), 
                  key=lambda i: self.gpu_devices[i]['memory'])
    
    def to_gpu(self, frame: np.ndarray, device_id: Optional[int] = None) -> Any:
        """Transfer frame to GPU memory."""
        if self.use_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            return gpu_frame
        elif self.use_cupy:
            return cp.asarray(frame)
        elif self.use_gpu:
            return cv2.UMat(frame)
        else:
            return frame
    
    def from_gpu(self, gpu_frame: Any) -> np.ndarray:
        """Transfer frame from GPU to CPU."""
        if hasattr(gpu_frame, 'download'):
            return gpu_frame.download()
        elif isinstance(gpu_frame, cp.ndarray):
            return cp.asnumpy(gpu_frame)
        elif isinstance(gpu_frame, cv2.UMat):
            return gpu_frame.get()
        else:
            return gpu_frame
    
    @monitor.time_section("gpu_color_conversion")
    def cvt_color(self, frame: Any, code: int, stream: Any = None) -> Any:
        """Color conversion on GPU if available."""
        if self.use_cuda and hasattr(frame, 'convertTo'):
            # CUDA color conversion
            gpu_frame = frame
            if stream is not None:
                return cv2.cuda.cvtColor(gpu_frame, code, stream=stream)
            else:
                return cv2.cuda.cvtColor(gpu_frame, code)
        elif self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.cvtColor(frame, code)
        elif self.use_cupy and isinstance(frame, cp.ndarray):
            # Use CuPy implementation
            return self._cupy_color_conversion(frame, code)
        else:
            return cv2.cvtColor(frame, code)
    
    def _cupy_color_conversion(self, frame: cp.ndarray, code: int) -> cp.ndarray:
        """Color conversion using CuPy."""
        if code == cv2.COLOR_BGR2HSV_FULL:
            # Implement BGR to HSV in CuPy
            frame_float = frame.astype(cp.float32) / 255.0
            
            b, g, r = frame_float[..., 0], frame_float[..., 1], frame_float[..., 2]
            
            v = cp.maximum(b, cp.maximum(g, r))
            
            m = cp.minimum(b, cp.minimum(g, r))
            c = v - m
            
            # Hue calculation
            h = cp.zeros_like(v)
            mask = c > 0
            
            # Red is max
            mask_r = mask & (r == v)
            h[mask_r] = 60 * (0 + (g[mask_r] - b[mask_r]) / c[mask_r])
            
            # Green is max
            mask_g = mask & (g == v)
            h[mask_g] = 60 * (2 + (b[mask_g] - r[mask_g]) / c[mask_g])
            
            # Blue is max
            mask_b = mask & (b == v)
            h[mask_b] = 60 * (4 + (r[mask_b] - g[mask_b]) / c[mask_b])
            
            h = (h + 360) % 360
            
            # Saturation calculation
            s = cp.zeros_like(v)
            s[mask] = c[mask] / v[mask]
            
            # Scale to 0-255
            h = (h * 255 / 360).astype(cp.uint8)
            s = (s * 255).astype(cp.uint8)
            v = (v * 255).astype(cp.uint8)
            
            return cp.stack([h, s, v], axis=-1)
        
        # Fallback to CPU
        return cv2.cvtColor(self.from_gpu(frame), code)
    
    @monitor.time_section("gpu_filter")
    def gaussian_blur(self, frame: Any, ksize: Tuple[int, int], 
                     sigma: float = 0, stream: Any = None) -> Any:
        """Gaussian blur on GPU."""
        if self.use_cuda and hasattr(frame, 'gaussianBlur'):
            if stream is not None:
                return cv2.cuda.GaussianBlur(frame, ksize, sigmaX=sigma, stream=stream)
            else:
                return cv2.cuda.GaussianBlur(frame, ksize, sigmaX=sigma)
        elif self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.GaussianBlur(frame, ksize, sigma)
        elif self.use_cupy and isinstance(frame, cp.ndarray):
            import cupyx.scipy.ndimage
            sigma_px = sigma if sigma > 0 else 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8
            return cupyx.scipy.ndimage.gaussian_filter(frame, sigma_px)
        else:
            return cv2.GaussianBlur(frame, ksize, sigma)
    
    @monitor.time_section("gpu_morphology")
    def morphology_ex(self, frame: Any, op: int, kernel: np.ndarray,
                     stream: Any = None) -> Any:
        """Morphological operations on GPU."""
        if self.use_cuda and hasattr(frame, 'morphologyEx'):
            gpu_kernel = cv2.cuda.createMorphologyFilter(op, frame.type(), kernel)
            if stream is not None:
                return gpu_kernel.apply(frame, stream=stream)
            else:
                return gpu_kernel.apply(frame)
        elif self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.morphologyEx(frame, op, kernel)
        else:
            return cv2.morphologyEx(frame, op, kernel)


# ============================================================================
#                           ENHANCED UI MANAGER
# ============================================================================

class EnhancedUIManager:
    """Enhanced UI manager with themes and advanced controls."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = EnhancedPerformanceMonitor()
        
        self.window_name = "Invisibility Cloak 5.0"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *config.frame_size)
        
        # UI state
        self.theme = self._create_theme(config.theme)
        self.show_mask = False
        self.show_original = False
        self.recording = False
        self.video_writer = None
        
        if config.show_controls:
            self._create_enhanced_controls()
    
    def _create_theme(self, theme_name: str) -> Dict[str, Any]:
        """Create UI theme."""
        themes = {
            "dark": {
                "bg_color": (20, 20, 30),
                "text_color": (0, 255, 0),
                "accent_color": (0, 200, 255),
                "warning_color": (0, 100, 255),
                "panel_alpha": 0.85
            },
            "light": {
                "bg_color": (240, 240, 245),
                "text_color": (30, 30, 30),
                "accent_color": (0, 120, 220),
                "warning_color": (0, 80, 200),
                "panel_alpha": 0.75
            },
            "matrix": {
                "bg_color": (0, 20, 0),
                "text_color": (0, 255, 100),
                "accent_color": (0, 255, 150),
                "warning_color": (0, 200, 100),
                "panel_alpha": 0.9
            }
        }
        return themes.get(theme_name, themes["dark"])
    
    def _create_enhanced_controls(self) -> None:
        """Create enhanced interactive controls."""
        # Main controls
        cv2.createTrackbar('Sensitivity', self.window_name,
                          int(self.config.color_sensitivity * 100), 150,
                          lambda x: None)
        cv2.createTrackbar('Smoothing', self.window_name,
                          int(self.config.mask_smoothing * 100), 100,
                          lambda x: None)
        cv2.createTrackbar('Feather', self.window_name,
                          int(self.config.feather_amount * 100), 100,
                          lambda x: None)
        
        # Advanced controls
        cv2.createTrackbar('Update Rate', self.window_name,
                          int(self.config.background_update_rate * 100), 100,
                          lambda x: None)
        cv2.createTrackbar('Min Area', self.window_name,
                          int(self.config.min_mask_area * 1000), 500,
                          lambda x: None)
    
    def update_controls(self) -> None:
        """Update control values from trackbars."""
        if not self.config.show_controls:
            return
        
        self.config.color_sensitivity = cv2.getTrackbarPos('Sensitivity', 
                                                          self.window_name) / 100.0
        self.config.mask_smoothing = cv2.getTrackbarPos('Smoothing', 
                                                       self.window_name) / 100.0
        self.config.feather_amount = cv2.getTrackbarPos('Feather', 
                                                       self.window_name) / 100.0
        self.config.background_update_rate = cv2.getTrackbarPos('Update Rate',
                                                              self.window_name) / 100.0
        self.config.min_mask_area = cv2.getTrackbarPos('Min Area',
                                                      self.window_name) / 1000.0
    
    def toggle_mask_overlay(self) -> None:
        """Toggle mask overlay display."""
        self.show_mask = not self.show_mask
    
    def toggle_original_view(self) -> None:
        """Toggle original frame view."""
        self.show_original = not self.show_original
    
    def toggle_recording(self, output_path: str = "output.mp4") -> None:
        """Toggle video recording."""
        if not self.recording:
            # Start recording
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, 
                self.config.target_fps, 
                self.config.frame_size
            )
            self.recording = True
            logger.info(f"Started recording to {output_path}")
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            logger.info("Stopped recording")
    
    @monitor.time_section("enhanced_ui_render")
    def render(self, frame: np.ndarray, stats: Dict[str, Any],
               mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Render enhanced UI elements on frame."""
        output = frame.copy()
        
        # Apply mask overlay if requested
        if self.show_mask and mask is not None:
            output = self._apply_mask_overlay(output, mask)
        
        # Show original if requested
        if self.show_original:
            output = self._show_original_view(output, frame)
        
        # Render statistics panel
        if self.config.show_stats:
            output = self._render_enhanced_stats(output, stats)
        
        # Render debug information
        if self.config.show_debug:
            output = self._render_enhanced_debug(output, stats)
        
        # Render recording indicator
        if self.recording:
            output = self._render_recording_indicator(output)
        
        # Write frame if recording
        if self.recording and self.video_writer is not None:
            self.video_writer.write(output)
        
        return output
    
    def _apply_mask_overlay(self, frame: np.ndarray, 
                           mask: np.ndarray) -> np.ndarray:
        """Apply mask overlay to frame."""
        if mask is None:
            return frame
        
        # Create colored overlay
        overlay = frame.copy()
        colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        
        # Blend with original
        alpha = 0.5
        overlay[mask > 0] = cv2.addWeighted(
            frame[mask > 0], 1 - alpha,
            colored_mask[mask > 0], alpha, 0
        )
        
        return overlay
    
    def _show_original_view(self, output: np.ndarray, 
                           original: np.ndarray) -> np.ndarray:
        """Show original frame in corner."""
        # Resize original
        scale = 0.25
        h, w = original.shape[:2]
        small_h, small_w = int(h * scale), int(w * scale)
        small_original = cv2.resize(original, (small_w, small_h))
        
        # Place in corner
        y_start, y_end = 10, 10 + small_h
        x_start, x_end = output.shape[1] - small_w - 10, output.shape[1] - 10
        
        output[y_start:y_end, x_start:x_end] = small_original
        
        # Add border
        cv2.rectangle(output, (x_start, y_start), (x_end, y_end),
                     self.theme['accent_color'], 2)
        
        # Add label
        cv2.putText(output, "Original", (x_start, y_start - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                   self.theme['text_color'], 1)
        
        return output
    
    def _render_enhanced_stats(self, frame: np.ndarray, 
                              stats: Dict[str, Any]) -> np.ndarray:
        """Render enhanced statistics panel."""
        panel_width = 400
        panel_height = 250
        
        # Create semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height),
                     self.theme['bg_color'], -1)
        
        # Statistics sections
        sections = [
            ("PERFORMANCE", [
                f"FPS: {stats.get('fps', 0):.1f}",
                f"Frame: {stats.get('frame_count', 0)}",
                f"Latency: {stats.get('pipeline_time', 0)*1000:.1f}ms",
                f"Mode: {stats.get('processing_mode', 'N/A').upper()}"
            ]),
            ("DETECTION", [
                f"Mask Area: {stats.get('mask_area', 0)*100:.1f}%",
                f"Confidence: {stats.get('confidence', 0):.2f}",
                f"Colors: {stats.get('num_colors', 0)}",
                f"Quality: {stats.get('mask_quality', 0):.2f}"
            ]),
            ("SYSTEM", [
                f"GPU: {'ON' if stats.get('gpu_enabled', False) else 'OFF'}",
                f"Threads: {stats.get('num_threads', 1)}",
                f"Memory: {stats.get('memory_usage', 0):.1f} MB",
                f"Cache: {stats.get('cache_hit_rate', 0):.1f}%"
            ])
        ]
        
        # Render sections
        y_offset = 40
        for section_title, section_items in sections:
            # Section title
            cv2.putText(overlay, section_title, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       self.theme['accent_color'], 1)
            y_offset += 20
            
            # Section items
            for item in section_items:
                cv2.putText(overlay, item, (30, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                           self.theme['text_color'], 1)
                y_offset += 18
            
            y_offset += 10  # Spacing between sections
        
        # Blend panel
        alpha = self.theme['panel_alpha']
        frame[10:panel_height, 10:panel_width] = cv2.addWeighted(
            frame[10:panel_height, 10:panel_width], 1 - alpha,
            overlay[10:panel_height, 10:panel_width], alpha, 0
        )
        
        return frame
    
    def _render_enhanced_debug(self, frame: np.ndarray, 
                              stats: Dict[str, Any]) -> np.ndarray:
        """Render enhanced debug information."""
        debug_info = [
            f"Frame: {frame.shape[1]}x{frame.shape[0]}",
            f"Buffer: {stats.get('buffer_size', 0)}",
            f"Timestamp: {time.time():.3f}",
            f"Bottlenecks: {len(stats.get('bottlenecks', {}))}",
        ]
        
        y_offset = frame.shape[0] - 10
        for info in reversed(debug_info):
            y_offset -= 20
            cv2.putText(frame, info, (frame.shape[1] - 300, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       self.theme['warning_color'], 1)
        
        return frame
    
    def _render_recording_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Render recording indicator."""
        # Red circle
        center = (frame.shape[1] - 30, 30)
        cv2.circle(frame, center, 8, (0, 0, 255), -1)
        
        # "REC" text
        cv2.putText(frame, "REC", (frame.shape[1] - 55, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 0, 255), 2)
        
        return frame
    
    def cleanup(self) -> None:
        """Cleanup UI resources."""
        if self.video_writer is not None:
            self.video_writer.release()


# ============================================================================
#                           ENHANCED MAIN PIPELINE
# ============================================================================

class EnhancedInvisibilityCloakPipeline:
    """Enhanced main processing pipeline with caching and optimization."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = EnhancedPerformanceMonitor()
        
        # Initialize enhanced components
        self.color_detector = EnhancedColorDetector(config)
        self.mask_processor = EnhancedMaskProcessor(config)
        self.background_manager = EnhancedBackgroundManager(config)
        self.gpu_accelerator = EnhancedGPUAccelerator(config) if config.enable_gpu else None
        self.ui_manager = EnhancedUIManager(config)
        
        # Enhanced state
        self.is_initialized = False
        self.current_frame = None
        self.current_mask = None
        self.frame_cache = deque(maxlen=config.cache_size)
        self.result_cache = deque(maxlen=config.cache_size)
        
        # Performance optimization
        self.use_pipeline = config.use_pipeline_optimization
        self.pipeline_stages = []
        self._init_pipeline_stages()
        
        # Thread/Process pool
        if config.enable_multithreading:
            self.thread_pool = ThreadPoolExecutor(max_workers=config.num_threads)
            self.process_pool = ProcessPoolExecutor(max_workers=2) if config.num_threads > 4 else None
        else:
            self.thread_pool = None
            self.process_pool = None
    
    def _init_pipeline_stages(self) -> None:
        """Initialize pipeline stages for optimization."""
        self.pipeline_stages = [
            self._stage_color_detection,
            self._stage_mask_processing,
            self._stage_background_restoration,
            self._stage_effect_application,
            self._stage_background_update
        ]
    
    @monitor.time_section("pipeline_initialize")
    def initialize(self, camera: cv2.VideoCapture) -> bool:
        """Initialize pipeline with enhanced background capture."""
        logger.info("Initializing enhanced pipeline...")
        
        background = self.background_manager.capture(camera)
        
        if background is None:
            logger.error("Pipeline initialization failed")
            return False
        
        # Warm up components
        self._warm_up_components(background)
        
        self.is_initialized = True
        logger.info("Enhanced pipeline initialized successfully")
        return True
    
    def _warm_up_components(self, background: np.ndarray) -> None:
        """Warm up components with initial data."""
        # Create a test frame
        test_frame = background.copy()
        
        # Warm up color detector
        _, _ = self.color_detector.detect(test_frame)
        
        # Warm up mask processor
        test_mask = np.zeros(test_frame.shape[:2], dtype=np.uint8)
        h, w = test_mask.shape
        cv2.rectangle(test_mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
        _ = self.mask_processor.process(test_mask, test_frame)
        
        # Warm up GPU if available
        if self.gpu_accelerator is not None:
            gpu_frame = self.gpu_accelerator.to_gpu(test_frame)
            _ = self.gpu_accelerator.cvt_color(gpu_frame, cv2.COLOR_BGR2HSV_FULL)
        
        logger.info("Components warmed up")
    
    @monitor.time_section("enhanced_pipeline_process")
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Enhanced frame processing with pipeline optimization."""
        if not self.is_initialized:
            return frame, {"error": "Pipeline not initialized"}
        
        start_time = time.perf_counter()
        self.current_frame = frame.copy()
        
        # Cache check
        cached_result = self._check_cache(frame)
        if cached_result is not None:
            return cached_result
        
        if self.use_pipeline and len(self.pipeline_stages) > 0:
            # Use optimized pipeline
            result, stats = self._execute_pipeline(frame)
        else:
            # Use sequential processing
            result, stats = self._execute_sequential(frame)
        
        # Update cache
        self._update_cache(frame, result, stats)
        
        # Update performance stats
        pipeline_time = time.perf_counter() - start_time
        self.monitor.update_fps(pipeline_time)
        self.monitor.frame_count += 1
        
        # Update UI controls
        self.ui_manager.update_controls()
        
        # Add enhanced statistics
        stats.update({
            "pipeline_time": pipeline_time,
            "frame_time": 1.0 / self.monitor.get_fps() if self.monitor.get_fps() > 0 else 0,
            "fps": self.monitor.get_fps(),
            "frame_count": self.monitor.frame_count,
            "gpu_enabled": self.config.enable_gpu,
            "processing_mode": self.config.processing_mode,
            "num_threads": self.config.num_threads,
            "cache_hit_rate": len(self.result_cache) / self.config.cache_size * 100 if self.config.cache_size > 0 else 0,
            "memory_usage": self._get_memory_usage()
        })
        
        return result, stats
    
    def _execute_pipeline(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute optimized processing pipeline."""
        intermediate_results = {}
        stats = {}
        
        # Execute stages
        for i, stage in enumerate(self.pipeline_stages):
            stage_name = stage.__name__.replace('_stage_', '')
            
            if i == 0:
                # First stage
                result, stage_stats = stage(frame, None)
            else:
                # Subsequent stages
                prev_result = intermediate_results.get(f"stage_{i-1}")
                result, stage_stats = stage(frame, prev_result)
            
            intermediate_results[f"stage_{i}"] = result
            stats[stage_name] = stage_stats
            
            # Early exit if error
            if result is None:
                break
        
        # Final result is from last stage
        final_result = intermediate_results.get(f"stage_{len(self.pipeline_stages)-1}", frame)
        
        # Combine stats
        combined_stats = {}
        for stage_stats in stats.values():
            if isinstance(stage_stats, dict):
                combined_stats.update(stage_stats)
        
        return final_result, combined_stats
    
    def _execute_sequential(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute sequential processing."""
        # 1. Color detection
        raw_mask, color_stats = self.color_detector.detect(frame)
        
        # 2. Mask processing
        processed_mask = self.mask_processor.process(raw_mask, frame)
        self.current_mask = processed_mask
        
        # 3. Background restoration
        restored_bg = self.background_manager.restore(frame, processed_mask)
        
        # 4. Apply cloak effect
        if self.background_manager.background is not None:
            output = self._apply_enhanced_effect(frame, processed_mask, restored_bg)
        else:
            output = frame
        
        # 5. Update background
        self.background_manager.update(frame, processed_mask)
        
        # Combine stats
        stats = {
            **color_stats,
            "mask_quality": self.mask_processor._calculate_mask_quality(processed_mask),
            "background_confidence": 0.9 if self.background_manager.background is not None else 0.0
        }
        
        return output, stats
    
    def _stage_color_detection(self, frame: np.ndarray, 
                              prev_result: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Pipeline stage: Color detection."""
        return self.color_detector.detect(frame)
    
    def _stage_mask_processing(self, frame: np.ndarray, 
                              prev_result: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Pipeline stage: Mask processing."""
        if prev_result is None or len(prev_result) != 2:
            return np.zeros(frame.shape[:2], dtype=np.uint8), {}
        
        mask, stats = prev_result
        processed_mask = self.mask_processor.process(mask, frame)
        self.current_mask = processed_mask
        
        return processed_mask, {"mask_processed": True}
    
    def _stage_background_restoration(self, frame: np.ndarray,
                                     prev_result: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Pipeline stage: Background restoration."""
        if prev_result is None:
            return frame, {}
        
        restored_bg = self.background_manager.restore(frame, prev_result)
        return restored_bg, {"background_restored": True}
    
    def _stage_effect_application(self, frame: np.ndarray,
                                 prev_result: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Pipeline stage: Effect application."""
        if prev_result is None or self.background_manager.background is None:
            return frame, {}
        
        if isinstance(prev_result, tuple):
            # prev_result is (mask, stats)
            mask = prev_result[0]
        else:
            # prev_result is restored background
            mask = self.current_mask if self.current_mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
            prev_result = self.background_manager.restore(frame, mask)
        
        output = self._apply_enhanced_effect(frame, mask, prev_result)
        return output, {"effect_applied": True}
    
    def _stage_background_update(self, frame: np.ndarray,
                                prev_result: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Pipeline stage: Background update."""
        if self.current_mask is not None:
            self.background_manager.update(frame, self.current_mask)
        
        # Return previous result unchanged
        return prev_result, {"background_updated": True}
    
    def _apply_enhanced_effect(self, frame: np.ndarray, mask: np.ndarray,
                              background: np.ndarray) -> np.ndarray:
        """Apply enhanced invisibility cloak effect."""
        if mask is None or background is None:
            return frame
        
        # Convert to float for high-quality blending
        frame_float = frame.astype(np.float32)
        background_float = background.astype(np.float32)
        
        # Create enhanced alpha mask
        alpha = mask.astype(np.float32) / 255.0
        
        # Apply sensitivity adjustment with non-linear curve
        sensitivity = self.config.color_sensitivity
        if sensitivity > 1.0:
            # Boost effect for sensitivity > 1.0
            alpha = np.power(alpha, 1.0 / sensitivity)
        else:
            alpha = alpha * sensitivity
        
        # Apply edge-aware blur to alpha for smoother transitions
        if self.config.feather_amount > 0:
            alpha = cv2.GaussianBlur(alpha, (15, 15), 0)
        
        # Expand alpha to 3 channels
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
        
        # Blend with gamma correction for better color preservation
        gamma = 1.0
        blended = np.power(
            frame_float * (1 - alpha_3ch) + 
            background_float * alpha_3ch,
            gamma
        )
        
        # Normalize and convert back
        output = np.clip(blended, 0, 255).astype(np.uint8)
        
        return output
    
    def _check_cache(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Check if result is in cache."""
        if len(self.frame_cache) == 0 or len(self.result_cache) == 0:
            return None
        
        # Simple frame difference check
        last_frame = self.frame_cache[-1]
        frame_diff = cv2.absdiff(frame, last_frame)
        diff_score = np.mean(frame_diff)
        
        # If frames are very similar, use cached result
        if diff_score < 5.0:  # Very small difference threshold
            self.monitor.increment("cache_hits")
            return self.result_cache[-1]
        
        self.monitor.increment("cache_misses")
        return None
    
    def _update_cache(self, frame: np.ndarray, result: np.ndarray,
                     stats: Dict[str, Any]) -> None:
        """Update frame and result cache."""
        self.frame_cache.append(frame.copy())
        self.result_cache.append((result.copy(), stats.copy()))
    
    def _get_memory_usage(self) -> float:
        """Get approximate memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def async_process(self, frame: np.ndarray) -> Any:
        """Process frame asynchronously."""
        if self.thread_pool is None:
            return self.process(frame)
        
        return self.thread_pool.submit(self.process, frame)
    
    def parallel_process_batch(self, frames: List[np.ndarray]) -> List[Any]:
        """Process multiple frames in parallel."""
        if self.process_pool is None:
            return [self.process(frame) for frame in frames]
        
        # Submit batch to process pool
        futures = [self.process_pool.submit(self.process, frame) for frame in frames]
        return [future.result() for future in futures]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        pipeline_stats = self.monitor.get_statistics()
        
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
                "gpu_devices": self.gpu_accelerator.gpu_devices if self.gpu_accelerator else [],
                "frame_size": self.config.frame_size,
                "cache_size": len(self.frame_cache),
                "memory_usage": self._get_memory_usage()
            }
        }


# ============================================================================
#                           ENHANCED MAIN APPLICATION
# ============================================================================

class EnhancedInvisibilityCloakSystem:
    """Enhanced main application class."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.pipeline = EnhancedInvisibilityCloakPipeline(self.config)
        
        self.is_running = False
        self.camera = None
        self.start_time = time.perf_counter()
        self.frame_counter = 0
        self.settings_file = Path("enhanced_cloak_settings.json")
        self.output_dir = Path(self.config.output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    def setup_camera(self) -> bool:
        """Enhanced camera setup with auto-configuration."""
        self.camera = cv2.VideoCapture(self.config.camera_index)
        
        if not self.camera.isOpened():
            # Try alternative camera indices
            for i in range(5):
                self.camera = cv2.VideoCapture(i)
                if self.camera.isOpened():
                    self.config.camera_index = i
                    logger.info(f"Found camera at index {i}")
                    break
        
        if not self.camera.isOpened():
            logger.error("Cannot open any camera")
            return False
        
        # Try to configure camera with optimal settings
        camera_settings = [
            (cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height),
            (cv2.CAP_PROP_FPS, self.config.target_fps),
            (cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size),
            (cv2.CAP_PROP_AUTOFOCUS, 0),
            (cv2.CAP_PROP_AUTO_EXPOSURE, 1),
            (cv2.CAP_PROP_EXPOSURE, -4),  # Manual exposure
            (cv2.CAP_PROP_GAIN, 0),  # Manual gain
        ]
        
        for prop, value in camera_settings:
            self.camera.set(prop, value)
        
        # Verify and log actual settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        
        # Update config with actual values
        self.config.frame_width = actual_width
        self.config.frame_height = actual_height
        
        return True
    
    def load_settings(self) -> bool:
        """Load settings from JSON file with version checking."""
        if not self.settings_file.exists():
            return False
        
        try:
            with open(self.settings_file, 'r') as f:
                saved = json.load(f)
            
            # Check version compatibility
            if 'version' in saved and saved['version'] >= "5.0":
                # Update config
                for key, value in saved.get('config', {}).items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                logger.info("Settings loaded successfully")
                return True
            else:
                logger.warning("Settings file is from older version")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False
    
    def save_settings(self) -> bool:
        """Save settings to JSON file with versioning."""
        try:
            settings_data = {
                'version': '5.0.0',
                'timestamp': time.time(),
                'config': self.config.to_dict()
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2)
            
            logger.info("Settings saved successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            return False
    
    def handle_keyboard(self, key: int) -> bool:
        """Enhanced keyboard handling."""
        actions = {
            # Basic controls
            ord('q'): lambda: setattr(self, 'is_running', False),
            ord(' '): self.toggle_pause,
            
            # Settings
            ord('s'): self.save_settings,
            ord('l'): self.load_settings,
            
            # Display modes
            ord('d'): self.toggle_debug,
            ord('m'): self.toggle_mask_overlay,
            ord('o'): self.toggle_original_view,
            ord('c'): self.toggle_controls,
            
            # Processing modes
            ord('1'): lambda: self.change_mode('fast'),
            ord('2'): lambda: self.change_mode('balanced'),
            ord('3'): lambda: self.change_mode('quality'),
            ord('4'): lambda: self.change_mode('dl'),
            
            # Color presets
            ord('g'): lambda: self.change_color('green'),
            ord('r'): lambda: self.change_color('red'),
            ord('b'): lambda: self.change_color('blue'),
            ord('u'): lambda: self.change_color('custom'),
            
            # Adjustments
            ord('+'): self.increase_sensitivity,
            ord('-'): self.decrease_sensitivity,
            ord('='): self.reset_sensitivity,
            
            # Background
            ord('n'): self.reset_background,
            ord('a'): self.capture_snapshot,
            
            # Recording
            ord('v'): lambda: self.toggle_recording(),
            
            # Themes
            ord('t'): self.cycle_theme,
            
            # Special
            27: lambda: setattr(self, 'is_running', False),  # ESC
            9: self.toggle_fullscreen,  # TAB
        }
        
        if key in actions:
            actions[key]()
            return True
        
        return False
    
    def toggle_pause(self) -> None:
        """Toggle processing pause."""
        self.pipeline.monitor.increment("paused" if not self.is_running else "resumed")
        self.is_running = not self.is_running
        logger.info(f"Processing {'paused' if not self.is_running else 'resumed'}")
    
    def toggle_mask_overlay(self) -> None:
        """Toggle mask overlay display."""
        self.pipeline.ui_manager.toggle_mask_overlay()
        logger.info(f"Mask overlay: {'ON' if self.pipeline.ui_manager.show_mask else 'OFF'}")
    
    def toggle_original_view(self) -> None:
        """Toggle original frame view."""
        self.pipeline.ui_manager.toggle_original_view()
        logger.info(f"Original view: {'ON' if self.pipeline.ui_manager.show_original else 'OFF'}")
    
    def toggle_controls(self) -> None:
        """Toggle UI controls."""
        self.config.show_controls = not self.config.show_controls
        if self.config.show_controls:
            self.pipeline.ui_manager._create_enhanced_controls()
        else:
            # Remove all trackbars
            cv2.destroyWindow(self.pipeline.ui_manager.window_name)
            cv2.namedWindow(self.pipeline.ui_manager.window_name, cv2.WINDOW_NORMAL)
        
        logger.info(f"Controls: {'SHOWN' if self.config.show_controls else 'HIDDEN'}")
    
    def toggle_debug(self) -> None:
        """Toggle debug mode."""
        self.config.show_debug = not self.config.show_debug
        logger.info(f"Debug: {'ON' if self.config.show_debug else 'OFF'}")
    
    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        current_prop = cv2.getWindowProperty(
            self.pipeline.ui_manager.window_name, 
            cv2.WND_PROP_FULLSCREEN
        )
        is_fullscreen = current_prop == cv2.WINDOW_FULLSCREEN
        
        if is_fullscreen:
            cv2.setWindowProperty(
                self.pipeline.ui_manager.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )
        else:
            cv2.setWindowProperty(
                self.pipeline.ui_manager.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
    
    def change_mode(self, mode: str) -> None:
        """Change processing mode."""
        old_mode = self.config.processing_mode
        self.config.processing_mode = mode
        self.pipeline.mask_processor._init_kernels()
        logger.info(f"Mode: {old_mode} -> {mode}")
    
    def change_color(self, color: str) -> None:
        """Change color preset."""
        if color in self.pipeline.color_detector.color_presets:
            self.pipeline.color_detector.current_color = color
            logger.info(f"Color: {color}")
    
    def increase_sensitivity(self) -> None:
        """Increase sensitivity."""
        self.config.color_sensitivity = min(1.5, self.config.color_sensitivity + 0.05)
        logger.info(f"Sensitivity: {self.config.color_sensitivity:.2f}")
    
    def decrease_sensitivity(self) -> None:
        """Decrease sensitivity."""
        self.config.color_sensitivity = max(0.0, self.config.color_sensitivity - 0.05)
        logger.info(f"Sensitivity: {self.config.color_sensitivity:.2f}")
    
    def reset_sensitivity(self) -> None:
        """Reset sensitivity to default."""
        self.config.color_sensitivity = 0.85
        logger.info(f"Sensitivity reset to: {self.config.color_sensitivity:.2f}")
    
    def reset_background(self) -> None:
        """Reset background."""
        logger.info("Resetting background...")
        if self.camera is not None:
            self.pipeline.background_manager.capture(self.camera)
    
    def capture_snapshot(self) -> None:
        """Capture and save snapshot."""
        if self.pipeline.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"snapshot_{timestamp}.jpg"
            cv2.imwrite(str(filename), self.pipeline.current_frame)
            logger.info(f"Snapshot saved: {filename}")
    
    def toggle_recording(self) -> None:
        """Toggle video recording."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"recording_{timestamp}.mp4"
        self.pipeline.ui_manager.toggle_recording(str(filename))
    
    def cycle_theme(self) -> None:
        """Cycle through UI themes."""
        themes = ["dark", "light", "matrix"]
        current_idx = themes.index(self.config.theme)
        next_idx = (current_idx + 1) % len(themes)
        self.config.theme = themes[next_idx]
        
        # Update UI manager theme
        self.pipeline.ui_manager.theme = self.pipeline.ui_manager._create_theme(self.config.theme)
        logger.info(f"Theme: {self.config.theme}")
    
    def run(self) -> None:
        """Enhanced main application loop."""
        logger.info("Starting Enhanced Invisibility Cloak System 5.0")
        
        if not self.setup_camera():
            return
        
        self.load_settings()
        
        if not self.pipeline.initialize(self.camera):
            self.camera.release()
            return
        
        self.is_running = True
        frame_times = deque(maxlen=60)
        
        try:
            while True:
                loop_start = time.perf_counter()
                
                if not self.is_running:
                    # Show paused message
                    display_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(display_frame, "PAUSED - Press SPACE to resume",
                               (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 255, 0), 2)
                    cv2.imshow(self.pipeline.ui_manager.window_name, display_frame)
                    
                    key = cv2.waitKey(100) & 0xFF
                    self.handle_keyboard(key)
                    continue
                
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.001)
                    continue
                
                # Resize and flip
                frame = cv2.resize(frame, self.config.frame_size)
                frame = cv2.flip(frame, 1)
                
                # Process frame
                output, stats = self.pipeline.process(frame)
                
                # Render UI
                output = self.pipeline.ui_manager.render(output, stats, 
                                                       self.pipeline.current_mask)
                
                # Display
                cv2.imshow(self.pipeline.ui_manager.window_name, output)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                self.handle_keyboard(key)
                
                # Check window
                if cv2.getWindowProperty(self.pipeline.ui_manager.window_name, 
                                        cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # Update performance tracking
                frame_time = time.perf_counter() - loop_start
                frame_times.append(frame_time)
                self.frame_counter += 1
                
                # Log periodically
                if self.frame_counter % 100 == 0:
                    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
                    fps_stats = self.pipeline.monitor.get_fps()
                    
                    logger.info(
                        f"Frames: {self.frame_counter}, "
                        f"FPS: {avg_fps:.1f} (stats: {fps_stats:.1f}), "
                        f"Mask: {stats.get('mask_area', 0)*100:.1f}%, "
                        f"Confidence: {stats.get('confidence', 0):.2f}"
                    )
                    
                    # Print performance bottlenecks periodically
                    if self.frame_counter % 500 == 0:
                        bottlenecks = self.pipeline.monitor.predict_bottleneck()
                        if bottlenecks:
                            logger.info("Performance bottlenecks:")
                            for section, score in bottlenecks.items():
                                logger.info(f"  {section}: {score:.2f}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Enhanced cleanup with statistics."""
        logger.info("Cleaning up...")
        
        self.save_settings()
        
        if self.camera is not None:
            self.camera.release()
        
        self.pipeline.ui_manager.cleanup()
        cv2.destroyAllWindows()
        
        if self.pipeline.thread_pool is not None:
            self.pipeline.thread_pool.shutdown(wait=True)
        
        if self.pipeline.process_pool is not None:
            self.pipeline.process_pool.shutdown(wait=True)
        
        # Print comprehensive summary
        total_time = time.perf_counter() - self.start_time
        avg_fps = self.frame_counter / total_time if total_time > 0 else 0
        
        # Get performance statistics
        perf_stats = self.pipeline.get_performance_stats()
        
        logger.info("=" * 60)
        logger.info("ENHANCED SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Frames processed: {self.frame_counter}")
        logger.info(f"Total runtime: {total_time:.1f}s")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Resolution: {self.config.frame_size}")
        logger.info(f"Processing mode: {self.config.processing_mode}")
        logger.info(f"GPU acceleration: {'Enabled' if self.config.enable_gpu else 'Disabled'}")
        logger.info(f"Threads used: {self.config.num_threads}")
        
        # Performance details
        if 'pipeline' in perf_stats and 'section_times' in perf_stats['pipeline']:
            logger.info("\nPerformance breakdown:")
            for section, times in perf_stats['pipeline']['section_times'].items():
                logger.info(f"  {section}: {times.get('mean', 0)*1000:.1f}ms "
                          f"(min: {times.get('min', 0)*1000:.1f}ms, "
                          f"max: {times.get('max', 0)*1000:.1f}ms)")
        
        logger.info("=" * 60)


# ============================================================================
#                           ENHANCED ENTRY POINT
# ============================================================================

def parse_enhanced_args() -> argparse.Namespace:
    """Parse enhanced command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced Invisibility Cloak System 5.0',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Camera settings
    camera_group = parser.add_argument_group('Camera Settings')
    camera_group.add_argument('--camera', type=int, default=0,
                            help='Camera index')
    camera_group.add_argument('--width', type=int, default=1280,
                            help='Frame width')
    camera_group.add_argument('--height', type=int, default=720,
                            help='Frame height')
    camera_group.add_argument('--fps', type=int, default=60,
                            help='Target FPS')
    camera_group.add_argument('--buffer', type=int, default=1,
                            help='Camera buffer size')
    
    # Processing settings
    processing_group = parser.add_argument_group('Processing Settings')
    processing_group.add_argument('--mode', 
                                 choices=['fast', 'balanced', 'quality', 'dl'],
                                 default='balanced', help='Processing mode')
    processing_group.add_argument('--gpu', action='store_true',
                                 help='Enable GPU acceleration')
    processing_group.add_argument('--no-gpu', dest='gpu', action='store_false',
                                 help='Disable GPU acceleration')
    parser.set_defaults(gpu=CUDA_AVAILABLE)
    processing_group.add_argument('--threads', type=int, default=None,
                                 help='Number of threads')
    processing_group.add_argument('--no-pipeline', dest='pipeline', 
                                 action='store_false',
                                 help='Disable pipeline optimization')
    parser.set_defaults(pipeline=True)
    
    # Color detection
    color_group = parser.add_argument_group('Color Detection')
    color_group.add_argument('--sensitivity', type=float, default=0.85,
                           help='Color sensitivity (0.0-1.5)')
    color_group.add_argument('--no-adaptive', dest='adaptive',
                           action='store_false',
                           help='Disable adaptive color detection')
    parser.set_defaults(adaptive=True)
    color_group.add_argument('--multi-color', action='store_true',
                           help='Enable multi-color detection')
    
    # Background settings
    bg_group = parser.add_argument_group('Background Settings')
    bg_group.add_argument('--bg-model', 
                         choices=['static', 'mog2', 'knn', 'u2net'],
                         default='mog2', help='Background model')
    bg_group.add_argument('--update-rate', type=float, default=0.05,
                         help='Background update rate')
    bg_group.add_argument('--bg-frames', type=int, default=30,
                         help='Background capture frames')
    
    # UI settings
    ui_group = parser.add_argument_group('UI Settings')
    ui_group.add_argument('--theme', choices=['dark', 'light', 'matrix'],
                         default='dark', help='UI theme')
    ui_group.add_argument('--no-ui', dest='show_ui', action='store_false',
                         help='Disable UI controls')
    ui_group.add_argument('--no-stats', dest='show_stats', action='store_false',
                         help='Disable statistics')
    parser.set_defaults(show_ui=True, show_stats=True)
    ui_group.add_argument('--debug', action='store_true',
                         help='Enable debug mode')
    
    # Performance
    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--profile', action='store_true',
                           help='Enable profiling')
    perf_group.add_argument('--cache-size', type=int, default=15,
                           help='Frame cache size')
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str, default='output',
                            help='Output directory')
    output_group.add_argument('--record', action='store_true',
                            help='Start recording immediately')
    
    return parser.parse_args()


def enhanced_main() -> None:
    """Enhanced main entry point."""
    args = parse_enhanced_args()
    
    config = SystemConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.fps,
        buffer_size=args.buffer,
        processing_mode=args.mode,
        enable_gpu=args.gpu,
        num_threads=args.threads or max(1, mp.cpu_count() - 1),
        use_pipeline_optimization=args.pipeline,
        color_sensitivity=args.sensitivity,
        adaptive_threshold=args.adaptive,
        use_multi_color=args.multi_color,
        background_model=args.bg_model,
        background_update_rate=args.update_rate,
        background_frames=args.bg_frames,
        theme=args.theme,
        show_controls=args.show_ui,
        show_stats=args.show_stats,
        show_debug=args.debug,
        cache_size=args.cache_size,
        output_dir=args.output_dir,
    )
    
    if args.profile:
        import cProfile
        import pstats
        from pstats import SortKey
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            system = EnhancedInvisibilityCloakSystem(config)
            if args.record:
                system.pipeline.ui_manager.toggle_recording(
                    f"{config.output_dir}/recording_startup.mp4"
                )
            system.run()
        finally:
            profiler.disable()
            
            stats = pstats.Stats(profiler)
            stats.sort_stats(SortKey.CUMULATIVE)
            stats.print_stats(30)
            
            # Save profile data
            profile_file = "enhanced_profile.prof"
            stats.dump_stats(profile_file)
            
            # Create flamegraph if snakeviz is available
            try:
                import snakeviz
                logger.info(f"Profile saved to '{profile_file}'")
                logger.info("Run 'snakeviz enhanced_profile.prof' to visualize")
            except ImportError:
                logger.info(f"Profile saved to '{profile_file}'")
                logger.info("Install snakeviz for visualization: pip install snakeviz")
    else:
        system = EnhancedInvisibilityCloakSystem(config)
        if args.record:
            system.pipeline.ui_manager.toggle_recording(
                f"{config.output_dir}/recording_startup.mp4"
            )
        system.run()


if __name__ == "__main__":
    enhanced_main()
