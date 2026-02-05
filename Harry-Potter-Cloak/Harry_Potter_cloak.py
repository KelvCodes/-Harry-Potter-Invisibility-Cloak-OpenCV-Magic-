
    use_multi_color: bool = True  # Detect multiple colors simultaneously
    max_colors: int = 3  # Maximum number of colors to detect
    
    # Background management
    background_frames: int = 30  # Number of frames to capture for background
    background_update_rate: float = 0.01  # How quickly background adapts (0-1)
    enable_background_restoration: bool = True  # Fill in missing background areas
    background_model: str = "mog2"  # Options: "static", "mog2", "knn"
    
    # Mask processing
    mask_smoothing: float = 0.7  # Temporal smoothing factor (0-1)
    temporal_stability: int = 5  # Number of frames to consider for stability
    feather_amount: float = 0.1  # Edge feathering amount (0-1)
    min_mask_area: float = 0.001  # Minimum mask area as % of frame (reject noise)
    max_mask_area: float = 0.4  # Maximum mask area as % of frame (prevent overflow)
    
    # Performance monitoring
    enable_profiling: bool = False  # Detailed performance analysis
    stats_window: int = 100  # Number of frames to average statistics over
    cache_size: int = 10  # Size of frame/mask cache for temporal operations
    
    # UI settings
    show_controls: bool = True  # Display interactive controls
    show_stats: bool = True  # Show performance statistics
    show_debug: bool = False  # Show debug information
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        """Return frame dimensions as tuple (width, height)."""
        return (self.frame_width, self.frame_height)
    
    @property
    def frame_area(self) -> int:
        """Calculate total number of pixels in frame."""
        return self.frame_width * self.frame_height


class ProcessingMode(Enum):
    """
    Enumeration for processing modes.
    Controls trade-off between speed and quality.
    """
    FAST = auto()      # Prioritize speed, lower quality
    BALANCED = auto()  # Balanced approach
    QUALITY = auto()   # Prioritize quality, lower speed
    
    @classmethod
    def from_string(cls, mode_str: str):
        """Convert string to ProcessingMode enum."""
        return {
            "fast": cls.FAST,
            "balanced": cls.BALANCED,
            "quality": cls.QUALITY
        }.get(mode_str.lower(), cls.BALANCED)


# =============================================================================
#                           PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """
    Comprehensive performance monitoring and profiling.
    Tracks execution times, FPS, and other metrics for optimization.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of samples to keep for rolling averages
        """
        self.window_size = window_size
        self.timings: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.start_time = time.perf_counter()  # High-resolution timer
        self.frame_count = 0
        
    def time_section(self, section_name: str) -> Callable:
        """
        Decorator to time a code section.
        
        Args:
            section_name: Name of the section being timed
            
        Returns:
            Decorator function
        """
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
        """Increment a named counter."""
        self.counters[counter_name] += amount
    
    def get_fps(self) -> float:
        """Calculate current frames per second."""
        total_time = time.perf_counter() - self.start_time
        if total_time == 0:
            return 0
        return self.frame_count / total_time
    
    def get_average_time(self, section_name: str) -> float:
        """Get average time for a specific section."""
        timings = self.timings.get(section_name, [])
        if not timings:
            return 0
        return np.mean(list(timings))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "fps": self.get_fps(),
            "frame_count": self.frame_count,
            "runtime": time.perf_counter() - self.start_time,
            "section_times": {},
            "counters": dict(self.counters)
        }
        
        # Calculate statistics for each timed section
        for section, timings in self.timings.items():
            if timings:
                timings_list = list(timings)
                stats["section_times"][section] = {
                    "mean": np.mean(timings_list),
                    "min": np.min(timings_list),
                    "max": np.max(timings_list),
                    "95th": np.percentile(timings_list, 95)  # 95th percentile
                }
        
        return stats
    
    def reset(self):
        """Reset all statistics."""
        self.timings.clear()
        self.counters.clear()
        self.start_time = time.perf_counter()
        self.frame_count = 0


# =============================================================================
#                           COLOR DETECTION SYSTEM
# =============================================================================

class ColorDetector:
    """
    Advanced color detection with adaptive learning.
    Uses multiple color spaces and machine learning for robust detection.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Color presets with enhanced ranges (HSV and LAB color spaces)
        self.color_presets = {
            'green': {
                'hsv_range': [(35, 40, 40), (85, 255, 255)],
                'lab_range': [(0, 0, 0), (255, 255, 255)],  # Will be updated
                'weight': 1.0  # Relative importance of this color
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
        self.adaptive_model = None  # Machine learning model for color adaptation
        self.color_history = deque(maxlen=100)  # History for adaptive learning
        
        # Initialize adaptive model if sklearn is available
        if SKLEARN_AVAILABLE and config.adaptive_threshold:
            self._init_adaptive_model()
    
    def _init_adaptive_model(self):
        """Initialize Gaussian Mixture Model for adaptive color learning."""
        try:
            # GMM with 3 components (shadows, midtones, highlights)
            self.adaptive_model = GaussianMixture(n_components=3, covariance_type='full')
        except:
            logger.warning("Failed to initialize adaptive color model")
            self.adaptive_model = None
    
    @monitor.time_section("color_detection")
    def detect_colors(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect colors in frame with multiple methods.
        
        Args:
            frame: Input BGR image
            
        Returns:
            Tuple of (mask, stats) where mask is binary mask of detected colors
        """
        # Convert to multiple color spaces for robust detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Hue-Saturation-Value
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)   # CIELAB (perceptually uniform)
        
        masks = []
        confidence_scores = []
        
        # Use multiple color spaces for robust detection
        for color_name, preset in self.color_presets.items():
            if color_name == 'custom' and not self.config.use_multi_color:
                continue
            
            # HSV detection (good for color segmentation)
            hsv_lower = np.array(preset['hsv_range'][0], dtype=np.uint8)
            hsv_upper = np.array(preset['hsv_range'][1], dtype=np.uint8)
            hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            
            # LAB detection for better color constancy (illumination invariant)
            lab_lower = np.array(preset['lab_range'][0], dtype=np.uint8)
            lab_upper = np.array(preset['lab_range'][1], dtype=np.uint8)
            lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
            
            # Combine masks (AND operation - must be in both color spaces)
            combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
            
            if np.count_nonzero(combined_mask) > 0:
                masks.append(combined_mask)
                
                # Calculate confidence based on color purity and area
                mask_area = np.count_nonzero(combined_mask) / self.config.frame_area
                if 0.01 < mask_area < 0.5:  # Reasonable area range
                    # Confidence decreases as area deviates from ideal (20%)
                    confidence = preset['weight'] * (1.0 - abs(0.2 - mask_area) / 0.2)
                    confidence_scores.append(confidence)
                else:
                    confidence_scores.append(0)
        
        # If no colors detected, return empty mask
        if not masks:
            return np.zeros(frame.shape[:2], dtype=np.uint8), {"confidence": 0}
        
        # Weighted combination of masks based on confidence
        total_confidence = sum(confidence_scores)
        if total_confidence > 0:
            combined_mask = np.zeros_like(masks[0], dtype=np.float32)
            for mask, confidence in zip(masks, confidence_scores):
                combined_mask += mask.astype(np.float32) * (confidence / total_confidence)
            combined_mask = np.clip(combined_mask, 0, 255).astype(np.uint8)
        else:
            combined_mask = masks[0]
        
        # Adaptive learning - update model with new samples
        if self.adaptive_model is not None and np.count_nonzero(combined_mask) > 100:
            self._update_adaptive_model(frame, combined_mask)
        
        # Prepare statistics
        stats = {
            "confidence": total_confidence / len(confidence_scores) if confidence_scores else 0,
            "num_colors": len(masks),
            "mask_area": np.count_nonzero(combined_mask) / self.config.frame_area
        }
        
        return combined_mask, stats
    
    def _update_adaptive_model(self, frame: np.ndarray, mask: np.ndarray):
        """
        Update adaptive color model with new samples.
        
        Args:
            frame: Original BGR frame
            mask: Binary mask of detected colors
        """
        try:
            # Extract pixels where mask is active
            masked_pixels = frame[mask > 0]
            if len(masked_pixels) > 1000:
                # Randomly sample for efficiency
                samples = masked_pixels[np.random.choice(len(masked_pixels), 1000, replace=False)]
                
                # Convert to HSV for model training
                samples_hsv = cv2.cvtColor(samples.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
                
                # Update Gaussian Mixture Model
                self.adaptive_model.partial_fit(samples_hsv)
                
                # Adjust color ranges based on learned distribution
                self._adjust_color_ranges()
        except Exception as e:
            logger.warning(f"Adaptive model update failed: {e}")
    
    def _adjust_color_ranges(self):
        """Adjust color ranges based on learned Gaussian Mixture Model."""
        if self.adaptive_model is None:
            return
        
        for color_name in self.color_presets:
            if color_name == 'custom':
                continue
            
            # Get model parameters
            means = self.adaptive_model.means_
            covariances = self.adaptive_model.covariances_
            
            # Find GMM component closest to current color preset
            current_hsv = np.mean(self.color_presets[color_name]['hsv_range'], axis=0)
            distances = [np.linalg.norm(mean - current_hsv) for mean in means]
            closest_idx = np.argmin(distances)
            
            # Adjust range based on learned distribution (mean ± 2σ)
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
    """
    Advanced mask processing with temporal stability and edge refinement.
    Cleans up raw detection masks for smooth, natural-looking cloak effect.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Initialize processing kernels based on processing mode
        self._init_kernels()
        
        # Temporal buffers for stability
        self.mask_history = deque(maxlen=config.temporal_stability)
        self.flow_history = deque(maxlen=3)
        
        # Optical flow for motion compensation
        self.prev_gray = None  # Previous frame in grayscale
        self.flow = None  # Optical flow field
    
    def _init_kernels(self):
        """Initialize morphological and blur kernels based on processing mode."""
        mode = ProcessingMode.from_string(self.config.processing_mode)
        
        if mode == ProcessingMode.FAST:
            # Small kernels for speed
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self.blur_kernel = (5, 5)
            self.iterations = 1
        elif mode == ProcessingMode.BALANCED:
            # Medium kernels for balanced performance
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self.blur_kernel = (7, 7)
            self.iterations = 2
        else:  # QUALITY mode
            # Large kernels for highest quality
            self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            self.blur_kernel = (9, 9)
            self.iterations = 3
    
    @monitor.time_section("mask_processing")
    def process_mask(self, mask: np.ndarray, frame: np.ndarray = None) -> np.ndarray:
        """
        Process mask with temporal stability and edge refinement.
        
        Args:
            mask: Raw binary mask from color detection
            frame: Optional current frame for temporal stabilization
            
        Returns:
            Cleaned and processed mask
        """
        if mask is None or mask.size == 0:
            return np.zeros(self.config.frame_size[::-1], dtype=np.uint8)
        
        processed_mask = mask.copy()
        
        # 1. Morphological operations to remove noise and fill holes
        # Closing: dilation followed by erosion (removes small holes)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        # Opening: erosion followed by dilation (removes small noise)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, self.morph_kernel)
        
        # 2. Remove small disconnected regions (noise removal)
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.config.frame_area * self.config.min_mask_area
        
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.drawContours(processed_mask, [contour], 0, 0, -1)  # Fill with black
        
        # 3. Apply Gaussian blur for smooth edges (reduces aliasing)
        if self.blur_kernel[0] > 1:
            processed_mask = cv2.GaussianBlur(processed_mask, self.blur_kernel, 0)
        
        # 4. Temporal stabilization using optical flow (reduces flickering)
        if frame is not None:
            processed_mask = self._apply_temporal_stabilization(processed_mask, frame)
        
        # 5. Feather edges for smooth transparency transitions
        if self.config.feather_amount > 0:
            processed_mask = self._feather_edges(processed_mask)
        
        # 6. Apply area constraints (prevent mask from covering too much/too little)
        mask_area = np.count_nonzero(processed_mask) / self.config.frame_area
        if mask_area > self.config.max_mask_area:
            # Scale down mask while preserving shape
            scale_factor = self.config.max_mask_area / mask_area
            processed_mask = self._scale_mask(processed_mask, scale_factor)
        
        return processed_mask
    
    def _apply_temporal_stabilization(self, mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """
        Apply temporal stabilization using optical flow.
        Warps previous mask to align with current frame motion.
        
        Args:
            mask: Current frame's mask
            frame: Current BGR frame
            
        Returns:
            Temporally stabilized mask
        """
        # Convert to grayscale for optical flow calculation
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5,  # pyramid scale
                3,    # pyramid levels
                15,   # window size
                3,    # iterations
                5,    # poly_n
                1.2,  # poly_sigma
                0     # flags
            )
            
            # Warp previous mask using optical flow
            if self.mask_history:
                prev_mask = self.mask_history[-1]
                h, w = prev_mask.shape
                
                # Create coordinate grid
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x + flow[..., 0]).astype(np.float32)
                map_y = (y + flow[..., 1]).astype(np.float32)
                
                # Warp previous mask to current frame position
                warped_mask = cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR)
                
                # Blend current mask with warped previous mask (temporal smoothing)
                alpha = self.config.mask_smoothing
                mask = cv2.addWeighted(mask, 1 - alpha, warped_mask, alpha, 0)
            
            self.flow_history.append(flow)
        
        # Update history for next frame
        self.prev_gray = gray
        self.mask_history.append(mask.copy())
        
        return mask
    
    def _feather_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        Feather mask edges for smooth transitions.
        Creates soft alpha channel at mask boundaries.
        
        Args:
            mask: Binary mask
            
        Returns:
            Mask with feathered edges
        """
        # Normalize mask to 0-1 range
        mask_float = mask.astype(np.float32) / 255.0
        
        # Calculate distance from edges (positive inside mask, negative outside)
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        # Blend original mask with distance transform for soft edges
        feathered = mask_float * (1 - self.config.feather_amount) + \
                   dist_transform * self.config.feather_amount
        
        return np.clip(feathered * 255, 0, 255).astype(np.uint8)
    
    def _scale_mask(self, mask: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Scale mask area while preserving shape.
        
        Args:
            mask: Binary mask
            scale_factor: Factor to scale mask area by (0-1)
            
        Returns:
            Scaled mask
        """
        # Find contours of mask regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        
        # Scale each contour individually
        scaled_mask = np.zeros_like(mask)
        for contour in contours:
            # Calculate centroid of contour
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
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
    """
    Advanced background management with restoration capabilities.
    Maintains background model and fills in missing areas using inpainting.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        self.background = None  # Current background image
        self.background_model = None  # Background subtraction model
        self.background_history = deque(maxlen=10)  # Background history for stabilization
        
        # Initialize background model based on configuration
        self._init_background_model()
        
        # Inpainting parameters for background restoration
        self.inpainting_radius = 3
    
    def _init_background_model(self):
        """Initialize background subtraction model."""
        if self.config.background_model == "mog2":
            # Mixture of Gaussians model (adaptive to lighting changes)
            self.background_model = cv2.createBackgroundSubtractorMOG2(
                history=500,  # Number of frames to consider
                varThreshold=16,  # Variance threshold
                detectShadows=False  # Don't detect shadows as foreground
            )
        elif self.config.background_model == "knn":
            # K-Nearest Neighbors model (handles complex backgrounds)
            self.background_model = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400,
                detectShadows=False
            )
        # "static" mode just uses a fixed background image
    
    @monitor.time_section("background_capture")
    def capture_background(self, camera: cv2.VideoCapture) -> Optional[np.ndarray]:
        """
        Capture initial background with motion detection.
        
        Args:
            camera: OpenCV VideoCapture object
            
        Returns:
            Background image or None if capture failed
        """
        frames = []
        motion_scores = []
        
        logger.info("Capturing background...")
        
        for i in range(self.config.background_frames):
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Preprocess frame
            frame = cv2.resize(frame, self.config.frame_size)
            frame = cv2.flip(frame, 1)  # Mirror for more intuitive interaction
            
            # Calculate motion score compared to previous frame
            if len(frames) > 0:
                prev_gray = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = cv2.absdiff(prev_gray, curr_gray)
                motion_score = np.mean(motion)
                motion_scores.append(motion_score)
                
                # Only keep frames with low motion (static background)
                if motion_score < 10:  # Threshold for motion
                    frames.append(frame)
            else:
                frames.append(frame)
            
            if len(frames) >= 20:  # Minimum samples for good background
                break
        
        if not frames:
            logger.error("Failed to capture background")
            return None
        
        # Use median for low-motion scenes, average for others
        if np.mean(motion_scores) < 5:
            # Median is robust to transient objects
            self.background = np.median(frames, axis=0).astype(np.uint8)
        else:
            # Average for more stable scenes
            self.background = np.mean(frames, axis=0).astype(np.uint8)
        
        # Initialize background model with captured frames
        if self.background_model is not None:
            for frame in frames:
                self.background_model.apply(frame)
        
        logger.info(f"Background captured from {len(frames)} frames")
        return self.background
    
    @monitor.time_section("background_update")
    def update_background(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """
        Update background model with new information from non-cloak areas.
        
        Args:
            frame: Current frame
            mask: Cloak mask (areas to NOT update background)
        """
        if self.background is None:
            self.background = frame.copy()
            return
        
        if self.background_model is not None:
            # Update background subtraction model
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
        
        # Adaptive background update in non-masked areas (where cloak isn't)
        inverse_mask = cv2.bitwise_not(mask)
        update_region = inverse_mask > 0
        
        if np.any(update_region):
            alpha = self.config.background_update_rate * 0.1  # Slower update
            
            # Update only in non-cloak areas
            self.background[update_region] = cv2.addWeighted(
                self.background[update_region], 1 - alpha,
                frame[update_region], alpha, 0
            ).astype(np.uint8)
        
        # Store in history for potential rollback or stabilization
        self.background_history.append(self.background.copy())
    
    @monitor.time_section("background_restoration")
    def restore_background(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Restore background in masked areas using inpainting.
        
        Args:
            frame: Current frame with missing background areas
            mask: Binary mask indicating areas to restore
            
        Returns:
            Frame with restored background
        """
        if self.background is None or not self.config.enable_background_restoration:
            return frame.copy()
        
        # Only inpaint if there are significant areas to restore
        if np.count_nonzero(mask) > 100:
            # Use Telea inpainting algorithm
            inpainted = cv2.inpaint(
                frame, mask,
                inpaintRadius=self.inpainting_radius,
                flags=cv2.INPAINT_TELEA  # Fast marching method
            )
            
            # Blend inpainted result with learned background
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
    """
    GPU acceleration using OpenCV's UMat and optional CuPy.
    Transparently accelerates operations when GPU is available.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.use_gpu = config.enable_gpu and cv2.ocl.haveOpenCL()
        self.use_cupy = CUDA_AVAILABLE and config.enable_gpu
        
        if self.use_gpu:
            cv2.ocl.setUseOpenCL(True)
            logger.info("OpenCL acceleration enabled")
        
        if self.use_cupy:
            logger.info("CuPy acceleration available")
        
        # Create GPU contexts for different operations
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
        """
        Convert frame to GPU memory.
        
        Args:
            frame: CPU numpy array
            
        Returns:
            GPU array (UMat or CuPy array)
        """
        if self.use_cupy:
            return cp.asarray(frame)
        elif self.use_gpu:
            return cv2.UMat(frame)
        else:
            return frame
    
    def from_gpu(self, gpu_frame: Any) -> np.ndarray:
        """
        Convert from GPU memory to CPU.
        
        Args:
            gpu_frame: GPU array
            
        Returns:
            CPU numpy array
        """
        if isinstance(gpu_frame, cp.ndarray):
            return cp.asnumpy(gpu_frame)
        elif isinstance(gpu_frame, cv2.UMat):
            return gpu_frame.get()
        else:
            return gpu_frame
    
    @monitor.time_section("gpu_color_conversion")
    def cvt_color(self, frame: Any, code: int) -> Any:
        """
        Color conversion on GPU if available.
        
        Args:
            frame: Input image
            code: OpenCV color conversion code
            
        Returns:
            Converted image
        """
        if self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.cvtColor(frame, code)
        elif self.use_cupy and isinstance(frame, cp.ndarray):
            # Convert to CPU for color conversion, then back to GPU
            return cv2.cvtColor(self.from_gpu(frame), code)
        else:
            return cv2.cvtColor(frame, code)
    
    @monitor.time_section("gpu_filter")
    def gaussian_blur(self, frame: Any, ksize: Tuple[int, int], sigma: float = 0) -> Any:
        """
        Gaussian blur on GPU if available.
        
        Args:
            frame: Input image
            ksize: Kernel size (width, height)
            sigma: Gaussian standard deviation
            
        Returns:
            Blurred image
        """
        if self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.GaussianBlur(frame, ksize, sigma)
        elif self.use_cupy and isinstance(frame, cp.ndarray):
            # Use CuPy's Gaussian filter
            import cupyx.scipy.ndimage
            sigma_px = sigma if sigma > 0 else 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8
            return cupyx.scipy.ndimage.gaussian_filter(frame, sigma_px)
        else:
            return cv2.GaussianBlur(frame, ksize, sigma)


# =============================================================================
#                           UI MANAGER
# =============================================================================

class UIManager:
    """
    Advanced UI manager with interactive controls.
    Displays statistics, controls, and debug information.
    """
    
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
        
        # Create interactive trackbars for real-time adjustment
        if self.show_controls:
            self._create_trackbars()
        
        # Cache for UI elements to avoid recomputation
        self.ui_cache = {}
    
    def _create_trackbars(self):
        """Create interactive trackbars for parameter adjustment."""
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
        """Callback for sensitivity trackbar change."""
        self.config.color_sensitivity = value / 100.0
    
    def _on_smoothing_change(self, value: int):
        """Callback for smoothing trackbar change."""
        self.config.mask_smoothing = value / 100.0
    
    def _on_feather_change(self, value: int):
        """Callback for feather trackbar change."""
        self.config.feather_amount = value / 100.0
    
    @monitor.time_section("ui_render")
    def render(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """
        Render UI elements on frame.
        
        Args:
            frame: Input frame
            stats: Statistics to display
            
        Returns:
            Frame with UI overlay
        """
        output = frame.copy()
        
        if self.show_stats:
            output = self._render_stats_panel(output, stats)
        
        if self.show_debug:
            output = self._render_debug_info(output, stats)
        
        return output
    
    def _render_stats_panel(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """
        Render statistics panel with performance metrics.
        
        Args:
            frame: Input frame
            stats: Statistics dictionary
            
        Returns:
            Frame with stats overlay
        """
        panel_height = 200
        panel_width = 350
        
        # Create semi-transparent background for stats panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), 
                     (0, 0, 0), -1)  # Black rectangle
        
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
        
        # Draw each line of text
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (20, 40 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 1)  # Green text
        
        # Blend overlay with original frame
        alpha = 0.7  # Transparency factor
        frame[10:panel_height, 10:panel_width] = cv2.addWeighted(
            frame[10:panel_height, 10:panel_width], 1 - alpha,
            overlay[10:panel_height, 10:panel_width], alpha, 0
        )
        
        return frame
    
    def _render_debug_info(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """
        Render debug information in corner of frame.
        
        Args:
            frame: Input frame
            stats: Statistics dictionary
            
        Returns:
            Frame with debug overlay
        """
        debug_text = [
            f"Frame Size: {frame.shape[1]}x{frame.shape[0]}",
            f"Memory: {sys.getsizeof(frame) / 1024:.1f} KB",
            f"Time/Frame: {stats.get('frame_time', 0)*1000:.1f}ms",
        ]
        
        # Draw debug info at top-right corner
        for i, text in enumerate(debug_text):
            cv2.putText(frame, text, (frame.shape[1] - 300, 30 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 0), 1)  # Yellow text
        
        return frame
    
    def create_debug_window(self, image: np.ndarray, title: str = "Debug"):
        """
        Create separate debug window for additional visualization.
        
        Args:
            image: Image to display
            title: Window title
        """
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image)


# =============================================================================
#                           MAIN PIPELINE
# =============================================================================

class InvisibilityCloakPipeline:
    """
    Main processing pipeline orchestrating all components.
    Manages the complete frame processing workflow.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Initialize all processing components
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
        """
        Initialize the pipeline by capturing background.
        
        Args:
            camera: Video capture object
            
        Returns:
            True if initialization successful
        """
        # Capture initial background
        background = self.background_manager.capture_background(camera)
        if background is None:
            logger.error("Failed to initialize pipeline: No background")
            return False
        
        self.is_initialized = True
        logger.info("Pipeline initialized successfully")
        return True
    
    @monitor.time_section("pipeline_process")
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (output_frame, statistics)
        """
        if not self.is_initialized:
            return frame, {"error": "Pipeline not initialized"}
        
        pipeline_start = time.perf_counter()
        
        # Store current frame for potential future use
        self.current_frame = frame.copy()
        
        # Step 1: Color detection - find cloak areas
        raw_mask, color_stats = self.color_detector.detect_colors(frame)
        
        # Step 2: Mask processing - clean up and stabilize
        processed_mask = self.mask_processor.process_mask(raw_mask, frame)
        self.current_mask = processed_mask
        
        # Step 3: Background restoration - fill in missing areas
        restored_background = self.background_manager.restore_background(
            frame, processed_mask
        )
        
        # Step 4: Apply cloak effect - blend with background
        if self.background_manager.background is not None:
            output_frame = self._apply_cloak_effect(
                frame, processed_mask, restored_background
            )
        else:
            output_frame = frame
        
        # Step 5: Update background model with new information
        self.background_manager.update_background(frame, processed_mask)
        
        # Step 6: Prepare comprehensive statistics
        pipeline_time = time.perf_counter() - pipeline_start
        stats = {
            **color_stats,  # Include color detection stats
            "pipeline_time": pipeline_time,
            "frame_time": 1.0 / self.monitor.get_fps() if self.monitor.get_fps() > 0 else 0,
            "fps": self.monitor.get_fps(),
            "frame_count": self.monitor.frame_count,
            "gpu_enabled": self.config.enable_gpu,
            "processing_mode": self.config.processing_mode,
        }
        
        # Update frame counter for FPS calculation
        self.monitor.frame_count += 1
        
        return output_frame, stats
    
    def _apply_cloak_effect(self, frame: np.ndarray, mask: np.ndarray, 
                           background: np.ndarray) -> np.ndarray:
        """
        Apply the invisibility cloak effect using alpha blending.
        
        Args:
            frame: Current frame
            mask: Cloak mask (0-255)
            background: Background to reveal
            
        Returns:
            Frame with cloak effect applied
        """
        # Convert mask to 3 channels and normalize to 0-1 range
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Blend factor based on sensitivity setting
        alpha = self.config.color_sensitivity
        
        # Perform alpha blending
        foreground = frame.astype(np.float32)
        background = background.astype(np.float32)
        
        # Blend: where mask is strong, show background; where weak, show frame
        blended = foreground * (1 - mask_3ch * alpha) + \
                 background * (mask_3ch * alpha)
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def async_process(self, frame: np.ndarray) -> Any:
        """
        Process frame asynchronously if threading is enabled.
        
        Args:
            frame: Input frame
            
        Returns:
            Future object or direct result
        """
        if self.thread_pool is None:
            return self.process_frame(frame)
        
        # Submit to thread pool for parallel processing
        future = self.thread_pool.submit(self.process_frame, frame)
        return future
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics from all components.
        
        Returns:
            Dictionary of performance metrics
        """
        pipeline_stats = self.monitor.get_statistics()
        
        # Combine statistics from all components
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
    """
    Main application class coordinating all components.
    Handles camera setup, user interaction, and main loop.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.pipeline = InvisibilityCloakPipeline(self.config)
        
        # Application state
        self.is_running = False
        self.camera = None
        
        # Performance tracking
        self.start_time = time.perf_counter()
        self.frame_counter = 0
        
        # Settings management
        self.settings_file = Path("cloak_settings_v4.json")
    
    def setup_camera(self) -> bool:
        """
        Setup camera with optimal settings for low latency.
        
        Returns:
            True if camera setup successful
        """
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
        """
        Load settings from JSON file.
        
        Returns:
            True if settings loaded successfully
        """
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
        """
        Save current settings to JSON file.
        
        Returns:
            True if settings saved successfully
        """
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
        """
        Handle keyboard input for interactive control.
        
        Args:
            key: Keyboard key code
            
        Returns:
            True if key was handled
        """
        actions = {
            ord('q'): lambda: setattr(self, 'is_running', False),  # Quit
            ord('s'): self.save_settings,  # Save settings
            ord('l'): self.load_settings,  # Load settings
            ord('d'): self.toggle_debug,   # Toggle debug mode
            ord('c'): self.toggle_controls,  # Toggle controls
            ord('r'): self.reset_background,  # Reset background
            ord('1'): lambda: self.change_color_preset('green'),  # Green cloak
            ord('2'): lambda: self.change_color_preset('red'),    # Red cloak
            ord('3'): lambda: self.change_color_preset('blue'),   # Blue cloak
            ord('4'): lambda: self.change_color_preset('custom'), # Custom color
            ord('f'): lambda: self.change_processing_mode('fast'),     # Fast mode
            ord('b'): lambda: self.change_processing_mode('balanced'), # Balanced mode
            ord('q'): lambda: self.change_processing_mode('quality'),  # Quality mode
            ord('+'): self.increase_sensitivity,  # Increase sensitivity
            ord('-'): self.decrease_sensitivity,  # Decrease sensitivity
            ord(' '): self.capture_background_now,  # Manual background capture
            27: lambda: setattr(self, 'is_running', False),  # ESC key
        }
        
        if key in actions:
            actions[key]()
            return True
        
        return False
    
    def toggle_debug(self):
        """Toggle debug information display."""
        self.config.show_debug = not self.config.show_debug
        logger.info(f"Debug mode: {'ON' if self.config.show_debug else 'OFF'}")
    
    def toggle_controls(self):
        """Toggle UI controls display."""
        self.config.show_controls = not self.config.show_controls
        logger.info(f"Controls: {'SHOWN' if self.config.show_controls else 'HIDDEN'}")
    
    def reset_background(self):
        """Reset background by capturing new one."""
        logger.info("Resetting background...")
        if self.camera is not None:
            self.pipeline.background_manager.capture_background(self.camera)
    
    def change_color_preset(self, preset: str):
        """Change active color preset for cloak detection."""
        if preset in self.pipeline.color_detector.color_presets:
            self.pipeline.color_detector.current_color = preset
            logger.info(f"Color preset changed to: {preset}")
    
    def change_processing_mode(self, mode: str):
        """Change processing mode (fast/balanced/quality)."""
        self.config.processing_mode = mode
        self.pipeline.mask_processor._init_kernels()  # Reinitialize kernels
        logger.info(f"Processing mode changed to: {mode}")
    
    def increase_sensitivity(self):
        """Increase color sensitivity by 5%."""
        self.config.color_sensitivity = min(1.0, self.config.color_sensitivity + 0.05)
        logger.info(f"Color sensitivity: {self.config.color_sensitivity:.2f}")
    
    def decrease_sensitivity(self):
        """Decrease color sensitivity by 5%."""
        self.config.color_sensitivity = max(0.0, self.config.color_sensitivity - 0.05)
        logger.info(f"Color sensitivity: {self.config.color_sensitivity:.2f}")
    
    def capture_background_now(self):
        """Manually capture background from current frame."""
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
        
        # Load saved settings
        self.load_settings()
        
        # Initialize pipeline
        if not self.pipeline.initialize(self.camera):
            self.camera.release()
            return
        
        # Main processing loop
        self.is_running = True
        frame_times = deque(maxlen=30)  # For FPS calculation
        
        try:
            while self.is_running:
                loop_start = time.perf_counter()
                
                # Read frame from camera
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.001)  # Small sleep on read failure
                    continue
                
                # Process frame through pipeline
                output_frame, stats = self.pipeline.process_frame(frame)
                
                # Apply UI overlay
                output_frame = self.pipeline.ui_manager.render(output_frame, stats)
                
                # Display result
                cv2.imshow(self.pipeline.ui_manager.window_name, output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self.handle_keyboard(key)
                
                # Check if window was closed
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
        """Cleanup resources and print performance summary."""
        logger.info("Cleaning up resources...")
        
        # Save current settings
        self.save_settings()
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
        
        # Close all OpenCV windows
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
        
        # Detailed component timing statistics
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
    """
    Parse command line arguments for flexible configuration.
    
    Returns:
        Parsed arguments namespace
    """
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
    parser.set_defaults(gpu=CUDA_AVAILABLE)  # Auto-detect GPU
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
    """Main entry point with optional profiling."""
    args = parse_arguments()
    
    # Create configuration from arguments
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
    
    # Optional profiling mode for performance analysis
    if args.profile:
        import cProfile
        import pstats
        from pstats import SortKey
        
        profiler = cProfile.Profile()
        profiler.enable()  # Start profiling
        
        try:
            system = InvisibilityCloakSystem(config)
            system.run()
        finally:
            profiler.disable()  # Stop profiling
            
            # Display profiling results
            stats = pstats.Stats(profiler)
            stats.sort_stats(SortKey.CUMULATIVE)  # Sort by total time
            stats.print_stats(20)  # Show top 20 functions
            
            # Save profiling data to file for detailed analysis
            stats.dump_stats('performance_profile.prof')
            logger.info("Profiling data saved to 'performance_profile.prof'")
    else:
        # Run normally without profiling overhead
        system = InvisibilityCloakSystem(config)
        system.run()


if __name__ == "__main__":
    main()
