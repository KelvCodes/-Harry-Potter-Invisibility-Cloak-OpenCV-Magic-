
    color_sensitivity: float = 0.8
    use_multi_color: bool = True
    max_colors: int = 3
    
    # Background management
    background_frames: int = 30
    background_update_rate: float = 0.01
    enable_background_restoration: bool = True
    background_model: str = "mog2"  # static, mog2, knn
    
    # Mask processing
    mask_smoothing: float = 0.7
    temporal_stability: int = 5
    feather_amount: float = 0.1
    min_mask_area: float = 0.001
    max_mask_area: float = 0.4
    
    # Performance
    enable_profiling: bool = False
    stats_window: int = 100
    cache_size: int = 10
    
    # UI settings
    show_controls: bool = True
    show_stats: bool = True
    show_debug: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        assert 0 < self.frame_width <= 4096, "Invalid frame width"
        assert 0 < self.frame_height <= 2160, "Invalid frame height"
        assert 1 <= self.target_fps <= 240, "Invalid FPS target"
        assert 0 <= self.color_sensitivity <= 1, "Invalid sensitivity"
        assert 0 <= self.mask_smoothing <= 1, "Invalid smoothing"
        assert 0 <= self.feather_amount <= 1, "Invalid feather amount"
        assert 0 <= self.min_mask_area < self.max_mask_area <= 1, "Invalid mask area bounds"
        
        if self.processing_mode not in ["fast", "balanced", "quality"]:
            logger.warning(f"Invalid processing mode: {self.processing_mode}, using balanced")
            self.processing_mode = "balanced"
        
        if self.background_model not in ["static", "mog2", "knn"]:
            logger.warning(f"Invalid background model: {self.background_model}, using mog2")
            self.background_model = "mog2"
    
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
    FAST = auto()
    BALANCED = auto()
    QUALITY = auto()
    
    @classmethod
    def from_string(cls, mode_str: str) -> 'ProcessingMode':
        mapping = {
            "fast": cls.FAST,
            "balanced": cls.BALANCED,
            "quality": cls.QUALITY
        }
        return mapping.get(mode_str.lower(), cls.BALANCED)


# ============================================================================
#                           PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """High-performance timing and statistics collection."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.timings: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.counters: Dict[str, int] = defaultdict(int)
        self.start_time = time.perf_counter()
        self.frame_count = 0
    
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
        total_time = time.perf_counter() - self.start_time
        return self.frame_count / total_time if total_time > 0 else 0.0
    
    def get_average_time(self, section_name: str) -> float:
        """Get average time for a section."""
        timings = self.timings.get(section_name, [])
        return np.mean(timings) if timings else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            "fps": self.get_fps(),
            "frame_count": self.frame_count,
            "runtime": time.perf_counter() - self.start_time,
            "section_times": {},
            "counters": dict(self.counters)
        }
        
        for section, timings in self.timings.items():
            if timings:
                timings_list = list(timings)
                stats["section_times"][section] = {
                    "mean": np.mean(timings_list),
                    "min": np.min(timings_list),
                    "max": np.max(timings_list),
                    "std": np.std(timings_list),
                    "95th": np.percentile(timings_list, 95)
                }
        
        return stats
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.timings.clear()
        self.counters.clear()
        self.start_time = time.perf_counter()
        self.frame_count = 0


# ============================================================================
#                           COLOR DETECTOR
# ============================================================================

class ColorDetector:
    """Advanced color detection with machine learning adaptation."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        # Color presets in HSV and LAB
        self.color_presets = {
            'green': {
                'hsv_range': [(35, 40, 40), (85, 255, 255)],
                'lab_range': [(0, 0, 0), (255, 255, 255)],
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
            }
        }
        
        self.current_color = 'green'
        self.adaptive_model = None
        self.color_history = deque(maxlen=100)
        
        self._init_adaptive_model()
    
    def _init_adaptive_model(self) -> None:
        """Initialize adaptive color learning model."""
        if SKLEARN_AVAILABLE and self.config.adaptive_threshold:
            try:
                self.adaptive_model = GaussianMixture(
                    n_components=3,
                    covariance_type='full'
                )
            except Exception as e:
                logger.warning(f"Failed to init adaptive model: {e}")
                self.adaptive_model = None
    
    @monitor.time_section("color_detection")
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect target colors in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        masks = []
        confidences = []
        
        for color_name, preset in self.color_presets.items():
            if color_name == 'custom' and not self.config.use_multi_color:
                continue
            
            # HSV detection
            hsv_lower = np.array(preset['hsv_range'][0], dtype=np.uint8)
            hsv_upper = np.array(preset['hsv_range'][1], dtype=np.uint8)
            hsv_mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            
            # LAB detection
            lab_lower = np.array(preset['lab_range'][0], dtype=np.uint8)
            lab_upper = np.array(preset['lab_range'][1], dtype=np.uint8)
            lab_mask = cv2.inRange(lab, lab_lower, lab_upper)
            
            # Combine masks
            combined_mask = cv2.bitwise_and(hsv_mask, lab_mask)
            
            if np.count_nonzero(combined_mask) > 0:
                masks.append(combined_mask)
                mask_area = np.count_nonzero(combined_mask) / self.config.frame_area
                
                # Calculate confidence
                if 0.01 < mask_area < 0.5:
                    confidence = preset['weight'] * (1.0 - abs(0.2 - mask_area) / 0.2)
                else:
                    confidence = 0
                confidences.append(confidence)
        
        if not masks:
            return np.zeros(frame.shape[:2], dtype=np.uint8), {"confidence": 0}
        
        # Weighted combination
        combined_mask = self._combine_masks(masks, confidences)
        
        # Update adaptive model
        if self.adaptive_model is not None and np.count_nonzero(combined_mask) > 100:
            self._update_model(frame, combined_mask)
        
        stats = {
            "confidence": np.mean(confidences) if confidences else 0,
            "num_colors": len(masks),
            "mask_area": np.count_nonzero(combined_mask) / self.config.frame_area
        }
        
        return combined_mask, stats
    
    def _combine_masks(self, masks: List[np.ndarray], 
                       confidences: List[float]) -> np.ndarray:
        """Combine multiple masks with confidence weights."""
        total_confidence = sum(confidences)
        
        if total_confidence > 0:
            combined = np.zeros_like(masks[0], dtype=np.float32)
            for mask, confidence in zip(masks, confidences):
                combined += mask.astype(np.float32) * (confidence / total_confidence)
            return np.clip(combined, 0, 255).astype(np.uint8)
        else:
            return masks[0]
    
    def _update_model(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Update adaptive color model with new samples."""
        try:
            masked_pixels = frame[mask > 0]
            if len(masked_pixels) > 1000:
                samples = masked_pixels[
                    np.random.choice(len(masked_pixels), 1000, replace=False)
                ]
                samples_hsv = cv2.cvtColor(
                    samples.reshape(-1, 1, 3), 
                    cv2.COLOR_BGR2HSV
                ).reshape(-1, 3)
                self.adaptive_model.partial_fit(samples_hsv)
                self._adjust_color_ranges()
        except Exception as e:
            logger.warning(f"Model update failed: {e}")
    
    def _adjust_color_ranges(self) -> None:
        """Adjust color ranges based on learned model."""
        if self.adaptive_model is None:
            return
        
        for color_name in self.color_presets:
            if color_name == 'custom':
                continue
            
            means = self.adaptive_model.means_
            covariances = self.adaptive_model.covariances_
            
            current_hsv = np.mean(self.color_presets[color_name]['hsv_range'], axis=0)
            distances = [np.linalg.norm(mean - current_hsv) for mean in means]
            closest_idx = np.argmin(distances)
            
            mean = means[closest_idx]
            std = np.sqrt(np.diag(covariances[closest_idx]))
            
            new_lower = np.clip(mean - 2 * std, 0, 255).astype(np.uint8)
            new_upper = np.clip(mean + 2 * std, 0, 255).astype(np.uint8)
            
            self.color_presets[color_name]['hsv_range'][0] = tuple(new_lower)
            self.color_presets[color_name]['hsv_range'][1] = tuple(new_upper)


# ============================================================================
#                           MASK PROCESSOR
# ============================================================================

class MaskProcessor:
    """Advanced mask processing with temporal stabilization."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        self._init_kernels()
        self.mask_history = deque(maxlen=config.temporal_stability)
        self.prev_gray = None
        self.flow_history = deque(maxlen=3)
    
    def _init_kernels(self) -> None:
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
    def process(self, mask: np.ndarray, 
                frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Process and refine mask."""
        if mask is None or mask.size == 0:
            return np.zeros(self.config.frame_size[::-1], dtype=np.uint8)
        
        processed = mask.copy()
        
        # Morphological operations
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, self.morph_kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, self.morph_kernel)
        
        # Remove small regions
        processed = self._remove_small_regions(processed)
        
        # Apply blur
        if self.blur_kernel[0] > 1:
            processed = cv2.GaussianBlur(processed, self.blur_kernel, 0)
        
        # Temporal stabilization
        if frame is not None:
            processed = self._stabilize_temporally(processed, frame)
        
        # Feather edges
        if self.config.feather_amount > 0:
            processed = self._feather_edges(processed)
        
        # Area constraints
        processed = self._apply_area_constraints(processed)
        
        return processed
    
    def _remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove small disconnected regions."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_area = self.config.frame_area * self.config.min_mask_area
        
        result = mask.copy()
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                cv2.drawContours(result, [contour], 0, 0, -1)
        
        return result
    
    def _stabilize_temporally(self, mask: np.ndarray, 
                             frame: np.ndarray) -> np.ndarray:
        """Apply temporal stabilization using optical flow."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None and self.mask_history:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            prev_mask = self.mask_history[-1]
            h, w = prev_mask.shape
            
            # Create coordinate grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + flow[..., 0]).astype(np.float32)
            map_y = (y + flow[..., 1]).astype(np.float32)
            
            # Warp previous mask
            warped = cv2.remap(prev_mask, map_x, map_y, cv2.INTER_LINEAR)
            
            # Blend with current mask
            alpha = self.config.mask_smoothing
            mask = cv2.addWeighted(mask, 1 - alpha, warped, alpha, 0)
            
            self.flow_history.append(flow)
        
        self.prev_gray = gray
        self.mask_history.append(mask.copy())
        
        return mask
    
    def _feather_edges(self, mask: np.ndarray) -> np.ndarray:
        """Feather mask edges for smooth transitions."""
        mask_float = mask.astype(np.float32) / 255.0
        
        # Distance transform
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        
        # Blend with original
        feathered = mask_float * (1 - self.config.feather_amount) + \
                   dist * self.config.feather_amount
        
        return np.clip(feathered * 255, 0, 255).astype(np.uint8)
    
    def _apply_area_constraints(self, mask: np.ndarray) -> np.ndarray:
        """Apply minimum and maximum area constraints."""
        mask_area = np.count_nonzero(mask) / self.config.frame_area
        
        if mask_area > self.config.max_mask_area:
            scale_factor = self.config.max_mask_area / mask_area
            mask = self._scale_mask(mask, scale_factor)
        
        return mask
    
    def _scale_mask(self, mask: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale mask while preserving shape."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return mask
        
        scaled = np.zeros_like(mask)
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            scaled_contour = contour.astype(np.float32)
            scaled_contour = (scaled_contour - [cx, cy]) * np.sqrt(scale_factor) + [cx, cy]
            scaled_contour = scaled_contour.astype(np.int32)
            
            cv2.drawContours(scaled, [scaled_contour], 0, 255, -1)
        
        return scaled


# ============================================================================
#                           BACKGROUND MANAGER
# ============================================================================

class BackgroundManager:
    """Background management with adaptive learning and restoration."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        self.background = None
        self.background_model = None
        self.background_history = deque(maxlen=10)
        self.inpainting_radius = 3
        
        self._init_background_model()
    
    def _init_background_model(self) -> None:
        """Initialize background subtraction model."""
        if self.config.background_model == "mog2":
            self.background_model = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=False
            )
        elif self.config.background_model == "knn":
            self.background_model = cv2.createBackgroundSubtractorKNN(
                history=500,
                dist2Threshold=400,
                detectShadows=False
            )
    
    @monitor.time_section("background_capture")
    def capture(self, camera: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Capture initial background with motion detection."""
        frames = []
        motion_scores = []
        
        logger.info("Capturing background...")
        
        for i in range(min(self.config.background_frames, 60)):  # Max 60 frames
            ret, frame = camera.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, self.config.frame_size)
            frame = cv2.flip(frame, 1)
            
            if frames:
                prev_gray = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                motion = cv2.absdiff(prev_gray, curr_gray)
                motion_score = np.mean(motion)
                
                if motion_score < 10:  # Low motion threshold
                    frames.append(frame)
                    motion_scores.append(motion_score)
            else:
                frames.append(frame)
            
            if len(frames) >= 20:
                break
        
        if not frames:
            logger.error("Background capture failed")
            return None
        
        # Use median for low motion, average otherwise
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
    def update(self, frame: np.ndarray, mask: np.ndarray) -> None:
        """Update background model with new information."""
        if self.background is None:
            self.background = frame.copy()
            return
        
        if self.background_model is not None:
            fg_mask = self.background_model.apply(frame)
            bg_from_model = self.background_model.getBackgroundImage()
            
            if bg_from_model is not None:
                alpha = self.config.background_update_rate
                self.background = cv2.addWeighted(
                    self.background, 1 - alpha,
                    bg_from_model, alpha, 0
                )
        
        # Update in non-masked areas
        inverse_mask = cv2.bitwise_not(mask)
        update_region = inverse_mask > 0
        
        if np.any(update_region):
            alpha = self.config.background_update_rate * 0.1
            self.background[update_region] = cv2.addWeighted(
                self.background[update_region], 1 - alpha,
                frame[update_region], alpha, 0
            ).astype(np.uint8)
        
        self.background_history.append(self.background.copy())
    
    @monitor.time_section("background_restoration")
    def restore(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Restore background in masked areas."""
        if self.background is None or not self.config.enable_background_restoration:
            return frame.copy()
        
        if np.count_nonzero(mask) > 100:
            # Inpaint missing areas
            inpainted = cv2.inpaint(
                frame, mask,
                inpaintRadius=self.inpainting_radius,
                flags=cv2.INPAINT_TELEA
            )
            
            # Blend with learned background
            alpha = 0.3
            restored = cv2.addWeighted(
                inpainted, alpha,
                self.background, 1 - alpha, 0
            )
            
            return restored
        
        return frame.copy()


# ============================================================================
#                           GPU ACCELERATOR
# ============================================================================

class GPUAccelerator:
    """GPU acceleration wrapper for OpenCV operations."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.use_gpu = config.enable_gpu and cv2.ocl.haveOpenCL()
        self.use_cupy = CUDA_AVAILABLE and config.enable_gpu
        
        if self.use_gpu:
            cv2.ocl.setUseOpenCL(True)
            logger.info("OpenCL acceleration enabled")
        
        if self.use_cupy:
            logger.info("CuPy acceleration available")
    
    def to_gpu(self, frame: np.ndarray) -> Any:
        """Transfer frame to GPU memory."""
        if self.use_cupy:
            return cp.asarray(frame)
        elif self.use_gpu:
            return cv2.UMat(frame)
        else:
            return frame
    
    def from_gpu(self, gpu_frame: Any) -> np.ndarray:
        """Transfer frame from GPU to CPU."""
        if isinstance(gpu_frame, cp.ndarray):
            return cp.asnumpy(gpu_frame)
        elif isinstance(gpu_frame, cv2.UMat):
            return gpu_frame.get()
        else:
            return gpu_frame
    
    @monitor.time_section("gpu_color_conversion")
    def cvt_color(self, frame: Any, code: int) -> Any:
        """Color conversion on GPU if available."""
        if self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.cvtColor(frame, code)
        elif self.use_cupy and isinstance(frame, cp.ndarray):
            # Convert to CPU temporarily
            return cv2.cvtColor(self.from_gpu(frame), code)
        else:
            return cv2.cvtColor(frame, code)
    
    @monitor.time_section("gpu_filter")
    def gaussian_blur(self, frame: Any, ksize: Tuple[int, int], 
                     sigma: float = 0) -> Any:
        """Gaussian blur on GPU if available."""
        if self.use_gpu and isinstance(frame, cv2.UMat):
            return cv2.GaussianBlur(frame, ksize, sigma)
        elif self.use_cupy and isinstance(frame, cp.ndarray):
            import cupyx.scipy.ndimage
            sigma_px = sigma if sigma > 0 else 0.3 * ((ksize[0] - 1) * 0.5 - 1) + 0.8
            return cupyx.scipy.ndimage.gaussian_filter(frame, sigma_px)
        else:
            return cv2.GaussianBlur(frame, ksize, sigma)


# ============================================================================
#                           UI MANAGER
# ============================================================================

class UIManager:
    """User interface manager with interactive controls."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.monitor = PerformanceMonitor()
        
        self.window_name = "Invisibility Cloak 4.0"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, *config.frame_size)
        
        if config.show_controls:
            self._create_controls()
    
    def _create_controls(self) -> None:
        """Create interactive trackbars."""
        cv2.createTrackbar('Sensitivity', self.window_name,
                          int(self.config.color_sensitivity * 100), 100,
                          lambda x: None)
        cv2.createTrackbar('Smoothing', self.window_name,
                          int(self.config.mask_smoothing * 100), 100,
                          lambda x: None)
        cv2.createTrackbar('Feather', self.window_name,
                          int(self.config.feather_amount * 100), 100,
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
    
    @monitor.time_section("ui_render")
    def render(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Render UI elements on frame."""
        output = frame.copy()
        
        if self.config.show_stats:
            output = self._render_stats(output, stats)
        
        if self.config.show_debug:
            output = self._render_debug(output, stats)
        
        return output
    
    def _render_stats(self, frame: np.ndarray, 
                     stats: Dict[str, Any]) -> np.ndarray:
        """Render statistics panel."""
        panel_height = 200
        panel_width = 350
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), 
                     (0, 0, 0), -1)
        
        # Statistics text
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
        
        # Draw text
        for i, text in enumerate(stats_text):
            cv2.putText(overlay, text, (20, 40 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 1)
        
        # Blend with original
        alpha = 0.7
        frame[10:panel_height, 10:panel_width] = cv2.addWeighted(
            frame[10:panel_height, 10:panel_width], 1 - alpha,
            overlay[10:panel_height, 10:panel_width], alpha, 0
        )
        
        return frame
    
    def _render_debug(self, frame: np.ndarray, 
                     stats: Dict[str, Any]) -> np.ndarray:
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


# ============================================================================
#                           MAIN PIPELINE
# ============================================================================

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
        
        # State
        self.is_initialized = False
        self.current_frame = None
        self.current_mask = None
        
        # Thread pool
        if config.enable_multithreading:
            self.thread_pool = ThreadPoolExecutor(max_workers=config.num_threads)
        else:
            self.thread_pool = None
    
    @monitor.time_section("pipeline_initialize")
    def initialize(self, camera: cv2.VideoCapture) -> bool:
        """Initialize pipeline with background capture."""
        background = self.background_manager.capture(camera)
        
        if background is None:
            logger.error("Pipeline initialization failed")
            return False
        
        self.is_initialized = True
        logger.info("Pipeline initialized successfully")
        return True
    
    @monitor.time_section("pipeline_process")
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process a single frame through the pipeline."""
        if not self.is_initialized:
            return frame, {"error": "Pipeline not initialized"}
        
        start_time = time.perf_counter()
        self.current_frame = frame.copy()
        
        # 1. Color detection
        raw_mask, color_stats = self.color_detector.detect(frame)
        
        # 2. Mask processing
        processed_mask = self.mask_processor.process(raw_mask, frame)
        self.current_mask = processed_mask
        
        # 3. Background restoration
        restored_bg = self.background_manager.restore(frame, processed_mask)
        
        # 4. Apply cloak effect
        if self.background_manager.background is not None:
            output = self._apply_effect(frame, processed_mask, restored_bg)
        else:
            output = frame
        
        # 5. Update background
        self.background_manager.update(frame, processed_mask)
        
        # 6. Update UI controls
        self.ui_manager.update_controls()
        
        # 7. Prepare statistics
        pipeline_time = time.perf_counter() - start_time
        stats = {
            **color_stats,
            "pipeline_time": pipeline_time,
            "frame_time": 1.0 / self.monitor.get_fps() if self.monitor.get_fps() > 0 else 0,
            "fps": self.monitor.get_fps(),
            "frame_count": self.monitor.frame_count,
            "gpu_enabled": self.config.enable_gpu,
            "processing_mode": self.config.processing_mode,
        }
        
        self.monitor.frame_count += 1
        
        return output, stats
    
    def _apply_effect(self, frame: np.ndarray, mask: np.ndarray,
                     background: np.ndarray) -> np.ndarray:
        """Apply invisibility cloak effect."""
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        alpha = self.config.color_sensitivity
        
        foreground = frame.astype(np.float32)
        background = background.astype(np.float32)
        
        blended = foreground * (1 - mask_3ch * alpha) + \
                 background * (mask_3ch * alpha)
        
        return np.clip(blended, 0, 255).astype(np.uint8)
    
    def async_process(self, frame: np.ndarray) -> Any:
        """Process frame asynchronously."""
        if self.thread_pool is None:
            return self.process(frame)
        
        return self.thread_pool.submit(self.process, frame)
    
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
                "frame_size": self.config.frame_size,
            }
        }


# ============================================================================
#                           MAIN APPLICATION
# ============================================================================

class InvisibilityCloakSystem:
    """Main application class."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.pipeline = InvisibilityCloakPipeline(self.config)
        
        self.is_running = False
        self.camera = None
        self.start_time = time.perf_counter()
        self.frame_counter = 0
        self.settings_file = Path("cloak_settings.json")
    
    def setup_camera(self) -> bool:
        """Setup camera with optimal settings."""
        self.camera = cv2.VideoCapture(self.config.camera_index)
        
        if not self.camera.isOpened():
            logger.error(f"Cannot open camera {self.config.camera_index}")
            return False
        
        # Configure camera
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Verify settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
        return True
    
    def load_settings(self) -> bool:
        """Load settings from JSON file."""
        if not self.settings_file.exists():
            return False
        
        try:
            with open(self.settings_file, 'r') as f:
                saved = json.load(f)
            
            # Update config
            for key, value in saved.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            logger.info("Settings loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return False
    
    def save_settings(self) -> bool:
        """Save settings to JSON file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
            
            logger.info("Settings saved")
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
            ord('1'): lambda: self.change_color('green'),
            ord('2'): lambda: self.change_color('red'),
            ord('3'): lambda: self.change_color('blue'),
            ord('f'): lambda: self.change_mode('fast'),
            ord('b'): lambda: self.change_mode('balanced'),
            ord('p'): lambda: self.change_mode('quality'),
            ord('+'): self.increase_sensitivity,
            ord('-'): self.decrease_sensitivity,
            27: lambda: setattr(self, 'is_running', False),  # ESC
        }
        
        if key in actions:
            actions[key]()
            return True
        
        return False
    
    def toggle_debug(self) -> None:
        """Toggle debug mode."""
        self.config.show_debug = not self.config.show_debug
        logger.info(f"Debug: {'ON' if self.config.show_debug else 'OFF'}")
    
    def toggle_controls(self) -> None:
        """Toggle UI controls."""
        self.config.show_controls = not self.config.show_controls
        logger.info(f"Controls: {'SHOWN' if self.config.show_controls else 'HIDDEN'}")
    
    def reset_background(self) -> None:
        """Reset background."""
        logger.info("Resetting background...")
        if self.camera is not None:
            self.pipeline.background_manager.capture(self.camera)
    
    def change_color(self, color: str) -> None:
        """Change color preset."""
        if color in self.pipeline.color_detector.color_presets:
            self.pipeline.color_detector.current_color = color
            logger.info(f"Color: {color}")
    
    def change_mode(self, mode: str) -> None:
        """Change processing mode."""
        self.config.processing_mode = mode
        self.pipeline.mask_processor._init_kernels()
        logger.info(f"Mode: {mode}")
    
    def increase_sensitivity(self) -> None:
        """Increase sensitivity."""
        self.config.color_sensitivity = min(1.0, self.config.color_sensitivity + 0.05)
        logger.info(f"Sensitivity: {self.config.color_sensitivity:.2f}")
    
    def decrease_sensitivity(self) -> None:
        """Decrease sensitivity."""
        self.config.color_sensitivity = max(0.0, self.config.color_sensitivity - 0.05)
        logger.info(f"Sensitivity: {self.config.color_sensitivity:.2f}")
    
    def run(self) -> None:
        """Main application loop."""
        logger.info("Starting Invisibility Cloak System 4.0")
        
        if not self.setup_camera():
            return
        
        self.load_settings()
        
        if not self.pipeline.initialize(self.camera):
            self.camera.release()
            return
        
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
                
                # Resize and flip
                frame = cv2.resize(frame, self.config.frame_size)
                frame = cv2.flip(frame, 1)
                
                # Process frame
                output, stats = self.pipeline.process(frame)
                
                # Render UI
                output = self.pipeline.ui_manager.render(output, stats)
                
                # Display
                cv2.imshow(self.pipeline.ui_manager.window_name, output)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                self.handle_keyboard(key)
                
                # Check window
                if cv2.getWindowProperty(self.pipeline.ui_manager.window_name, 
                                        cv2.WND_PROP_VISIBLE) < 1:
                    break
                
                # Update performance
                frame_time = time.perf_counter() - loop_start
                frame_times.append(frame_time)
                self.frame_counter += 1
                
                # Log periodically
                if self.frame_counter % 100 == 0:
                    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
                    logger.info(f"FPS: {avg_fps:.1f}, "
                               f"Mask: {stats.get('mask_area', 0)*100:.1f}%")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up...")
        
        self.save_settings()
        
        if self.camera is not None:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        if self.pipeline.thread_pool is not None:
            self.pipeline.thread_pool.shutdown(wait=True)
        
        # Print summary
        total_time = time.perf_counter() - self.start_time
        avg_fps = self.frame_counter / total_time if total_time > 0 else 0
        
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info(f"Frames: {self.frame_counter}")
        logger.info(f"Time: {total_time:.1f}s")
        logger.info(f"FPS: {avg_fps:.2f}")
        logger.info(f"Size: {self.config.frame_size}")
        logger.info("=" * 50)


# ============================================================================
#                           ENTRY POINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Invisibility Cloak System 4.0',
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
                       help='Number of threads')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-ui', dest='show_ui', action='store_false',
                       help='Disable UI controls')
    parser.set_defaults(show_ui=True)
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    config = SystemConfig(
        camera_index=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.fps,
        processing_mode=args.mode,
        enable_gpu=args.gpu,
        num_threads=args.threads or max(1, mp.cpu_count() - 1),
        enable_profiling=args.profile,
        show_controls=args.show_ui,
        show_debug=args.debug,
    )
    
    if args.profile:
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            system = InvisibilityCloakSystem(config)
            system.run()
        finally:
            profiler.disable()
            
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
            stats.dump_stats('profile.prof')
            logger.info("Profile saved to 'profile.prof'")
    else:
        system = InvisibilityCloakSystem(config)
        system.run()


if __name__ == "__main__":
    main()
