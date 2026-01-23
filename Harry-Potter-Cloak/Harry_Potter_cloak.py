
    """Main configuration class for the invisibility cloak system."""
    camera_index: int = 0
    frame_width: int = 1280  # Increased for better quality
    frame_height: int = 720
    background_frames: int = 60  # Reduced for faster capture
    fps_window: int = 30
    mask_history_size: int = 7
    gpu_acceleration: bool = False
    enable_multiprocessing: bool = True
    min_mask_ratio: float = 0.01
    max_mask_ratio: float = 0.4
    adaptive_background: bool = True
    background_learning_rate: float = 0.001
    logging_level: int = logging.INFO
    
    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self.frame_width, self.frame_height)

class ProcessingMode(Enum):
    """Processing modes for different quality/performance trade-offs."""
    FAST = "fast"      # Optimized for performance
    BALANCED = "balanced"  # Default balanced mode
    QUALITY = "quality"    # Highest quality, slower

# =============================================================================
#                           HSV COLOR MANAGEMENT
# =============================================================================

@dataclass
class HSVColor:
    """HSV color range with adaptive adjustment capabilities."""
    lower: np.ndarray
    upper: np.ndarray
    
    def __post_init__(self):
        self.original_lower = self.lower.copy()
        self.original_upper = self.upper.copy()
    
    def adjust_for_lighting(self, lighting_factor: float) -> None:
        """Adjust HSV ranges based on lighting conditions."""
        # Reduce saturation range in bright conditions
        if lighting_factor > 0.7:
            self.lower[1] = int(self.original_lower[1] * 0.8)
            self.upper[1] = int(min(255, self.original_upper[1] * 1.1))
        # Expand value range in dark conditions
        elif lighting_factor < 0.3:
            self.lower[2] = int(self.original_lower[2] * 0.6)
            self.upper[2] = int(min(255, self.original_upper[2] * 1.2))
        else:
            self.lower = self.original_lower.copy()
            self.upper = self.original_upper.copy()

class HSVManager:
    """Manages HSV color detection with adaptive capabilities."""
    
    HSV_PRESETS = {
        '1': (np.array([50, 40, 40]), np.array([80, 255, 255])),   # Green
        '2': (np.array([0, 120, 70]), np.array([10, 255, 255])),   # Red 1
        '3': (np.array([170, 120, 70]), np.array([180, 255, 255])), # Red 2
        '4': (np.array([100, 40, 40]), np.array([140, 255, 255])),  # Blue
        '5': (np.array([20, 100, 100]), np.array([30, 255, 255])),  # Yellow
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.current_preset = '1'
        self.auto_mode = False
        self.trackbar_window = "HSV Controls"
        
    def setup_trackbars(self) -> None:
        """Setup HSV adjustment trackbars."""
        cv2.namedWindow(self.trackbar_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.trackbar_window, 400, 300)
        
        cv2.createTrackbar("LH", self.trackbar_window, 50, 180, lambda x: None)
        cv2.createTrackbar("LS", self.trackbar_window, 100, 255, lambda x: None)
        cv2.createTrackbar("LV", self.trackbar_window, 100, 255, lambda x: None)
        cv2.createTrackbar("UH", self.trackbar_window, 80, 180, lambda x: None)
        cv2.createTrackbar("US", self.trackbar_window, 255, 255, lambda x: None)
        cv2.createTrackbar("UV", self.trackbar_window, 255, 255, lambda x: None)
    
    def get_hsv_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current HSV range based on mode."""
        if self.auto_mode:
            return HSV_PRESETS[self.current_preset]
        else:
            return self._read_trackbars()
    
    def _read_trackbars(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read HSV values from trackbars."""
        lower = np.array([
            cv2.getTrackbarPos("LH", self.trackbar_window),
            cv2.getTrackbarPos("LS", self.trackbar_window),
            cv2.getTrackbarPos("LV", self.trackbar_window)
        ])
        upper = np.array([
            cv2.getTrackbarPos("UH", self.trackbar_window),
            cv2.getTrackbarPos("US", self.trackbar_window),
            cv2.getTrackbarPos("UV", self.trackbar_window)
        ])
        return lower, upper
    
    def set_preset(self, preset_key: str) -> None:
        """Set HSV preset and update trackbars."""
        if preset_key in self.HSV_PRESETS:
            self.current_preset = preset_key
            lower, upper = self.HSV_PRESETS[preset_key]
            
            # Update trackbars
            cv2.setTrackbarPos("LH", self.trackbar_window, lower[0])
            cv2.setTrackbarPos("LS", self.trackbar_window, lower[1])
            cv2.setTrackbarPos("LV", self.trackbar_window, lower[2])
            cv2.setTrackbarPos("UH", self.trackbar_window, upper[0])
            cv2.setTrackbarPos("US", self.trackbar_window, upper[1])
            cv2.setTrackbarPos("UV", self.trackbar_window, upper[2])

# =============================================================================
#                           FRAME PROCESSOR
# =============================================================================

class FrameProcessor:
    """Handles frame processing with multiple optimization strategies."""
    
    def __init__(self, config: Config):
        self.config = config
        self.mode = ProcessingMode.BALANCED
        self.gpu_available = self._check_gpu()
        
        # Initialize kernels based on mode
        self._init_kernels()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4) if config.enable_multiprocessing else None
        
    def _check_gpu(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0 and self.config.gpu_acceleration:
                logging.info("CUDA GPU acceleration available and enabled")
                return True
        except:
            pass
        logging.info("GPU acceleration not available or disabled")
        return False
    
    def _init_kernels(self) -> None:
        """Initialize processing kernels based on mode."""
        if self.mode == ProcessingMode.FAST:
            self.morph_kernel = np.ones((3, 3), np.uint8)
            self.blur_size = (5, 5)
            self.erode_iterations = 1
            self.dilate_iterations = 1
        elif self.mode == ProcessingMode.BALANCED:
            self.morph_kernel = np.ones((5, 5), np.uint8)
            self.blur_size = (7, 7)
            self.erode_iterations = 2
            self.dilate_iterations = 2
        else:  # QUALITY
            self.morph_kernel = np.ones((7, 7), np.uint8)
            self.blur_size = (9, 9)
            self.erode_iterations = 3
            self.dilate_iterations = 3
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame with flip and resize."""
        frame = cv2.flip(frame, 1)
        return cv2.resize(frame, self.config.frame_size, interpolation=cv2.INTER_LINEAR)
    
    def create_mask(self, hsv_frame: np.ndarray, lower: np.ndarray, 
                   upper: np.ndarray) -> np.ndarray:
        """Create refined mask from HSV range."""
        # Create base mask
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Morphological operations
        mask = cv2.erode(mask, self.morph_kernel, iterations=self.erode_iterations)
        mask = cv2.dilate(mask, self.morph_kernel, iterations=self.dilate_iterations)
        
        # Apply blur based on mode
        if self.mode != ProcessingMode.FAST:
            mask = cv2.GaussianBlur(mask, self.blur_size, 0)
        
        return mask
    
    def feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply edge-aware feathering to mask."""
        # Convert to float for processing
        mask_float = mask.astype(np.float32) / 255.0
        
        # Use guided filter for edge-aware smoothing
        if self.mode == ProcessingMode.QUALITY and hasattr(cv2, 'ximgproc'):
            try:
                mask_float = cv2.ximgproc.guidedFilter(
                    mask_float, mask_float, radius=10, eps=0.01
                )
            except:
                pass
        
        # Apply distance transform for smooth edges
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Combine with original mask
        feathered = mask_float * dist
        
        return (feathered * 255).astype(np.uint8)
    
    def stabilize_masks(self, masks: Deque[np.ndarray]) -> np.ndarray:
        """Apply temporal stabilization to masks."""
        if len(masks) < 2:
            return masks[-1] if masks else np.zeros(self.config.frame_size, dtype=np.uint8)
        
        # Weighted average with recent masks having higher weight
        weights = np.linspace(0.1, 1.0, len(masks))
        weights /= weights.sum()
        
        stabilized = np.zeros_like(masks[0], dtype=np.float32)
        for mask, weight in zip(masks, weights):
            stabilized += mask.astype(np.float32) * weight
        
        return stabilized.astype(np.uint8)

# =============================================================================
#                           BACKGROUND MANAGER
# =============================================================================

class BackgroundManager:
    """Manages background capture and maintenance."""
    
    def __init__(self, config: Config):
        self.config = config
        self.background = None
        self.adaptive_bg = None
        self.last_update = time.time()
    
    def capture_background(self, cap: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Capture stable background with motion detection."""
        frames = []
        prev_frame = None
        motion_threshold = 5.0
        
        logging.info("Capturing background... Please remain still.")
        
        for i in range(self.config.background_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, self.config.frame_size)
            frame = cv2.flip(frame, 1)
            
            # Check for motion
            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame)
                motion = np.mean(diff)
                
                if motion > motion_threshold:
                    logging.debug(f"Frame {i}: Motion detected ({motion:.2f}), skipping")
                    continue
            
            frames.append(frame)
            prev_frame = frame.copy()
            
            # Show progress
            progress = int((i + 1) / self.config.background_frames * 100)
            cv2.putText(frame, f"Capturing: {progress}%", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Background Capture", frame)
            cv2.waitKey(1)
        
        cv2.destroyWindow("Background Capture")
        
        if not frames:
            logging.error("No stable frames captured")
            return None
        
        # Use median for robust background
        self.background = np.median(frames, axis=0).astype(np.uint8)
        self.adaptive_bg = self.background.copy()
        
        logging.info("Background captured successfully")
        return self.background
    
    def update_adaptive(self, current_frame: np.ndarray, mask: np.ndarray) -> None:
        """Update adaptive background where cloak is not present."""
        if not self.config.adaptive_background:
            return
        
        # Only update areas without cloak
        inverse_mask = cv2.bitwise_not(mask)
        
        # Slow adaptive update
        self.adaptive_bg = cv2.addWeighted(
            self.adaptive_bg, 0.999,
            cv2.bitwise_and(current_frame, current_frame, mask=inverse_mask), 0.001, 0
        )

# =============================================================================
#                           VISUALIZATION & UI
# =============================================================================

class UIManager:
    """Manages UI elements and visualization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.show_mask = False
        self.show_fps = True
        self.show_stats = True
        
    def draw_control_panel(self, frame: np.ndarray, info: Dict[str, Any]) -> None:
        """Draw control panel with information."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw information
        y_offset = 40
        line_height = 25
        
        stats = [
            f"FPS: {info.get('fps', 0):.1f}",
            f"Mask Ratio: {info.get('mask_ratio', 0)*100:.1f}%",
            f"Alpha: {info.get('alpha', 1.0):.2f}",
            f"Mode: {'AUTO' if info.get('auto_mode', False) else 'MANUAL'}",
            f"Processing: {info.get('processing_mode', 'BALANCED')}",
            f"GPU: {'ON' if info.get('gpu_enabled', False) else 'OFF'}",
        ]
        
        for i, text in enumerate(stats):
            cv2.putText(frame, text, (20, y_offset + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw controls hint
        controls = "Q:Quit A:Auto B:Bg R:Reset [/:Alpha M:Mask 1-5:Presets"
        cv2.putText(frame, controls, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_mask_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Overlay mask visualization on frame."""
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 0] = 0  # Blue channel for mask
        mask_colored[:, :, 1] = 0
        
        return cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

# =============================================================================
#                           MAIN APPLICATION
# =============================================================================

class InvisibilityCloak:
    """Main invisibility cloak application."""
    
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
        self.last_valid_hsv = None
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=self.config.logging_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """Initialize camera with error handling."""
        try:
            cap = cv2.VideoCapture(self.config.camera_index)
            
            # Try to set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
            
            if not cap.isOpened():
                logging.error(f"Failed to open camera at index {self.config.camera_index}")
                # Try other indices
                for i in range(3):
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        logging.info(f"Found camera at index {i}")
                        break
            
            return cap if cap.isOpened() else None
            
        except Exception as e:
            logging.error(f"Camera initialization error: {e}")
            return None
    
    def run(self) -> None:
        """Main application loop."""
        logging.info("Starting Invisibility Cloak System")
        
        # Initialize camera
        cap = self.initialize_camera()
        if cap is None:
            logging.error("No camera available. Exiting.")
            return
        
        # Setup UI
        self.hsv_manager.setup_trackbars()
        cv2.namedWindow("Invisibility Cloak", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Invisibility Cloak", *self.config.frame_size)
        
        # Capture initial background
        if not self.bg_manager.capture_background(cap):
            logging.error("Failed to capture background. Exiting.")
            cap.release()
            return
        
        self.running = True
        prev_time = time.time()
        
        try:
            while self.running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to read frame")
                    time.sleep(0.01)
                    continue
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time)
                prev_time = current_time
                self.fps_history.append(fps)
                
                # Update frame counter
                self.frame_count += 1
                
                # Display
                cv2.imshow("Invisibility Cloak", processed_frame)
                
                # Handle keyboard input
                self.handle_input()
                
                # Break on 'q' or window close
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty("Invisibility Cloak", cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.cleanup(cap)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the pipeline."""
        # Preprocess
        frame = self.frame_processor.preprocess(frame)
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get HSV range
        lower, upper = self.hsv_manager.get_hsv_range()
        
        # Create and refine mask
        mask = self.frame_processor.create_mask(hsv, lower, upper)
        
        # Update mask history and stabilize
        self.mask_history.append(mask)
        mask = self.frame_processor.stabilize_masks(self.mask_history)
        
        # Feather mask edges
        mask = self.frame_processor.feather_mask(mask)
        
        # Calculate mask ratio for adaptive adjustments
        mask_ratio = np.count_nonzero(mask) / mask.size
        
        # Adaptive HSV adjustment based on mask ratio
        if mask_ratio < self.config.min_mask_ratio and self.last_valid_hsv:
            lower, upper = self.last_valid_hsv
            mask = self.frame_processor.create_mask(hsv, lower, upper)
        elif self.config.min_mask_ratio < mask_ratio < self.config.max_mask_ratio:
            self.last_valid_hsv = (lower.copy(), upper.copy())
        
        # Update adaptive background
        self.bg_manager.update_adaptive(frame, mask)
        
        # Create invisibility effect
        inverse_mask = cv2.bitwise_not(mask)
        
        # Use adaptive background if available
        background = self.bg_manager.adaptive_bg if self.config.adaptive_background \
                    else self.bg_manager.background
        
        # Apply cloak effect
        cloak_area = cv2.bitwise_and(background, background, mask=mask)
        visible_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)
        
        # Blend
        result = cv2.addWeighted(cloak_area, self.alpha, visible_area, 1 - self.alpha, 0)
        
        # UI overlays
        if self.ui_manager.show_mask:
            result = self.ui_manager.draw_mask_overlay(result, mask)
        
        # Draw info panel
        info = {
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'mask_ratio': mask_ratio,
            'alpha': self.alpha,
            'auto_mode': self.hsv_manager.auto_mode,
            'processing_mode': self.frame_processor.mode.value.upper(),
            'gpu_enabled': self.frame_processor.gpu_available,
        }
        self.ui_manager.draw_control_panel(result, info)
        
        return result
    
    def handle_input(self) -> None:
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            self.running = False
        elif key == ord('a'):
            self.hsv_manager.auto_mode = not self.hsv_manager.auto_mode
            logging.info(f"Auto mode: {'ON' if self.hsv_manager.auto_mode else 'OFF'}")
        elif key == ord('b'):
            logging.info("Recapturing background...")
            # Need to pass cap to capture_background, handled in main loop
        elif key == ord('r'):
            self.reset_parameters()
        elif key == ord('['):
            self.alpha = max(0.0, self.alpha - 0.05)
            logging.info(f"Alpha: {self.alpha:.2f}")
        elif key == ord(']'):
            self.alpha = min(1.0, self.alpha + 0.05)
            logging.info(f"Alpha: {self.alpha:.2f}")
        elif key == ord('m'):
            self.ui_manager.show_mask = not self.ui_manager.show_mask
            logging.info(f"Show mask: {self.ui_manager.show_mask}")
        elif key == ord('s'):
            self.save_settings()
        elif key == ord('l'):
            self.load_settings()
        elif ord('1') <= key <= ord('5'):
            self.hsv_manager.set_preset(chr(key))
            logging.info(f"Loaded preset {chr(key)}")
        elif key == ord('f'):
            self.frame_processor.mode = ProcessingMode.FAST
            logging.info("Mode: FAST")
        elif key == ord('b'):
            self.frame_processor.mode = ProcessingMode.BALANCED
            logging.info("Mode: BALANCED")
        elif key == ord('q'):
            self.frame_processor.mode = ProcessingMode.QUALITY
            logging.info("Mode: QUALITY")
    
    def reset_parameters(self) -> None:
        """Reset all parameters to defaults."""
        self.alpha = 1.0
        self.hsv_manager.auto_mode = False
        self.mask_history.clear()
        self.fps_history.clear()
        self.last_valid_hsv = None
        self.hsv_manager.set_preset('1')
        logging.info("All parameters reset to defaults")
    
    def save_settings(self) -> None:
        """Save current settings to file."""
        settings = {
            'alpha': self.alpha,
            'auto_mode': self.hsv_manager.auto_mode,
            'current_preset': self.hsv_manager.current_preset,
            'show_mask': self.ui_manager.show_mask,
            'processing_mode': self.frame_processor.mode.value,
        }
        
        try:
            with open('cloak_settings.json', 'w') as f:
                json.dump(settings, f)
            logging.info("Settings saved successfully")
        except Exception as e:
            logging.error(f"Failed to save settings: {e}")
    
    def load_settings(self) -> None:
        """Load settings from file."""
        try:
            with open('cloak_settings.json', 'r') as f:
                settings = json.load(f)
            
            self.alpha = settings.get('alpha', 1.0)
            self.hsv_manager.auto_mode = settings.get('auto_mode', False)
            self.hsv_manager.set_preset(settings.get('current_preset', '1'))
            self.ui_manager.show_mask = settings.get('show_mask', False)
            
            mode_str = settings.get('processing_mode', 'balanced')
            self.frame_processor.mode = ProcessingMode(mode_str)
            
            logging.info("Settings loaded successfully")
        except FileNotFoundError:
            logging.warning("Settings file not found")
        except Exception as e:
            logging.error(f"Failed to load settings: {e}")
    
    def cleanup(self, cap: cv2.VideoCapture) -> None:
        """Cleanup resources."""
        logging.info("Cleaning up resources...")
        
        # Release camera
        cap.release()
        
        # Calculate and log statistics
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        logging.info(f"Session statistics:")
        logging.info(f"  Frames processed: {self.frame_count}")
        logging.info(f"  Average FPS: {avg_fps:.2f}")
        logging.info(f"  Total time: {elapsed:.2f}s")
        
        # Close all windows
        cv2.destroyAllWindows()
        
        if self.frame_processor.executor:
            self.frame_processor.executor.shutdown()
        
        logging.info("Application terminated")

# =============================================================================
#                           ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Create configuration with optional GPU acceleration
    config = Config(
        frame_width=1280,
        frame_height=720,
        gpu_acceleration=True,  # Set to True if you have CUDA-compatible GPU
        enable_multiprocessing=True,
        adaptive_background=True
    )
    
    # Create and run the application
    app = InvisibilityCloak(config)
    app.run()
