e=np.uint8)),  # Blue
    '5': (np.array([20, 100, 100], dtype=np.uint8), 
          np.array([30, 255, 255], dtype=np.uint8)),   # Orange
}


# =============================================================================
#                           PERFORMANCE UTILITIES
# =============================================================================

def timer_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.debug(f"{func.__name__} executed in {end_time-start_time:.4f}s")
        return result
    return wrapper


class PerformanceTimer:
    """Context manager for timing code blocks."""
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


# =============================================================================
#                           HSV COLOR MANAGER
# =============================================================================

class HSVColorManager:
    """Manages HSV color ranges for cloak detection."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.current_preset = '1'
        self.auto_mode = False
        self._current_range = HSV_PRESETS['1']
    
    def get_hsv_range(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current HSV color range."""
        return self._current_range
    
    def set_preset(self, preset_key: str) -> None:
        """Set a predefined HSV color preset."""
        if preset_key in HSV_PRESETS:
            self.current_preset = preset_key
            self._current_range = HSV_PRESETS[preset_key]
            logging.info(f"HSV preset changed to {preset_key}")
        else:
            logging.warning(f"Invalid HSV preset: {preset_key}")


# =============================================================================
#                           FRAME PROCESSOR
# =============================================================================

class FrameProcessor:
    """Processes video frames for cloak effect."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.processing_mode = ProcessingMode.BALANCED
        self._initialize_kernels()
    
    def _initialize_kernels(self):
        """Initialize processing kernels based on mode."""
        if self.processing_mode == ProcessingMode.FAST:
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self._blur_size = 5
            self._erode_iterations = 1
            self._dilate_iterations = 1
        elif self.processing_mode == ProcessingMode.BALANCED:
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self._blur_size = 7
            self._erode_iterations = 2
            self._dilate_iterations = 2
        else:  # QUALITY
            self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            self._blur_size = 9
            self._erode_iterations = 3
            self._dilate_iterations = 3
    
    @timer_decorator
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess input frame."""
        # Mirror the frame
        frame = cv2.flip(frame, 1)
        
        # Resize if needed
        if frame.shape[:2] != self.config.frame_size[::-1]:
            frame = cv2.resize(frame, self.config.frame_size, 
                             interpolation=cv2.INTER_LINEAR)
        
        return frame
    
    @timer_decorator
    def create_color_mask(self, hsv_frame: np.ndarray, 
                         lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Create mask based on HSV color range."""
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Apply morphological operations
        if self._erode_iterations > 0:
            mask = cv2.erode(mask, self._morph_kernel, 
                           iterations=self._erode_iterations)
        
        if self._dilate_iterations > 0:
            mask = cv2.dilate(mask, self._morph_kernel, 
                            iterations=self._dilate_iterations)
        
        # Apply Gaussian blur for smooth edges
        if self._blur_size > 1:
            mask = cv2.GaussianBlur(mask, (self._blur_size, self._blur_size), 0)
        
        return mask
    
    @timer_decorator
    def feather_mask_edges(self, mask: np.ndarray) -> np.ndarray:
        """Feather mask edges for smooth transitions."""
        if np.count_nonzero(mask) == 0:
            return mask
        
        # Convert to float for processing
        mask_float = mask.astype(np.float32) / 255.0
        
        # Calculate distance transform for edge smoothing
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        cv2.normalize(distance, distance, 0, 1.0, cv2.NORM_MINMAX)
        
        # Combine with original mask
        feathered = mask_float * distance
        
        return (feathered * 255).astype(np.uint8)
    
    @timer_decorator
    def stabilize_mask_temporally(self, masks: Deque[np.ndarray]) -> np.ndarray:
        """Apply temporal stabilization to mask sequence."""
        if not masks:
            return np.zeros(self.config.frame_size[::-1], dtype=np.uint8)
        
        if len(masks) == 1:
            return masks[0]
        
        # Weighted average of recent masks
        num_masks = len(masks)
        weights = np.linspace(0.1, 1.0, num_masks)
        weights /= weights.sum()
        
        # Calculate weighted average
        stabilized = np.zeros_like(masks[0], dtype=np.float32)
        for mask, weight in zip(masks, weights):
            stabilized += mask.astype(np.float32) * weight
        
        # Clip to valid range
        np.clip(stabilized, 0, 255, out=stabilized)
        return stabilized.astype(np.uint8)


# =============================================================================
#                           BACKGROUND MANAGER
# =============================================================================

class BackgroundManager:
    """Manages background capture and updates."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.background = None
        self.adaptive_background = None
    
    @timer_decorator
    def capture_background(self, video_capture: cv2.VideoCapture) -> Optional[np.ndarray]:
        """Capture background from video feed."""
        frames = []
        previous_gray = None
        motion_threshold = 5000
        
        logging.info("Capturing background...")
        
        for i in range(self.config.background_frames):
            ret, frame = video_capture.read()
            if not ret:
                continue
            
            # Preprocess frame
            frame = cv2.resize(frame, self.config.frame_size)
            frame = cv2.flip(frame, 1)
            
            # Detect motion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                diff = cv2.absdiff(gray, previous_gray)
                motion = np.sum(diff)
                
                if motion > motion_threshold:
                    continue  # Skip frames with motion
            
            frames.append(frame)
            previous_gray = gray
            
            if len(frames) >= 20:  # Enough samples
                break
        
        if not frames:
            logging.error("Failed to capture background")
            return None
        
        # Calculate median background
        self.background = np.median(frames, axis=0).astype(np.uint8)
        self.adaptive_background = self.background.copy()
        
        logging.info(f"Background captured from {len(frames)} frames")
        return self.background
    
    @timer_decorator
    def update_adaptive_background(self, current_frame: np.ndarray, 
                                  mask: np.ndarray) -> None:
        """Update adaptive background model."""
        if not self.config.adaptive_background or self.adaptive_background is None:
            return
        
        # Only update areas not covered by cloak
        inverse_mask = cv2.bitwise_not(mask)
        update_region = inverse_mask.astype(bool)
        
        # Exponential moving average update
        learning_rate = self.config.background_learning_rate
        
        self.adaptive_background[update_region] = (
            (1 - learning_rate) * self.adaptive_background[update_region] + 
            learning_rate * current_frame[update_region]
        ).astype(np.uint8)


# =============================================================================
#                           UI MANAGER
# =============================================================================

class UIManager:
    """Manages user interface elements."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.show_mask_overlay = False
        
    def draw_control_panel(self, frame: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
        """Draw control panel overlay on frame."""
        panel_height = 180
        panel_width = 350
        
        # Create semi-transparent overlay
        overlay = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        
        # Display statistics
        display_text = [
            f"FPS: {stats.get('fps', 0):.1f}",
            f"Mask: {stats.get('mask_ratio', 0)*100:.1f}%",
            f"Alpha: {stats.get('alpha', 1.0):.2f}",
            f"Mode: {'AUTO' if stats.get('auto_mode', False) else 'MAN'}",
            f"Quality: {stats.get('processing_mode', 'BAL')}",
            f"GPU: {'ON' if stats.get('gpu_enabled', False) else 'OFF'}",
        ]
        
        for i, text in enumerate(display_text):
            cv2.putText(overlay, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0), 2)
        
        # Blend overlay with frame
        frame[10:10+panel_height, 10:10+panel_width] = cv2.addWeighted(
            frame[10:10+panel_height, 10:10+panel_width], 0.3,
            overlay, 0.7, 0
        )
        
        return frame


# =============================================================================
#                           MAIN APPLICATION
# =============================================================================

class InvisibilityCloakSystem:
    """Main invisibility cloak application."""
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self._setup_logging()
        
        # Initialize system components
        self.hsv_manager = HSVColorManager(self.config)
        self.frame_processor = FrameProcessor(self.config)
        self.background_manager = BackgroundManager(self.config)
        self.ui_manager = UIManager(self.config)
        
        # Application state
        self.is_running = False
        self.alpha_blend = 1.0
        
        # Performance tracking
        self.frame_counter = 0
        self.start_time = time.perf_counter()
        self.fps_history = deque(maxlen=self.config.fps_window)
        self.mask_history = deque(maxlen=self.config.mask_history_size)
        
        # Color tracking
        self.last_valid_hsv = HSV_PRESETS['1']
    
    def _setup_logging(self):
        """Configure logging system."""
        logging.basicConfig(
            level=self.config.logging_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """Initialize video camera."""
        video_capture = cv2.VideoCapture(self.config.camera_index)
        
        if not video_capture.isOpened():
            logging.error(f"Cannot open camera at index {self.config.camera_index}")
            return None
        
        # Configure camera settings
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        logging.info("Camera initialized successfully")
        return video_capture
    
    def process_video_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame through the cloak pipeline."""
        # Preprocess frame
        processed_frame = self.frame_processor.preprocess_frame(frame)
        
        # Convert to HSV color space
        hsv_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        
        # Get current color range
        lower_hsv, upper_hsv = self.hsv_manager.get_hsv_range()
        
        # Create color mask
        color_mask = self.frame_processor.create_color_mask(hsv_frame, 
                                                           lower_hsv, 
                                                           upper_hsv)
        
        # Update mask history and stabilize
        self.mask_history.append(color_mask)
        stable_mask = self.frame_processor.stabilize_mask_temporally(self.mask_history)
        
        # Feather mask edges
        feathered_mask = self.frame_processor.feather_mask_edges(stable_mask)
        
        # Calculate mask coverage
        mask_coverage = np.count_nonzero(feathered_mask) / self.config.frame_area
        
        # Adaptive color range adjustment
        if mask_coverage < self.config.min_mask_ratio and self.last_valid_hsv:
            lower_hsv, upper_hsv = self.last_valid_hsv
            color_mask = self.frame_processor.create_color_mask(hsv_frame, 
                                                               lower_hsv, 
                                                               upper_hsv)
            self.mask_history[-1] = color_mask
        elif (self.config.min_mask_ratio < mask_coverage < self.config.max_mask_ratio):
            self.last_valid_hsv = (lower_hsv.copy(), upper_hsv.copy())
        
        # Update adaptive background
        self.background_manager.update_adaptive_background(processed_frame, 
                                                          feathered_mask)
        
        # Apply cloak effect
        if self.background_manager.adaptive_background is not None:
            # Extract cloak and visible regions
            inverse_mask = cv2.bitwise_not(feathered_mask)
            background = self.background_manager.adaptive_background
            
            # Combine background and foreground
            background_region = cv2.bitwise_and(background, background, 
                                                mask=feathered_mask)
            foreground_region = cv2.bitwise_and(processed_frame, processed_frame, 
                                                mask=inverse_mask)
            
            # Apply alpha blending
            if self.alpha_blend < 1.0:
                output_frame = cv2.addWeighted(background_region, self.alpha_blend,
                                              foreground_region, 1 - self.alpha_blend, 0)
            else:
                output_frame = cv2.add(background_region, foreground_region)
        else:
            output_frame = processed_frame
        
        # Add UI overlay
        output_frame = self.ui_manager.draw_control_panel(output_frame, {
            'fps': np.mean(self.fps_history) if self.fps_history else 0,
            'mask_ratio': mask_coverage,
            'alpha': self.alpha_blend,
            'auto_mode': self.hsv_manager.auto_mode,
            'processing_mode': self.frame_processor.processing_mode.value[:3].upper(),
            'gpu_enabled': False,  # Placeholder for GPU status
        })
        
        return output_frame
    
    def handle_user_input(self):
        """Process keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        # Number keys for color presets
        if ord('1') <= key <= ord('5'):
            self.hsv_manager.set_preset(chr(key))
            return
        
        # Control keys
        key_actions = {
            ord('q'): lambda: setattr(self, 'is_running', False),
            ord('a'): self._toggle_auto_mode,
            ord('r'): self._reset_parameters,
            ord('['): lambda: self._adjust_alpha(-0.05),
            ord(']'): lambda: self._adjust_alpha(0.05),
            ord('m'): self._toggle_mask_overlay,
            ord('s'): self._save_settings,
            ord('l'): self._load_settings,
            ord('f'): lambda: self._set_processing_mode(ProcessingMode.FAST),
            ord('b'): lambda: self._set_processing_mode(ProcessingMode.BALANCED),
            ord('q'): lambda: self._set_processing_mode(ProcessingMode.QUALITY),
        }
        
        if key in key_actions:
            key_actions[key]()
    
    def _toggle_auto_mode(self):
        self.hsv_manager.auto_mode = not self.hsv_manager.auto_mode
        mode = "ON" if self.hsv_manager.auto_mode else "OFF"
        logging.info(f"Auto mode: {mode}")
    
    def _reset_parameters(self):
        self.alpha_blend = 1.0
        self.hsv_manager.auto_mode = False
        self.mask_history.clear()
        self.hsv_manager.set_preset('1')
        logging.info("Parameters reset to defaults")
    
    def _adjust_alpha(self, delta: float):
        self.alpha_blend = max(0.0, min(1.0, self.alpha_blend + delta))
        logging.info(f"Alpha blend: {self.alpha_blend:.2f}")
    
    def _toggle_mask_overlay(self):
        self.ui_manager.show_mask_overlay = not self.ui_manager.show_mask_overlay
        state = "shown" if self.ui_manager.show_mask_overlay else "hidden"
        logging.info(f"Mask overlay: {state}")
    
    def _set_processing_mode(self, mode: ProcessingMode):
        self.frame_processor.processing_mode = mode
        self.frame_processor._initialize_kernels()
        logging.info(f"Processing mode: {mode.value}")
    
    def _save_settings(self):
        """Save current settings to file."""
        settings = {
            'alpha': self.alpha_blend,
            'auto_mode': self.hsv_manager.auto_mode,
            'current_preset': self.hsv_manager.current_preset,
            'show_mask': self.ui_manager.show_mask_overlay,
            'processing_mode': self.frame_processor.processing_mode.value,
        }
        
        try:
            with open('cloak_settings.json', 'w') as settings_file:
                json.dump(settings, settings_file, indent=2)
            logging.info("Settings saved successfully")
        except Exception as error:
            logging.error(f"Failed to save settings: {error}")
    
    def _load_settings(self):
        """Load settings from file."""
        try:
            with open('cloak_settings.json', 'r') as settings_file:
                settings = json.load(settings_file)
            
            self.alpha_blend = settings.get('alpha', 1.0)
            self.hsv_manager.auto_mode = settings.get('auto_mode', False)
            self.hsv_manager.set_preset(settings.get('current_preset', '1'))
            self.ui_manager.show_mask_overlay = settings.get('show_mask', False)
            
            mode_value = settings.get('processing_mode', 'balanced')
            self._set_processing_mode(ProcessingMode(mode_value))
            
            logging.info("Settings loaded successfully")
        except FileNotFoundError:
            logging.warning("Settings file not found, using defaults")
        except Exception as error:
            logging.error(f"Failed to load settings: {error}")
    
    def run(self):
        """Main application loop."""
        logging.info("Starting Invisibility Cloak System")
        
        # Initialize camera
        camera = self.initialize_camera()
        if camera is None:
            return
        
        # Capture initial background
        if not self.background_manager.capture_background(camera):
            camera.release()
            return
        
        # Create display window
        window_name = "Invisibility Cloak"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, *self.config.frame_size)
        
        self.is_running = True
        
        # Performance tracking
        frame_times = deque(maxlen=30)
        last_frame_time = time.perf_counter()
        
        try:
            while self.is_running:
                # Read frame from camera
                ret, frame = camera.read()
                if not ret:
                    time.sleep(0.005)
                    continue
                
                # Process frame
                with PerformanceTimer() as timer:
                    output_frame = self.process_video_frame(frame)
                self.frame_counter += 1
                
                # Update FPS calculation
                current_time = time.perf_counter()
                frame_duration = current_time - last_frame_time
                last_frame_time = current_time
                
                frame_times.append(frame_duration)
                fps = 1.0 / np.mean(frame_times) if frame_times else 0
                self.fps_history.append(fps)
                
                # Display output
                cv2.imshow(window_name, output_frame)
                
                # Handle user input
                self.handle_user_input()
                
                # Check for window close
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
        except KeyboardInterrupt:
            logging.info("Application interrupted by user")
        except Exception as error:
            logging.error(f"Runtime error: {error}", exc_info=True)
        finally:
            self._shutdown(camera)
    
    def _shutdown(self, camera: cv2.VideoCapture):
        """Clean shutdown of application resources."""
        logging.info("Shutting down application...")
        
        # Release resources
        camera.release()
        cv2.destroyAllWindows()
        
        # Performance report
        total_time = time.perf_counter() - self.start_time
        if total_time > 0:
            average_fps = self.frame_counter / total_time
            
            logging.info("=" * 50)
            logging.info("PERFORMANCE SUMMARY")
            logging.info(f"Total frames processed: {self.frame_counter}")
            logging.info(f"Average FPS: {average_fps:.2f}")
            logging.info(f"Total runtime: {total_time:.2f} seconds")
            logging.info("=" * 50)


# =============================================================================
#                           ENTRY POINT
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Invisibility Cloak System - Real-time background replacement'
    )
    parser.add_argument('--profile', action='store_true',
                       help='Enable performance profiling')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration (if available)')
    parser.add_argument('--fast', action='store_true',
                       help='Use fast processing mode')
    parser.add_argument('--width', type=int, default=960,
                       help='Frame width')
    parser.add_argument('--height', type=int, default=540,
                       help='Frame height')
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Create configuration
    config = SystemConfig(
        gpu_acceleration=args.gpu,
        enable_multithreading=True,
        adaptive_background=True,
        frame_width=args.width,
        frame_height=args.height,
    )
    
    # Optional profiling
    if args.profile:
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            cloak_system = InvisibilityCloakSystem(config)
            cloak_system.run()
        finally:
            profiler.disable()
            
            # Save profiling data
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats(20)
            stats.dump_stats('performance_profile.prof')
    else:
        # Run normally
        cloak_system = InvisibilityCloakSystem(config)
        cloak_system.run()


if __name__ == "__main__":
    main()
