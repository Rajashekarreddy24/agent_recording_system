import asyncio
import boto3
import cv2
import json
import logging
import mss
import mss.tools
import numpy as np
import os
import platform
import pyautogui
import shutil
import subprocess
import tempfile
import threading
import time
from botocore.exceptions import ClientError
from ctypes import windll, wintypes
from datetime import datetime, time
from dataclasses import dataclass
from django.core.cache import cache
from pathlib import Path
from PIL import Image, ImageGrab
from queue import Queue
from typing import Dict, List, Optional, Tuple
from enum import Enum
import aiofiles
import queue
import ctypes

# Windows-specific imports
if platform.system() == "Windows":
    import win32api
    import win32con
    import win32gui
    import win32security
    import win32ts
    import win32ui
    import pygetwindow as gw

@dataclass
class RecordingMetadata:
    """Metadata for recording session"""
    ticket_id: str
    agent_name: str
    recording_id: str
    start_time: datetime
    resolution: Tuple[int, int]
    fps: int
    custom_metadata: Dict = None

class EnhancedActionRecorder:
    """Enhanced action recorder with more detailed action capture"""

    def __init__(self, ticket_id=None):
        self.ticket_id = ticket_id
        self.record_thread = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        # Directory for recordings
        if not os.path.exists('recordings'):
            os.makedirs('recordings')

    def start_recording(self):
        """Start recording user actions"""
        if self.record_thread and self.record_thread.is_alive():
            self.logger.warning("Recording already in progress, cannot start a new session.")
            return  # Prevent starting a new thread if recording is ongoing
        
        self.stop_event.clear()  # Clear the stop signal
        self.record_thread = threading.Thread(target=self._record_screen)
        self.record_thread.start()
        cache.set(f'is_recording_{self.ticket_id}', True)  # Update cache state
        self.logger.info(f"Recording started for ticket ID: {self.ticket_id}")

    def stop_recording(self):
        """Stop recording user actions"""
        if self.record_thread and self.record_thread.is_alive():
            self.stop_event.set()  # Signal to stop the recording thread
            self.record_thread.join()  # Wait for the thread to finish
            cache.set(f'is_recording_{self.ticket_id}', False)  # Update cache state
            self.logger.info(f"Recording stopped for ticket ID: {self.ticket_id}")
        else:
            self.logger.warning("No recording in progress to stop.")

    def _record_screen(self):
        """Record screen in real-time"""
        try:
            screen_size = pyautogui.size()
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"recordings/{self.ticket_id}_{timestamp}.mp4"
            out = cv2.VideoWriter(file_name, fourcc, 20.0, (screen_size.width, screen_size.height))

            while not self.stop_event.is_set():
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to correct color format
                out.write(frame)
                self.logger.debug("Recording frame captured.")
                time.sleep(0.05)  # Adjust as needed

            out.release()  # Stop recording and release file
            self.logger.info(f"Recording saved successfully as {file_name}")

        except Exception as e:
            self.logger.error(f"Error during screen recording: {str(e)}")

class EnhancedVideoRecorder:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.state_validator = RecordingStateValidator()
        
        # Initialize directories
        self.output_dir = Path(config.get('output_dir', 'recordings'))
        self.temp_dir = Path(config.get('temp_dir', 'temp_recordings'))
        self.backup_dir = Path(config.get('backup_dir', 'backup_recordings'))
        
        for directory in [self.output_dir, self.temp_dir, self.backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize state variables
        self.recording = False
        self.current_recording = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.frame_queue = Queue(maxsize=30)
        self.writer = None

    def _aenter_(self):
        """Async context manager entry"""
        return self

    async def _aexit_(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        if exc_type:
            self.logger.error(f"Error in context: {exc_val}")
            return False
        return True

    async def start_recording(self, ticket_id: str, agent_name: str, **kwargs) -> Dict:
        """Start recording with async support"""
        try:
            if self.recording:
                return {
                    'status': 'already_recording',
                    'recording_id': self.current_recording.get('recording_id')
                }

            # Generate recording ID
            recording_id = f"{agent_name}{ticket_id}{int(datetime.now().timestamp())}"
            output_path = self.output_dir / f"{recording_id}.mp4"

            # Initialize video writer
            self.writer = await self._initialize_writer(output_path, **kwargs)
            if not self.writer:
                raise RuntimeError("Failed to initialize video writer")

            # Set up recording information
            self.current_recording = {
                'recording_id': recording_id,
                'ticket_id': ticket_id,
                'agent_name': agent_name,
                'output_path': str(output_path),
                'start_time': datetime.now(),
                'metadata': kwargs
            }

            # Start recording
            self.recording = True
            
            # Start recording task
            asyncio.create_task(self._recording_task())

            self.logger.info(f"Started recording: {recording_id}")
            return {
                'status': 'started',
                'recording_id': recording_id,
                'output_path': str(output_path)
            }

        except Exception as e:
            self.logger.error(f"Failed to start recording: {str(e)}")
            await self._handle_error(e)
            raise

    async def stop_recording(self) -> Dict:
        """Stop recording with async support"""
        if not self.recording:
            return {'status': 'not_recording'}

        try:
            # Signal recording to stop
            self.recording = False
            self._stop_event.set()

            # Wait for recording task to complete
            await asyncio.sleep(0.5)  # Give time for last frames

            # Release writer
            if self.writer:
                self.writer.release()
                self.writer = None

            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - self.current_recording['start_time']).total_seconds()

            result = {
                'status': 'completed',
                'recording_id': self.current_recording['recording_id'],
                'output_path': self.current_recording['output_path'],
                'duration': duration,
                'metadata': self.current_recording['metadata']
            }

            self.logger.info(f"Stopped recording: {result['recording_id']}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to stop recording: {str(e)}")
            await self._handle_error(e)
            raise

    async def _initialize_writer(self, output_path: Path, **kwargs) -> cv2.VideoWriter:
        """Initialize video writer with given settings"""
        try:
            fps = kwargs.get('fps', 30)
            resolution = kwargs.get('resolution', (1920, 1080))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                resolution
            )
            
            if not writer.isOpened():
                raise RuntimeError("Failed to open video writer")
                
            return writer
            
        except Exception as e:
            self.logger.error(f"Writer initialization failed: {str(e)}")
            raise

    async def _recording_task(self):
        """Async recording task"""
        try:
            while self.recording and not self._stop_event.is_set():
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    self.writer.write(frame)
                else:
                    await asyncio.sleep(0.001)  # Small delay to prevent CPU overuse
                    
        except Exception as e:
            self.logger.error(f"Recording task error: {str(e)}")
            await self._handle_error(e)

    async def _handle_error(self, error: Exception):
        """Handle errors during recording"""
        try:
            self.logger.error(f"Recording error occurred: {str(error)}")
            
            # Save error information
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'error_type': type(error)._name_,
                'error_message': str(error),
                'recording_id': self.current_recording.get('recording_id') if self.current_recording else None
            }
            
            # Save error log
            error_log_path = self.output_dir / 'error_logs'
            error_log_path.mkdir(exist_ok=True)
            error_file = error_log_path / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
            
            # Cleanup
            await self.cleanup()
            
        except Exception as e:
            self.logger.error(f"Error handling failed: {str(e)}")

    async def cleanup(self):
        """Cleanup resources"""
        try:
            self._stop_event.set()
            self.recording = False
            
            if self.writer:
                self.writer.release()
                self.writer = None
            
            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except:
                    break
            
            self.current_recording = None
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    def add_frame(self, frame: np.ndarray):
        """Add frame to processing queue"""
        if self.recording and not self.frame_queue.full():
            self.frame_queue.put(frame)
# Usage example
async def main():
    config = {
        'output_dir': 'recordings',
        'temp_dir': 'temp_recordings',
        'backup_dir': 'backup_recordings'
    }
    
    recorder = EnhancedVideoRecorder(config)
    
    try:
        # Start recording
        result = await recorder.start_recording(
            ticket_id="TICKET123",
            agent_name="agent_smith",
            fps=30,
            resolution=(1920, 1080)
        )
        print(f"Recording started: {result}")
        
        # Simulate frame capture
        for _ in range(300):  # 10 seconds at 30 fps
            # Your frame capture code here
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Example frame
            recorder.add_frame(frame)
            await asyncio.sleep(1/30)  # Simulate frame rate
        
        # Stop recording
        stop_result = await recorder.stop_recording()
        print(f"Recording completed: {stop_result}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await recorder.cleanup()

class ScreenRecorder:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.recording = False
        self.paused = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.current_recording = None
        self.recording_thread = None
        self.monitor = self._get_primary_monitor()

    def _get_primary_monitor(self) -> Dict[str, int]:
        """Get primary monitor dimensions using Win32"""
        try:
            monitor = {
                'left': 0,
                'top': 0,
                'width': win32api.GetSystemMetrics(win32con.SM_CXSCREEN),
                'height': win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
            }
            self.logger.info(f"Monitor dimensions: {monitor}")
            return monitor
        except Exception as e:
            self.logger.error(f"Failed to get monitor dimensions: {e}")
            # Fallback to standard resolution
            return {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}

    def _capture_screen_win32(self) -> Optional[np.ndarray]:
        """Capture screen using Win32 API"""
        try:
            # Get handles for the desktop window
            hwnd = win32gui.GetDesktopWindow()
            width = self.monitor['width']
            height = self.monitor['height']

            # Create device contexts and bitmap
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap and select it into DC
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
            saveDC.SelectObject(saveBitMap)

            # Copy screen content
            result = saveDC.BitBlt(
                (0, 0), (width, height),
                mfcDC, (0, 0),
                win32con.SRCCOPY
            )

            # Convert bitmap to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype='uint8')
            img.shape = (height, width, 4)

            # Cleanup
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

            # Convert from BGRA to BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        except Exception as e:
            self.logger.error(f"Win32 screen capture failed: {e}")
            return self._capture_screen_fallback()

    def _capture_screen_fallback(self) -> Optional[np.ndarray]:
        """Fallback screen capture using PyAutoGUI"""
        try:
            screenshot = pyautogui.screenshot()
            frame = np.array(screenshot)
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"Fallback screen capture failed: {e}")
            return None

    def start_recording(self, output_name: Optional[str] = None) -> str:
        """Start recording the screen"""
        if self.recording:
            self.logger.warning("Recording already in progress")
            return self.current_recording

        try:
            if not output_name:
                output_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            output_path = self.output_dir / f"{output_name}.mp4"
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Test capture first
            test_frame = self._capture_screen_win32()
            if test_frame is None:
                raise Exception("Failed to capture test frame")

            # Initialize video writer
            self.writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.config.get('fps', 30),
                (self.monitor['width'], self.monitor['height'])
            )

            if not self.writer.isOpened():
                raise Exception("Failed to initialize video writer")

            self.recording = True
            self.current_recording = str(output_path)
            
            # Start recording thread
            self.recording_thread = threading.Thread(
                target=self._record_screen,
                daemon=True
            )
            self.recording_thread.start()

            self.logger.info(f"Started recording: {output_path}")
            return self.current_recording

        except Exception as e:
            self.logger.error(f"Error starting recording: {str(e)}")
            self.recording = False
            self.current_recording = None
            raise

    def stop_recording(self) -> Optional[str]:
        """Stop the current recording"""
        if not self.recording:
            return None

        try:
            self.recording = False
            if self.recording_thread:
                self.recording_thread.join(timeout=5.0)

            if hasattr(self, 'writer'):
                self.writer.release()

            recorded_path = self.current_recording
            self.current_recording = None

            self.logger.info("Recording stopped successfully")
            return recorded_path

        except Exception as e:
            self.logger.error(f"Error stopping recording: {str(e)}")
            raise

    def _record_screen(self):
        """Main recording loop"""
        last_frame_time = time.time()
        frame_interval = 1.0 / self.config.get('fps', 30)
        
        while self.recording:
            try:
                current_time = time.time()
                
                # Maintain consistent frame rate
                if current_time - last_frame_time < frame_interval:
                    time.sleep(frame_interval - (current_time - last_frame_time))
                    
                if not self.paused:
                    frame = self._capture_screen_win32()
                    
                    if frame is not None:
                        self.writer.write(frame)
                        
                        # Add to frame queue for preview
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                            
                last_frame_time = time.time()
                
            except Exception as e:
                self.logger.error(f"Recording error: {str(e)}")
                break

    def take_screenshot(self, output_name: Optional[str] = None) -> str:
        """Take a single screenshot"""
        try:
            if not output_name:
                output_name = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            output_path = self.output_dir / f"{output_name}.png"
            
            frame = self._capture_screen_win32()
            if frame is not None:
                cv2.imwrite(str(output_path), frame)
                self.logger.info(f"Screenshot saved: {output_path}")
                return str(output_path)
                
        except Exception as e:
            self.logger.error(f"Screenshot failed: {str(e)}")
            raise

    def pause_recording(self):
        """Pause the current recording"""
        self.paused = True
        self.logger.info("Recording paused")

    def resume_recording(self):
        """Resume the current recording"""
        self.paused = False
        self.logger.info("Recording resumed")

    def get_preview_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame for preview"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def _del_(self):
        """Cleanup resources"""
        self.stop_recording()

if __name__ == "_main_":
    asyncio.run(main())
    asyncio.run(main())

class MSSScreenRecorder:
    def __init__(self, output_dir: str = "recordings"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recording = False
        
        # Initialize MSS
        self.sct = mss.mss()
        
        # Get the primary monitor
        self.monitor = self.sct.monitors[0]  # Primary monitor
        self.width = self.monitor["width"]
        self.height = self.monitor["height"]
        
        self.logger.info(f"Initialized recorder with resolution: {self.width}x{self.height}")

    def capture_frame(self):
        """Capture a single frame using MSS"""
        try:
            # Capture the screen
            screenshot = self.sct.grab(self.monitor)
            
            # Convert to numpy array
            frame = np.array(screenshot)
            
            # Convert from BGRA to BGR
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None

    def record(self, duration: int = 10, fps: int = 10, output_name: str = None):
        """Record screen for specified duration"""
        try:
            if not output_name:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_name = f"recording_{timestamp}.avi"
            
            output_path = self.output_dir / output_name
            self.logger.info(f"Starting recording to {output_path}")
            
            # Test capture first
            test_frame = self.capture_frame()
            if test_frame is None:
                raise Exception("Failed to capture test frame")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (self.width, self.height)
            )
            
            if not out.isOpened():
                raise Exception("Failed to create video writer")

            self.recording = True
            start_time = time.time()
            frames_captured = 0
            frame_interval = 1.0 / fps
            
            self.logger.info("Recording started")
            
            while self.recording and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                frame = self.capture_frame()
                if frame is not None:
                    out.write(frame)
                    frames_captured += 1
                    
                    # Calculate sleep time to maintain FPS
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
                self.logger.debug(f"Captured frame {frames_captured}")
            
            out.release()
            self.logger.info(f"Recording completed. Captured {frames_captured} frames")
            
            # Convert to MP4 for better compatibility
            mp4_path = str(output_path).replace('.avi', '.mp4')
            self.convert_to_mp4(str(output_path), mp4_path)
            
            # Remove the AVI file
            os.remove(str(output_path))
            
            return mp4_path
            
        except Exception as e:
            self.logger.error(f"Recording failed: {e}")
            if 'out' in locals():
                out.release()
            raise
        finally:
            self.recording = False

    def convert_to_mp4(self, input_path: str, output_path: str):
        """Convert AVI to MP4 using OpenCV"""
        try:
            # Read the AVI file
            cap = cv2.VideoCapture(input_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create MP4 writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )
            
            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            # Clean up
            cap.release()
            out.release()
            
            self.logger.info(f"Converted to MP4: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            raise

    def stop(self):
        """Stop recording"""
        self.recording = False
        
    def _del_(self):
        """Clean up MSS instance"""
        try:
            self.sct.close()
        except:
            pass

def test_recording():
    """Test the screen recorder"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create recorder
        recorder = MSSScreenRecorder("recordings")
        
        # Record for 5 seconds
        logger.info("Starting test recording...")
        output_path = recorder.record(duration=5, fps=10)
        
        # Verify file exists and has size
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            logger.info(f"Recording successful. File size: {size/1024/1024:.2f} MB")
            return True
        else:
            logger.error("Recording file not found")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "_main_":
    success = test_recording()
    print(f"Test {'succeeded' if success else 'failed'}")

# Windows Terminal Services API definitions
class WTSINFOEX_LEVEL1(ctypes.Structure):
    fields = [
        ("SessionId", ctypes.c_ulong),
        ("SessionState", ctypes.c_ulong),
        ("SessionFlags", ctypes.c_ulong),
    ]

class WTSINFOEX(ctypes.Structure):
    fields = [
        ("Level", ctypes.c_ulong),
        ("Data", WTSINFOEX_LEVEL1),
    ]


class RDSScreenRecorder:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.recording = False
        self.paused = False
        self.frame_queue = queue.Queue(maxsize=30)
        self.current_recording = None
        self.recording_thread = None
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Set DPI awareness
        try:
            windll.user32.SetProcessDPIAware()
        except Exception as e:
            self.logger.warning(f"Failed to set DPI awareness: {e}")
            
        # Initialize screen dimensions
        self.monitor = self._get_screen_dimensions()

    def _get_screen_dimensions(self) -> Dict[str, int]:
        """Get screen dimensions using Windows API"""
        try:
            width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
            height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
            left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
            top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
            
            return {
                'left': left,
                'top': top,
                'width': width,
                'height': height
            }
        except Exception as e:
            self.logger.error(f"Failed to get screen dimensions: {e}")
            # Fallback to default resolution
            return {
                'left': 0,
                'top': 0,
                'width': 1920,
                'height': 1080
            }

    def _create_memory_dc(self):
        """Create memory DC for screen capture"""
        try:
            # Get handles
            hwnd = win32gui.GetDesktopWindow()
            hwndDC = win32gui.GetWindowDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            
            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(
                mfcDC, 
                self.monitor['width'], 
                self.monitor['height']
            )
            saveDC.SelectObject(saveBitMap)
            
            return hwnd, hwndDC, mfcDC, saveDC, saveBitMap
        except Exception as e:
            self.logger.error(f"Failed to create memory DC: {e}")
            return None

    def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture screen with checks to ensure resources are properly managed."""
        hwnd = win32gui.GetDesktopWindow()
        hwndDC = win32gui.GetWindowDC(hwnd)
        if not hwndDC:
            self.logger.error("Failed to get hwndDC")
            return None
        
        try:
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, self.monitor['width'], self.monitor['height'])
            saveDC.SelectObject(saveBitMap)
            
            # Capture screen using BitBlt
            saveDC.BitBlt((0, 0), (self.monitor['width'], self.monitor['height']), mfcDC, (self.monitor['left'], self.monitor['top']), win32con.SRCCOPY)
            
            # Convert bitmap to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8)
            img.shape = (self.monitor['height'], self.monitor['width'], 4)
            
            # Convert BGRA to BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        except Exception as e:
            self.logger.error(f"Screen capture failed: {e}")
            return None
        finally:
            # Ensure DC cleanup
            if saveBitMap:
                win32gui.DeleteObject(saveBitMap.GetHandle())
            if saveDC:
                saveDC.DeleteDC()
            if mfcDC:
                mfcDC.DeleteDC()
            if hwndDC:
                win32gui.ReleaseDC(hwnd, hwndDC)


    def start_recording(self, output_name: Optional[str] = None) -> str:
        """Start recording the screen"""
        if self.recording:
            return self.current_recording

        try:
            if not output_name:
                output_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Setup paths
            temp_path = self.temp_dir / f"{output_name}_temp.avi"
            final_path = self.output_dir / f"{output_name}.mp4"

            # Test capture with multiple attempts
            test_frame = None
            for attempt in range(3):
                test_frame = self._capture_screen()
                if test_frame is not None:
                    break
                time.sleep(1)

            if test_frame is None:
                raise Exception("Failed to capture test frame after multiple attempts")

            # Initialize video writer with MJPG codec for better compatibility
            self.writer = cv2.VideoWriter(
                str(temp_path),
                cv2.VideoWriter_fourcc(*'mp4V'),
                self.config.get('fps', 10),
                (self.monitor['width'], self.monitor['height']),
                True
            )

            if not self.writer.isOpened():
                raise Exception("Failed to initialize video writer")

            self.recording = True
            self.current_recording = str(final_path)
            self.temp_recording = str(temp_path)

            # Start recording thread
            self.recording_thread = threading.Thread(
                target=self._record_screen,
                daemon=True
            )
            self.recording_thread.start()

            return self.current_recording

        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")
            self.recording = False
            self.current_recording = None
            raise

    def _record_screen(self):
        """Main recording loop"""
        last_frame_time = time.time()
        frame_interval = 1.0 / self.config.get('fps', 10)
        failed_captures = 0
        max_failed_captures = 5

        while self.recording:
            try:
                current_time = time.time()
                
                if current_time - last_frame_time < frame_interval:
                    time.sleep(frame_interval - (current_time - last_frame_time))

                if not self.paused:
                    frame = self._capture_screen()
                    
                    if frame is not None:
                        self.writer.write(frame)
                        failed_captures = 0
                        
                        if not self.frame_queue.full():
                            self.frame_queue.put(frame)
                    else:
                        failed_captures += 1
                        if failed_captures >= max_failed_captures:
                            self.logger.error("Too many failed captures, stopping recording")
                            break

                last_frame_time = time.time()

            except Exception as e:
                self.logger.error(f"Recording error: {e}")
                break

    def stop_recording(self) -> Optional[str]:
        """Stop recording and convert to final format"""
        if not self.recording:
            return None

        try:
            self.recording = False
            if self.recording_thread:
                self.recording_thread.join(timeout=5.0)

            if hasattr(self, 'writer'):
                self.writer.release()

            # Convert to MP4 with H.264 codec
            if os.path.exists(self.temp_recording):
                cap = cv2.VideoCapture(self.temp_recording)
                
                # Initialize MP4 writer with H.264 codec
                out = cv2.VideoWriter(
                    self.current_recording,
                    cv2.VideoWriter_fourcc(*'H264'),
                    self.config.get('fps', 10),
                    (self.monitor['width'], self.monitor['height']),
                    True
                )

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)

                cap.release()
                out.release()

                # Cleanup temporary file
                os.remove(self.temp_recording)

            return self.current_recording

        except Exception as e:
            self.logger.error(f"Failed to stop recording: {e}")
            raise

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'writer'):
                self.writer.release()
            
            if os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def _del_(self):
        """Destructor"""
        self.cleanup()

class RecordingState(Enum):
    """Enum defining all possible states of the recording system"""
    
    # Define each state with a string value
    IDLE = 'idle'
    INITIALIZING = 'initializing'
    RECORDING = 'recording'
    PAUSED = 'paused'
    STOPPING = 'stopping'
    ERROR = 'error'
    RECOVERING = 'recovering'
    CLEANUP = 'cleanup'
    FAILED = 'failed'

    def __str__(self) -> str:
        """Return the string value of the state"""
        return self.value

    @classmethod
    def from_string(cls, state_str: str) -> 'RecordingState':
        """Create RecordingState from string"""
        try:
            return cls(state_str.lower())
        except ValueError:
            raise ValueError(f"Invalid state string: {state_str}")

class RecordingStateValidator:
    """Validates state transitions and manages state history"""
    
    # Define valid state transitions
    VALID_TRANSITIONS = {
        RecordingState.IDLE: {
            RecordingState.INITIALIZING,
            RecordingState.ERROR
        },
        RecordingState.INITIALIZING: {
            RecordingState.RECORDING,
            RecordingState.ERROR,
            RecordingState.FAILED
        },
        RecordingState.RECORDING: {
            RecordingState.PAUSED,
            RecordingState.STOPPING,
            RecordingState.ERROR
        },
        RecordingState.PAUSED: {
            RecordingState.RECORDING,
            RecordingState.STOPPING,
            RecordingState.ERROR
        },
        RecordingState.STOPPING: {
            RecordingState.CLEANUP,
            RecordingState.ERROR,
            RecordingState.FAILED
        },
        RecordingState.ERROR: {
            RecordingState.RECOVERING,
            RecordingState.CLEANUP,
            RecordingState.FAILED
        },
        RecordingState.RECOVERING: {
            RecordingState.RECORDING,
            RecordingState.ERROR,
            RecordingState.FAILED,
            RecordingState.CLEANUP
        },
        RecordingState.CLEANUP: {
            RecordingState.IDLE,
            RecordingState.ERROR,
            RecordingState.FAILED
        },
        RecordingState.FAILED: {
            RecordingState.CLEANUP,
            RecordingState.IDLE
        }
    }

    def __init__(self):
        self.current_state = RecordingState.IDLE
        self.state_history: List[Tuple[RecordingState, datetime]] = []
        self.logger = logging.getLogger(__name__)

    def can_transition(self, from_state: RecordingState, to_state: RecordingState) -> bool:
        """Check if state transition is valid"""
        return to_state in self.VALID_TRANSITIONS.get(from_state, set())

    def transition(self, to_state: RecordingState) -> bool:
        """Attempt to transition to new state"""
        if self.can_transition(self.current_state, to_state):
            self.logger.info(f"State transition: {self.current_state} -> {to_state}")
            self.state_history.append((self.current_state, datetime.now()))
            self.current_state = to_state
            return True
        else:
            self.logger.warning(
                f"Invalid state transition: {self.current_state} -> {to_state}"
            )
            return False

    def get_state_history(self) -> List[Tuple[RecordingState, datetime]]:
        """Get history of state transitions"""
        return self.state_history.copy()

    def get_time_in_state(self, state: RecordingState) -> float:
        """Get total time spent in a specific state"""
        total_time = 0.0
        state_times = [(s, t) for s, t in self.state_history if s == state]
        
        for i in range(len(state_times) - 1):
            total_time += (state_times[i+1][1] - state_times[i][1]).total_seconds()
            
        # Add time for current state if it matches
        if self.current_state == state and state_times:
            total_time += (datetime.now() - state_times[-1][1]).total_seconds()
            
        return total_time

    def reset(self):
        """Reset state to IDLE"""
        self.state_history.append((self.current_state, datetime.now()))
        self.current_state = RecordingState.IDLE

class StateContext:
    """Context manager for state transitions"""
    
    def __init__(self, validator: RecordingStateValidator, state: RecordingState):
        self.validator = validator
        self.new_state = state
        self.previous_state = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Enter new state"""
        self.previous_state = self.validator.current_state
        if not self.validator.transition(self.new_state):
            raise ValueError(
                f"Invalid state transition: {self.previous_state} -> {self.new_state}"
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit state and handle errors"""
        if exc_type is not None:
            # Error occurred, transition to ERROR state
            self.logger.error(f"Error in state {self.new_state}: {exc_val}")
            self.validator.transition(RecordingState.ERROR)
        elif self.new_state != RecordingState.FAILED:
            # Successful execution, return to previous state
            self.validator.transition(self.previous_state)

# Usage example:
async def example_usage():
    validator = RecordingStateValidator()
    
    try:
        # Start recording process
        async with StateContext(validator, RecordingState.INITIALIZING):
            # Initialization code here
            await asyncio.sleep(1)  # Simulating initialization
        
        async with StateContext(validator, RecordingState.RECORDING):
            # Recording code here
            await asyncio.sleep(5)  # Simulating recording
            
        async with StateContext(validator, RecordingState.STOPPING):
            # Cleanup code here
            await asyncio.sleep(1)  # Simulating cleanup
            
    except Exception as e:
        print(f"Error occurred: {e}")
        # Handle error appropriately
    
    finally:
        print(f"Final state: {validator.current_state}")
        print("State history:")
        for state, timestamp in validator.get_state_history():
            print(f"{timestamp}: {state}")

class SupportTicketResolutionAgent:
    def __init__(self, screen_recorder, db_manager):
        self.screen_recorder = screen_recorder
        self.db_manager = db_manager
        self.recording_steps = []

    def record_resolution_steps(self, ticket_id):
        """
        Record the steps taken by the technician to resolve a support ticket.
        
        Args:
            ticket_id (str): Unique identifier for the support ticket.
        """
        print(f"== Recording Resolution Steps for Ticket: {ticket_id} ==")

        # Start recording the screen
        video_filepath = self.screen_recorder.start_recording(ticket_id)
        print(f"Recording started. File saved at: {video_filepath}")

        self.recording_steps = []

        try:
            while True:
                # Capture the current screen
                screenshot = np.array(pyautogui.screenshot())

                # Record the current mouse position and any keyboard/mouse events
                mouse_x, mouse_y = pyautogui.position()
                mouse_button = pyautogui.mouseDown() or pyautogui.mouseUp()
                keyboard_key = pyautogui.keyDown() or pyautogui.keyUp()

                self.recording_steps.append({
                    'screenshot': screenshot,
                    'mouse_x': mouse_x,
                    'mouse_y': mouse_y,
                    'mouse_button': mouse_button,
                    'keyboard_key': keyboard_key,
                    'timestamp': datetime.now()
                })

                time.sleep(0.1)  # Capture a new frame every 0.1 seconds

        except KeyboardInterrupt:
            # Stop recording when the user presses Ctrl+C
            recording_data = self.screen_recorder.stop_recording()
            self.db_manager.save_recording_metadata(recording_data)
            print("Recording stopped. Resolution steps saved.")

    def replay_resolution_steps(self, ticket_id):
        """
        Replay the recorded steps to resolve a support ticket.
        
        Args:
            ticket_id (str): Unique identifier for the support ticket.
        """
        print(f"== Replaying Resolution Steps for Ticket: {ticket_id} ==")

        # Retrieve the recorded steps from the database
        self.recording_steps = self.db_manager.get_recorded_steps(ticket_id)

        for step in self.recording_steps:
            # Display the screenshot
            cv2.imshow('Resolution Replay', step['screenshot'])
            cv2.waitKey(1)

            # Replay the mouse and keyboard actions
            pyautogui.moveTo(step['mouse_x'], step['mouse_y'])
            if step['mouse_button']:
                pyautogui.mouseDown() if step['mouse_button'] == 'down' else pyautogui.mouseUp()
            if step['keyboard_key']:
                pyautogui.keyDown(step['keyboard_key']) if step['keyboard_key'] == 'down' else pyautogui.keyUp(step['keyboard_key'])

            time.sleep((step['timestamp'] - self.recording_steps[0]['timestamp']).total_seconds())

        cv2.destroyAllWindows()
        print("Resolution steps replay completed.")

class SimpleScreenRecorder:
    def __init__(self, output_dir: str = "recordings"):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recording = False
        
        # Disable pyautogui safety features
        pyautogui.FAILSAFE = False
        
        # Get screen size
        self.width, self.height = pyautogui.size()
        self.logger.info(f"Screen size: {self.width}x{self.height}")

    def capture_frame(self):
        """Capture a single frame"""
        try:
            # Take screenshot
            screenshot = pyautogui.screenshot()
            # Convert to numpy array
            frame = np.array(screenshot)
            # Convert from RGB to BGR
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None

    def record(self, duration: int = 10, fps: int = 10, output_name: str = None):
        """Record screen for specified duration"""
        try:
            if not output_name:
                output_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
            
            output_path = self.output_dir / output_name
            
            # Test capture first
            test_frame = self.capture_frame()
            if test_frame is None:
                raise Exception("Failed to capture test frame")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (self.width, self.height)
            )
            
            if not out.isOpened():
                raise Exception("Failed to create video writer")

            self.recording = True
            start_time = time.time()
            frames_captured = 0
            
            self.logger.info(f"Starting recording to {output_path}")
            
            while self.recording and (time.time() - start_time) < duration:
                frame = self.capture_frame()
                if frame is not None:
                    out.write(frame)
                    frames_captured += 1
                time.sleep(1/fps)  # Control frame rate
            
            out.release()
            self.logger.info(f"Recording completed. Captured {frames_captured} frames")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Recording failed: {e}")
            if 'out' in locals():
                out.release()
            raise

    def stop(self):
        """Stop recording"""
        self.recording = False

def test_recording():
    """Test the screen recorder"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create recorder
        recorder = SimpleScreenRecorder("recordings")
        
        # Record for 5 seconds
        logger.info("Starting test recording...")
        output_path = recorder.record(duration=5, fps=10)
        
        # Verify file exists and has size
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            logger.info(f"Recording successful. File size: {size/1024/1024:.2f} MB")
            return True
        else:
            logger.error("Recording file not found")
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "_main_":
    success = test_recording()
    print(f"Test {'succeeded' if success else 'failed'}")


