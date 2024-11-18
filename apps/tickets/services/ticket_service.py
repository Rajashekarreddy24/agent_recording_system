from typing import Dict, Optional, List
import logging
from pathlib import Path
from datetime import datetime
import asyncio

from typing import Optional, Dict, Any
import requests
import os
from pathlib import Path
import logging
import json
import hashlib
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from datetime import datetime

import requests
import logging
from django.conf import settings
from typing import Dict
import cv2
import numpy as np
import pytesseract
import pyautogui
import tensorflow as tf
from pathlib import Path
import pm4py
import logging
from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass
from keras import models, layers
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import torch
import torch.nn as nn
from PIL import Image
import os
import tempfile
import pandas as pd
import torch.nn.functional as F
from typing import List, Callable
import cv2
from cv2 import legacy

legacy = cv2.legacy if hasattr(cv2, 'legacy') else None
# tracker = legacy.TrackerCSRT_create()
# tracker = cv2.TrackerCSRT_create()
tracker = cv2.TrackerMIL_create()  # or use cv2.legacy.TrackerMIL()


@dataclass
class ProcessSequence:
    action_sequence: List[str]
    screen_states: List[Dict]
    interaction_points: List[Tuple[int, int]]
    extracted_text: List[str]
    confidence: float

@dataclass
class ProcessedFrame:
    frame_number: int
    timestamp: float
    text_content: str
    mouse_position: Tuple[int, int]
    click_detected: bool
    key_pressed: Optional[str]
    screen_changes: Dict[str, any]

class VideoUploadManager:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.upload_base_url = config['api_base_url']
        
        # Initialize S3 client if using S3
        if config.get('use_s3', False):
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config['aws_access_key'],
                aws_secret_access_key=config['aws_secret_key'],
                region_name=config['aws_region']
            )
        else:
            self.s3_client = None
            
        # For tracking upload progress
        self._upload_progress = {}
        self._upload_lock = threading.Lock()

    def upload_video(self, 
                    video_path: str, 
                    ticket_id: str,
                    metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Upload video file to central system"""
        try:
            file_path = Path(video_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)
            
            # Prepare metadata
            upload_metadata = {
                'ticket_id': ticket_id,
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_hash': file_hash,
                'upload_date': datetime.utcnow().isoformat(),
                'technician_metadata': metadata
            }

            # Get upload URL and token
            upload_info = self._get_upload_url(upload_metadata)

            if self.config.get('use_s3', False):
                # Upload to S3
                return self._upload_to_s3(
                    file_path,
                    upload_info['bucket'],
                    upload_info['key'],
                    upload_metadata
                )
            else:
                # Upload directly to API
                return self._upload_to_api(
                    file_path,
                    upload_info['url'],
                    upload_info['token'],
                    upload_metadata
                )

        except Exception as e:
            self.logger.error(f"Upload failed: {str(e)}")
            raise

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_upload_url(self, metadata: Dict) -> Dict[str, str]:
        """Get pre-signed upload URL from API"""
        try:
            response = requests.post(
                f"{self.upload_base_url}/get-upload-url",
                json=metadata,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get upload URL: {str(e)}")
            raise

    def _upload_to_s3(self, 
                     file_path: Path,
                     bucket: str,
                     key: str,
                     metadata: Dict) -> Dict[str, Any]:
        """Upload file to S3 with progress tracking"""
        try:
            # Initialize progress tracking
            upload_id = str(time.time())
            self._upload_progress[upload_id] = {
                'total': file_path.stat().st_size,
                'uploaded': 0,
                'status': 'uploading'
            }

            # Create progress callback
            def progress_callback(bytes_transferred):
                with self._upload_lock:
                    self._upload_progress[upload_id]['uploaded'] += bytes_transferred

            # Upload file
            self.s3_client.upload_file(
                str(file_path),
                bucket,
                key,
                Callback=progress_callback,
                ExtraArgs={'Metadata': {
                    'ticket_id': metadata['ticket_id'],
                    'upload_date': metadata['upload_date']
                }}
            )

            # Update progress
            with self._upload_lock:
                self._upload_progress[upload_id]['status'] = 'completed'

            # Notify API about successful upload
            self._notify_upload_complete({
                'bucket': bucket,
                'key': key,
                'metadata': metadata
            })

            return {
                'status': 'success',
                'location': f"s3://{bucket}/{key}",
                'metadata': metadata
            }

        except Exception as e:
            with self._upload_lock:
                self._upload_progress[upload_id]['status'] = 'failed'
            raise

    def _upload_to_api(self, 
                      file_path: Path,
                      url: str,
                      token: str,
                      metadata: Dict) -> Dict[str, Any]:
        """Upload file directly to API"""
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'video/mp4')}
                headers = {
                    'Authorization': f'Bearer {token}',
                    'X-Upload-Metadata': json.dumps(metadata)
                }
                
                response = requests.post(
                    url,
                    files=files,
                    headers=headers
                )
                response.raise_for_status()
                
                return response.json()

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API upload failed: {str(e)}")
            raise

    def _notify_upload_complete(self, upload_info: Dict):
        """Notify API about completed upload"""
        try:
            response = requests.post(
                f"{self.upload_base_url}/upload-complete",
                json=upload_info,
                headers=self._get_headers()
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to notify upload completion: {str(e)}")
            # Don't raise here, as upload was successful

    def get_upload_progress(self, upload_id: str) -> Dict[str, Any]:
        """Get progress of specific upload"""
        with self._upload_lock:
            return self._upload_progress.get(upload_id, {})

    def _get_headers(self) -> Dict[str, str]:
        """Get common API headers"""
        return {
            'Authorization': f'Bearer {self.config["api_token"]}',
            'Content-Type': 'application/json'
        }

    def cleanup_progress(self, upload_id: str):
        """Clean up progress tracking for completed upload"""
        with self._upload_lock:
            self._upload_progress.pop(upload_id, None)

class IntegratedTicketVideoSystem:
    def _init_(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize components
        self.video_processor = VideoProcessingEngine(config.get('processing_config', {}))
        self.video_uploader = VideoUploadManager(config.get('upload_config', {}))
        
        self.logger.info("Integrated ticket video system initialized")

    async def process_ticket_recording(self,
                                     ticket_id: str,
                                     video_path: str,
                                     metadata: Dict = None) -> Dict:
        """Complete process from recording to agent creation"""
        try:
            # 1. Process video for automation sequence
            self.logger.info(f"Processing video for ticket {ticket_id}")
            process_sequence = self.video_processor.process_video(video_path)
            
            # 2. Enhance metadata with processed information
            enhanced_metadata = self._enhance_metadata(metadata, process_sequence)
            
            # 3. Upload video with enhanced metadata
            upload_result = await self.video_uploader.upload_video_for_ticket(
                ticket_id=ticket_id,
                video_path=video_path,
                resolution_steps=process_sequence.action_sequence,
                tags=enhanced_metadata.get('tags', []),
                compression_settings=self.config.get('compression_settings'),
                metadata=enhanced_metadata
            )
            
            # 4. Track processing status
            processing_result = await self._track_processing(
                upload_result['video_id'],
                process_sequence
            )
            
            return {
                'ticket_id': ticket_id,
                'video_id': upload_result['video_id'],
                'process_sequence': process_sequence,
                'processing_result': processing_result,
                'metadata': enhanced_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process ticket recording: {str(e)}")
            raise

    def _enhance_metadata(self,
                         original_metadata: Optional[Dict],
                         process_sequence: ProcessSequence) -> Dict:
        """Enhance metadata with processed information"""
        metadata = original_metadata or {}
        
        # Add processing information
        metadata.update({
            'processing_date': datetime.utcnow().isoformat(),
            'sequence_confidence': process_sequence.confidence,
            'action_count': len(process_sequence.action_sequence),
            'extracted_text': process_sequence.extracted_text,
            'interaction_points': process_sequence.interaction_points,
            'automation_potential': self._calculate_automation_potential(process_sequence)
        })
        
        # Add extracted tags
        metadata['tags'] = list(set(
            metadata.get('tags', []) +
            self._extract_tags(process_sequence)
        ))
        
        return metadata

    def _calculate_automation_potential(self,
                                     sequence: ProcessSequence) -> float:
        """Calculate automation potential score"""
        try:
            # Factors affecting automation potential:
            # 1. Sequence confidence
            confidence_score = sequence.confidence
            
            # 2. Action consistency
            action_consistency = self._calculate_action_consistency(
                sequence.action_sequence
            )
            
            # 3. Interaction point stability
            interaction_stability = self._calculate_interaction_stability(
                sequence.interaction_points
            )
            
            # 4. Text recognition quality
            text_quality = self._calculate_text_quality(
                sequence.extracted_text
            )
            
            # Weighted combination
            potential = (
                confidence_score * 0.3 +
                action_consistency * 0.3 +
                interaction_stability * 0.2 +
                text_quality * 0.2
            )
            
            return min(max(potential, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculating automation potential: {e}")
            return 0.5

    async def _track_processing(self,
                              video_id: str,
                              sequence: ProcessSequence) -> Dict:
        """Track video processing status"""
        try:
            max_attempts = self.config.get('max_tracking_attempts', 30)
            attempt = 0
            
            while attempt < max_attempts:
                status = await self.video_uploader.check_processing_status(video_id)
                
                if status['status'] == 'completed':
                    return {
                        'status': 'success',
                        'agent_id': status.get('created_agent_id'),
                        'processing_time': status.get('processing_time')
                    }
                elif status['status'] == 'failed':
                    # Attempt recovery if confidence is high
                    if sequence.confidence > 0.8:
                        await self.video_uploader.retry_failed_processing(video_id)
                    else:
                        return {
                            'status': 'failed',
                            'error': status.get('processing_error')
                        }
                
                attempt += 1
                await asyncio.sleep(10)  # Wait 10 seconds between checks
            
            return {'status': 'timeout'}
            
        except Exception as e:
            self.logger.error(f"Error tracking processing: {e}")
            return {'status': 'error', 'error': str(e)}

    def _extract_tags(self, sequence: ProcessSequence) -> List[str]:
        """Extract relevant tags from sequence"""
        tags = set()
        
        # Add tags based on actions
        for action in sequence.action_sequence:
            if 'CLICK' in action:
                tags.add('mouse-interaction')
            if 'KEY_PRESS' in action:
                tags.add('keyboard-input')
                
        # Add tags based on extracted text
        for text in sequence.extracted_text:
            # Add common system operation tags
            if 'settings' in text.lower():
                tags.add('system-settings')
            if 'password' in text.lower():
                tags.add('password-management')
            if 'error' in text.lower():
                tags.add('error-handling')
                
        return list(tags)

class TicketSystemIntegration:
    def __init__(self):
        self.api_url = settings.TICKET_SYSTEM_API_URL
        self.api_key = settings.TICKET_SYSTEM_API_KEY
        self.headers = {'Authorization': f'Bearer {self.api_key}'}

    def sync_ticket_status(self, ticket_id: str, status: str) -> bool:
        try:
            response = requests.post(
                f"{self.api_url}/tickets/{ticket_id}/status",
                json={'status': status},
                headers=self.headers
            )
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Failed to sync ticket status: {e}")
            return False

    def get_ticket_details(self, ticket_id: str) -> Dict:
        try:
            response = requests.get(
                f"{self.api_url}/tickets/{ticket_id}",
                headers=self.headers
            )
            return response.json() if response.status_code == 200 else {}
        except Exception as e:
            logging.error(f"Failed to get ticket details: {e}")
            return {}

class VideoProcessingEngine:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize OCR
        self.tesseract_config = '--oem 3 --psm 6'
        pytesseract.pytesseract.tesseract_cmd = config.get(
            'tesseract_path', 
            r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        )
        
        # Initialize ML models
        self.sequence_model = self._initialize_sequence_model()
        self.action_detector = self._initialize_action_detector()
        self.frame_processor = self._setup_frame_processor()

        # Setup processing pipeline
        self.frame_processor = self._setup_frame_processor()
        
        self.logger.info("Video processing engine initialized")

    def process_video(self, video_path: str) -> ProcessSequence:
        """Process video and extract automation sequence"""
        try:
            # Read video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            processed_frames = []
            action_sequence = []
            
            self.logger.info(f"Processing video: {frame_count} frames at {fps} FPS")
            
            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number = len(processed_frames)
                timestamp = frame_number / fps
                
                # Process frame
                processed_frame = self._process_frame(
                    frame, 
                    frame_number, 
                    timestamp
                )
                processed_frames.append(processed_frame)
                
                # Detect actions
                actions = self._detect_actions(processed_frame)
                if actions:
                    action_sequence.extend(actions)
            
            cap.release()
            
            # Extract process sequence
            process_sequence = self._extract_process_sequence(
                processed_frames,
                action_sequence
            )
            
            # Generate process map
            self._generate_process_map(process_sequence)
            
            return process_sequence
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            raise

    def _process_frame(self, 
                      frame: np.ndarray,
                      frame_number: int,
                      timestamp: float) -> ProcessedFrame:
        """Process individual frame"""
        try:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform OCR
            text_content = pytesseract.image_to_string(
                rgb_frame,
                config=self.tesseract_config
            )
            
            # Get mouse position (if available in frame)
            mouse_pos = self._detect_mouse_position(rgb_frame)
            
            # Detect clicks
            click_detected = self._detect_click(rgb_frame, frame_number)
            
            # Detect key presses
            key_pressed = self._detect_key_press(rgb_frame)
            
            # Detect screen changes
            screen_changes = self._detect_screen_changes(
                rgb_frame,
                frame_number
            )
            
            return ProcessedFrame(
                frame_number=frame_number,
                timestamp=timestamp,
                text_content=text_content,
                mouse_position=mouse_pos,
                click_detected=click_detected,
                key_pressed=key_pressed,
                screen_changes=screen_changes
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            raise

    def _detect_mouse_position(self, 
                             frame: np.ndarray) -> Tuple[int, int]:
        """Detect mouse cursor position in frame"""
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Apply template matching for cursor
            cursor_template = self._load_cursor_template()
            result = cv2.matchTemplate(
                gray,
                cursor_template,
                cv2.TM_CCOEFF_NORMED
            )
            
            # Get position of best match
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.8:  # Confidence threshold
                return max_loc
            return None
            
        except Exception as e:
            self.logger.warning(f"Mouse detection failed: {str(e)}")
            return None

    def _detect_click(self, 
                     frame: np.ndarray,
                     frame_number: int) -> bool:
        """Detect mouse clicks in frame"""
        try:
            # Use the action detector model
            frame_tensor = self._preprocess_frame_for_model(frame)
            prediction = self.action_detector.predict(frame_tensor)
            return prediction[0] > 0.5  # Click confidence threshold
            
        except Exception as e:
            self.logger.warning(f"Click detection failed: {str(e)}")
            return False

    def _detect_key_press(self, frame: np.ndarray) -> Optional[str]:
        """Detect key presses in frame"""
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Perform OCR in specific regions
            roi = gray[100:200, 100:200]  # Adjust ROI as needed
            text = pytesseract.image_to_string(
                roi,
                config='--psm 6'
            )
            
            # Process text to detect key presses
            # This is a simplified version - enhance based on needs
            if text.strip():
                return text.strip()
            return None
            
        except Exception as e:
            self.logger.warning(f"Key press detection failed: {str(e)}")
            return None

    def _detect_screen_changes(self, 
                             frame: np.ndarray,
                             frame_number: int) -> Dict:
        """Detect changes in screen content"""
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Compare with previous frame if available
            if hasattr(self, 'previous_frame'):
                # Calculate frame difference
                diff = cv2.absdiff(self.previous_frame, gray)
                
                # Apply threshold
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                
                # Find changed regions
                contours, _ = cv2.findContours(
                    thresh,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                changes = []
                for contour in contours:
                    if cv2.contourArea(contour) > 100:  # Min area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        changes.append({
                            'x': x, 'y': y,
                            'width': w, 'height': h
                        })
                
                self.previous_frame = gray
                return {'changes': changes}
            
            self.previous_frame = gray
            return {'changes': []}
            
        except Exception as e:
            self.logger.warning(f"Screen change detection failed: {str(e)}")
            return {'changes': []}

    def _initialize_sequence_model(self) -> tf.keras.Model:
        """Initialize sequence learning model"""
        model = Sequential([
            input(shape=(None, 128)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def _initialize_action_detector(self) -> torch.nn.Module:
        """Initialize action detection model"""
        class ActionDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(32 * 56 * 56, 512)
                self.fc2 = nn.Linear(512, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, 32 * 56 * 56)
                x = F.relu(self.fc1(x))
                x = self.sigmoid(self.fc2(x))
                return x
                
        return ActionDetector()

    def _extract_process_sequence(self,
                                processed_frames: List[ProcessedFrame],
                                action_sequence: List[str]) -> ProcessSequence:
        """Extract process sequence from processed frames"""
        try:
            # Group frames by action
            action_groups = []
            current_group = []
            
            for frame in processed_frames:
                if frame.click_detected or frame.key_pressed:
                    if current_group:
                        action_groups.append(current_group)
                        current_group = []
                current_group.append(frame)
                
            if current_group:
                action_groups.append(current_group)
            
            # Extract sequence information
            sequence = []
            screen_states = []
            interaction_points = []
            extracted_text = []
            
            for group in action_groups:
                # Get representative frame from group
                rep_frame = group[len(group)//2]
                
                # Extract action
                action = self._classify_action(rep_frame)
                if action:
                    sequence.append(action)
                    
                # Extract screen state
                screen_states.append(rep_frame.screen_changes)
                
                # Extract interaction point
                if rep_frame.mouse_position:
                    interaction_points.append(rep_frame.mouse_position)
                    
                # Extract text
                if rep_frame.text_content.strip():
                    extracted_text.append(rep_frame.text_content.strip())
            
            # Calculate confidence
            confidence = self._calculate_sequence_confidence(sequence)
            
            return ProcessSequence(
                action_sequence=sequence,
                screen_states=screen_states,
                interaction_points=interaction_points,
                extracted_text=extracted_text,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Process sequence extraction failed: {str(e)}")
            raise

    def _classify_action(self, frame: ProcessedFrame) -> Optional[str]:
        """Classify action from frame data"""
        try:
            if frame.click_detected:
                return "CLICK"
            elif frame.key_pressed:
                return f"KEY_PRESS_{frame.key_pressed}"
            elif frame.screen_changes.get('changes'):
                return "SCREEN_CHANGE"
            return None
            
        except Exception as e:
            self.logger.warning(f"Action classification failed: {str(e)}")
            return None

    def _calculate_sequence_confidence(self, 
                                    sequence: List[str]) -> float:
        """Calculate confidence score for extracted sequence"""
        try:
            if not sequence:
                return 0.0
                
            # Factors affecting confidence:
            # 1. Sequence length
            length_score = min(len(sequence) / 10, 1.0)
            
            # 2. Action variety
            unique_actions = len(set(sequence))
            variety_score = unique_actions / len(sequence)
            
            # 3. Pattern recognition
            pattern_score = self._detect_patterns(sequence)
            
            # Combine scores
            confidence = (length_score * 0.3 + 
                        variety_score * 0.3 + 
                        pattern_score * 0.4)
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5

    def _generate_process_map(self, sequence: ProcessSequence):

        """Generate process map using PM4Py"""
        try:
            # Convert sequence to event log format
            events = []
            for i, action in enumerate(sequence.action_sequence):
                events.append({
                    'case:concept:name': '1',
                    'concept:name': action,
                    'time:timestamp': time.time() + i
                })
            
            # Create log
            log = pm4py.format_dataframe(
                pd.DataFrame(events),
                case_id='case:concept:name',
                activity_key='concept:name',
                timestamp_key='time:timestamp'
            )
            
            # Generate process map
            process_tree = pm4py.discover_process_tree_inductive(log)
            
            # Save visualization
            pm4py.save_vis_process_tree(
                process_tree,
                f"process_map_{int(time.time())}.png"
            )
            
        except Exception as e:
            self.logger.error(f"Process map generation failed: {str(e)}")
            raise

    def _setup_frame_processor(self) -> Dict:
        """Initialize frame processing components"""
        try:
            # Create tracker factory
            tracker_types = {
                'CSRT': cv2.legacy.TrackerCSRT_create,
                'KCF': cv2.legacy.TrackerKCF_create,
                'MOSSE': cv2.legacy.TrackerMOSSE_create
            }
            
            selected_tracker = self.config.get('tracker_type', 'KCF')
            if selected_tracker not in tracker_types:
                self.logger.warning(f"Invalid tracker type {selected_tracker}, defaulting to KCF")
                selected_tracker = 'KCF'

            processors = {
                'motion_detector': cv2.createBackgroundSubtractorMOG2(
                    history=500,
                    varThreshold=16,
                    detectShadows=False
                ),
                'face_detector': cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                ),
                'text_detector': pytesseract.image_to_string,
                'frame_enhancer': self._create_frame_enhancer(),
                'tracker_factory': tracker_types[selected_tracker],
                'preprocessors': self._setup_preprocessors(),
                'postprocessors': self._setup_postprocessors()
            }

            # Validate processor initialization
            for name, processor in processors.items():
                if processor is None:
                    raise ValueError(f"Failed to initialize {name}")

            self.logger.info("Frame processor components initialized successfully")
            return processors

        except Exception as e:
            self.logger.error(f"Frame processor setup failed: {str(e)}")
            raise

    def _create_frame_enhancer(self) -> Dict:
        """Create frame enhancement pipeline"""
        return {
            'denoise': lambda frame: cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                10,
                10,
                7,
                21
            ),
            'sharpen': lambda frame: cv2.filter2D(
                frame,
                -1,
                np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
            ),
            'contrast': lambda frame: cv2.convertScaleAbs(
                frame,
                alpha=1.1,
                beta=10
            )
        }

    def _setup_preprocessors(self) -> List[Callable]:
        """Setup frame preprocessing pipeline"""
        return [
            # Color space conversion
            lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            
            # Resize if needed
            lambda frame: cv2.resize(
                frame,
                (
                    self.config.get('frame_width', 1920),
                    self.config.get('frame_height', 1080)
                )
            ) if frame.shape[:2] != (
                self.config.get('frame_height', 1080),
                self.config.get('frame_width', 1920)
            ) else frame,
            
            # Normalization
            lambda frame: frame.astype(np.float32) / 255.0,
            
            # Add padding if needed
            lambda frame: np.pad(
                frame,
                ((0, 0), (0, 0), (0, 0)),
                mode='constant'
            ) if frame.shape[2] < 3 else frame
        ]

    def _setup_postprocessors(self) -> List[Callable]:
        """Setup frame postprocessing pipeline"""
        return [
            # Denormalize
            lambda frame: (frame * 255).astype(np.uint8),
            
            # Apply color correction
            lambda frame: cv2.convertScaleAbs(
                frame,
                alpha=self.config.get('color_alpha', 1.0),
                beta=self.config.get('color_beta', 0)
            ),
            
            # Apply final enhancements
            lambda frame: self._apply_final_enhancements(frame)
        ]

    def _apply_final_enhancements(self, frame: np.ndarray) -> np.ndarray:
        """Apply final enhancements to frame"""
        try:
            # Apply configured enhancements
            if self.config.get('enable_denoising', True):
                frame = self.frame_processor['frame_enhancer']['denoise'](frame)
            
            if self.config.get('enable_sharpening', True):
                frame = self.frame_processor['frame_enhancer']['sharpen'](frame)
            
            if self.config.get('enable_contrast', True):
                frame = self.frame_processor['frame_enhancer']['contrast'](frame)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"Final enhancement failed: {str(e)}")
            return frame

    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float) -> Dict:
        """Process a single frame using the frame processor pipeline"""
        try:
            # Apply preprocessors
            for preprocessor in self.frame_processor['preprocessors']:
                frame = preprocessor(frame)

            # Detect motion
            motion_mask = self.frame_processor['motion_detector'].apply(frame)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.frame_processor['face_detector'].detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Extract text
            text = self.frame_processor['text_detector'](frame)
            
            # Update tracking if active
            tracking_result = None
            if self.tracker:
                tracking_result = self.update_object_tracking(frame)
            
            # Apply postprocessors
            for postprocessor in self.frame_processor['postprocessors']:
                frame = postprocessor(frame)

            return {
                'processed_frame': frame,
                'motion_detected': np.any(motion_mask > 0),
                'faces_detected': len(faces) if isinstance(faces, np.ndarray) else 0,
                'extracted_text': text,
                'tracking_result': tracking_result,
                'frame_number': frame_number,
                'timestamp': timestamp
            }

        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            return {
                'processed_frame': frame,
                'error': str(e),
                'frame_number': frame_number,
                'timestamp': timestamp
            }

    def start_object_tracking(self, frame: np.ndarray, box: Tuple[int, int, int, int]):
        """Start tracking an object in the frame"""
        try:
            # Create new tracker instance
            self.tracker = self.frame_processor['tracker_factory']()
            
            # Initialize tracker
            success = self.tracker.init(frame, box)
            if success:
                self.tracking_box = box
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to start object tracking: {str(e)}")
            return False

    def stop_object_tracking(self):
        """Stop object tracking"""
        if hasattr(self, 'tracking_box'):
            delattr(self, 'tracking_box')

    def update_object_tracking(self, frame: np.ndarray) -> Optional[Tuple[bool, Tuple]]:
        """Update object tracking"""
        try:
            if self.tracker:
                success, box = self.tracker.update(frame)
                if success:
                    self.tracking_box = box
                    return True, box
                else:
                    self.tracking_box = None
                    return False, None
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to update object tracking: {str(e)}")
            return None
