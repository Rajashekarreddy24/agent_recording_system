from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.core.cache import cache
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from config import settings

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist

from .models import Ticket, Activity, TicketInfo, Action, Pattern
from apps.recordings.models import Recording
from .serializers import (
    TicketInfoSerializer,
    ActionSerializer,
    PatternSerializer)

from apps.recordings.serializers import(
    RecordingSerializer,
    StartRecordingSerializer,
    StopRecordingSerializer,
    RecordingStatusSerializer,
    RecordingResponseSerializer,
)

from apps.tickets.services.ticket_service import TicketSystemIntegration
from apps.recordings.services.recorder import (
    EnhancedActionRecorder,
    EnhancedVideoRecorder,
    SimpleScreenRecorder,
    MSSScreenRecorder,
)
from utils.exceptions import generate_activity_dataframe, export_to_csv

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import datetime
import os
import json
import yaml
import csv
import re
import logging
import threading
import time
import cv2
import numpy as np
import pyautogui
import mss
import pytesseract
import pygetwindow as gw
import difflib
import hashlib
import psutil
import win32gui
import win32process

from typing import List, Dict, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from queue import Queue
import asyncio

is_recording = False
ticket_system = TicketSystemIntegration()
active_monitors = {}
recording_thread = None
recording = None

class ActivityMonitor:
    def __init__(self, ticket_id: str):
        self.ticket_id = ticket_id
        self.last_activity_time = datetime.now()
        self.current_window = None
        self.recording = False

    def start_monitoring(self):
        self.recording = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.recording = False

    def _monitor_loop(self):
        while self.recording:
            try:
                window = win32gui.GetForegroundWindow()
                if window != self.current_window:
                    self._log_window_change(window)
                    self.current_window = window
                
                if self._check_inactivity():
                    self._prompt_for_update()
                
                time.sleep(1)
            except Exception as e:
                logging.error(f"Monitoring error: {e}")

    def _log_window_change(self, window):
        try:
            window_title = win32gui.GetWindowText(window)
            pid = win32process.GetWindowThreadProcessId(window)[1]
            process = psutil.Process(pid)
            app_name = process.name()
           
            # Use Django ORM to log activity in the database
            Activity.objects.create(
                ticket_id=self.ticket_id,
                timestamp=datetime.now(),
                application=app_name,
                action=f"Switched to: {window_title}",
                category='WINDOW_CHANGE'
            )
           
            self.last_activity_time = datetime.now()
        except Exception as e:
            logging.error(f"Failed to log window change: {e}")

    def _check_inactivity(self) -> bool:
        inactivity_threshold = getattr(settings, 'INACTIVITY_THRESHOLD', 300)  # default threshold 5 mins
        return (datetime.now() - self.last_activity_time).seconds > inactivity_threshold

    def _prompt_for_update(self):
        logging.info("User inactive, prompt for update.")
        # Implement prompt logic, perhaps sending a notification or updating a status in the database.
@dataclass
class ActionContext:
    window_title: str
    active_element: str
    parent_element: str
    screen_region: Tuple[int, int, int, int]  # (left, top, width, height)

class PatternMatcher:
    """Enhanced pattern matching for ticket classification"""

    def __init__(self):
        self.patterns_db = {}
        self.load_patterns()

    def load_patterns(self):
        """Load patterns from YAML configuration."""
        try:
            with open('patterns.yaml', 'r') as f:
                self.patterns_db = yaml.safe_load(f)
        except FileNotFoundError:
            self.patterns_db = {
                'categories': {
                    'password_reset': [
                        r'(?i)password.{0,10}reset',
                        r'(?i)forgot.{0,10}password',
                        r'(?i)change.{0,10}password'
                    ],
                    'access_request': [
                        r'(?i)request.{0,10}access',
                        r'(?i)permission.{0,10}needed',
                        r'(?i)grant.{0,10}access'
                    ],
                    'software_install': [
                        r'(?i)install.{0,10}software',
                        r'(?i)new.{0,10}application',
                        r'(?i)download.{0,10}program'
                    ]
                },
                'priorities': {
                    'high': [
                        r'(?i)urgent',
                        r'(?i)critical',
                        r'(?i)emergency'
                    ],
                    'medium': [
                        r'(?i)normal',
                        r'(?i)standard',
                        r'(?i)regular'
                    ],
                    'low': [
                        r'(?i)low',
                        r'(?i)minor',
                        r'(?i)whenever'
                    ]
                }
            }
            # Save default patterns
            self.save_patterns()

    def save_patterns(self):
        """Save patterns to YAML file."""
        with open('patterns.yaml', 'w') as f:
            yaml.dump(self.patterns_db, f)

    def match_category(self, text: str) -> str:
        """Match text against category patterns."""
        max_score = 0
        category = 'unknown'
        
        for cat, patterns in self.patterns_db.get('categories', {}).items():
            score = sum(len(re.findall(pattern, text)) for pattern in patterns)
            if score > max_score:
                max_score = score
                category = cat
                
        return category

    def extract_environment_info(self, text: str) -> Dict[str, str]:
        """Extract system environment information from the given text."""
        env_info = {}
        patterns = {
            'os': r'OS:\s*(.*?)(?:\n|$)',
            'browser': r'Browser:\s*(.*?)(?:\n|$)',
            'software_version': r'Version:\s*(.*?)(?:\n|$)'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                env_info[key] = match.group(1).strip()
        return env_info

    def learn_new_pattern(self, category: str, text: str):
        """Learn new patterns from successful categorizations."""
        words = text.lower().split()
        for i in range(len(words) - 2):
            pattern = r'(?i)' + r'.{0,10}'.join(words[i:i + 3])
            if category in self.patterns_db['categories']:
                if pattern not in self.patterns_db['categories'][category]:
                    self.patterns_db['categories'][category].append(pattern)
            else:
                self.patterns_db['categories'][category] = [pattern]
        self.save_patterns()

def dashboard(request):
    tickets = Ticket.objects.all()
    return render(request, 'technician_activities/dashboard.html', {'tickets': tickets})

def sync_ticket(request, ticket_id):
    if request.method == 'POST':
        status = request.POST.get('status')
        success = ticket_system.sync_ticket_status(ticket_id, status)
        if success:
            return JsonResponse({'status': 'success', 'message': 'Ticket status synced successfully.'})
        return JsonResponse({'status': 'error', 'message': 'Failed to sync ticket status.'}, status=500)

def view_ticket(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    # activities = Activity.objects.filter(ticket=ticket)
    # external_ticket_details = ticket_system.get_ticket_details(ticket_id)
    # return render(request, 'technician_activities/report.html', {
    #     'ticket': ticket,
    #     'activities': activities,
    #     'external_ticket_details': external_ticket_details,
    # })
    return render(request, 'technician_activities/recording_template.html', {'ticket': ticket})

def download_report(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    activities = Activity.objects.filter(ticket=ticket)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{ticket_id}_report.csv"'
    writer = csv.writer(response)
    writer.writerow(['Activity ID', 'Ticket ID', 'Timestamp', 'Application', 'Action', 'Notes', 'Duration', 'Category', 'Automated Flag'])
    for activity in activities:
        writer.writerow([
            activity.id,
            activity.ticket.ticket_id,
            activity.timestamp,
            activity.application,
            activity.action,
            activity.notes,
            activity.duration,
            activity.category,
            activity.automated_flag
        ])
    return response

def download_report_pdf(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    activities = Activity.objects.filter(ticket=ticket)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{ticket_id}_report.pdf"'
    
    p = canvas.Canvas(response, pagesize=letter)
    width, height = letter
    p.drawString(100, height - 50, f"Report for Ticket ID: {ticket_id}")
    y_position = height - 90
    for activity in activities:
        p.drawString(100, y_position, f"Activity ID: {activity.id}, Action: {activity.action}, Timestamp: {activity.timestamp}")
        y_position -= 20
    p.showPage()
    p.save()
    return response

def start_monitoring(request, ticket_id):
    if ticket_id not in active_monitors:
        monitor = ActivityMonitor(ticket_id)
        monitor.start_monitoring()
        active_monitors[ticket_id] = monitor
        return JsonResponse({'status': f'Monitoring started for ticket {ticket_id}'})
    return JsonResponse({'status': f'Monitoring already active for ticket {ticket_id}'})

def stop_monitoring(request, ticket_id):
    monitor = active_monitors.get(ticket_id)
    if monitor:
        monitor.stop_monitoring()
        del active_monitors[ticket_id]
        return JsonResponse({'status': f'Monitoring stopped for ticket {ticket_id}'})
    return JsonResponse({'status': f'No active monitor found for ticket {ticket_id}'})

def generate_activity_report(request, ticket_id):
    ticket = get_object_or_404(Ticket, ticket_id=ticket_id)
    start_date = request.GET.get('start_date', '2023-01-01')
    end_date = request.GET.get('end_date', datetime.datetime.now().strftime('%Y-%m-%d'))
    activities = Activity.get_activity_report(ticket_id, start_date, end_date)
    df = generate_activity_dataframe(activities)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="activity_report_{ticket_id}.csv"'
    df.to_csv(response, index=False)
    return response

def generate_time_analysis(request, ticket_id):
    analysis = Activity.get_time_analysis(ticket_id)
    return JsonResponse(analysis)

class TicketInfoViewSet(viewsets.ModelViewSet):
    queryset = TicketInfo.objects.all()
    serializer_class = TicketInfoSerializer

class ActionViewSet(viewsets.ModelViewSet):
    queryset = Action.objects.all()
    serializer_class = ActionSerializer
    
class PatternViewSet(viewsets.ModelViewSet):
    queryset = Pattern.objects.all()
    serializer_class = PatternSerializer

def ticket_list(request):
    """View to display the list of tickets."""
    tickets = TicketInfo.objects.all()  
    return render(request, 'tickets/ticket_list.html', {'tickets': tickets})

def ticket_detail(request, ticket_id ):
    """View to display details of a specific ticket."""
    ticket = get_object_or_404(TicketInfo, ticket_id=ticket_id)  # Get ticket by primary key
    return render(request, 'tickets/ticket_detail.html', {'ticket': ticket})

class EnhancedTicketAnalyzer(View):

    """Enhanced ticket analyzer with better pattern recognition."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pattern_matcher = PatternMatcher()
        self.ocr_queue = Queue()
        self.latest_ocr_result = ""
        self.start_ocr_worker()

    def start_ocr_worker(self):
        """Start background OCR processing."""
        def worker():
            while True:
                frame = self.ocr_queue.get()
                if frame is None:
                    break
                self._process_frame_ocr(frame)

        self.ocr_thread = threading.Thread(target=worker, daemon=True)
        self.ocr_thread.start()

    def _process_frame_ocr(self, frame):
        """Process OCR in background."""
        try:
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(thresh, config=custom_config)
            # Store result in class variable
            self.latest_ocr_result = text
        except Exception as e:
            logging.error(f"OCR processing error: {str(e)}")

    def post(self, request, *args, **kwargs):
        """Handle POST requests for analyzing frames."""
        frame_file = request.FILES['frame']
        
        # Read the frame using OpenCV
        file_bytes = np.asarray(bytearray(frame_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process the frame with OCR
        self.ocr_queue.put(frame)

        # Get the latest OCR result after processing (this may need synchronization)
        text_result = self.latest_ocr_result

        # Extract environment info from the OCR result (if needed)
        env_info = self.pattern_matcher.extract_environment_info(text_result)

        return render(request, 'tickets/analyze.html', {'text_result': text_result, 'env_info': env_info})

    def get(self, request, *args, **kwargs):
        """Render the analyze page."""
        return render(request, 'tickets/analyze.html')
enhanced_analyzer = EnhancedTicketAnalyzer()

def analyze_frame(request):

    """Analyze a video frame for ticket information."""
    if request.method == 'POST':
        # Assume frame is uploaded as a file (you can modify this as needed)
        frame_file = request.FILES['frame']
        
        # Read the frame using OpenCV
        file_bytes = np.asarray(bytearray(frame_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process the frame with OCR
        enhanced_analyzer.ocr_queue.put(frame)

        # Get the latest OCR result after processing (this may need synchronization)
        text_result = enhanced_analyzer.latest_ocr_result

        
        env_info = enhanced_analyzer.extract_environment_info(text_result)

        return render(request, 'tickets/analyze.html', {'text_result': text_result, 'env_info': env_info})

    return render(request, 'tickets/analyze.html')

class StartRecordingView(View):
    @csrf_exempt
    def post(self, request, ticket_id):
        """Start recording for the given ticket_id."""
        recorder = EnhancedActionRecorder(ticket_id)
        recorder.start_recording()
        # return JsonResponse({"status": "Recording started", "ticket_id": ticket_id})
        return render(request, 'technician_activities/recording_in_progress.html', {'ticket_id': ticket_id})

class StopRecordingView(View):
    @csrf_exempt
    def post(self, request, ticket_id):
        """Stop recording for the given ticket_id."""
        recorder = EnhancedActionRecorder(ticket_id)
        recorder.stop_recording()
        return JsonResponse({"status": "Recording stopped", "ticket_id": ticket_id})

class RecordActionView(View):
    """View to handle recording actions."""
    
    def get(self, request, ticket_id):
        """Render the recording UI page."""
        return render(request, 'tickets/record.html', {'ticket_id': ticket_id})
    
    def post(self, request, ticket_id):
        action_recorder = (ticket_id ==ticket_id)
        
        # Check if we are starting or stopping recording
        if request.POST.get('action') == 'start':
            if not cache.get(f'is_recording_{ticket_id}'):
                action_recorder.start_recording()
                messages.success(request, 'Recording in progress')
                return redirect('recording_status', ticket_id=ticket_id)  # Redirect to recording status page
            else:
                messages.info(request, 'Recording is already in progress for this ticket.')
                return redirect('recording_status', ticket_id=ticket_id)

        elif request.POST.get('action') == 'stop':
            if cache.get(f'is_recording_{ticket_id}'):
                action_recorder.stop_recording()
                messages.success(request, 'Recording has been stopped.')
                return redirect('dashboard')  # Redirect to the dashboard
            else:
                messages.info(request, 'No recording in progress to stop.')
                return redirect('recording_status', ticket_id=ticket_id)

        return render(request, 'record.html', {'ticket_id': ticket_id})

class RecordingStatusView(View):
    """View to display recording status"""
    
    def get(self, request, ticket_id):
        is_recording = cache.get('is_recording', False)
        status_message = "Recording is in progress..." if is_recording else "Recording has been stopped."
        return render(request, 'tickets/recording_status.html', {
            'ticket_id': ticket_id,
            'status_message': status_message
        })

def record_screen(ticket_id):
    global is_recording, recording

    screen_size = pyautogui.size()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"recordings/{ticket_id}_{timestamp}.mp4"
    recording = cv2.VideoWriter(file_name, fourcc, 20.0, (screen_size.width, screen_size.height))

    while is_recording:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        recording.write(frame)
        time.sleep(0.05)  # Adjust as needed for smooth recording

    recording.release()  # Stop recording and release file

@csrf_exempt
def start_recording(request, ticket_id):
    global is_recording, recording_thread
    if is_recording:
        # return JsonResponse({'status': 'Recording is already in progress', 'ticket_id': ticket_id})
        return render(request, 'technician_activities/recording_in_progress.html', {'ticket_id': ticket_id})
    
    is_recording = True
    recording_thread = threading.Thread(target=record_screen, args=(ticket_id,))
    recording_thread.start()
    return render(request, 'technician_activities/recording_in_progress.html', {'ticket_id': ticket_id})

@csrf_exempt
def stop_recording(request, ticket_id):
    global is_recording, recording_thread
    if not is_recording:
        return JsonResponse({'status': 'No recording in progress', 'ticket_id': ticket_id})
        # return render(request, 'technician_activities/stop_recording.html', {'ticket_id': ticket_id, 'status': 'Recording stopped'})
    
    is_recording = False  # Signal to stop recording
    recording_thread.join()  # Wait for recording thread to finish
    # return render(request, 'technician_activities/stop_recording.html', {'ticket_id': ticket_id, 'status': 'Recording stopped'})
    return JsonResponse({'status': 'Recording stopped successfully', 'redirect_url': f'/stop_recording/{ticket_id}/'})

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def record_screen_view(request):
    duration_seconds = int(request.GET.get('duration', 10))
    try:
        recorder = SimpleScreenRecorder(output_dir="recordings")
        output_path = recorder.record(
            duration=duration_seconds,
            fps=10,
            output_name=f"screen_recording_{int(time.time())}.avi"
        )
        return JsonResponse({"success": True, "output_path": output_path})
    except Exception as e:
        logging.error(f"Recording failed: {e}")
        return JsonResponse({"success": False, "error": str(e)})

def record_screen(duration_seconds: int = 10):
    recorder = None
    try:
        # Initialize recorder
        recorder = MSSScreenRecorder(output_dir="recordings")
        
        # Start recording
        output_path = recorder.record(
            duration=duration_seconds,
            fps=10
        )
        
        print(f"Recording saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Recording failed: {e}")
        return JsonResponse({"success": False, "error": str(e)})
    finally:
        if recorder:
            recorder.stop()

class RecordingViewSet(viewsets.ModelViewSet):
    queryset = Recording.objects.all()
    serializer_class = RecordingSerializer
    recorder = None

    def get_serializer_class(self):
        """Return appropriate serializer class based on action"""
        if self.action == 'start_recording':
            return StartRecordingSerializer
        elif self.action == 'stop_recording':
            return StopRecordingSerializer
        elif self.action == 'recording_status':
            return RecordingStatusSerializer
        elif self.action == 'retrieve' or self.action == 'list':
            return RecordingResponseSerializer
        return self.serializer_class

    @action(detail=False, methods=['post'])
    def start_recording(self, request, *args, **kwargs):
        """Start recording endpoint"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        ticket_id = serializer.validated_data['ticket_id']
        agent_name = serializer.validated_data['agent_name']

        try:
            # Check for existing recording
            existing_recording = Recording.objects.filter(
                ticket_id=ticket_id,
                agent_name=agent_name,
                status='recording'
            ).first()

            if existing_recording:
                return Response(
                    {
                        'error': 'Active recording already exists',
                        'recording_id': existing_recording.recording_id
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create recording
            recording = Recording.objects.create(
                ticket_id=ticket_id,
                agent_name=agent_name,
                status='initializing',
                metadata=serializer.validated_data.get('metadata', {})
            )

            # Start recording
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self.recorder.start_recording(
                    ticket_id=ticket_id,
                    agent_name=agent_name,
                    fps=serializer.validated_data.get('fps', 30),
                    resolution=tuple(serializer.validated_data.get('resolution', [1920, 1080]))
                )
            )

            # Update recording
            recording.recording_id = result['recording_id']
            recording.output_path = result['output_path']
            recording.status = 'recording'
            recording.save()

            response_serializer = RecordingResponseSerializer(recording)
            return Response(response_serializer.data)

        except Exception as e:
            if 'recording' in locals():
                recording.status = 'error'
                recording.error_message = str(e)
                recording.save()

            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )