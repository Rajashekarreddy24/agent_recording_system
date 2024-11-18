from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import logging
from .services.recorder import ScreenRecorder, RDSScreenRecorder, EnhancedVideoRecorder

#Initialize ScreenRecorder with a config dictionary
# screen_recorder = ScreenRecorder(config={"output_dir": "newrecordings", "fps": 30})

# @csrf_exempt
# @require_http_methods(["POST"])
# def start_recording(request, agent_name, ticket_id=None):
#     """
#     Start screen recording for an agent
#     """
#     try:
#         filepath = screen_recorder.start_recording(output_name=f"{agent_name}_{ticket_id}")

#         if filepath:
#             return JsonResponse({
#                 "status": "success",
#                 "filepath": filepath
#             })
#         else:
#             return JsonResponse({
#                 "status": "error",
#                 "message": "Recording already in progress"
#             }, status=400)
#     except Exception as e:
#         print('Start recording failed------------>')
#         logging.error(f"Failed to start recording: {str(e)}")
#         return JsonResponse({
#             "status": "error",
#             "message": str(e)
#         }, status=500)

# @csrf_exempt
# @require_http_methods(["POST"])
# def stop_recording(request, agent_name=None, ticket_id=None):
#     """
#     Stop the current screen recording
#     """
#     try:
#         recorded_path = screen_recorder.stop_recording()
#         return JsonResponse({
#             "status": "success",
#             "message": f"Recording stopped for Agent: {agent_name}, Ticket: {ticket_id}",
#             "filepath": recorded_path
#         })
#     except Exception as e:
#         logging.error(f"Failed to stop recording: {str(e)}")
#         return JsonResponse({
#             "status": "error",
#             "message": str(e)
#         }, status=500)

# @require_http_methods(["GET"])
# def get_recording_status(request):
#     """
#     Get current recording status
#     """
#     return JsonResponse({
#         "is_recording": screen_recorder.recording
#     })


# Initialize RDSScreenRecorder with a configuration dictionary
# screen_recorder = RDSScreenRecorder(config={"output_dir": "newrecordings", "fps": 30})

# @csrf_exempt
# @require_http_methods(["POST"])
# def start_recording(request, agent_name, ticket_id=None):
#     """
#     Start screen recording for an agent
#     """
#     try:
#         # Adjusting for RDSScreenRecorder method
#         filepath = screen_recorder.start_recording(output_name=f"{agent_name}_{ticket_id}")

#         if filepath:
#             return JsonResponse({
#                 "status": "success",
#                 "filepath": filepath
#             })
#         else:
#             return JsonResponse({
#                 "status": "error",
#                 "message": "Recording already in progress"
#             }, status=400)
#     except Exception as e:
#         logging.error(f"Failed to start recording: {str(e)}")
#         return JsonResponse({
#             "status": "error",
#             "message": str(e)
#         }, status=500)

# @csrf_exempt
# @require_http_methods(["POST"])
# def stop_recording(request, agent_name=None, ticket_id=None):
#     """
#     Stop the current screen recording
#     """
#     try:
#         recorded_path = screen_recorder.stop_recording()
#         return JsonResponse({
#             "status": "success",
#             "message": f"Recording stopped for Agent: {agent_name}, Ticket: {ticket_id}",
#             "filepath": recorded_path
#         })
#     except Exception as e:
#         logging.error(f"Failed to stop recording: {str(e)}")
#         return JsonResponse({
#             "status": "error",
#             "message": str(e)
#         }, status=500)

# @require_http_methods(["GET"])
# def get_recording_status(request):
#     """
#     Get current recording status
#     """
#     return JsonResponse({
#         "is_recording": screen_recorder.recording
#     })


# # Initialize the EnhancedRecorder instance
# screen_recorder = EnhancedVideoRecorder(config={"output_dir": "newrecordings", "processing_config": {}})

# @csrf_exempt
# @require_http_methods(["POST"])
# def start_recording(request, agent_name, ticket_id=None):
#     """
#     Start screen recording for an agent
#     """

#     try:
#         # Adjust start recording to pass the correct parameters
#         resolution = (1920, 1080)  # You can modify this as needed
#         fps = 30  # You can adjust the FPS if necessary

#         # Start recording and handle it asynchronously
#         recording_info = screen_recorder.start_recording(
#             ticket_id=ticket_id,
#             agent_name= None,
#             resolution=resolution,
#             fps=fps
#         )
#         if recording_info['status'] == 'started':

#             return JsonResponse({
#                 "status": "success",
#                 "recording_id": recording_info['recording_id'],
#                 "output_path": recording_info['output_path'],
#                 "ticket_id": ticket_id
#             })
#         else:
#             return JsonResponse({
#                 "status": "error",
#                 "message": "Recording already in progress"
#             }, status=400)

#     except Exception as e:
#         logging.error(f"Failed to start recording: {str(e)}")
#         return JsonResponse({
#             "status": "error",
#             "message": str(e)
#         }, status=500)

# @csrf_exempt
# @require_http_methods(["POST"])
# def stop_recording(request, agent_name=None, ticket_id=None):
#     """
#     Stop the current screen recording
#     """
#     try:
#         # Stop the recording and retrieve the result
#         result =  screen_recorder.stop_recording()
        
#         if result['status'] == 'completed':
#             return JsonResponse({
#                 "status": "success",
#                 "message": f"Recording stopped for Agent: {agent_name}, Ticket: {ticket_id}",
#                 "filepath": result['recording_info']['output_path'],
#                 "duration": result['recording_info']['duration'],
#                 "actions_detected": result['recording_info']['detected_actions']
#             })
#         else:
#             return JsonResponse({
#                 "status": "error",
#                 "message": "No recording in progress"
#             }, status=400)

#     except Exception as e:
#         logging.error(f"Failed to stop recording: {str(e)}")
#         return JsonResponse({
#             "status": "error",
#             "message": str(e)
#         }, status=500)

# @require_http_methods(["GET"])
# def get_recording_status(request):
#     """
#     Get current recording status
#     """
#     return JsonResponse({
#         "is_recording": screen_recorder.recording,
#         "current_recording": screen_recorder.current_recording if screen_recorder.current_recording else None
#     })


from rest_framework import viewsets, status
from rest_framework.decorators import action, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
import asyncio
from asgiref.sync import async_to_sync
from django.core.exceptions import ObjectDoesNotExist
from datetime import datetime
from .models import Recording
from .serializers import *

@permission_classes([AllowAny])  # Or use IsAuthenticated if you need authentication
class RecordingViewSet(viewsets.ModelViewSet):
    queryset = Recording.objects.all()
    serializer_class = RecordingSerializer
    recorder = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.recorder:
            config = {
                'output_dir': 'recordings',
                'temp_dir': 'temp_recordings',
                'backup_dir': 'backup_recordings'
            }
            self.recorder = EnhancedVideoRecorder(config)

    def get_permissions(self):
        """
        Set custom permissions for different actions
        """
        if self.action in ['start_recording', 'stop_recording']:
            permission_classes = [AllowAny]
        else:
            permission_classes = [IsAuthenticated]
        return [permission() for permission in permission_classes]
    
    @action(detail=False, methods=['post'], permission_classes=[AllowAny])
    def start_recording(self, request, *args, **kwargs):
        print("""Start recording endpoint""")
        try:
            print('------------------->')
            ticket_id = kwargs.get('ticket_id')
            agent_name = kwargs.get('agent_name')

            if not ticket_id or not agent_name:
                return Response(
                    {'error': 'ticket_id and agent_name are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create serializer with combined data
            data = {
                'ticket_id': ticket_id,
                'agent_name': agent_name,
                **request.data
            }
            serializer = StartRecordingSerializer(data=data)
            
            if not serializer.is_valid():
                return Response(
                    {'error': serializer.errors},
                    status=status.HTTP_400_BAD_REQUEST
                )

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

            # Create recording record
            recording = Recording.objects.create(
                ticket_id=ticket_id,
                agent_name=agent_name,
                status='initializing',
                metadata=serializer.validated_data.get('metadata', {})
            )

            # Run async recording in sync context
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self._start_recording_async(
                    ticket_id=ticket_id,
                    agent_name=agent_name,
                    fps=serializer.validated_data.get('fps', 30),
                    resolution=tuple(serializer.validated_data.get('resolution', [1920, 1080]))
                )
            )

            # Update recording record
            recording.recording_id = result['recording_id']
            recording.output_path = result['output_path']
            recording.status = 'recording'
            recording.save()

            return Response({
                'status': 'success',
                'recording_id': recording.recording_id,
                'output_path': recording.output_path,
                'ticket_id': ticket_id,
                'agent_name': agent_name
            })

        except Exception as e:
            if 'recording' in locals():
                recording.status = 'error'
                recording.error_message = str(e)
                recording.save()

            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['post'], permission_classes=[AllowAny])
    def stop_recording(self, request, *args, **kwargs):
        """Stop recording endpoint"""
        try:
            ticket_id = kwargs.get('ticket_id')
            agent_name = kwargs.get('agent_name')

            if not ticket_id or not agent_name:
                return Response(
                    {'error': 'ticket_id and agent_name are required'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            try:
                recording = Recording.objects.get(
                    ticket_id=ticket_id,
                    agent_name=agent_name,
                    status='recording'
                )
            except ObjectDoesNotExist:
                return Response(
                    {
                        'error': 'No active recording found',
                        'ticket_id': ticket_id,
                        'agent_name': agent_name
                    },
                    status=status.HTTP_404_NOT_FOUND
                )

            # Run async stop in sync context
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = loop.run_until_complete(self._stop_recording_async())

            # Update recording record
            recording.status = 'stopped'
            recording.end_time = datetime.now()
            recording.save()

            return Response({
                'status': 'success',
                'recording_id': recording.recording_id,
                'output_path': recording.output_path,
                'ticket_id': ticket_id,
                'agent_name': agent_name,
                'duration': (recording.end_time - recording.start_time).total_seconds()
            })

        except Exception as e:
            if 'recording' in locals():
                recording.status = 'error'
                recording.error_message = str(e)
                recording.save()

            return Response(
                {'error': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )