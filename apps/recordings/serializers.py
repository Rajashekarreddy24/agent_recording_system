
from rest_framework import serializers
from .models import Recording

class RecordingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recording
        fields = '_all'  # Using 'all_' as a string, not tuple/list
        # Or explicitly list fields:
        # fields = [
        #     'id',
        #     'ticket_id',
        #     'agent_name',
        #     'recording_id',
        #     'output_path',
        #     'start_time',
        #     'end_time',
        #     'status',
        #     'error_message',
        #     'metadata'
        # ]

class StartRecordingSerializer(serializers.Serializer):
    ticket_id = serializers.CharField(required=True)
    agent_name = serializers.CharField(required=True)
    fps = serializers.IntegerField(default=30)
    resolution = serializers.ListField(
        child=serializers.IntegerField(),
        default=[1920, 1080],
        min_length=2,
        max_length=2
    )
    metadata = serializers.DictField(required=False, default=dict)

class StopRecordingSerializer(serializers.Serializer):
    ticket_id = serializers.CharField(required=True)
    agent_name = serializers.CharField(required=True)

class RecordingStatusSerializer(serializers.Serializer):
    ticket_id = serializers.CharField(required=True)
    agent_name = serializers.CharField(required=True)

class RecordingResponseSerializer(serializers.ModelSerializer):
    duration = serializers.FloatField(read_only=True)
    
    class Meta:
        model = Recording
        fields = [
            'status',
            'recording_id',
            'ticket_id',
            'agent_name',
            'output_path',
            'start_time',
            'end_time',
            'duration',
            'metadata'
        ]

