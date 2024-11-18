from django.db import models
from django.utils import timezone
import uuid
from apps.tickets.models import Ticket

class Recording(models.Model):
    STATUS_CHOICES = [
        ('initializing', 'Initializing'),
        ('recording', 'Recording'),
        ('stopped', 'Stopped'),
        ('error', 'Error')
    ]

    ticket_id = models.CharField(max_length=100)
    agent_name = models.CharField(max_length=100)
    recording_id = models.CharField(max_length=200, unique=True)
    output_path = models.CharField(max_length=500)
    start_time = models.DateTimeField(default=timezone.now)
    end_time = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='initializing'
    )
    error_message = models.TextField(null=True, blank=True)
    metadata = models.JSONField(default=dict)

    class Meta:
        ordering = ['-start_time']
        indexes = [
            models.Index(fields=['ticket_id', 'agent_name', 'status']),
            models.Index(fields=['recording_id'])
        ]

    def __str__(self):
        return f"{self.recording_id} - {self.ticket_id} ({self.status})"
    

class VideoRecording(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    start_time = models.DateTimeField(auto_now_add=True)
    duration = models.FloatField()
    filepath = models.CharField(max_length=255)
    filesize = models.BigIntegerField()
    resolution = models.CharField(max_length=50)
    fps = models.IntegerField()