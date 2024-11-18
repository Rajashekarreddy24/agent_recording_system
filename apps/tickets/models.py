from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
class Ticket(models.Model):
    ticket_id = models.CharField(max_length=255, primary_key=True)
    status = models.CharField(max_length=100)
    description = models.TextField()
    category = models.CharField(max_length=100)
    tags = models.CharField(max_length=255)
    priority = models.CharField(max_length=50)
    start_time = models.DateTimeField()
    last_updated = models.DateTimeField(auto_now=True)
    resolution_time = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(blank=True, null=True)
    automated_prompts = models.IntegerField(default=0)
    integration_status = models.CharField(max_length=100, blank=True, null=True)

class Activity(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    application = models.CharField(max_length=100)
    action = models.CharField(max_length=100)
    notes = models.TextField(blank=True, null=True)
    duration = models.IntegerField(default=0)
    category = models.CharField(max_length=100, blank=True, null=True)
    automated_flag = models.BooleanField(default=False)

    
    @staticmethod
    def get_activity_report(ticket_id, start_date, end_date):
        return Activity.objects.filter(
            ticket__ticket_id=ticket_id,
            timestamp__range=(start_date, end_date)
        ).order_by('-timestamp')

    @staticmethod
    def get_time_analysis(ticket_id):
        data = Activity.objects.filter(ticket__ticket_id=ticket_id).aggregate(
            total_time=models.Sum('duration'),
            avg_duration=models.Avg('duration'),
            activity_count=models.Count('id')
        )
        return data


class Prompt(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    prompt_type = models.CharField(max_length=100)
    response = models.TextField()
    response_time = models.DateTimeField(blank=True, null=True)

class TicketInfo(models.Model):
    ticket_id = models.ForeignKey(Ticket, on_delete=models.CASCADE, related_name='ticket_infos', to_field='ticket_id')
    category = models.CharField(max_length=50)
    description = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    priority = models.CharField(max_length=20)

    def __str__(self):
        return f"Info for Ticket {self.ticket.ticket_id}"

class Action(models.Model):
    action_type = models.CharField(max_length=50)
    target = models.CharField(max_length=100)
    parameters = models.JSONField()
    timestamp = models.DateTimeField(auto_now_add=True)
    screen_location = models.JSONField()  # Store as dict for (x, y) coordinates


class Pattern(models.Model):
    category = models.CharField(max_length=100)
    pattern = models.TextField()

    def __str__(self):
        return f"{self.category}: {self.pattern}"