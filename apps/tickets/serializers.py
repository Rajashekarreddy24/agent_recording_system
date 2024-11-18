from rest_framework import serializers
from .models import Ticket, Activity, Prompt
from apps.recordings.models import Recording
from rest_framework import serializers
from .models import Pattern
from rest_framework import serializers
from .models import TicketInfo, Action

class TicketSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ticket
        fields = '__all__'

class ActivitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Activity
        fields = '__all__'

class PromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prompt
        fields = '__all__'

class TicketInfoSerializer(serializers.ModelSerializer):
    class Meta:
        model = TicketInfo
        fields = '__all__'

class ActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Action
        fields = '__all__'

class PatternSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pattern
        fields = '__all__'



