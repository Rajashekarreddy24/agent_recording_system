from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import RecordingViewSet

router = DefaultRouter()
router.register(r'recordings', RecordingViewSet, basename='recording')

urlpatterns = [
    path('', include(router.urls)),
    path('api-auth/', include('rest_framework.urls')),  # Add this for authentication
    path('agent/start-recording/<str:ticket_id>/<str:agent_name>/', 
         RecordingViewSet.as_view({'post': 'start_recording'}), 
         name='start-recording'),
    path('agent/stop-recording/<str:ticket_id>/<str:agent_name>/', 
         RecordingViewSet.as_view({'post': 'stop_recording'}), 
         name='stop-recording'),
]
