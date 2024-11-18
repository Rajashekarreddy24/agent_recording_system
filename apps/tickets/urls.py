from django.urls import path, include
from apps.tickets import views
from .views import *
urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('ticket/<str:ticket_id>/', views.view_ticket, name='view_ticket'),
    path('start_recording/<str:ticket_id>/', views.start_recording, name='start_recording'),
    path('stop_recording/<str:ticket_id>/', views.stop_recording, name='stop_recording'),   
    path('ticket/<str:ticket_id>/download-report/', views.download_report, name='download_report'),
    path('time-analysis/<str:ticket_id>/', views.generate_time_analysis, name='generate_time_analysis'),
    path('ticket_list/', views.ticket_list, name='ticket_list'),
    path('tickets/<str:ticket_id>/', views.ticket_detail, name= 'ticket_detail'),
    path('Actions/', views.ActionViewSet.as_view({'get': 'list', 'post': 'create'}), name='ticket_action_list'), 
    path('analyze/<int:pk>/',EnhancedTicketAnalyzer.as_view(), name='analyze_ticket'),  
    path('analyze/', analyze_frame, name='analyze_frame'),  
    path('record/<str:ticket_id>/', RecordActionView.as_view(), name='record_action'),
    path('recording_status/<str:ticket_id>/', RecordingStatusView.as_view(), name='recording_status'), 
    path('simple/recording/', views.record_screen_view, name='recording'),
    path('mss/recording/', views.record_screen, name='mssrecording'),
    path('api/start-recording/<str:ticket_id>/<str:agent_name>/', RecordingViewSet.as_view({'post': 'start_recording'})),
    path('api/stop-recording/<str:ticket_id>/<str:agent_name>/', RecordingViewSet.as_view({'post': 'stop_recording'})),
    path('api/recording-status/<str:ticket_id>/<str:agent_name>/', RecordingViewSet.as_view({'get': 'recording_status'})),

]