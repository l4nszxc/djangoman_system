# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('detect/', views.detect_sign, name='detect_sign'),
    path('camera-detect/', views.camera_feed, name='camera_detect'),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('upload/', views.upload_image, name='upload'),
    path('history/', views.history, name='history'),
]