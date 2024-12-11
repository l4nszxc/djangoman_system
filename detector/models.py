# models.py
from django.db import models

class RoadSignImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    source = models.CharField(max_length=10, choices=[('camera', 'Camera'), ('upload', 'Upload')], default='upload')
    label = models.CharField(max_length=50, blank=True)
    description = models.TextField(blank=True)
    confidence = models.FloatField(blank=True, null=True)
    detected_at = models.DateTimeField(auto_now_add=True)