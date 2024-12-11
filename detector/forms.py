from django import forms
from .models import RoadSignImage

class RoadSignForm(forms.ModelForm):
    class Meta:
        model = RoadSignImage
        fields = ['image']
