from django import forms


import os
from django.core.exceptions import ValidationError

class DocumentForm(forms.Form):
    docfile = forms.FileField(label='Select a file')

    def clean_docfile(self):
        file = self.cleaned_data['docfile']
        # Validate file size (max 10 MB)
        if file.size > 10 * 1024 * 1024:
            raise ValidationError('File size must be under 10 MB.')
        # Validate file type (simple check by extension)
        valid_video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        ext = os.path.splitext(file.name)[1].lower()
        if ext not in valid_video_extensions:
            raise ValidationError('Only video files are allowed.')
        return file
