# story_app/urls.py (Recommended name)

from django.urls import path
from .views import GeneratorView, StoryGalleryView  # Import the new view

urlpatterns = [
    path('', GeneratorView.as_view(), name='home'),
    path('gallery/', StoryGalleryView.as_view(), name='gallery'), # Add this line
]