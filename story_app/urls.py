from django.urls import path
from .views import GeneratorView

# This is where you map URLs to Views.
# The `name` parameter is important as it's used by `{% url '...' %}` in templates.
urlpatterns = [
    # The path '' means the root of your app.
    # We've named it 'home' to match your index.html form action.
    path('', GeneratorView.as_view(), name='home'),
]
