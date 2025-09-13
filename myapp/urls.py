from django.urls import path
from .views import my_view, result_view

urlpatterns = [
    path('', my_view, name='my-view'),
    path('result/', result_view, name='result-view'),
]
