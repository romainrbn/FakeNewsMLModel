from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('input', views.inputView, name='input'),
    path('result', views.result, name='result'),
    path('all', views.all, name='all'),
]