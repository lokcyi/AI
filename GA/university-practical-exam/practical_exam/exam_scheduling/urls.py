from django.urls import path

from . import views

app_name = "exam_scheduling"
urlpatterns = [
    path("", views.index, name="index"),
    path("schedule", views.schedule, name="schedule"),
    path("display", views.display, name="display"),
]
