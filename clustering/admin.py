from django.contrib import admin
from .models import ClusteringSession


@admin.register(ClusteringSession)
class ClusteringSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'original_filename', 'created_at')
    search_fields = ('id', 'original_filename')
