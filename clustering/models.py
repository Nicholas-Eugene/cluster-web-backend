from django.db import models
import uuid


class ClusteringSession(models.Model):
    """Stores uploaded dataset and results for a clustering session."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    original_filename = models.CharField(max_length=255, blank=True)
    parameters = models.JSONField(default=dict)
    results = models.JSONField(default=dict, blank=True)

    def __str__(self) -> str:
        return f"Session {self.id}"
