from django.db import migrations, models
import uuid


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='ClusteringSession',
            fields=[
                ('id', models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('original_filename', models.CharField(max_length=255, blank=True)),
                ('parameters', models.JSONField(default=dict)),
                ('results', models.JSONField(default=dict, blank=True)),
            ],
        ),
    ]

