# Generated by Django 5.1.2 on 2024-12-07 18:50

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detector', '0002_roadsignimage_source'),
    ]

    operations = [
        migrations.AddField(
            model_name='roadsignimage',
            name='confidence',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='roadsignimage',
            name='description',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='roadsignimage',
            name='detected_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='roadsignimage',
            name='label',
            field=models.CharField(blank=True, max_length=50),
        ),
    ]
