# Generated by Django 5.1.2 on 2024-12-07 18:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detector', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='roadsignimage',
            name='source',
            field=models.CharField(choices=[('camera', 'Camera'), ('upload', 'Upload')], default='upload', max_length=10),
        ),
    ]
