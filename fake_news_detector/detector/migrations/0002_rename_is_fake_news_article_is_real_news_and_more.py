# Generated by Django 4.0 on 2021-12-08 13:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('detector', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='article',
            old_name='is_fake_news',
            new_name='is_real_news',
        ),
        migrations.AddField(
            model_name='article',
            name='subject',
            field=models.CharField(default='Unknown', max_length=200),
            preserve_default=False,
        ),
    ]
