import os
import csv
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'KAMCO.settings')
django.setup()

from myapp.models import QueryInsight  # Adjust the import as necessary

# Define the CSV file name
csv_file_path = 'myapp_queryinsight.csv'

# Open the CSV file for writing
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header
    writer.writerow([field.name for field in QueryInsight._meta.fields])

    # Write the data rows
    for obj in QueryInsight.objects.all():
        writer.writerow([getattr(obj, field.name) for field in QueryInsight._meta.fields])

print(f'Successfully exported to {csv_file_path}')
