from django.db import models
from django.contrib.auth.models import User

class Department(models.Model):
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class UploadedPDF(models.Model):
    file = models.FileField(upload_to='pdfs/')
    name = models.CharField(max_length=255)
    department = models.ForeignKey(Department, on_delete=models.CASCADE, null=True, default=None)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)  # Allow null temporarily
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.user.username if self.user else 'No User'} - {self.department.name if self.department else 'No Department'})"

class UserVectorStore(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    vector_store_path = models.CharField(max_length=255)

    class Meta:
        unique_together = ('user', 'department')
    
    
    
class QueryInsight(models.Model):
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    topic = models.TextField(null=True, blank=True)  
    frequency = models.IntegerField(default=1)
    last_queried = models.DateTimeField(auto_now=True)

# models.py
class QueryLog(models.Model):
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    query = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Query in {self.department.name} at {self.timestamp}"

    class Meta:
        ordering = ['-timestamp']
