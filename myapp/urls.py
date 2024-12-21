from django.urls import path
from .views import select_department, home, knowledge_agent, universal_agent, view_pdf,insight_analysis

#For main

from django.urls import path
from .views import (
    select_department, 
    home, 
    knowledge_agent, 
    universal_agent, 
    view_pdf, 
    insight_analysis,
    register_view,
    login_view,
    logout_view
)

urlpatterns = [
    path('', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    path('department/', select_department, name='select_department'),
    path('home/', home, name='home'),
    path('knowledge_agent/', knowledge_agent, name='knowledge_agent'),
    path('universal_agent/', universal_agent, name='universal_agent'),
    path('view_pdf/<str:pdf_name>/', view_pdf, name='view_pdf'),
    path('insights/', insight_analysis, name='insight_analysis'),
]

# For analysis

# urlpatterns = [
#     path('', insight_analysis, name='insight_analysis'),
# ]





