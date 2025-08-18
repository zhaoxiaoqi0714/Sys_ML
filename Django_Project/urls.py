"""
URL configuration for Django_Project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from myapp import views
from django.conf.urls.static import static
from django.conf import settings
# URL和函数的对应关系

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', views.home, name='home'),
    path('upload/', views.upload_file_html, name='upload_file_html'),  # 用于单文件上传
    path('upload/upload_data/', views.upload_data, name='upload_files'),  # 用于多文件上传和参数保存
    path('analysis/', views.analysis, name='analysis'),
    path('start_analysis/', views.start_analysis, name='start_analysis'),  # 数据分析
    path('download_result/', views.download_result, name='download_result'),  # 数据下载
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
