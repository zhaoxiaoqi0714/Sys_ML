import os
import json
import logging
import threading
from django.http import FileResponse
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_protect, csrf_exempt
from django.utils.text import get_valid_filename
from django.conf import settings
from urllib.parse import unquote
from myapp.Sys_ML_main import run_analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create your views here.
def home(request):
    return render(request, 'home.html')  # Render the home page template

def upload_file_html(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_path = os.path.join('upload', uploaded_file.name)  # File save path
        file_name = default_storage.save(file_path, uploaded_file)  # Save the file
        file_url = default_storage.url(file_name)  # Get the file access URL
        return JsonResponse({'message': 'File upload successful!', 'file_url': file_url})
    return render(request, 'upload.html')  # Render the upload page

@csrf_protect
def upload_data(request):
    if request.method == 'POST':
        project_name = request.POST.get('project-name')
        if not project_name:
            return JsonResponse({'message': 'Project name cannot be empty', 'success': False})

        project_dir = os.path.join(settings.MEDIA_ROOT, project_name)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)

        # Save files
        matrix_file = request.FILES.get('matrix-file')
        label_file = request.FILES.get('label-file')
        save_file(matrix_file, project_dir, matrix_file.name)
        save_file(label_file, project_dir, label_file.name)
        # Check if test data is provided
        has_test_data = 'HadTest' in request.POST
        if has_test_data:
            test_matrix_file = request.FILES.get('test-matrix-file')
            test_label_file = request.FILES.get('test-label-file')
            save_file(test_matrix_file, project_dir, test_matrix_file.name)
            save_file(test_label_file, project_dir, test_label_file.name)

        # Save parameters as a JSON file
        form_data = {
            'project_name': project_dir,
            'HadLabel': 'HadLabel' in request.POST,
            'HadTest': has_test_data,
            'Recommend': 'Recommend' in request.POST,
            'recommendOption': request.POST.get('recommendOption', 'all'),
            'missingValueMethod': request.POST.getlist('missingValueMethod'),
            'normalizationMethod': request.POST.getlist('normalizationMethod'),
            'MLAnalysisMLAnalysis': 'MLAnalysis' in request.POST,
            'SurvivalAnalysis': 'SurvivalAnalysis' in request.POST,
            'LoadUni': 'LoadUni' in request.POST,
            'Ensemble': 'Ensemble' in request.POST,
            'Imbalance': 'Imbalance' in request.POST,
            'ML_Plotting': request.POST.get('ML_Plotting', 'off') == 'on',
            'Unsup_analysis': 'Unsup_analysis' in request.POST,
            'CV': int(request.POST.get('CV', 10)),
            'SA_cofactor': 'SA_cofactor' in request.POST,
            'LoadSAUni': 'LoadSAUni' in request.POST,
            'SA_Plotting': 'SA_Plotting' in request.POST,
            'SA_cofactor_list': request.POST.get('SA_cofactor_list', ''),
            'ML_Methods': request.POST.getlist('ML_Methods')
        }

        json_file_path = os.path.join(project_dir, 'project_params.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(form_data, json_file, indent=4)

        # Redirect to analysis.html and pass project_dir
        return redirect(f'/analysis/?project_dir={project_dir}')
    else:
        return JsonResponse({'message': 'Only POST requests are supported', 'success': False})

def save_file(file, file_dir, file_name):
    if file:
        file_path = os.path.join(file_dir, get_valid_filename(file_name))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
        with open(file_path, 'wb') as f:
            f.write(file.read())

def analysis(request):
    # Get the URL parameter project_dir
    project_dir = request.GET.get('project_dir', '')

    # Assume the analysis is completed, and render the analysis.html template
    return render(request, 'analysis.html', {'project_dir': project_dir})

@csrf_exempt
def start_analysis(request):
    if request.method == 'POST':
        try:
            # Parse JSON data from the request body
            data = json.loads(request.body)
            project_dir = data.get('project_dir')
            if not project_dir or not os.path.exists(project_dir):
                return JsonResponse({'message': 'Invalid project path', 'success': False})

            # Run the analysis and generate the result file
            result_file = run_analysis(project_dir)
            if not result_file or not os.path.exists(result_file):
                return JsonResponse({'message': 'Result file not generated', 'success': False})

            # Return a success response with the download link
            return JsonResponse({
                'message': 'Data analysis started',
                'success': True,
                'download_url': f'/download_result/?file_path={result_file}'  # Provide the download link
            })
        except Exception as e:
            logger.error(f'Failed to start data analysis: {str(e)}')
            return JsonResponse({'message': f'Failed to start data analysis: {str(e)}', 'success': False})
    else:
        return JsonResponse({'message': 'Only POST requests are supported', 'success': False})

def download_result(request):
    file_path = request.GET.get('file_path')  # Get the file path
    if not file_path or not os.path.exists(file_path):
        return JsonResponse({'message': 'File does not exist', 'success': False})

    # Provide the file for download
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))

def serve_file(file_path):
    """
    A generic function to serve files for download
    """
    try:
        with open(file_path, 'rb') as file:
            response = FileResponse(file)
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    except Exception as e:
        logger.error(f'File download failed: {str(e)}')
        return JsonResponse({'message': 'File download failed', 'success': False})