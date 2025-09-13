from django.conf import settings
from django.urls import reverse
from django.http import HttpResponseRedirect

import os
from django.shortcuts import render

from .models import Document
from .forms import DocumentForm

from .analysis import run_analysis_on_uploaded_video
from .file_transfer import FileTransfer

def result_view(request):

    # List files in the output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),  'myapp', 'static', 'myapp', 'analysis', 'resaults')
    output_files = []
    if os.path.exists(output_dir):
        output_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    # Assume output is served from /output/ (adjust if needed)
    file_url_prefix = '/static/myapp/analysis/resaults/'
    context = {'output_files': output_files, 'file_url_prefix': file_url_prefix}
    return render(request, 'result.html', context)




def my_view(request):
    print("Great! You're using Python 3.6+. If you fail here, use the right version.")
    message = 'Upload as many files as you want!'
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()


            # Clear the output folder before running analysis
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),  'myapp', 'static', 'myapp', 'analysis', 'resaults')
            if os.path.exists(output_dir):
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            import shutil
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')

            # Run analysis on the uploaded file
            uploaded_file_path = newdoc.docfile.path
            run_analysis_on_uploaded_video(uploaded_file_path)
            # Move output files to static/myapp/analysis/resaults
            FileTransfer.move_output_to_static()

            # Remove file from media directory and database after analysis
            try:
                if os.path.exists(uploaded_file_path):
                    os.remove(uploaded_file_path)
            except Exception as e:
                print(f"Error deleting file: {e}")
            newdoc.delete()

            # Redirect to the result page after analysis
            return HttpResponseRedirect(reverse('result-view'))
        else:
            message = 'The form is not valid. Fix the following error:'
    else:
        form = DocumentForm()  # An empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    context = {'documents': documents, 'form': form, 'message': message}
    return render(request, 'list.html', context)

