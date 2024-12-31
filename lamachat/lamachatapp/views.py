from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

def index(request):
    return render(request,"index.html")

#@csrf_exempt    
def res(request):
    msg = ""
    if request.method == "POST":
        msg = request.POST['msg']
    return HttpResponse(msg)
