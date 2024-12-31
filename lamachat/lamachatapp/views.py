from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from lamachatapp.backend import pdflrn,pdfres
# Create your views here.

def index(request):
    return render(request,"index.html")

#@csrf_exempt    
def res(request):
    msg = ""
    if request.method == "POST":
        msg = request.POST['msg']
    response = pdfres(msg)
    return HttpResponse(response)

def pdf(request):
    return pdflrn("/content/test.pdf")
    
    
