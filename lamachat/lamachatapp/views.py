from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from lamachatapp.backend import pdflrn,pdfres,chat
# Create your views here.

def index(request):
    return render(request,"index.html")

@csrf_exempt    
def res(request):
    msg = ""
    if request.method == "POST":
        msg = request.POST['msg']
    #return StreamingHttpResponse(pdfres(msg,"data","llama3.2","nomic-embed-text"))
    return HttpResponse(pdfres(msg,"data","llama3.2","nomic-embed-text"))

def pdf(request):
    return HttpResponse(pdflrn("aaa.pdf","data","nomic-embed-text"))


@csrf_exempt
def chatai(request):
    msg = ""
    if request.method == "POST":
        msg = request.POST['msg']
        return StreamingHttpResponse(chat(msg))
    return render(request,"chat.html")
    
    
    
