from app import app
from flask import render_template,send_file,request
import tempfile
from flask.ext.cors import cross_origin
import cv2
from StringIO import StringIO
import os
from PIL import Image

#hack to get around 404s
@app.route("/styles/vendor-dc9008c5.css")
def hack1():
    return app.send_static_file("/styles/vendor-dc9008c5.css")

@app.route("/styles/main-e078d42e.css")
def hack2():
    return app.send_static_file("/styles/main-e078d42e.css")

@app.route('/')
def index():
    #return render_template("/static/index.html")
    return app.send_static_file("index.html")

@app.route('/test')
def test():
    return "Hello Wor"

def gaussianBlur(img,blrSize=3):
    return cv2.GaussianBlur(img,(blrSize,blrSize),0)
	

#treshhold the image after seprating it into subimages and counting the edges as a heuristic
def filterBackgroundNoise(img,subimageDimX=30,subimageDimY=30):
    for ix in range(img.shape[0]/subimageDimX) :
        for iy in range(img.shape[1]/subimageDimY) :
            # Extracting sub image
            xStartPixel = ix*subimageDimX
            yStartPixel = iy*subimageDimY
            xEndPixel = ix*subimageDimX + subimageDimX
            yEndPixel = iy*subimageDimY + subimageDimY
            subImage = blr[xStartPixel:xEndPixel,yStartPixel:yEndPixel]
            # Thresholding using Otsu
            varianceThreshold = 10
            if np.var(subImage.ravel()) > varianceThreshold :
            	thresh,thresholdedImg = cv2.threshold(subImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            else :
            	thresholdedImg = 0*subImage
            # Adding the subimage the result image
            resultImage[xStartPixel:xEndPixel,yStartPixel:yEndPixel] = thresholdedImg    
    return resultImage

def noiseRemoval(img,kSize=3):
    kernel = np.ones((kSize,kSize),np.uint8)
    return cv2.morphologyEx(resultImage, cv2.MORPH_OPEN, kernel)

#jsut reads from filesys now, change for upload later
def getImg():
    lena = cv2.imread('app/lena512.bmp',0)
    return lena

#use the specified transofrm on the image and return the result
def transform(img,method,params):
if (method == "gaussianBlur"):
    return gaussianBlur(img,**params)
elif (method =="noiseRemoval"):
    return noiseRemoval(img,**params)
elif (method== "filterBackgroundNoise"):
    return filterBackgroundNoise(img,**params)

@app.route('/analyze',methods=['POST','GET'])
def analyze():
	json = request.get_json(force=True) 	

@app.route('/render',methods=['POST','GET'])
@cross_origin()
def render():
    orig=getImg()
    #get the image, for now just use one from the server
    img_io = StringIO()
    #stringio for return of image without storing on disk
    json = request.get_json(force=True)
    #json request specifiyng the order of operations and at what point we return the result
    intermediate=lena
    for i in range(json["showUntil"]:
        intermediate=transform(intermediate,**json["transformations"][i])
    result = Image.fromarray(intermediate)
    result.save(img_io,'BMP')
    img_io.seek(0)
    return send_file(img_io,mimetype='image/bmp',attachment_filename='does_not_matter.bmp',as_attachment=True)

def analyze(img,transformations,analysis):
    pass

@app.route('/batch')
def batch():
    #get list from json
    #get transformations from json
    #get analysis to be done from json


@app.route('/sobelimg.bmp')
def sobel():
    img_io = StringIO()
    im = open('lena.bmp','rb')
    #print os.getcwd()
#    lena = cv2.imread('app/static/lena.bmp',0)
    trafolena = cv2.Canny(lena,100,200)
    im = Image.fromarray(trafolena)
    im.save(img_io,'BMP')
    img_io.seek(0)
    return send_file(img_io,mimetype='image/bmp',attachment_filename='does_not_matter.bmp',as_attachment=True)
