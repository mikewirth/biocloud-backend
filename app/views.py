from app import app
from flask import render_template,send_file,request,jsonify
import tempfile
from flask.ext.cors import cross_origin
import cv2
from StringIO import StringIO
import os
from PIL import Image
import numpy as np

#hack to get around 404s, might not be necessary
@app.route("/styles/vendor-dc9008c5.css")
def hack1():
    return app.send_static_file("/styles/vendor-dc9008c5.css")

@app.route("/styles/main-e078d42e.css")
def hack2():
    return app.send_static_file("/styles/main-e078d42e.css")
#endhack
#helper functions
def gaussianBlur(img,blrSize=3):
    return cv2.GaussianBlur(img,(blrSize,blrSize),0)
	

#treshhold the image after seprating it into subimages and counting the edges as a heuristic
def filterBackgroundNoise(img,subimageSize=30,varianceTreshhold=10):
    for ix in range(img.shape[0]/subimageSize) :
        for iy in range(img.shape[1]/subimageSize) :
            # Extracting sub image
            xStartPixel = ix*subimageSize
            yStartPixel = iy*subimageSize
            xEndPixel = ix*subimageSize + subimageSize
            yEndPixel = iy*subimageSize + subimageSize
            subImage = blr[xStartPixel:xEndPixel,yStartPixel:yEndPixel]
            # Thresholding using Otsu
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
    lena = cv2.imread('app/sampleimages/lena512.bmp',0)
    return lena

#use the specified transofrm on the image and return the result
def transform(img,method,parameters=None):
    if (method == "gaussianBlur"):
        return gaussianBlur(img,**parameters)
    elif (method =="noiseRemoval"):
        return noiseRemoval(img,**parameters)
    elif (method== "filterBackgroundNoise"):
        return filterBackgroundNoise(img,**parameters)

#analysis 
def get_feature(img,featID):
    if (featID=="vesselDiameter"):
        return vesselDiameter(img)
    #elif (featID==):

def vesselDiameter(img):
    pass#code for vessel estimation


def analyze(img,transformations,analysis):
    feature_report ={}
    for t in transformations:
        img = transform(img,**t)
    for a in analysis:
        feature_report[a]=get_feature(img,a)

#io functions
def get_average(reportlist,analysis):
    avg = {}
    try:
        for feature in analysis:
            avg[feature]= sum([x[feature] for x in reportlist])/len(reportlist)
    except:
        print("there was an error, most probably a feature is not easily averaged")
    return {"average":avg}

def read_Img_from_HDD(img):
    return cv2.imread("app/sampleimages/"+img)

#routes
@app.route('/')
def index():
    #return render_template("/static/index.html")
    return app.send_static_file("index.html")

@app.route('/test')
def test():
    return "Hello Wor"


@app.route('/render',methods=['POST','GET'])
@cross_origin()
def preview():
    try:
        orig=getImg()
        #get the image, for now just use one from the server
        img_io = StringIO()
        #stringio for return of image without storing on disk
        json = request.get_json(force=True)
        #json request specifiyng the order of operations and at what point we return the result
        intermediate=orig
        for i in range(json["showUntil"]):
            intermediate=transform(intermediate,**json["transformations"][i])
        result = Image.fromarray(intermediate)
        result.save(img_io,'BMP')
        img_io.seek(0)
    except:
        print("Error, most likely we cannot find the image under sampleimages")
    return send_file(img_io,mimetype='image/bmp',attachment_filename='does_not_matter.bmp',as_attachment=True)


@app.route('/batch')
def batch():
    json = request.get_json(force=True)
    imglist = json["imagelist"]
    transformations = json["transformations"]
    analysis = json["analysis"]
    reportlist = [analyze(read_Img_from_HDD(img),transformations,analysis) for img in imglist]
    reportlist = [get_average(reportlist,analysis)] + reportlist
    return jsonify(results=reportlist)


"""@app.route('/sobelimg.bmp')
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
    """
