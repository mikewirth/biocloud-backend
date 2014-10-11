from app import app
from flask import render_template,send_file,request,jsonify
import tempfile
from flask.ext.cors import cross_origin
import cv2
from StringIO import StringIO
import os
from PIL import Image
import numpy as np
class printer(object):
    def __init__(self, f):
	"""
	If there are no decorator arguments, the function
	to be decorated is passed to the constructor.
	"""
	self.f = f

    def __call__(self, *args):
        """
        The __call__ method is not called until the
        decorated function is called.
        """
        print "Inside "+self.f.__name__
        return self.f(*args)
#hack to get around 404s, might not be necessary
@app.route("/styles/vendor-dc9008c5.css")
def hack1():
    return app.send_static_file("/styles/vendor-dc9008c5.css")

@app.route("/styles/main-e078d42e.css")
def hack2():
    return app.send_static_file("/styles/main-e078d42e.css")
#endhack
#helper functions
@printer
def gaussianBlur(img,blrSize=3):
    return cv2.GaussianBlur(img,(blrSize,blrSize),0)
	

#treshhold the image after seprating it into subimages and counting the edges as a heuristic
@printer
def filterBackgroundNoise(img,subimageSize=30,varianceThreshold=10):
    print("in resultImage")
    resultImage = 0*img 
    for ix in range(img.shape[0]/subimageSize) :
        for iy in range(img.shape[1]/subimageSize) :
            # Extracting sub image
            xStartPixel = ix*subimageSize
            yStartPixel = iy*subimageSize
            xEndPixel = ix*subimageSize + subimageSize
            yEndPixel = iy*subimageSize + subimageSize
            subImage = img[xStartPixel:xEndPixel,yStartPixel:yEndPixel]
            # Thresholding using Otsu
            if np.var(subImage.ravel()) > varianceThreshold :
            	thresh,thresholdedImg = cv2.threshold(subImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            else :
            	thresholdedImg = 0*subImage
            # Adding the subimage the result image
            resultImage[xStartPixel:xEndPixel,yStartPixel:yEndPixel] = thresholdedImg    
    return resultImage

@printer
def noiseRemoval(img,kSize=3):
    kernel = np.ones((kSize,kSize),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#jsut reads from filesys now, change for upload later
def getImg():
    #lena = cv2.imread('app/sampleimages/lena.bmp',0)
    lena=cv2.imread('app/sampleimages/Exp1/Images/P00-1_00D1.tif',0)
    #print(lena)
    return lena

def removeHoles(img,kSize=3):
    kernel = np.ones((kSize,kSize),np.uint8)
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img_closed


@printer
def skeletonize(img):
    originalimg=img
    size=np.size(img)
    #Create Skeleton Array
    skelzeros=np.zeros(img.shape,np.uint8)
    skel=skelzeros
    #Set a structural cross element for the morphilogical opening
    element=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    #Condition for while loop
    done=False
    #Itteratively Open unit. This will delete thin lines. By subtracting the result from the original binary 
    #image we get that skeletonised segment. We then keep iterating until the entire skeleton is revealed
    while(not done):
        eroded=cv2.erode(img,element)
        temp=cv2.dilate(eroded,element)
        temp2=cv2.subtract(img,temp)
        skel=cv2.bitwise_or(skel,temp2)
        img=eroded.copy()
        #When to stop the itteration
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done=True
    return skel

@printer
def measureBV(img,pixelsize=5):
    skel = skeletonize(img)
    p=pixelsize
    #Area is the sum of the binary image x pixel size^2
    area=np.sum(img)*p**2
    #Length is the sum of the skeleton picels * pixel size
    length=np.sum(skel)*p
    #Average diameter is the Area/length
    diameter=float(area)/length
    return (area,length,diameter)


#use the specified transofrm on the image and return the result
def transform(img,method,parameters=None):
    if (method == "gaussianBlur"):
        return gaussianBlur(img,**parameters)
    elif (method =="noiseRemoval"):
        return noiseRemoval(img,**parameters)
    elif (method== "filterBackgroundNoise"):
        return filterBackgroundNoise(img,**parameters)
    elif (method== "removeHoles"):
        return removeHoles(img,**parameters)
    elif (method== "skeletonize"):
        return skeletonize(img)
    else:
        print("no known transform")
        return img




def analyze(img,method,parameters):
    if method=="vesselWidth":#TODO change to apropriate
        a,l,d= measureBV(img,parameters['pixelSize'])
        return {"diameter":d,'length':l,'area':a}
    return {'error':'unkown analysis'}

#io functions
def get_average(reportlist,analysis):
    avg = {}
    try:
        for feature in analysis:
            avg[feature]= sum([x[feature] for x in reportlist])/len(reportlist)
    except Exception as e:
        print("there was an error, most probably a feature is not easily averaged")
        print(str(e))
    return {"average":avg}

def read_Img_from_HDD(img):#FIXME does not work properly
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
        report_list=[]
        #json request specifiyng the order of operations and at what point we return the result
        intermediate=orig
        print(json)
	actionslist = json['actions']
        for action in actionslist:
            actiontype=action['type']
            if actiontype == "transformation":
                intermediate=transform(intermediate,action['method'],action['parameters'])
            if actiontype == "analysis":
                report_list.append(analyze(intermediate,action['method'],action['parameters']))
                print(report_list)
                #TODO figure out a way to serve this as jsonalongside with the image
        result = Image.fromarray(intermediate)
        result.save(img_io,'BMP')
        img_io.seek(0)
	return send_file(img_io,mimetype='image/bmp',attachment_filename='does_not_matter.bmp',as_attachment=True)
    except Exception as e:
        print("Error, most likely we cannot find the image under sampleimages")
        print(str(e))
	return app.send_static_file("index.html")


@app.route('/imglist')
@cross_origin()
def imglist():
    return jsonify(
    results=({'name':"Lena",'id':1,'images':['static/lena.bmp']},
	    {'name':'mice','id':2,'images':['static/sampleimages/Exp1/Images/'+ x for x in os.listdir('app/sampleimages/Exp1/Images')]},
	    {'name':'dinosaurs','id':3,'images':['static/sampleimages/Exp2/Images/'+ x for x in os.listdir('app/sampleimages/Exp1/Images')]		}
)
    )

@app.route('/batch',methods=['POST','GET'])
@cross_origin()
def batch():
    json = request.get_json(force=True)
    print(json)
    imglist = ['/Exp1/Images/'+ x for x in os.listdir('app/sampleimages/Exp1/Images')]#json["imagelist"]
    actionslist = json["actions"]
    ziplist=[]
    showlist=[]
    analyze_list=[]
    for imgid in imglist:
        try:
           img = read_Img_from_HDD(imgid)
           for i,action in zip(range(len(actionslist)),actionslist):
               actiontype=action["type"]
               if actiontype == "transformation":
                   img=transform(img,method=action["method"],parameters=action["parameters"])
               elif actiontype == "analysis":
                   analyze_list.append({action["method"]+'_step'+str(i):analyze(img,method=action["method"],parameters=action["parameters"])})
               elif actiontype == "show":
                   showlist.append({imgid+"onstep"+str(i):img})
               elif actiontype == "zip":
                   ziplist.append(imgid+"onstep"+str(i))
        except Exception as e: 
            print(str(e))

    #if showlist/ziplist not empty send jsonified images/create onetime download link
    return jsonify(results=analyze_list)


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
