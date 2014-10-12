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
def dumpArgs(func):
    '''Decorator to print function call details - parameters names and effective values'''
    def wrapper(*func_args, **func_kwargs):
        arg_names = func.func_code.co_varnames[:func.func_code.co_argcount]
        args = func_args[:len(arg_names)]
        defaults = func.func_defaults or ()
        args = args + defaults[len(defaults) - (func.func_code.co_argcount - len(args)):]
        params = zip(arg_names, args)
        args = func_args[len(arg_names):]
        if args: params.append(('args', args))
        if func_kwargs: params.append(('kwargs', func_kwargs))
        print func.func_name + ' (' + ', '.join('%s = %r' % p for p in params) + ' )'
        return func(*func_args, **func_kwargs)
    return wrapper  

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

#Return the crop of an image absed off certain inputs	
def CropTool(img,top=0,bottom=0,left=0,right=0):
    print img.shape
    size=img.shape
    print size[0]

    if top+bottom<size[0] and left+right<size[1]:
        crop=img[top:size[0]-bottom,left:size[1]-right]
        print 'Cropping Image'
    else:
        print 'Croping Border is larger than image size, please re-adjust the cropping dimensions'
        Error=True
        return Error

    return crop
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
def noiseRemoval(img,kSize=3,iterations=1):
    kernel = np.ones((kSize,kSize),np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

#jsut reads from filesys now, change for upload later
def getImg():
    #lena = cv2.imread('app/sampleimages/lena.bmp',0)
    lena=cv2.imread('app/sampleimages/cellCountDataset/dna-0.png',0)
    #print(lena)
    return lena

def removeHoles(img,kSize=3,iterations=1):
    kernel = np.ones((kSize,kSize),np.uint8)
    img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img_closed

#PJ's Edge detection method using the Canny Detector
def edge_detection(img,min_val=15,max_val=80):


    blur=cv2.medianBlur(img,3)
    #testcase = cv2.Sobel(testcase,cv2.CV_64F,1,0,ksize=3)

    cv2.multiply(blur,-1)

    testcase = cv2.Canny(blur,25,80,3)
    testcase2=cv2.Canny(blur,min_val,max_val,3)
    kernel = np.ones((3,3),np.uint8)
    dialation=cv2.dilate(testcase2,kernel,iterations =4)
    erosion=cv2.erode(dialation,kernel,iterations=4)



    result=erosion
    
    return result

@printer
def skeletonize(img):
    #forcibly binarize to ensure idiot proofness
    junk,img = cv2.threshold(img, 127,255,cv2.THRESH_BINARY)
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
        zeros = size - int(np.sum(img)/255)
        if zeros==size:
            done=True
    return skel

@printer
def thresholding(img):
    thresh,img_thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return img_thresh

@printer
def watershed(img) :
    # Identifying the sure background area
    kernel = np.ones((3,3),np.uint8)
    sure_bg = cv2.dilate(img_thresh,kernel,iterations=5)
    # Identifying the sure foreground area
    dist_transform = cv2.distanceTransform(img_thresh,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    # Finding unknown region
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, img_markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    img_markers = img_markers+1
    # Now, mark the region of unknown with zero
    img_markers[unknown==255] = 0
    # Doing the watershedding
    img_markers = cv2.watershed(img_col,img_markers)
    # Return
    return img_markers

@printer
def cellSegmentation(img):
    # Converting the image to greyscale
    img_orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurring the image
    img_blur = gaussianBlur(img_orig_gray,blrSize=21)
    # Theresholding (dynamic)
    img_thresh = thresholding(img_blur)
    # Removing noise
    img_denoised = noiseRemoval(img_thresh,kernalSize=3,iterations=2)
    # Removing the holes
    img_deholed = removeHoles(img_denoised,kernalSize=21,iterations=50)
    # Segmenting the cells
    img_segmented = watershed(img_deholed,img)
    # Returning the segmentated image
    return img_segmented

@printer
def measureBV(img,pixelsize=1):
    skel = skeletonize(img)
    p=pixelsize
    #Area is the sum of the binary image x pixel size^2
    area=np.sum(img)*p**2
    #Length is the sum of the skeleton picels * pixel size
    length=np.sum(skel)*p
    #Average diameter is the Area/length
    diameter=float(area)/length
    return (area,length,diameter)

@printer
def countSegments(img) :
    vals = []
    for px in img.ravel() :
        if px not in vals :
            vals.append(px)
    return len(vals) - 1

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
    elif (method== "thresholding"):
        return thresholding(img)
    elif (method== "watershed"):
        return watershed(img)
    elif (method== "cellSegmentation"):
        return cellSegmentation(img)
    elif (method=='CropTool'):
        return CropTool(img,**parameters)
    elif (method=='edge_detection'):
        return edge_detection(img,**parameters)
    else:
        print("no known transform")
        return img


def analyze(img,method,parameters):
    if method=="vesselWidth":#TODO change to apropriate
        a,l,d= measureBV(img,parameters['pixelSize'])
        return {"diameter":d,'length':l,'area':a}
    if method=="countCells":
        n = countSegments(img)
        return {"number":n}
    return {'error':'unkown analysis'}

#io functions
def get_average(reportlist):
    avg = {}
    print(reportlist)
    for feature in reportlist[0]['data'].keys():
        avg[feature]= sum([x['data'][feature] for x in reportlist])/len(reportlist)
#    try:
#        for feature in reportlist[0]['data'].keys():
#            avg[feature]= sum([x['data'][feature] for x in reportlist])/len(reportlist)
#    except Exception as e:
#        print("there was an error, most probably a feature is not easily averaged")
#        print(str(e))
    return avg

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
    imglist = imglist[:8]
    print imglist
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
                   analyze_list.append({"method":action["method"],'data':analyze(img,method=action["method"],parameters=action["parameters"]),'step':i,'image':imgid})
               elif actiontype == "show":
                   showlist.append({imgid+"onstep"+str(i):img})
               elif actiontype == "zip":
                   ziplist.append(imgid+"onstep"+str(i))
        except Exception as e: 
            print(str(e))

    #if showlist/ziplist not empty send jsonified images/create onetime download link

    #return jsonify(results=analyze_list,average=get_average(analyze_list))
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
