from app import app
from flask import render_template,send_file,request
import tempfile
from flask.ext.cors import cross_origin
#import cv2
from StringIO import StringIO
import os
#import Image

@app.route('/')
def index():
    return render_template("/static/index.html")

@app.route('/test')
def test():
    return "Hello Wor"


@app.route('/render',methods=['POST','GET'])
@cross_origin()
def render():
    img_io = StringIO()
    return send_file("lena.bmp",mimetype='image/bmp',attachment_filename='does_not_matter.bmp',as_attachment=True)

@app.route('/sobelimg.bmp')
def sobel():
    img_io = StringIO()
    im = open('lena.bmp','rb')
    #print os.getcwd()
#    lena = cv2.imread('app/static/lena.bmp',0)
 #   trafolena = cv2.Canny(lena,100,200)
 #   im = Image.fromarray(trafolena)
    im.save(img_io,'BMP')
    img_io.seek(0)
    return send_file(img_io,mimetype='image/bmp',attachment_filename='does_not_matter.bmp',as_attachment=True)
