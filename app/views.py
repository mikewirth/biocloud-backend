from app import app
from flask import render_template,send_file
import tempfile
import cv2
from StringIO import StringIO
import os
import Image

@app.route('/')
def index():
    return render_template("/static/index.html")


@app.route('/render',methods=['POST'])
def render():
    return "Hello World Render"

@app.route('/sobelimg.bmp')
def sobel():
    img_io = StringIO()
    #print os.getcwd()
    lena = cv2.imread('app/static/lena.bmp',0)
    trafolena = cv2.Canny(lena,100,200)
    im = Image.fromarray(trafolena)
    im.save(img_io,'BMP')
    img_io.seek(0)
    return send_file(img_io,mimetype='image/bmp',attachment_filename='does_not_matter.bmp',as_attachment=True)
