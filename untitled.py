import sys
import os
import os.path as osp
from flask import Flask,request,render_template,request,redirect, url_for, send_from_directory
from werkzeug import secure_filename
import argparse
import _init_modules
import align_face
import extract_feature
import numpy as np
from os import listdir
from os.path import isfile, join
import time
from os.path import basename

this_dir = osp.dirname(__file__)
print this_dir
modelDir = osp.join(this_dir, 'mxnet-face','model')
print modelDir
dlib_model_dir = osp.join(modelDir, 'dlib')


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/home/att/PycharmProjects/untitled/uploads/'
app.config['faceDB'] = '/home/att/PycharmProjects/untitled/faceDB/'
app.config['ALLOWED_EXTENSIONS'] = set(['png','jpg','jpeg'])
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024 # 15M
app.config['ALIEN_FOLDER'] = '/home/att/PycharmProjects/untitled/mxnet-face/alienresult/'
app.config['THRESHOLD'] = 0.4
# Port
app.config['OUTSIDE_PORT_NUMBER'] = 8088


def faceAlien():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputDir', type=str, help="Input image directory.",
                        default='/home/att/PycharmProjects/untitled/uploads/')
    parser.add_argument('--opencv-det', action='store_true', default=False,
                        help='True means using opencv model for face detection(because sometimes dlib'
                             'face detection will failed')
    parser.add_argument('--opencv-model', type=str, default='../model/opencv/cascade.xml',
                        help="Path to dlib's face predictor.")
    parser.add_argument('--only-crop', action='store_true', default=False,
                        help='True : means we only use face detection and crop the face area\n'
                             'False : both face detection and then do face alignment')
    parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                        default=os.path.join(dlib_model_dir, "shape_predictor_68_face_landmarks.dat"))

    parser.add_argument('--landmarks', type=str,
                        choices=['outerEyesAndNose', 'innerEyesAndBottomLip'],
                        help='The landmarks to align to.', default='innerEyesAndBottomLip')
    parser.add_argument(
        '--outputDir', type=str, help="Output directory of aligned images.",
        default='/home/att/PycharmProjects/untitled/mxnet-face/alienresult/')
    parser.add_argument('--pad', type=float, nargs='+',
                        help="pad (left,top,right,bottom) for face detection region")
    parser.add_argument('--ts', type=float,
                        help="translation(,ts) the proportion position of eyes downward so that..."
                             " we can reserve more area of forehead",
                        default='0.1')
    parser.add_argument('--size', type=int, help="Default image size.",
                        default=128)
    parser.add_argument('--ext', type=str, help="Default image extension.",
                        default='jpg')
    parser.add_argument('--fallbackLfw', type=str,
                        help="If alignment doesn't work, fallback to copying the deep funneled version from this directory..")
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    return args

def loadDB():
    onlyfiles = [f for f in listdir(app.config['faceDB']) if isfile(join(app.config['faceDB'], f))]
    for np_name in onlyfiles:
        db[np_name] = np.load(os.path.join(app.config['faceDB'],np_name))
    # print onlyfiles
    # print db

def findRightPerson(queryFeature):
    start_time = time.time()
    dist = sys.float_info.min
    name = ''
    output_string= ''
    for key, value in db.iteritems():
        dist_tmp = extract_feature.cal_dist(queryFeature[0],value[0])
        #print key + '\t'+ str(dist_tmp)
        str_tmp= key + '\t'+ str(dist_tmp) + '\n'
        output_string += str_tmp
        if dist_tmp>dist:
            dist = dist_tmp
            name = key
    #print output_string
    data_loading_time = time.time() - start_time
    print('Time For Find Person: ' + str(data_loading_time) + '\n')

    return name , str(dist)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



@app.route('/')
def index():
    #return render_template('index.html',input ='jj',image_file=url_for('output_file', filename='face11.png'))
    #return render_template('index.html',input ='jj',image_file='sample3/face11.png')
    return render_template('index.html')






@app.route('/upload',methods=['POST'])
def upload():
    file = request.files['file']
    os.path.abspath(__file__)
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        #process image
        absfilename = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_extension = os.path.splitext(filename)[1]
        args.ext = file_extension
        print args.ext

        align_face.alignMain_xiaobo(args,ailen,absfilename,filename)
        aligned_face_path = os.path.join(app.config['ALIEN_FOLDER'],filename)

        if not os.path.isfile(aligned_face_path): return 'Unable to align the face, bad picture.'

        print 'numpy: ' + filename
        output = extract_feature.extractFeature(aligned_face_path, symbol, model_args, model_auxs)
        #np.save(os.path.join(app.config['faceDB'],filename),output)
        name,score = findRightPerson(output)
        #return  'ok'
        if score<app.config['THRESHOLD']:
            name = '-1'

        print 'score ' + str(score) +'   '+name

        print '@@@' + aligned_face_path

        return '['+ name +']'
        #return render_template('ret.html', output=name+score,
        #                       image_file=url_for('output_file_upload', filename=filename),db_imge=url_for('output_file_aligned', filename=name[:-4]))


@app.route('/outputUpload/<filename>')
def output_file_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/outputAligned/<filename>')
def output_file_aligned(filename):
    return send_from_directory(app.config['ALIEN_FOLDER'],filename)

if __name__ == '__main__':

    db = {}
    args = faceAlien();
    ailen = align_face.getAlign(args)
    symbol, model_args, model_auxs = extract_feature.loadFaceRecognitionModel();
    loadDB()
    app.run(
        host = "0.0.0.0",
        port = int(app.config['OUTSIDE_PORT_NUMBER']),
        debug = False
    )
