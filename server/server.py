
#!/usr/bin/env python

import tornado.ioloop
import tornado.web

import hashlib

import uuid, json, base64
import numpy as np
import sys, os
import os.path

# Opencv, skimage
import dlib
import numpy as np
import pickle


#### Use this to configure the service

parameters = { 
    'port': 5000, 
    'enable_paretopoint': True, 
} 

Paretopoint = "./result/Paretopoint.json"

class UploadModelHandler(tornado.web.RequestHandler): 
    
    """ 
    Lets the client upload an model. Returns a token that corresponds to the uploaded 
    file. 
    """ 
    def post(self): 

        print('---------------------') 
        print("Got an upload request") 

        email_address = self.get_argument('name')
        student_id = self.get_argument('student_id')
        pseudonym = self.get_argument('pseudonym')
        
        if email_address == '':
            login_response = "{'error': true, 'msg': 'Please enter your name.'}"
        elif student_id == '':
            login_response = "{'error': true, 'msg': 'Please enter your student id.'}"
        else:
            login_response = "{'error': true, 'msg': 'Thank You, checking the model'}"

        self.write(json.dumps(login_response))


        # now get the image
        image_bytes = self.request.body 

        folder = "./uploads/%s_%s"% (email_address, student_id)
        trial = 0
        if not os.path.exists(folder):
            os.mkdir(folder)
        else:
            for f in os.listdir(folder):
                id = int(f.)
                if id > trial:
                    trial = id

        model_name = "./uploads/model_%d.py" % trial
        main_name = "./uploads/main_%d.py" % trial
        checkpoint_name = "./uploads/checkpoint_%d.pt" % trial

        with open(model_name, 'wt+') as fp: 
            fp.write(image_bytes) 

        with open(main_name, 'wt+') as fp: 
            fp.write(image_bytes) 

        with open(checkpoint_name, 'wb+') as fp: 
            fp.write(image_bytes) 
 
        print(".. saving the uploads done") 
 
        # now call the shell to run it to make sure it's working         
         
        # updates the global paretopoint and save it
        paretopoint = json.load(Paretopoint)
        paretopoint[pseudonym + ":" + str(trial)] = [perplexity, time_ratio]
        json.dump(paretopoint, Paretopoint)


        self.set_header("Content-Type", "application/json") 
        ret = [{}] 
        ret[0]['token'] = str(fname) 
 
        self.write(json.dumps(ret)) 
        return 
 
class ParetopointHandler(tornado.web.RequestHandler): 

    def post(self): 
        print('-----------------------') 
        print('Request to change emotion') 
 
        jsonBytes = self.request.body 
        reqDict = json.loads(jsonBytes) 
 
        uuidImage = reqDict['image'] 
        uuidSelection = reqDict['selection'] 
        destEmotion = reqDict['emotion'] 
 
        md5Image = None 
        md5Selection = None 
 
        with open('./uploads/%s.jpg' % uuidImage, 'rb') as fp: 
            md5Image = hashlib.md5(fp.read()).hexdigest() 
 
        with open('./uploads/%s.jpg' % uuidSelection, 'rb') as fp: 
            md5Selection = hashlib.md5(fp.read()).hexdigest() 
 
        cachefile = "./cache/%s-%s-%s.jpg" % (md5Image, md5Selection, destEmotion) 
 
        if not os.path.exists(cachefile): 
            print('     .. calculating') 
 
            fileImage =  "./uploads/%s.jpg" % uuidImage 
            fileSelection = "./uploads/%s.jpg" % uuidSelection 
            image = cv2.imread(fileImage) 
            selection = cv2.imread(fileSelection, 0) 
            (rows, cols, channels) = image.shape 
 
            output = emotion.change_emotion(fileImage, fileSelection, destEmotion) 
            cv2.imwrite(cachefile, output) 
            print("      .. done calculating") 







        # updates the global paretopoint and save it
        paretopoint = json.load(Paretopoint)
        paretopoint[pseudonym + ":" + str(trial)] = [perplexity, time_ratio]
        json.dump(paretopoint, Paretopoint)


        self.set_header("Content-Type", "application/json") 
        ret = [{}] 
        ret[0]['tweaked'] = '' 
 
        with open(cachefile, 'rb') as fp: 
            ret[0]['tweaked'] = base64.b64encode(fp.read()) 
 
        self.write(json.dumps(ret)) 
        print(' .. done') 
        os.remove(cachefile) 
        return 

def make_app():
    
    routes = []
    routes.append( (r'/upload', UploadModelHandler) )

    if parameters['enable_paretopoint']:
        print('Get Paretopoint: Enabled')
        routes.append( (r'/Paretopoint', ParetopointHandler) )
    else:
        print('Get Paretopoint: Disabled')

    return tornado.web.Application(routes)


def main():

    application = make_app()
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
