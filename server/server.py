
#!/usr/bin/env python

import tornado.ioloop
import tornado.web

import hashlib

import uuid, json, base64
import numpy as np
import sys, os
import os.path

import dlib
import numpy as np
import pickle
import logging
try:
    from urllib.parse import unquote
except ImportError:
    # Python 2.
    from urllib import unquote


#### Use this to configure the service

parameters = { 
    'port': 5000, 
    'enable_paretopoint': True,
    'enable_uploading': True, 
} 

Paretopoint = "./result/Paretopoint.json"
Paretopoint_private = "./result/Paretopoint_private.json"

class UploadModelHandler(tornado.web.RequestHandler): 
    
    """ 
    Lets the client upload an model. Returns a token that corresponds to the uploaded 
    file. 
    """

    def post(self): 

        print('---------------------') 
        print("Got an upload request") 

        name = self.get_argument('name')
        student_id = self.get_argument('student_id')
        pseudonym = self.get_argument('pseudonym')
        
        if name == '':
            login_response = "{'error': true, 'msg': 'Please enter your name.'}"
            self.write(json.dumps(login_response))
            return 
        elif student_id == '':
            login_response = "{'error': true, 'msg': 'Please enter your student id.'}"
            self.write(json.dumps(login_response))
            return 
        else:
            login_response = "{'error': true, 'msg': 'Thank You'}"


        # now get the model and files 
        for field_name, files in self.request.files.items():
            for info in files:
                filename, content_type = info['filename'], info['content_type']
                body = info['body']
                if filename == "model_module":
                    model_module = body
                elif filename == "main_module":
                    main_module = body
                elif filename == "checkpoint":
                    check_point = body    
                else:
                    print("error\n")
                print('POST "%s" "%s" %d bytes', filename, content_type, len(body))


        folder = "./uploads/%s_%s" % (email_address, student_id)
        trial = 0
        if not os.path.exists(folder):
            os.mkdir(folder)
        else:
            for f in os.listdir(folder):
                id = int(f.)
                if id > trial:
                    trial = id


        # folder/studentid/__trial  
        model_name = "./uploads/%s_%s/model_%d.py" % (email_address, student_id, trial)
        main_name = "./uploads/%s_%s/main_%d.py" % (email_address, student_id, trial)
        checkpoint_name = "./uploads/%s_%s/checkpoint_%d.pt" % (email_address, student_id, trial)

        with open(model_name, 'wt+') as fp: 
            fp.write(model_module) 

        with open(main_name, 'wt+') as fp: 
            fp.write(main_module) 

        with open(checkpoint_name, 'wb+') as fp: 
            fp.write(check_point) 
 
        print(".. saving the uploads done") 


        # now call the shell to run it to make sure it's working         
        
        cmd_eval = "python %s " % main_name
        cmd_gen = "python %s " % main_name
        os.system.command(cmd_eval)
        os.system.command(cmd_gen)
        
        # check the running is ok
        if not os.path.exists():
            login_response = "{'error': true, 'msg': 'Thank you, but the model running failed, please check your submission'}"
            self.write(json.dumps(login_response))
            return 


        # updates the global paretopoint and save it
        paretopoint = json.load(Paretopoint)
        paretopoint[pseudonym + ":::" + str(trial)] = [perplexity, time_ratio]
        json.dump(paretopoint, Paretopoint)


        # updates the private paretopoint and save it (only for grading purpose)
        aretopoint_private = json.load(Paretopoint_private)
        paretopoint_private[name + "_" + student_id + "_" + str(trial)] = [perplexity, time_ratio]
        json.dump(paretopoint_private, Paretopoint_private)

        # if submission accepted
        login_response = "{'error': true, 'msg': 'Thank you, submission is accepted'}"
        self.write(json.dumps(login_response))
        return 
 
class ParetopointHandler(tornado.web.RequestHandler): 

    def post(self):

        print('-----------------------') 
        print('Request to get the Paretopoint') 
 
        self.set_header("Content-Type", "application/json") 
        ret = [{}] 
        ret[0]['Paretopoint'] = '' 
 
        with open(Paretopoint, 'rb') as fp: 
            ret[0]['Paretopoint'] = base64.b64encode(fp.read()) 
 
        self.write(json.dumps(ret)) 
        print(' .. done') 
        return 

def make_app():
    
    routes = []
    if parameters['enable_uploading']:
        print('Uploading: Enabled')
        routes.append( (r'/upload', UploadModelHandler) )
    else:
        print('Uploading: Disabled')

    if parameters['enable_paretopoint']:
        print('Get Paretopoint: Enabled')
        routes.append( (r'/Paretopoint', ParetopointHandler) )
    else:
        print('Get Paretopoint: Disabled')

    return tornado.web.Application(routes)

def main():

    application = make_app()
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(parameters['port'])
    tornado.ioloop.IOLoop.instance().start()
