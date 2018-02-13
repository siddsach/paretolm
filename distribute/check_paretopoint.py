# coding: utf-8
import argparse
import time
import math
import urllib2
import matplotlib as plt
from __future__ import print_function
import mimetypes
import os
import sys
from functools import partial
from uuid import uuid4
try:
    from urllib.parse import quote
except ImportError:
    from urllib import quote

from tornado import gen, httpclient, ioloop
from tornado.options import define, options



parser = argparse.ArgumentParser(description='Submit your model/check the pareto point through http')
                    
# three components must submit                    
parser.add_argument('--model', type=str, default='./checkpoint/',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--model_module', type=str, default='model.py',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--main_module', type=str, default='main.py',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')

# fake identity for pareto point display only
parser.add_argument('--pseudonym', type=str, default='./checkpoint/',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
# real identity information
parser.add_argument('--name', type=str, default='haiwang',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--student_id', type=str, default='104416XX',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')


args = parser.parse_args()


########################################################### Http submision utils ###################################
@gen.coroutine
def multipart_producer(boundary, filenames, write):
    boundary_bytes = boundary.encode()

    for filename in filenames:
        filename_bytes = filename.encode()
        mtype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        buf = (
            (b'--%s\r\n' % boundary_bytes) +
            (b'Content-Disposition: form-data; name="%s"; filename="%s"\r\n' %
             (filename_bytes, filename_bytes)) +
            (b'Content-Type: %s\r\n' % mtype.encode()) +
            b'\r\n'
        )
        yield write(buf)
        with open(filename, 'rb') as f:
            while True:
                # 16k at a time.
                chunk = f.read(16 * 1024)
                if not chunk:
                    break
                yield write(chunk)

        yield write(b'\r\n')

    yield write(b'--%s--\r\n' % (boundary_bytes,))

@gen.coroutine
def post(filenames, id_info):
    client = httpclient.AsyncHTTPClient()
    boundary = uuid4().hex
    headers = {'Content-Type': 'multipart/form-data; boundary=%s' % boundary}
    producer = partial(multipart_producer, boundary, filenames)
    response = yield client.fetch('htttp://128.135.8.238/dl/upload',
                                  method='POST',
                                  headers=headers,
                                  body_producer=producer)

    print(response)


def submit_current_model():
    
    pseudonym = args.pseudonym
    name = args.name
    student_id = args.student_id

    model_module = args.model_module
    main_module = args.main_module

    submissions = [args.main_module, args.model_module, args.model]
    ioloop.IOLoop.current().run_sync(lambda: post(submissions))

def fetch_current_status():

    req = urllib2.Request(url='htttp://128.135.8.238/dl')
    f = urllib2.urlopen(req)
    status = f.read()

    plt.ylabel('Perplexity')
    plt.xlabel('Ratio: Training Time/Baseline time')
    plt.axis([0, 10, 0, 300])

    all_points_x = []
    all_points_y = []
    labels = []

    for key in status.keys():
        labels.append(key)
        all_points_x.append(status[key][0])
        all_points_y.append(status[key][1])

    fig, ax = plt.subplots()
    ax.scatter(all_points_x, all_points_y)

    for i, label in enumerate(labels):
        ax.annotate(label, (all_points_x[i], all_points_y[i]))


def main():

    print("Use this code to submit your model or check the current paretopoint")
    print("Options: \n")
    print("0: submit your model and code\n")
    print("1: check current paretopoint and display it\n")
    option = input("Type your choice: ")
    if option == 0:
        response = submit_current_model()
        if response:
            print("Submission accepted")
        else:
            print("Submission not accepted")
    else:
        fetch_current_status()        
    
if __name__== "__main__":
  main()