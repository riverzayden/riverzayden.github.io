# -- coding: utf-8 --
from flask import Flask,request, jsonify
from flask_restful import Resource, Api
from flask_restful import reqparse
import requests
import argparse
import time
import numpy as np
from scipy.misc import imread
from flask import jsonify
from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import sys
import json
import os
import time
import cv2
import ast
from b_preprocessing import pre_processing
from b_postprocessing import post_processing

parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", type=str, default=None,
                help='Tensorflow Serving Model Name')
parser.add_argument("-signature", "--signature", type=str, default=None,
                help='Tensorflow Serving Signature Name')
parser.add_argument("-inputs", "--inputs", type=str, default=None,
                help='Tensorflow Serving Input List Name')
parser.add_argument("-outputs", "--outputs", type=str, default=None,
                help='outputs for post' )
parser.add_argument("-image_size", "--image_size", type=int, default=None,
                help='image_size ' )
parser.add_argument("-image_dimension", "--image_dimension", type=int, default=None,
                help='image_dimension ' )
args = parser.parse_args()

def type_val(result, output_name, output_type):
    if output_type =='DT_FLOAT':
        out = result.outputs[output_name].float_val
        status =0
    elif output_type =='DT_INT64':
        out = result.outputs[output_name].int64_val
        status= 0
    else:
        status=1000
    return out, status
        


def run(host, port, image, model,input_info, output_info, signature_name):
    t= time.time()
    # channel = grpc.insecure_channel('%s:%d' % (host, port))
    
    channel = implementations.insecure_channel(host, port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    data= image
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = signature_name
    result={}
    try:
        for k, v in input_info.items():
            if v=="TRUE":
                request.inputs[k].CopyFrom(make_tensor_proto(data, shape=list(data.shape)))
            else:
                request.inputs[k].CopyFrom(make_tensor_proto(v, shape=[1]))
        result_predict = stub.Predict(request, 1.0)
        print("result")
        print( result_predict)
        print("-----------------------------")
        try:
            final = {}
            for k, v in output_info.items():
                final[k] = list(type_val(result_predict, k, v)[0])
            result['status']=0
            result['message']="Success"
            result['output'] = final
            return result
        except Exception as ee:
            result['status'] = 342
            result['message']=" Final Predict API ( Output Error ) "
            return result
    except Exception as ee:
        print(ee)
        result['status']= 343
        result['message']= "Final Predict Tensorflow-sErving Start is Failed Error "
        return result 

        


app = Flask(__name__)
api = Api(app)
app.config['JSON_AS_ASCII'] = False



class post_process_inference(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        
        host = json_data['host']
        port = json_data['port']
        image = json_data['image']
        model = args.model
        signature_name = args.signature
        output_info =  json.loads(args.outputs)
        input_info = json.loads(args.inputs)
        
        image_size = args.image_size
        image_dimension = args.image_dimension
        image = pre_processing(image,image_size,image_dimension )
        print('-------------',image.shape)

        #image = pre_processing(image)


        prob = run(host, port, image, model, input_info, output_info, signature_name)
        print("=====================================")
        print(prob)
        if prob['status']==0:
            output = prob['output']            
            process_output = post_processing(output)
        prob['output']= process_output
        return jsonify(prob)


class pre_inference(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        host = json_data['host']
        port = json_data['port']
        image = json_data['image']
        model = args.model
        signature_name = args.signature
        output_info =  json.loads(args.outputs)
        input_info = json.loads(args.inputs)
        
        image_size = args.image_size
        image_dimension = args.image_dimension
        image = pre_processing(image,image_size,image_dimension )
        print('-------------',image.shape)
        #image = pre_processing(image)


        prob = run(host, port, image, model, input_info, output_info, signature_name)
        print("=====================================")
        print(prob)
        return jsonify(prob)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

class app_shutdown(Resource):
    def post(self):
        shutdown_server()
        return jsonify('Server Shutting Down...')

api.add_resource(post_process_inference,'/inference/post')
api.add_resource(pre_inference,'/inference/pre')
api.add_resource(app_shutdown, '/shutdown')
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5102,debug=True)
