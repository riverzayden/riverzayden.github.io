import argparse
import numpy as np
from grpc.beta import implementations
from tensorflow.contrib.util import make_tensor_proto

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import sys
import json
import os
import time
import cv2
import argparse
import ast
from b_preprocessing import pre_processing
from b_postprocessing import post_processing
parser = argparse.ArgumentParser()
parser.add_argument("-prepost", "--prepost", type=str, default="pre", help="image Processing Post or Pre" )
parser.add_argument("-port", "--port", type=int, default=8500, help="Tensorflow Serving Port" )

parser.add_argument("-image", "--image", type=str, default=None,help='Image Path')
parser.add_argument("-model", "--model", type=str, default=None,
                help='Tensorflow Serving Model Name')
parser.add_argument("-signature", "--signature", type=str, default=None,
                help='Tensorflow Serving Signature Name')
parser.add_argument("-inputs", "--inputs", type=str, default=None,
                help='Tensorflow Serving Input List Name')
parser.add_argument("-outputs", "--outputs", type=str, default=None,
                help='outputs for post' )
parser.add_argument("-target_path", "--target_path", type=str, default=None,
                help='outputs for post' )
parser.add_argument("-image_size", "--image_size", type=int, default=None,
                help='outputs for post' )
parser.add_argument("-image_dimension", "--image_dimension", type=int, default=None,
                help='image_dimension ft' )
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
    return out
try:
    channel = implementations.insecure_channel('localhost',args.port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
except Exception as EE:
    print('Channel Error: ' ,EE)
    result={}
    result['status']=321
    result['message'] = "Custom Pre-Predict Channel Error"
    sys.exit(1)

try:
    image_size = args.image_size
    image_dimension = args.image_dimension
    image = pre_processing(args.image,image_size,image_dimension)
except Exception as EE:
    print('Pre Processing Error:', EE)
    result={}
    result['status']=322
    result['message'] = "Custom Pre-Predict Processing Error"
    sys.exit(1)
try:
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.model
    request.model_spec.signature_name = args.signature
except Exception as EE:
    print(EE)
    result={}
    result['status']=323
    result['message'] = "Custom Pre-Predict Request Error"
    sys.exit(1)  

try:
    input_info = ast.literal_eval(args.inputs)
    for k, v in input_info.items():
        if v=="TRUE":
            request.inputs[k].CopyFrom(make_tensor_proto(image, shape=list(image.shape)))
        else:
            request.inputs[k].CopyFrom(make_tensor_proto(v, shape=[1]))
    result = stub.Predict(request, 1.0)
except Exception as EE:
    print("InPut ERROR:" ,EE)
    result={}
    result['status']=324
    result['message'] = "Error : Custom Pre-Predict Input Name Do Not Match with TRAIN DATA"
    sys.exit(1)  
if args.prepost =='post':
    final={}
    output_modify={}
    for i in ast.literal_eval(args.outputs):
        split_1 = str(result.outputs[i]).split('dtype: ')
        dtype= split_1[1].split('\ntensor_shape')
        output_modify[i] =dtype[0]
        final[i] = list(type_val(result, i, dtype[0]))
    final_2 = post_processing(final)
    print(final_2)
    with open(os.path.join(args.target_path,'post_result.json'),'w') as f:
        f.write(json.dumps(final_2))
    with open(os.path.join(args.target_path,'output_type.json'),'w') as f:
        f.write(json.dumps(output_modify))
else:
    final={}
    output_modify={}
    for i in ast.literal_eval(args.outputs):
        split_1 = str(result.outputs[i]).split('dtype: ')
        dtype= split_1[1].split('\ntensor_shape')
        output_modify[i] =dtype[0]
        final[i] = list(type_val(result, i, dtype[0]))
    with open(os.path.join(args.target_path,'pre_result.json'),'w') as f:
        f.write(json.dumps(final))
    print(final)
    with open(os.path.join(args.target_path,'output_type.json'),'w') as f:
        f.write(json.dumps(output_modify))

