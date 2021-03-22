# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import grpc
import logging
import agent_pb2 as agent
import agent_pb2_grpc as agent_grpc
import struct
import numpy as np
import uuid

class EdgeAgentClient(object):
    """ Helper class that uses the Edge Agent stubs to
        communicate with the SageMaker Edge Agent through unix socket.
        
        To generate the stubs you need to use protoc. First install/update:
        pip3 install -U grpcio-tools grpcio protobuf
        then generate the code using the provided agent.proto file
        
        python3 -m grpc_tools.protoc \
            --proto_path=$PWD/agent/docs/api --python_out=. --grpc_python_out=. $PWD/agent/docs/api/agent.proto
        
    """
    def __init__(self, channel_path):
        # connect to the agent and list the models
        self.channel = grpc.insecure_channel('unix://%s' % channel_path )
        self.agent = agent_grpc.AgentStub(self.channel)
        self.model_map = {}
    
    def __update_models_list__(self):
        models_list = self.agent.ListModels(agent.ListModelsRequest())
        self.model_map = {m.name:{'in': m.input_tensor_metadatas, 'out': m.output_tensor_metadatas} for m in models_list.models}
        return self.model_map
    
    def capture_data(self, model_name, input_tensor, output_tensor):
        try:
            req = agent.CaptureDataRequest()
            req.model_name = model_name
            req.capture_id = str(uuid.uuid4())
            req.input_tensors.append( input_tensor )
            req.output_tensors.append( output_tensor )
            resp = self.agent.CaptureData(req)
        except Exception as e:
            logging.error(e)
            
    def predict(self, model_name, x, capture=False):        
        """
        Invokes the model and get the predictions
        """
        try:
            if self.model_map.get(model_name) is None:
                return None
            # Create a request
            req = agent.PredictRequest()
            req.name = model_name
            # Then load the data into a temp Tensor
            tensor = agent.Tensor()
            meta = self.model_map[model_name]['in'][0]
            tensor.tensor_metadata.name = meta.name
            tensor.tensor_metadata.data_type = meta.data_type
            num_floats = 1
            for s in meta.shape: tensor.tensor_metadata.shape.append(s); num_floats *= s
            tensor.byte_data = struct.pack('%df' % num_floats, *x.flatten())
            req.tensors.append(tensor)
            
            # Invoke the model
            resp = self.agent.Predict(req)
            if capture:
                out_tensor = agent.Tensor()
                out_tensor.tensor_metadata.name = resp.tensors[0].tensor_metadata.name
                out_tensor.tensor_metadata.data_type = resp.tensors[0].tensor_metadata.data_type
                for i in resp.tensors[0].tensor_metadata.shape: out_tensor.tensor_metadata.shape.append(i)
                out_tensor.byte_data = bytes(resp.tensors[0].byte_data)
                self.capture_data(model_name, tensor, out_tensor)

            # Parse the output
            meta = self.model_map[model_name]['out'][0]
            shape = []
            num_floats = 1            
            tensor = resp.tensors[0]
            for s in tensor.tensor_metadata.shape: num_floats *= s; shape += [s]
            data = np.array(struct.unpack('%df' % num_floats, tensor.byte_data))
            return data.reshape(shape)
        except Exception as e:
            logging.error(e)        
            return None

    def is_model_loaded(self, model_name):
        return self.model_map.get(model_name) is not None
    
    def load_model(self, model_name, model_path):
        """ Load a new model into the Edge Agent if not loaded yet"""
        try:
            if self.is_model_loaded(model_name):
                logging.info( "Model %s was already loaded" % model_name )
                return self.model_map
            req = agent.LoadModelRequest()
            req.url = model_path
            req.name = model_name
            resp = self.agent.LoadModel(req)

            return self.__update_models_list__()            
        except Exception as e:
            logging.error(e)        
            return None
        
    def unload_model(self, model_name):
        """ UnLoad model from the Edge Agent"""
        try:
            if not self.is_model_loaded(model_name):
                logging.info( "Model %s was not loaded" % model_name )
                return self.model_map
            
            req = agent.UnLoadModelRequest()
            req.name = model_name
            resp = self.agent.UnLoadModel(req)
            
            return self.__update_models_list__()
        except Exception as e:
            logging.error(e)        
            return None
