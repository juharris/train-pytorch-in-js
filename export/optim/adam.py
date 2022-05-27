# Copied and slightly modified from https://github.com/microsoft/onnxruntime/blob/e70ae3303dc57096d1b1ee51483e8789cad51941/orttraining/orttraining/python/training/optim/adam.py

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# optimizer.py

from onnx import helper
from onnx import TensorProto

class AdamOnnxGraphBuilder:
    def __init__(self,
                 named_params,
                 bias_correction=True,
                 betas=(0.9,
                        0.999),
                 eps=1e-6,
                 weight_decay=0.,
                 max_norm_clip=1.0):

        # Initialize the attributes with constructor arguments.
        # Note that learning rate is not an attribute but an input to the graph
        self.params = named_params
        self.bias_correction = bias_correction
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_norm_clip = max_norm_clip

    def export(self):
        # Export the optimizer graph to an onnx model format

        graph_nodes = []
        graph_inputs = []
        graph_outputs = []
        for idx, (name, param) in enumerate(self.params):
            # prepare node (and graph) inputs and outputs
            node_input_names = [name+'.learning_rate', # learning rate
                                name+'.step', # training step (used for beta correction)
                                name, # weight to be updated
                                name+'.gradient', # gradient of the weight to be used for update
                                name+'.exp_avg', # first order moment for this weight
                                name+'.exp_avg_sq', # second order moment for this weight
                                # These aren't needed and cause problems in ORT Web because float16 is not supported.
                                # name+'.mixed_precision', # mixed precision weight representation (required if computation to be done in mp)
                                # name+'.loss_scaler', # used for gradient scaling
                                # name+'.global_gradient_norm', # used for gradient scaling
                                # name+'.should_update', # whether or not to skip updating the weights
            ]

            node_inputs = [
                helper.make_tensor_value_info(name+'.learning_rate', TensorProto.FLOAT , [1]),
                helper.make_tensor_value_info(name+'.step', TensorProto.INT64, [1]),
                helper.make_tensor_value_info(name, TensorProto.FLOAT , list(param.shape)),
                helper.make_tensor_value_info(name+'.gradient', TensorProto.FLOAT , list(param.shape)),
                helper.make_tensor_value_info(name+'.exp_avg', TensorProto.FLOAT , list(param.shape)),
                helper.make_tensor_value_info(name+'.exp_avg_sq', TensorProto.FLOAT , list(param.shape)),
                # These aren't needed and cause problems in ORT Web because float16 is not supported.
                # helper.make_tensor_value_info(name+'.mixed_precision', TensorProto.FLOAT16 , [0]),
                # helper.make_tensor_value_info(name+'.loss_scaler', TensorProto.FLOAT, []),
                # helper.make_tensor_value_info(name+'.global_gradient_norm', TensorProto.FLOAT, []),
                # helper.make_tensor_value_info(name+'.should_update', TensorProto.BOOL, [1]),
            ]
            graph_inputs.extend(node_inputs)

            node_output_names = [name+'.step.out', # step out
                                 name+'.exp_avg.out', # first order moment output
                                 name+'.exp_avg_sq.out', # second order moment output
                                 name+'.out', # updated weights
                                 name+'.gradient.out', # gradients output
                                 # Not needed and causes problems in ORT Web because float16 is not supported.
                                #  name+'.mixed_precision.out',  # updated mixed precision weights
                                 ]

            node_outputs = [
                helper.make_tensor_value_info(name+'.step.out', TensorProto.INT64, [1]),
                helper.make_tensor_value_info(name+'.exp_avg.out', TensorProto.FLOAT , list(param.shape)),
                helper.make_tensor_value_info(name+'.exp_avg_sq.out', TensorProto.FLOAT , list(param.shape)),
                helper.make_tensor_value_info(name+'.out', TensorProto.FLOAT , list(param.shape)),
                helper.make_tensor_value_info(name+'.gradient.out', TensorProto.FLOAT , list(param.shape)),
                # Not needed and causes problems in ORT Web because float16 is not supported.
                # helper.make_tensor_value_info(name+'.mixed_precision.out', TensorProto.FLOAT16, [0])
            ]
            graph_outputs.extend(node_outputs)

            # node attributes
            node_attributes = {
                'alpha': self.betas[0], # beta1
                'beta': self.betas[1], # beta2
                'lambda': self.weight_decay, # weight decay
                'epsilon': self.eps, # epsilon
                'do_bias_correction': 1 if self.bias_correction else 0, # bias_correction
                'weight_decay_mode': 1, # weight decay mode 1 implies transformers adamw 0 implies pytorch adamw
                'max_norm_clip': self.max_norm_clip # used for gradient scaling
            }

            # gradient scaling equation:
            # if global_gradient_norm > loss_scaler*max_norm_clip: global_gradient_norm / max_norm_clip
            # else: loss_scaler*max_norm_clip

            # make the node
            optimizer_node = helper.make_node("AdamOptimizer",
                                              node_input_names,
                                              node_output_names,
                                              name=f"AdamOptimizer{idx}",
                                              domain='com.microsoft',
                                              **node_attributes)

            graph_nodes.append(optimizer_node)

        # make the graph and the model
        graph = helper.make_graph(graph_nodes, 'AdamOptimizerGraph', graph_inputs, graph_outputs)
        model = helper.make_model(graph, producer_name='adam_graph_builder',
                                  opset_imports=[helper.make_opsetid('com.microsoft', 1)])
        return model