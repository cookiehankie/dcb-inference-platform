# generate_proto.py
from grpc_tools import protoc

protoc.main([
    'grpc_tools.protoc',
    '--proto_path=.',
    '--python_out=.',
    '--grpc_python_out=.',
    'proto/inference.proto',
])
