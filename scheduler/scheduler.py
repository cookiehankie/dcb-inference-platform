import asyncio
import threading
import time
from queue import Queue, Empty
from typing import List, Dict

import grpc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 导入生成的 gRPC 类
import proto.inference_pb2 as inference_pb2
import proto.inference_pb2_grpc as inference_pb2_grpc

# Scheduler 配置
BATCH_INTERVAL = 0.01  # 10ms
MAX_BATCH_SIZE = 32
GRPC_WORKER_ADDRESS = 'localhost:50051'

# FastAPI 服务
app = FastAPI()

# 内存队列，元素为 (InferenceRequest, asyncio.Future)
request_queue: Queue = Queue()

# 连接到 Worker 的 gRPC stub
channel = grpc.insecure_channel(GRPC_WORKER_ADDRESS)
stub = inference_pb2_grpc.InferenceServiceStub(channel)


class InferenceItem(BaseModel):
    request_id: str
    model_name: str
    input_ids: List[int]
    max_length: int
    priority: int = 0
    enable_kv_prune: bool = False
    kv_prune_k: int = 0
    enable_speculative: bool = False
    speculative_steps: int = 0


@app.post("/infer")
async def infer(req: InferenceItem):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    # 构造 gRPC 请求
    grpc_req = inference_pb2.InferenceRequest(
        request_id=req.request_id,
        model_name=req.model_name,
        input_ids=req.input_ids,
        max_length=req.max_length,
        priority=req.priority,
        enable_kv_prune=req.enable_kv_prune,
        kv_prune_k=req.kv_prune_k,
        enable_speculative=req.enable_speculative,
        speculative_steps=req.speculative_steps,
    )
    # 入队
    request_queue.put((grpc_req, future))
    try:
        # 等待结果，超时 5s
        tokens = await asyncio.wait_for(future, timeout=5.0)
        return {"request_id": req.request_id, "tokens": tokens}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")


def batcher_loop():
    while True:
        time.sleep(BATCH_INTERVAL)
        batch_items = []
        futures = []
        # 收集批次
        while len(batch_items) < MAX_BATCH_SIZE:
            try:
                item, fut = request_queue.get_nowait()
                batch_items.append(item)
                futures.append(fut)
            except Empty:
                break
        if not batch_items:
            continue
        # 构造 BatchRequest
        batch_req = inference_pb2.BatchRequest(requests=batch_items)
        try:
            batch_resp = stub.BatchInfer(batch_req)
            # 按 request_id 分组响应
            resp_map: Dict[str, List[int]] = {}
            for resp in batch_resp.responses:
                resp_map.setdefault(resp.request_id, []).append(resp.token)
            # 完成 futures
            for item, fut in zip(batch_items, futures):
                tokens = resp_map.get(item.request_id, [])
                fut.set_result(tokens)
        except Exception as e:
            for _, fut in futures:
                fut.set_exception(e)


# 启动批次调度线程
threading.Thread(target=batcher_loop, daemon=True).start()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
