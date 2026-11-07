"""Replay the original FP32 scorer, with exact input shapes and current query values.

No score/projection cache, padding, precision change, or altered sorting rule.
Only one shape is retained. Shape changes rebuild the graph and are charged to decode.
The returned output aliases graph storage and is consumed on the same CUDA stream.
"""
import time
import torch

_CAPTURE_STREAM = None


class GraphRanker:
    def __init__(self, original):
        self.original = original
        self.signature = None
        self.graph = None
        self.inputs = None
        self.output = None
        self.build_seconds = 0.
        self.builds = 0
        self.calls = 0
        self.resident_bytes = 0

    @torch.inference_mode()
    def __call__(self, model, packed):
        assert not model.training
        signature = (id(model), tuple((k, tuple(v.shape), v.dtype, v.device)
                                     for k, v in sorted(packed.items())))
        if signature != self.signature:
            self._build(model, packed, signature)
        else:
            for k, v in packed.items():
                self.inputs[k].copy_(v)
        self.graph.replay()
        self.calls += 1
        return self.output

    def _build(self, model, packed, signature):
        torch.cuda.synchronize()
        start = time.perf_counter()
        self.graph = self.inputs = self.output = None
        # Preserve actual values/strides while warming; no uninitialized model inputs.
        self.inputs = {k: v.clone() for k, v in packed.items()}
        global _CAPTURE_STREAM
        if _CAPTURE_STREAM is None:
            _CAPTURE_STREAM = torch.cuda.Stream()
        stream = _CAPTURE_STREAM
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                warm = self.original(model, self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        del warm
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.output = self.original(model, self.inputs)
        torch.cuda.synchronize()
        self.resident_bytes = (torch.cuda.memory_allocated() - before +
                               sum(v.numel() * v.element_size() for v in self.inputs.values()))
        self.signature = signature
        self.build_seconds += time.perf_counter() - start
        self.builds += 1

    def metrics(self):
        return dict(builds=self.builds, calls=self.calls, build_seconds=self.build_seconds,
                    resident_MiB=self.resident_bytes / 2**20)
