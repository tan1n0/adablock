"""Append-only buffers; generated suffixes reused, immutable prompt prefix shared."""
import torch
from transformers import DynamicCache

BUFFERS={}

def reset_registry():
    BUFFERS.clear()

class BufferCache(DynamicCache):
    def __init__(self,capacity=0):
        super().__init__()
        self.capacity=capacity
        self.buffers={}

    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        self.append_new_layers(layer_idx)
        layer=self.layers[layer_idx]
        start=layer.get_seq_length()
        end=start+key_states.shape[-2]
        if layer_idx not in self.buffers:
            assert self.capacity>=end
            shape=list(key_states.shape);shape[-2]=self.capacity
            self.buffers[layer_idx]=(torch.empty(shape,dtype=key_states.dtype,device=key_states.device),
                                     torch.empty(shape,dtype=value_states.dtype,device=value_states.device))
        k,v=self.buffers[layer_idx]
        BUFFERS[k.data_ptr()]=k
        BUFFERS[v.data_ptr()]=v
        assert end<=k.shape[-2]
        k[...,start:end,:].copy_(key_states);v[...,start:end,:].copy_(value_states)
        layer.keys=k[...,:end,:];layer.values=v[...,:end,:]
        return layer.keys,layer.values

    @classmethod
    def from_legacy_cache(cls,pairs):
        # Native-cache short probes still use the original implementation.
        if pairs[0][0].data_ptr() not in BUFFERS:
            return DynamicCache.from_legacy_cache(pairs)
        obj=cls(BUFFERS[pairs[0][0].data_ptr()].shape[-2])
        for i,(k,v) in enumerate(pairs):
            obj.append_new_layers(i)
            obj.layers[i].keys=k;obj.layers[i].values=v
            obj.buffers[i]=(BUFFERS[k.data_ptr()],BUFFERS[v.data_ptr()])
        return obj

def append_pair(oldk,oldv,k,v):
    position=oldk.shape[-2]
    kb,vb=BUFFERS[oldk.data_ptr()],BUFFERS[oldv.data_ptr()]
    assert kb is not None and vb is not None and kb.ndim==vb.ndim==4
    assert position+k.shape[-2]<=kb.shape[-2]
    end=position+k.shape[-2]
    kb[...,position:end,:].copy_(k);vb[...,position:end,:].copy_(v)
    return kb[...,:end,:],vb[...,:end,:]
