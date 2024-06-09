import numpy as np
np.random.seed(0)
import timeit

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from utils import softmax, linear

def attention(q, k, v, mask): #[n_q, d_k] [n_k, d_k] [n_k, d_v] [n_q, n_k] -> [n_q, d_v]
    #when not caching n_q=n_k
    #when using caching n_q=1, n_k=1+len(previously generated tokens)
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head, kvcache=None): #[n_seq, n_embd] -> [n_seq, n_embd]
    #qkv projection
    x = linear(x, **c_attn) #[n_seq, n_emb] -> [n_seq, 3*n_embed]
    #logger.info(f"x.shape :{x.shape}")
    #split into q, k , v
    qkv = np.split(x, 3, axis=-1) #[n_seq, 3*n_embd] -> [3, n_seq, n_embd]
    #logger.info(f"qkv.shape :{np.array(qkv).shape}")
    if kvcache:
        new_q, new_k, new_v = qkv
        old_k, old_v = kvcache

        k = np.vstack([old_k, new_k])
        v = np.vstack([old_v, new_v])

        qkv = [new_q, k, v]

    current_cache = [qkv[1], qkv[2]]
    
    #split into heads
    
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [n_head, 3, n_seq, n_embd/n_head]
    #logger.debug(f"qkv.shape :{np.array(qkv).shape}")
    #causal masking: this mask will be added to the Q.Kt befor performing softmax
    if kvcache:
        causal_mask = np.zeros((1, k.shape[0]))
    else:
        causal_mask = (1 - np.tri(x.shape[0])) * -1e10

    #perform attention
    out_heads = [ attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]

    #merge heads
    x = np.hstack(out_heads) # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    #out proj
    x = linear(x, **c_proj)
    return x, current_cache

if __name__ =="__main__":

    t = 5
    n_head = 2
    d_hidden = n_head*8 

    c_attn = {"w":np.random.randn(d_hidden, 3*d_hidden), "b":np.random.randn(1, 3*d_hidden)}
    c_proj = {"w":np.random.randn(d_hidden, d_hidden), "b":np.random.randn(1, d_hidden)}

    np.random.seed(0)
    inputs = np.random.randn(t, d_hidden)
    
    kvcache = None
    start = timeit.default_timer()
    for idx in range(500):  # auto-regressive decode loop
        logits, kvcache = mha(inputs, c_attn, c_proj, n_head=n_head, kvcache=kvcache)  # model forward pass
        inputs = softmax(logits[-1].reshape(1, -1))  # get prediction and use as input

    logger.info(f"Time with KV-cache {timeit.default_timer()-start}")


    np.random.seed(0)
    inputs_nokv = np.random.randn(t, d_hidden)

    kvcache = None
    start = timeit.default_timer()
    for idx in range(500):  # auto-regressive decode loop
        logits, kvcache = mha(inputs_nokv, c_attn, c_proj, n_head=n_head, kvcache=None)  # model forward pass
        new_inp = softmax(logits[-1].reshape(1, -1))
        inputs_nokv = np.vstack([inputs_nokv, new_inp])  # append prediction to input

    logger.info(f"Time without KV-cache {timeit.default_timer()-start}")
    
   
    print(f"Last token w KV-cache: {inputs[0]}")
    print(f"Last token w/o KV-cache: {inputs_nokv[-1]}")
    print(f"diff: {inputs[0] - inputs_nokv[-1]}")