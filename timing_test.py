import time

t0 = time.time()
import torch
print(f"import torch:              {time.time()-t0:.2f}s")

t0 = time.time()
from sentence_transformers import SentenceTransformer
print(f"import sentence_transformers: {time.time()-t0:.2f}s")

t0 = time.time()
m = SentenceTransformer("all-MiniLM-L6-v2")
print(f"load model:               {time.time()-t0:.2f}s")

t0 = time.time()
m.encode("running shoes")
print(f"encode one query:         {time.time()-t0:.2f}s")

t0 = time.time()
import chromadb
print(f"import chromadb:          {time.time()-t0:.2f}s")
