import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

# 1.a
def embed(sentence):
  punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  for ele in sentence:
    if ele in punc:
        sentence = sentence.replace(ele, "")
  w = sentence.lower().split()
  words = sorted(w)
  index = 0
  tokens = {}
  tokenize = []
  for word in words:
    if word not in tokens:
      tokens[word] = index
      index += 1
  for word in w:
    tokenize.append(tokens[word])
  tensor = torch.tensor(tokenize)
  embed_func = nn.Embedding(len(tensor),16)
  return embed_func(tensor)
temp = embed("Attention is all you need for now.")
print(temp.shape)

# 1.b
def context(embeddings):
  dq = 24
  dk = 24
  dv = 28
  q = nn.Linear(dq, 16)
  k = nn.Linear(dk, 16)
  v = nn.Linear(dv, 16)
  nn.init.normal_(q.weight, mean=0, std=0.01)
  nn.init.normal_(k.weight, mean=0, std=0.01)
  nn.init.normal_(v.weight, mean=0, std=0.01)
  qx = torch.matmul(embeddings,q.weight)
  kx = torch.matmul(embeddings, k.weight)
  vx = torch.matmul(embeddings,v.weight)
  w = []
  for i in range(len(qx)):
    w.append((qx[i]@kx[i])/math.sqrt(dk))
  smax = nn.Softmax(dim=0)
  a = smax(torch.tensor(w))
  c = [0]*len(a)
  for i in range(len(a)):
    currA = float(a[i])
    currV = sum(v.weight[i])
    c[i] += currA*currV
  return torch.tensor(c)
print(context(temp).shape)

# 2
def multiheaded(embeddings, n_heads):
  c = [context(embeddings) for _ in range(n_heads)]
  C = torch.stack(c)
  return C
print(multiheaded(temp,5).shape)

# 3
'''
Intuitively, it seems that cross-attention would be better at contextual understanding than multi-headed attention
By swapping the key and values for each token, the context surrounding each embedding
would be compared with the others, leading to an improved "understanding of the context".
This would be useful for cases where contextual information is important, such as when the
training text is highly technical.
'''