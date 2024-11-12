import dtlpy as dl
import clip
from PIL import Image
import time
import torch
import numpy as np
import torch

dl.setenv('rc')
m = dl.models.get(None, '6732dfe92aa895346cc469e9')

local_path = m.artifacts.download(local_path='.')
model, preprocess = clip.load("ViT-B/32", device='cpu')

checkpoint = torch.load(r"best.pt", map_location='cpu')

model.load_state_dict(checkpoint['model_state_dict'])



dl.setenv('prod')
mm = dl.models.get(model_id="673334351881e27f94cbb1ca")
mm.metrics.list().print()