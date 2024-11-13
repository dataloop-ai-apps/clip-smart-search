import clip
import torch
import tqdm
import dtlpy as dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from model_adapter import ClipAdapter

# From https://github.com/pytorch/pytorch/blob/2efe4d809fdc94501fc38bf429e9a8d4205b51b6/torch/utils/tensorboard/_pytorch_graph.py#L384


device = "cuda" if torch.cuda.is_available() else "cpu"

project = dl.projects.get('smart image search')
# model_entity = project.models.get(model_name='CLIP ViT-B/32 SFT-PSicV')
model_entity = project.models.get(model_name='clip-smart-search-o44in_2024_11_11-T11_03_29')
model_path = r'/best.pt'
# model_path = r'C:\Users\Yaya Tang\.dataloop\models\clclip-smart-search-o44in_2024_11_11-T11_03_29\best.pt'
# model_path = r'C:\Users\Yaya Tang\PycharmProjects\clip-smart-search\tmp\6731d515bba6dc4ca4667227\output\best.pt'

# app = ClipAdapter()
# app.load_from_model(model_entity=model_entity, overwrite=False)
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

##################
# prepare images #
##################

# img_paths = model_entity.dataset.download()
data_path = r"/tmp/6731d515bba6dc4ca4667227/datasets/672cc229e773c08bacdfedad/train"
img_paths, _ = ClipAdapter.get_images_and_text(data_path=data_path, overwrite=False)

images = []
# pbar = tqdm.tqdm(total=len(img_paths))
for img_path in img_paths:
    print(img_path)
    image = Image.open(Path(img_path)).convert("RGB")
    images.append(preprocess(image))
    # pbar.update()

image_input = torch.tensor(np.stack(images)).to(device)
with torch.no_grad():
    image_features = model.encode_image(image_input)
    # text_features = model.encode_text(text_tokens)
    # logits_per_image, logits_per_text = model(images, text_features)
    # probs = logits_per_text.softmax(dim=-1).cpu().numpy()
image_features /= image_features.norm(dim=-1, keepdim=True)

# create text/query feature
QUERY_STRING = "alumininum can"

text_tokens = clip.tokenize([QUERY_STRING]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Pick the top 10 most similar images for the text/query
result = cosine_similarity(text_features.cpu().numpy(), image_features.cpu().numpy())

results_dict = {'name': [], 'prob': [], 'filepath': []}

pbar = tqdm.tqdm(total=len(img_paths))
for i, img_path in enumerate(img_paths):
    results_dict['name'].append(Path(img_path).name)
    results_dict['prob'].append(result[0][i])
    results_dict['filepath'].append(img_path)
    results_dict.update()
    pbar.update()

results_df = pd.DataFrame(results_dict)
results_df.sort_values(by=['prob'], ascending=False, inplace=True)

print(results_df.iloc[:9][['name', 'prob']])

plt.figure(figsize=(16, 16))
for i, img_path in enumerate(results_df['filepath'].iloc[:9]):
    plt.subplot(3, 3, i + 1)
    image = Image.open(img_path).convert("RGB")
    plt.imshow(image)

plt.suptitle(f"Query: {QUERY_STRING}")
plt.show()
