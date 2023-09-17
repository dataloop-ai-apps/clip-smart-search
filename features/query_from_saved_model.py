import clip
import datetime
import math
import os
import torch
import tqdm
import dtlpy as dl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import umap.umap_ as umap
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


def is_image_file(filename):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return any(filename.lower().endswith(ext) for ext in img_extensions)


def save_features(image_features, save_dir=None, file_name=None):
    if save_dir is None:
        save_dir = os.getcwd()
    if file_name is None:
        file_name = 'image_features'
    save_path = os.path.join(save_dir, file_name)
    np.save(save_path, image_features.cpu().numpy())
    print(f'Saved image features to {save_path}')
    return save_path


MODEL_FROM_SAVED = False
FEATURES_FROM_SAVED = False
save_dir = r"../output"
os.makedirs(save_dir, exist_ok=True)
dataset_name = 'TACO'
img_dir = Path(r'C:\Users\Yaya Tang\Documents\DATASETS\TACO dataloop\items\raw')

# load saved, trained model
device = "cuda" if torch.cuda.is_available() else "cpu"

if MODEL_FROM_SAVED is True:
    checkpoint_path = r"checkpoints\model_1_epoch_19.pt"
    model, preprocess = clip.load("ViT-B/32", device=device)  # Must set jit=False for training
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

# load images to embed and query
all_paths = []
for path, subdir, files in os.walk(os.path.join(os.getcwd(), img_dir)):
    for name in files:
        all_paths.append(os.path.join(path, name))

img_paths = [str(filepath) for filepath in all_paths if is_image_file(str(filepath))]

if FEATURES_FROM_SAVED is True:
    image_features = np.load(os.path.join(save_dir, 'image_features.npy'))
    image_features = torch.tensor(image_features).to(device)
else:
    # prepare images
    images = []
    pbar = tqdm.tqdm(total=len(img_paths))
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        images.append(preprocess(image))
        pbar.set_description(f"Processing {img_path}...")
        pbar.update()

    # create image features
    image_input = torch.tensor(np.stack(images)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    save_features(image_features, save_dir, 'image_features')

# create text/query feature
QUERY_STRING = "styrofoam container"
NUM_RESULTS = 10

model = model.eval()

text_tokens = clip.tokenize([QUERY_STRING]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Pick the top N most similar images for the text/query
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

results_df = results_df.iloc[:NUM_RESULTS][['name', 'prob', 'filepath']]
print(results_df[['name', 'prob']])

if MODEL_FROM_SAVED is True:
    checkpoint_name = Path(checkpoint_path).name
    model_name = f'SAVED-{checkpoint_name}'
else:
    model_name = 'PRETRAINED'

num_grid = math.isqrt(NUM_RESULTS)
subplot_dims = num_grid + 1 if num_grid ** 2 < NUM_RESULTS else num_grid

plt.figure(figsize=(10, 10))
for i, img_path in enumerate(results_df['filepath'].iloc[:NUM_RESULTS]):
    plt.subplot(subplot_dims, subplot_dims, i + 1)
    image = Image.open(img_path).convert("RGB")
    plt.imshow(image)

plt.suptitle(f"Query: {QUERY_STRING}, found {len(results_df)}, {model_name} model")
plot_filename = f"clip_query_results_{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
save_path = os.path.join(save_dir, plot_filename)
plt.savefig(save_path)
print(f'Saved query results to {save_path}')

# UMAP
all_features = torch.cat((image_features, text_features), 0)
reducer = umap.UMAP(random_state=42, metric='cosine')
# embedding = reducer.fit_transform(image_features.cpu())
embedding = reducer.fit_transform(all_features.cpu())

# update lists to include the query string as the last item
names = [Path(path).name for path in img_paths]
results = results_df['name'].tolist()
query_returned = ['results' if name in results else '0' for name in names]

names.append('query')
query_returned.append('query')
img_paths.append('query')

thumbs_df = pd.DataFrame(embedding, columns=['x', 'y'])
thumbs_df['filename'] = img_paths
thumbs_df['name'] = names
thumbs_df['query_returned'] = query_returned

# plot all images
plt.figure(figsize=(15, 10))
sns.scatterplot(x=thumbs_df['x'], y=thumbs_df['y'], hue=np.array(thumbs_df['query_returned']), palette="deep")
plt.axis('off')
plt.title(f'UMAP of {model_name} CLIP features, query: {QUERY_STRING}')

plot_filename = f"umap_{dataset_name}_all_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
save_path = os.path.join(save_dir, plot_filename)
plt.savefig(save_path)

# plot only returned images
df_filt = thumbs_df[thumbs_df['query_returned'] != '0']
plt.figure(figsize=(15, 10))
sns.scatterplot(x=df_filt['x'], y=df_filt['y'], hue=np.array(df_filt['query_returned']), palette="deep")
plt.axis('off')
plt.title(f'UMAP, CLIP model {model_name} features, only returned images for query {QUERY_STRING}')

plot_filename = f"umap_{dataset_name}_returned_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
save_path = os.path.join(save_dir, plot_filename)
plt.savefig(save_path)
print(f'Saved UMAP fig to {save_path}')
