##################
# notebook setup #
##################
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

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Viridis

######################
# load model, images #
######################

save_dir = r"output"
os.makedirs(save_dir, exist_ok=True)

# load saved, trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# load images
def is_image_file(filename):
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return any(filename.lower().endswith(ext) for ext in img_extensions)


# load images to embed and query
img_dir = Path(r"C:\Users\Yaya Tang\Documents\DATASETS\TACO dataloop\items\raw")
img_paths = [str(img_path) for img_path in img_dir.glob("*") if is_image_file(str(img_path))]

# prepare images
images_np = []
pbar = tqdm.tqdm(total=len(img_paths))
for img_path in img_paths:
    img_np = Image.open(img_path).convert("RGB")
    images_np.append(preprocess(img_np))
    pbar.set_description(f"Processing {img_path}...")
    pbar.update()

#####################################
# generate descriptions from labels #
#####################################
# create descriptions for each image from its annotations
import dtlpy as dl


def create_descrip(labels: list):
    description = "a photo"
    if len(labels) != 0:
        description += " of a "
        for i, label in enumerate(labels):
            if i < len(labels) - 1:
                description += f"{label}, "
            else:
                description += f"and {label}."
    return description


# setup dtlpy
dl.setenv('prod')
if dl.token_expired():
    dl.login()

dl_dataset = dl.datasets.get(dataset_id='64c27e74615b1c5d7d576776')

# for training, adding annotations as descriptions
all_labels = dl_dataset.labels
new_label_names = [label.tag for label in all_labels]

# create text descriptions from labels
items = list(dl_dataset.items.list().all())
lookup = dict()
pbar = tqdm.tqdm(total=len(items))
for i, item in enumerate(items):
    item_name = item.name
    annotations = item.annotations.list()
    item_labels = []
    for annotation in annotations:
        item_labels.append(str(annotation.label).split(".")[-1])
    lookup[item_name] = item_labels
    pbar.update()

#########################
# create image features #
#########################
image_input = torch.tensor(np.stack(images_np)).to(device)
with torch.no_grad():
    image_features = model.encode_image(image_input)

image_features /= image_features.norm(dim=-1, keepdim=True)

# create query feature
QUERY_STRING = "cigarettes on the sidewalk"
NUM_RESULTS = 20

text_tokens = clip.tokenize([QUERY_STRING]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
text_features /= text_features.norm(dim=-1, keepdim=True)

# get top results
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

# plot returned images
num_grid = math.isqrt(NUM_RESULTS)
subplot_dims = num_grid + 1 if num_grid ** 2 < NUM_RESULTS else num_grid

plt.figure(figsize=(12, 12))
plt.tight_layout()
for i, img_path in enumerate(results_df['filepath'].iloc[:NUM_RESULTS]):
    plt.subplot(subplot_dims, subplot_dims, i + 1)
    image = Image.open(img_path).convert("RGB")
    plt.text(0, -1, f'{Path(img_path).name}', verticalalignment="bottom")
    plt.imshow(image)

plt.suptitle(f"Query: {QUERY_STRING}, found {len(results_df)}, on fine-tuned CLIP")
plot_filename = f"clip_query_results_PRETRAINED_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
save_path = os.path.join(save_dir, plot_filename)
plt.savefig(save_path)
print(f'Saved query results to {save_path}')

############
# TRAINING #
############


#############
# load data #
#############

BATCH_SIZE = 32
NUM_EPOCHS = 20
MODEL_ITERATION = "1_TACO100"


def get_data():
    data_pairs = pd.read_csv(r"C:\Users\Yaya Tang\Documents\DATASETS\TACO 100\taco_100_INPUTS_nb.csv")
    return data_pairs['filepath'], data_pairs['img_description']


class image_title_dataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.title = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        title = self.title[idx]
        return image, title


# load data
random_seed = 11
torch.manual_seed(random_seed)

list_image_path, list_txt = get_data()
dataset = image_title_dataset(list_image_path, list_txt)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


###############
# train model #
###############

# train model
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6,
                             weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

for epoch in range(NUM_EPOCHS):
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    for batch in dataloader:
        optimizer.zero_grad()

        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        # forward pass
        logits_per_image, logits_per_text = model(images, texts)

        # calc loss + backprop
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {total_loss.item():.4f}")
    pbar.update()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    },
        rf"C:\Users\Yaya Tang\PycharmProjects\clip-smart-search\checkpoints\model_{MODEL_ITERATION}_epoch_{epoch + 1}.pt")

######################################
# recreate images + query embeddings #
######################################
# create image features
model = model.eval()
image_input = torch.tensor(np.stack(images_np)).to(device)
with torch.no_grad():
    image_features = model.encode_image(image_input)

image_features /= image_features.norm(dim=-1, keepdim=True)

# create query embedding
QUERY_STRING = "cigarettes on the sidewalk"
NUM_RESULTS = 20

text_tokens = clip.tokenize([QUERY_STRING]).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
text_features /= text_features.norm(dim=-1, keepdim=True)

# get top results
results = cosine_similarity(text_features.cpu().numpy(), image_features.cpu().numpy())

results_dict = {'name': [], 'prob': [], 'filepath': []}

pbar = tqdm.tqdm(total=len(img_paths))
for i, img_path in enumerate(img_paths):
    results_dict['name'].append(Path(img_path).name)
    results_dict['prob'].append(results[0][i])
    results_dict['filepath'].append(img_path)
    results_dict.update()
    pbar.update()

results_df = pd.DataFrame(results_dict)
results_df.sort_values(by=['prob'], ascending=False, inplace=True)

results_df = results_df.iloc[:NUM_RESULTS][['name', 'prob', 'filepath']]
print(results_df[['name', 'prob']])

# plot top K nearest images to query

num_grid = math.isqrt(NUM_RESULTS)
subplot_dims = num_grid + 1 if num_grid ** 2 < NUM_RESULTS else num_grid

plt.figure(figsize=(12, 12))
plt.tight_layout()
for i, img_path in enumerate(results_df['filepath'].iloc[:NUM_RESULTS]):
    plt.subplot(subplot_dims, subplot_dims, i + 1)
    image = Image.open(img_path).convert("RGB")
    plt.text(0, -1, f'{Path(img_path).name}', verticalalignment="bottom")
    plt.imshow(image)

plt.suptitle(f"Query: {QUERY_STRING}, found {len(results_df)}, on fine-tuned CLIP")
plot_filename = f"clip_query_results_{MODEL_ITERATION}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
save_path = os.path.join(save_dir, plot_filename)
plt.savefig(save_path)
print(f'Saved query results to {save_path}')

####################################
# UMAP reduction and visualization #
####################################
# umap reduction and joining images with query
# concatenate both image + query features and reduce with UMAP

all_features = torch.cat((image_features, text_features), 0)
reducer = umap.UMAP(random_state=42, metric='cosine')
embedding = reducer.fit_transform(all_features.cpu())

# output_file(filename=r"dataloop\output\interactive_umap_TACO.html",
#             title="TACO umap plotting fastdup features")

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

# umap viz
plt.figure(figsize=(15, 10))
sns.scatterplot(x=thumbs_df['x'], y=thumbs_df['y'], hue=np.array(thumbs_df['query_returned']), palette="deep")
plt.axis('off')
plt.title('UMAP of pretrained CLIP features, with returned images')
plt.show()

plt.savefig(f"umap_TACO_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png")

#############################
# make descrips from labels #
#############################


# ###################
# # interactive viz #
# ###################
# # copied from kaggle nb
# dataset.transform = None
# image_urls = b64_image_files(images_generator(dataset))
#
# source = ColumnDataSource(data=dict(
#     x=thumbs_df['x'],
#     y=thumbs_df['y'],
#     # label=[str(l) for l in labels],
#     # prediction=[str(p) for p in predictions],
#     # success=[str(s) for s in success],
#     # desc=descs,
#     imgs=img_paths,
#     # image_urls=image_urls,
# ))
#
# TOOLTIPS = """
#     <div>
#         <div>
#             <img
#                 src="@image_urls" height="200" alt="@image_urls" width="200"
#                 style="float: left; margin: 0px 15px 15px 0px;"
#                 border="2"
#             ></img>
#         </div>
#         <div>
#             <div style="font-size: 15px; font-weight: bold;">Label: @label</div>
#             <div style="font-size: 15px; font-weight: bold;">Predicted: @prediction</div>
#             <div style="font-size: 12px; font-weight: bold;">Success: @success</div>
#             <div style="font-size: 12px;">@desc</div>
#             <div style="font-size: 12px; color: #966;">[$index]</div>
#         </div>
#     </div5
# """
#
# p = figure(plot_width=1000, plot_height=600, tooltips=TOOLTIPS,
#            title="UMAP: Mouse over the dots")
#
# mapper = factor_cmap(field_name='label', palette=Category10[5], factors=['0', '1', '2', '3', '4'])
#
# p.scatter('x', 'y',
#           color=mapper,
#           marker=factor_mark('success', ['circle', 'x'], [str(True), str(False)]),
#           size=10,
#           fill_alpha=0.5,
#           legend_field="desc",
#           source=source)
#
# p.legend.orientation = "vertical"
# p.legend.location = "top_right"
# output_file("umap.html")
# show(p)
