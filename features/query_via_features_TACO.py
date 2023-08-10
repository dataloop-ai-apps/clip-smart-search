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


# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# info about preprocessing for the model:
# dataset images: center crop & normalize pixel intensity
# dataset texts: padded to 77 tokens long

dl.setenv('prod')
ONLINE = False

if ONLINE is True:
    # download the image, and then download the labels in the image
    dataset = dl.datasets.get(dataset_id='64c27e74615b1c5d7d576776')  # TACO trash
    img_paths = dataset.download(annotation_options=dl.VIEW_ANNOTATION_OPTIONS_JSON)  # data paths
    img_dir = Path(list(img_paths)[0]).parent
else:
    img_dir = Path(r'C:\Users\Yaya Tang\Documents\DATASETS\TACO dataloop\items\raw')
    img_paths = [str(img_path) for img_path in img_dir.glob("*")]

# for training, adding annotations as descriptions
# all_labels = dataset.labels
# new_label_names = [label.tag for label in all_labels]
#
# # create text descriptions from labels
# items = dataset.items.list().all()
# item_labels_lookup = {}
# for item in items:
#     item_name = item.name
#     annotations = item.annotations.list()
#     item_labels = []
#     for annotation in annotations:
#         item_labels.append(str(annotation.label).split(".")[-1])
#     item_labels_lookup[item_name] = item_labels


# embed feature vectors for each image + the query string
# feature_vectors = {}

# prepare images
images = []
print(img_paths)

pbar = tqdm.tqdm(total=len(img_paths))
for img_path in img_paths:
    image = Image.open(img_path).convert("RGB")
    images.append(preprocess(image))
    pbar.update()

# create image features
image_input = torch.tensor(np.stack(images)).to(device)
with torch.no_grad():
    image_features = model.encode_image(image_input)
    # text_features = model.encode_text(text_tokens)
    # logits_per_image, logits_per_text = model(images, text_features)  # TODO what do these lines do exactly?
    # probs = logits_per_text.softmax(dim=-1).cpu().numpy()
image_features /= image_features.norm(dim=-1, keepdim=True)

# create text/query feature
QUERY_STRING = "food container"

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
