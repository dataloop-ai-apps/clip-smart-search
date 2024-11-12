import logging
import os
import torch
import tqdm
import dtlpy as dl
import numpy as np
import pandas as pd

from clip import clip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from model_adapter import ClipAdapter
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

dl.setenv('rc')
logger = logging.getLogger('clip-smart-search')

project = dl.projects.get('smart image search')
dataset = project.datasets.get(dataset_name='TACO 100')
model_entity = dl.models.get(model_id='6732dfe92aa895346cc469e9')  # trained tuesday morning

# embed images
# app = ClipAdapter()
# app.device = torch.cpu
# app.load_from_model(model_entity=model_entity, overwrite=False)
# app.embed_dataset(dataset=dataset, upload_features=True)

model_path = 'best.pt'
model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)
checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# get the feature vectors
# image_features = project.feature_sets.get(feature_set_name=model_entity.name)

img_paths = dataset.download(local_path=os.path.join(os.getcwd(), '.dataloop'),
                             annotation_options=dl.VIEW_ANNOTATION_OPTIONS_JSON,
                             overwrite=False)
imgs_list = list(img_paths)

images = []
pbar = tqdm.tqdm(total=len(imgs_list))
for img_path in imgs_list:
    image = Image.open(img_path).convert("RGB")
    images.append(preprocess(image))
    pbar.update()

# create image features
image_input = torch.tensor(np.stack(images)).to("cpu")
with torch.no_grad():
    image_features = model.encode_image(image_input)
    # text_features = model.encode_text(text_tokens)
    # logits_per_image, logits_per_text = model(images, text_features)
    # probs = logits_per_text.softmax(dim=-1).cpu().numpy()
image_features /= image_features.norm(dim=-1, keepdim=True)

# create text/query feature
QUERY_STRING = "alumininum can outside"

text_tokens = clip.tokenize([QUERY_STRING]).to("cpu")
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Pick the top 10 most similar images for the text/query
result = cosine_similarity(text_features.cpu().numpy(), image_features.cpu().numpy())

results_dict = {'name': [], 'prob': [], 'filepath': []}

pbar = tqdm.tqdm(total=len(imgs_list))
for i, img_path in enumerate(imgs_list):
    results_dict['name'].append(Path(img_path).name)
    results_dict['prob'].append(result[0][i])
    results_dict['filepath'].append(img_path)
    results_dict.update()
    pbar.update()

results_df = pd.DataFrame(results_dict)
results_df.sort_values(by=['prob'], ascending=False, inplace=True)

print(results_df.iloc[:9][['name', 'prob']])


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.suptitle(f"Query: {QUERY_STRING}")

for i, ax in enumerate(axes):
    img = mpimg.imread(results_df['filepath'].iloc[i])
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Best matching image {i + 1}')

plt.tight_layout()
plt.show()


# plt.figure(figsize=(16, 16))
# for i, img_path in enumerate(results_df['filepath'].iloc[:9]):
#     plt.subplot(3, 3, i + 1)
#     image = Image.open(img_path).convert("RGB")
#     plt.imshow(image)
#
# plt.suptitle(f"Query: {QUERY_STRING}")
# plt.show()


def create_feature_set(self, project: dl.Project):
    try:
        feature_set = project.feature_sets.get(feature_set_name=self.feature_set_name)
        logger.info(f'Feature Set found! name: {feature_set.name}, id: {feature_set.id}')
    except dl.exceptions.NotFound:
        logger.info('Feature Set not found. creating...')
        feature_set = project.feature_sets.create(name=self.feature_set_name,
                                                  entity_type=dl.FeatureEntityType.ITEM,
                                                  project_id=project.id,
                                                  set_type='clip',
                                                  size=512)
        logger.info(f'Feature Set created! name: {feature_set.name}, id: {feature_set.id}')
        # binaries = project.datasets._get_binaries_dataset()
        # buffer = BytesIO()
        # buffer.write(json.dumps({}, default=lambda x: None).encode())
        # buffer.seek(0)
        # buffer.name = "clip_feature_set.json"
        # feature_set_item = binaries.items.upload(
        #     local_path=buffer,
        #     item_metadata={
        #         "system": {
        #             "clip_feature_set_id": feature_set.id
        #         }
        #     }
        # )
    self.feature_set = feature_set
