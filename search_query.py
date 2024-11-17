import logging
import os
import datetime
import torch
import tqdm
import dtlpy as dl
import numpy as np
import pandas as pd

from clip import clip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from model_adapter import ClipAdapter

dl.setenv('prod')
logger = logging.getLogger('clip-smart-search')

project = dl.projects.get('Model mgmt demo')
dataset = project.datasets.get(dataset_name='taco 100')
model_entity_id = "6739b9f92ebe8b4bdd2a4a2c"

QUERY_STRING = "cigarette butts on the ground"
DEVICE = "cpu"

###################
# download images #
###################
img_paths = dataset.download(local_path=os.path.join(os.getcwd(), '.dataloop'),
                             annotation_options=dl.VIEW_ANNOTATION_OPTIONS_JSON,
                             overwrite=False)
imgs_list = list(img_paths)
# image_features = project.feature_sets.get(feature_set_name=model_entity.name)

#######################
# load original model #
#######################
# get original model features
device = "cuda" if torch.cuda.is_available() else "cpu"
model_orig, preprocess_orig = clip.load("ViT-B/32", device=DEVICE)
# data_path = r"C:\Users\Yaya Tang\PycharmProjects\clip-smart-search\tmp\6731d515bba6dc4ca4667227\datasets\672cc229e773c08bacdfedad\train"
# img_paths, _ = ClipAdapter.get_images_and_text(data_path=data_path, overwrite=False)

########################
# load finetuned model #
########################
# model_entity = dl.models.get(model_id=model_entity_id)
# model_entity.artifacts.download(local_path='.', overwrite=True)
model_path_sft = os.path.join(os.getcwd(), 'best.pt')

# embed images
app = ClipAdapter()
model_entity = dl.models.get(model_id="673334351881e27f94cbb1ca")
app.device = "cpu"
app.load_from_model(model_entity=model_entity, overwrite=False)
app.embed_dataset(dataset=dataset, upload_features=True)

# # get finetuned model features
model_sft, preprocess_sft = clip.load("ViT-B/32", device=DEVICE, jit=False)
checkpoint = torch.load(model_path_sft, map_location=DEVICE, weights_only=True)
model_sft.load_state_dict(checkpoint['model_state_dict'])
model_sft.eval()


def get_features(imgs_list, model, preprocess):
    # get the feature vectors
    images = []
    pbar = tqdm.tqdm(total=len(imgs_list))
    for img_path in imgs_list:
        image = Image.open(img_path).convert("RGB")
        images.append(preprocess(image))
        pbar.update()

    image_input = torch.tensor(np.stack(images)).to(DEVICE)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features


def query_result(imgs_list, model, query_string, image_features):
    # create text/query feature
    text_tokens = clip.tokenize(query_string).to("cpu")
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

    return results_df


# compare search results between two models
img_fts_orig = get_features(imgs_list, model_orig, preprocess_orig)
img_fts_sft = get_features(imgs_list, model_sft, preprocess_sft)

results_orig = query_result(imgs_list, model_orig, QUERY_STRING, img_fts_orig)
results_sft = query_result(imgs_list, model_sft, QUERY_STRING, img_fts_sft)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
fig.suptitle(f"Query: {QUERY_STRING}", fontsize=16)  # Add a window title

# Display images from the first DataFrame in the first row
for i in range(3):
    img = mpimg.imread(results_orig['filepath'].iloc[i])
    axes[0, i].imshow(img)
    axes[0, i].axis('off')
    axes[0, i].set_title(f'clip - Best matching image {i + 1}')

# Display images from the second DataFrame in the second row
for i in range(3):
    img = mpimg.imread(results_sft['filepath'].iloc[i])
    axes[1, i].imshow(img)
    axes[1, i].axis('off')
    axes[1, i].set_title(f'SFT - Best matching image {i + 1}')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
plt.show()
plt.savefig(f'clip_vs_sft_{datetime.datetime.now().strftime("%Y_%m_%d-T%H_%M_%S")}.png')



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
