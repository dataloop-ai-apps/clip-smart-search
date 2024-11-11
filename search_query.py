import logging
import os
import dtlpy as dl

from PIL import Image
from model_adapter import ClipAdapter

logger = logging.getLogger('clip-smart-search')

project = dl.projects.get('smart image search')
# model_entity = project.models.get(model_name='CLIP ViT-B/32 SFT-PSicV')
model_entity = project.models.get(model_name='clip-smart-search-o44in_2024_11_11-T11_03_29')

app = ClipAdapter()
app.load_from_model(model_entity=model_entity, overwrite=False)

dataset = project.datasets.get(dataset_name='TACO 100')
app.embed_dataset(dataset=dataset, upload_features=True)



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


def homemade_embed():
    data_path = dataset.download(local_path=os.path.join(os.getcwd(), '.dataloop'),
                                 annotation_options=dl.VIEW_ANNOTATION_OPTIONS_JSON)
    img_paths, captions = app.get_images_and_text(data_path=data_path)

    images = []
    for img_path in img_paths:
        img = Image.open(img_path)
        images.append(app.preprocess(img))

    embeddings = app.embed(images)

    # upload the feature vectors

    app._upload_model_features()


# features = dataset.features.get(feature_name=model_entity.name)


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
