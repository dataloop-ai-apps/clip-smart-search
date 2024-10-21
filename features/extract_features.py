import tqdm
from PIL import Image, ImageFile
import dtlpy as dl
import logging
import torch
import time
import clip
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os

logger = logging.getLogger('[CLIP-SEARCH]')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClipExtractor(dl.BaseServiceRunner):
    def __init__(self, project=None):
        if project is None:
            project = self.service_entity.project
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.feature_set = None
        self.feature_set_name = 'clip-feature-set'
        # self.feature_vector_entities = list()
        self.create_feature_set(project=project)

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
            binaries = project.datasets._get_binaries_dataset()
            buffer = BytesIO()
            buffer.write(json.dumps({}, default=lambda x: None).encode())
            buffer.seek(0)
            buffer.name = "clip_feature_set.json"
            feature_set_item = binaries.items.upload(
                local_path=buffer,
                item_metadata={
                    "system": {
                        "clip_feature_set_id": feature_set.id
                    }
                }
            )
        self.feature_set = feature_set
        # self.feature_vector_entities = [fv.entity_id for fv in self.feature_set.features.list().all()]

    def extract_from_data(self, image: Image.Image = None, text=None):
        if image is not None:
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            features = self.model.encode_image(image)
        elif text is not None:
            tokens = clip.tokenize([text], context_length=77).to(self.device)
            features = self.model.encode_text(tokens)
        else:
            raise ValueError('Either image or text is required')
        return features[0].cpu().detach().numpy().tolist()

    def extract_item(self, item: dl.Item) -> dl.Item:
        # if item.id in self.feature_vector_entities:
        #     logger.info(f'Item {item.id} already has feature vector')
        #     return item
        logger.info(f'Started on item id: {item.id}, filename: {item.filename}')
        tic = time.time()
        # assert False
        if 'image/' in item.mimetype:
            orig_image = Image.fromarray(item.download(save_locally=False, to_array=True))
            # orig_image = Image.open(filepath)
            features = self.extract_from_data(image=orig_image)
        elif 'text/' in item.mimetype:
            text = item.download(save_locally=False).read().decode()
            # TODO get the length of input text currently hardcoded to 200
            features = self.extract_from_data(text=text[:200])
        else:
            raise ValueError(f'Unsupported mimetype for clip: {item.mimetype}')
        output = features[0].cpu().detach().numpy().tolist()
        try:
            self.feature_set.features.create(value=output, entity=item)
        except dl.exceptions.BadRequest:
            logger.error(f'Error creating feature vector for item id: {item.id}, feature vector already exists!')
        logger.info(f'Done. runtime: {(time.time() - tic):.2f}[s]')
        return item

    def extract_dataset(self, dataset: dl.Dataset, query=None, progress=None):
        filters = dl.Filters()
        filters.add(field='metadata.system.mimetype', values="image/*", method=dl.FILTERS_METHOD_OR)
        filters.add(field='metadata.system.mimetype', values="text/*", method=dl.FILTERS_METHOD_OR)

        items_path = os.path.join(os.getcwd(), dataset.id)
        pages = dataset.items.list(filters=filters)

        pbar = tqdm.tqdm(total=pages.items_count)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.extract_item, obj) for obj in pages.all()]
            done_count = 0
            previous_update = 0
            while futures:
                done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                done_count += len(done)
                pbar.update(len(done))
                current_progress = done_count * 100 // pages.items_count

                if (current_progress // 10) % 10 > previous_update:
                    previous_update = (current_progress // 10) % 10
                    if progress is not None:
                        progress.update(progress=previous_update * 10)
                    else:
                        logger.info(f'Extracted {done_count} out of {pages.items_count} items')
        return dataset


if __name__ == "__main__":
    project = dl.projects.get(project_id='')
    app = ClipExtractor(project=project)
    dataset = dl.datasets.get(dataset_id='')
    app.extract_dataset(dataset=dataset)
