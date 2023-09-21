from PIL import Image, ImageFile
import dtlpy as dl
import logging
import torch
import tqdm
import time
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger('[CLIP-SEARCH]')
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ClipExtractor(dl.BaseServiceRunner):

    def __init__(self, context: dict):
        project = context.get('project')
        logger.info(f'Initinggggg for project: {project.id}, {project.name}')
        model, preprocess = clip.load("ViT-B/32", device=device)
        self.model = model
        self.preprocess = preprocess
        try:
            feature_set = project.feature_sets.get(feature_set_name='clip-image-search')
        except dl.exceptions.NotFound:
            feature_set = project.feature_sets.create(name='clip-image-search',
                                                      entity_type=dl.FeatureEntityType.ITEM,
                                                      project_id=project.id,
                                                      set_type='clip',
                                                      size=512)
        self.feature_set = feature_set

    def extract_item(self, item: dl.Item) -> dl.Item:
        logger.info(f'Started on item id: {item.id}, filename: {item.filename}')
        tic = time.time()
        # assert False
        orig_image = Image.fromarray(item.download(save_locally=False, to_array=True))
        # orig_image = Image.open(filepath)
        image = self.preprocess(orig_image).unsqueeze(0).to(device)
        image_features = self.model.encode_image(image)
        output = image_features[0].cpu().detach().numpy().tolist()
        self.feature_set.features.create(value=output, entity_id=item.id)
        logger.info(f'Done. runtime: {(time.time() - tic):.2f}[s]')
        return item

    def extract_dataset(self, dataset: dl.Dataset):
        # TODO parallel
        pages = dataset.items.list()
        pbar = tqdm.tqdm(total=pages.items_count)
        with torch.no_grad():
            for item in pages.all():
                _ = self.extract_item(item=item)
                pbar.update()


if __name__ == "__main__":
    app = dl.AppModule(name='clip-extractor',
                       description='Extract image feature for free text search')
