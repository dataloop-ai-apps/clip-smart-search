# batch_size must larger than 1

import os
import clip
import json
import logging
import shutil
import dtlpy as dl
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGenerator, DatasetGeneratorTorch
from dtlpy.utilities.reports import Report, FigOptions, ConfusionMatrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class ImageTextDataset(Dataset):
    def __init__(self, list_image_path, list_txt, preprocess):
        self.image_path = list_image_path
        # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.title = clip.tokenize(list_txt)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = self.preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        title = self.title[idx]
        return image, title


logger = logging.getLogger('[CLIP-SEARCH]')
ImageFile.LOAD_TRUNCATED_IMAGES = True


# clip available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for CLIP text and image embedding model from OpenAI')
class ClipAdapter(dl.BaseModelAdapter):
    """
    Model Adapter for CLIP text and image embedding model from OpenAI
    """

    def __init__(self, model_entity: dl.Model = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_entity is None:
            raise ValueError('model_entity must be provided')

        self.arch_name = model_entity.configuration.get("model_name", "ViT-B/32")
        if self.arch_name not in clip.available_models():
            raise ValueError(f"Model {self.arch_name} is not an available architecture for CLIP.")
        self.model_name = "CLIP " + self.arch_name

        self.feature_set = None
        self.feature_set_name = 'clip-{}-feature-set'.format(self.arch_name)

        super().__init__(model_entity=model_entity)

    def load(self, local_path, **kwargs):
        model_filename = self.configuration.get('weights_filename', 'best.pt')
        model_filepath = os.path.join(local_path, model_filename) if Path(
            model_filename).stem not in clip.available_models() \
            else model_filename
        self.model, self.preprocess = clip.load(name=self.arch_name, device=self.device)
        if os.path.isfile(model_filepath):
            checkpoint = torch.load(model_filepath, map_location=self.device)
            # Use these 3 lines if you use default model setting (not training setting) of the clip.
            # checkpoint['model_state_dict']["input_resolution"] = self.model.input_resolution  # default is 224
            # checkpoint['model_state_dict']["context_length"] = self.model.context_length  # default is 77
            # checkpoint['model_state_dict']["vocab_size"] = self.model.vocab_size

            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.info("No previously saved model found, loading from default pre-trained weights.")
        self.model.eval()
        logger.info("Loaded model {} successfully".format(self.model_name))

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally
            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        weights_filename = self.model_entity.configuration.get('weights_filename', 'best.pt')
        model_path = os.path.join(local_path, weights_filename)
        torch.save(self.model, model_path)
        logger.info("Model saved to {}".format(model_path))

    def embed(self, batch, **kwargs):
        embeddings = []
        with torch.no_grad():
            for item in batch:
                if isinstance(item, str):
                    self.adapter_defaults.upload_features = True
                    text = item
                    # TODO get the length of input text currently hardcoded to 200
                    tokens = clip.tokenize([text[:200]], context_length=77).to(self.device)
                    features = self.model.encode_text(tokens)
                elif isinstance(item, np.ndarray):
                    item_img = Image.fromarray(item)
                    image = self.preprocess(item_img).unsqueeze(0).to(self.device)
                    features = self.model.encode_image(image)
                else:
                    raise ValueError(f'Unsupported mimetype for CLIP: {type(item)}')
                embedding = features[0].cpu().detach().numpy().tolist()
                embeddings.append(embedding)
        return embeddings

    def train(self, data_path, output_path, **kwargs):
        self.model.train()

        batch_size = self.configuration.get('batch_size', 32)
        num_epochs = self.configuration.get('num_epochs', 20)
        learning_rate = self.configuration.get('learning_rate', 5e-5)
        betas = self.configuration.get('betas', (0.9, 0.98))
        episilon = self.configuration.get('episilon', 1e-6)
        weight_decay = self.configuration.get('weight_decay', 0.2)
        on_epoch_end_callback = kwargs.get('on_epoch_end_callback')

        # early stopping params
        best_loss = np.inf
        best_iter = 0
        not_improving_epochs = 0
        early_stop = self.configuration.get('early_stopping', False)
        early_stopping_epochs = self.configuration.get('early_stopping_epochs', 5)

        # os.makedirs(output_path, exist_ok=True)
        # os.makedirs(os.path.join(output_path, 'weights'), exist_ok=True)

        self.model.to(device=self.device)
        logger.info("Model set to train mode.")

        ################
        # prepare data #
        ################

        train_filter = self.model_entity.metadata['system']['subsets']['train']['filter']
        val_filter = self.model_entity.metadata['system']['subsets']['validation']['filter']

        train_items = dataset.items.download(filters=dl.Filters(custom_filter=train_filter))
        val_items = dataset.items.download(filters=dl.Filters(custom_filter=val_filter))

        train_dataset = ImageTextDataset(*self.get_image_text_pairs(os.path.join(data_path, 'train')),
                                         self.preprocess)
        val_dataset = ImageTextDataset(*self.get_image_text_pairs(os.path.join(data_path, 'validation')),
                                       self.preprocess)

        dataloaders = {'train': DataLoader(train_dataset,
                                           batch_size=batch_size),
                       'val': DataLoader(val_dataset,
                                         batch_size=batch_size)}
        logger.debug("Train and validation data loaders created")

        #################
        # prepare model #
        #################
        if self.device == "cpu":
            self.model.float()

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=learning_rate,
                                     betas=betas,
                                     eps=episilon,
                                     weight_decay=weight_decay)

        for epoch in range(num_epochs):
            with tqdm(dataloaders['train'], unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{num_epochs}...")
                for idx, batch in enumerate(tepoch):
                    optimizer.zero_grad()

                    images, texts = batch
                    images = images.to(self.device)
                    texts = texts.to(self.device)

                    # forward pass
                    logits_per_image, logits_per_text = self.model(images, texts)

                    # calc loss + backprop
                    ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
                    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text,
                                                                                      ground_truth)) / 2
                    total_loss.backward()
                    if self.device == "cpu":
                        optimizer.step()
                    else:
                        convert_models_to_fp32(self.model)
                        optimizer.step()
                        clip.model.convert_weights(self.model)
                    tepoch.set_postfix(Training_loss=f"{total_loss.item():.4f}")

            with tqdm(dataloaders['val'], unit="batch") as vepoch:
                vepoch.set_description("  Validation...")
                for batch in vepoch:
                    optimizer.zero_grad()
                    images, texts = batch
                    images = images.to(self.device)
                    texts = texts.to(self.device)

                    # forward pass
                    logits_per_image, logits_per_text = self.model(images, texts)

                    # calc loss
                    ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
                    val_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                    vepoch.set_postfix(Validation_loss=f"{val_loss.item():.4f}")

            if val_loss < best_loss:
                not_improving_epochs = 0
                best_loss = val_loss
                best_iter = epoch + 1
                logger.info(
                    f'Validation loss decreased ({best_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
                # torch.save(self.model.state_dict(), os.path.join(output_path, 'best.pth'))  # saving ckpt TODO remove and break
                break
            else:
                not_improving_epochs += 1
            if not_improving_epochs > early_stopping_epochs and early_stop:
                if ((epoch + 1) - best_iter) > early_stopping_epochs:
                    print("Early stop achieved at epoch ", epoch + 1)
                    break

    # def prepare_item_func(self, item: dl.Item) -> dl.Item:
    #     try:
    #         caption = item.description
    #     except TypeError:
    #         logger.warning(f"Item {item.id} has no description. Using blank description.")
    #         caption = ''
    #     new_name = Path(item.name).stem + '.json'
    #     new_path = '/promptItems' + item.dir
    #
    #     prompt_item = dl.PromptItem(name=new_name)
    #     prompt = dl.Prompt(key='image_text')
    #     prompt.add_element(mimetype=dl.PromptType.IMAGE, value=item.stream)
    #     prompt.add_element(mimetype=dl.PromptType.TEXT, value=caption)
    #     prompt_item.prompts.append(prompt)
    #
    #     new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=new_path)
    #     return new_item

    def convert_from_dtlpy(self, data_path, **kwargs):
        # Subsets validation
        subsets = self.model_entity.metadata.get("system", dict()).get("subsets", None)
        if 'train' not in subsets:
            raise ValueError('Could not find train set. CLIP requires train and validation set for training. '
                             'Add a train set DQL filter in the dl.Model metadata')
        if 'validation' not in subsets:
            raise ValueError('Could not find validation set. CLIP requires train and validation set for training. '
                             'Add a validation set DQL filter in the dl.Model metadata')

        for subset, filters_dict in subsets.items():
            filters = dl.Filters(custom_filter=filters_dict)
            filters.add_join(field='type', values='text')
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(f'Could not find free-text annotations in subset {subset}. '
                                 f'Cannot train without annotations in the data subsets.')

        self.move_annotation_files(os.path.join(data_path, 'train'))
        self.move_annotation_files(os.path.join(data_path, 'validation'))

    @staticmethod
    def move_annotation_files(data_path):
        logger.debug(f"Data path: {data_path}")
        path = Path(data_path)
        json_files = (path / 'json').rglob("*.json")
        logger.debug(f"Json files: {json_files}")
        img_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
        item_files = []
        for ext in img_extensions:
            item_files += (path / 'items').rglob(f"*.{ext}")
        for src, dst in zip([json_files, item_files], ['json', 'items']):
            for src_file in src:
                if not os.path.exists(os.path.join(data_path, dst, os.path.basename(src_file))):
                    shutil.move(src_file, os.path.join(data_path, dst, os.path.basename(src_file)))
        for root, dirs, files in os.walk(data_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    @staticmethod
    def get_image_text_pairs(data_path):
        logger.debug(f"Data path: {data_path}")
        path = Path(data_path)
        json_files = (path / 'json').rglob("*.json")
        logger.debug(f"Json files: {json_files}")
        img_extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
        item_files = []
        item_captions = []
        for ext in img_extensions:
            item_files += (path / 'items').rglob(f"*.{ext}")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            item_captions.append(data['description'])
        return item_files, item_captions


if __name__ == "__main__":
    # dl.setenv('prod')
    # project = dl.projects.get(project_name='Model mgmt demo')
    # model = project.models.get(model_id='670ccfa3bb9423df793e5216')
    # model.configuration['model_name'] = 'ViT-B/32'
    # model.configuration['featureSetName'] = 'wohoooooooooox'
    # model.configuration['embeddings_size'] = 512
    # model.name = 'CLIP ViT-B/32'
    #
    # app = ClipAdapter(model_entity=model)
    # dataset = project.datasets.get(dataset_name='taco mini')
    # item = dataset.items.get(item_id='670cc97f74e80d85f07e950c')
    # # app.embed_items(items=[item])
    # new_model = model.clone(model_name=model.name+' FT', dataset=dataset)
    #
    # app.train_model(model=new_model)

    dl.setenv('rc')
    project = dl.projects.get(project_name='smart image search')
    dataset = project.datasets.get(dataset_name='TACO 100')
    # item = dataset.items.get(item_id='670cc97f74e80d85f07e950c')
    # model = project.models.get(model_id='670ebac88834bc76cf60abe1')  # yolov8

    model = project.models.get(model_id='670ebac88834bc76cf60abe1')  # yolo model
    # model = project.models.create(model_name='CLIP ViT-B/32', model_type='adapter')
    model.configuration = {'model_name': 'ViT-B/32',
                           'embeddings_size': 512,
                           'num_epochs': 20}
    model_filters = model.metadata.get('system', None)
    if model_filters is None:
        model.metadata['system'] = {}
    model.metadata['system']['subsets'] = {}

    train_filters = dl.Filters()
    train_filters.add(field='metadata.system.tags.train', values=True)
    val_filters = dl.Filters()
    val_filters.add(field='metadata.system.tags.validation', values=True)

    model.metadata['system']['subsets']['train'] = train_filters.prepare()
    model.metadata['system']['subsets']['validation'] = val_filters.prepare()
    # model.input_type = ['image', 'text']
    # model.output_type = ['box', 'classification']
    model.name = 'CLIP ' + model.configuration['model_name']

    app = ClipAdapter(model_entity=model)
    new_model = model.clone(model_name=model.name + ' SFT', dataset=dataset)
    new_model.output_type = 'text'
    app.train_model(model=new_model)