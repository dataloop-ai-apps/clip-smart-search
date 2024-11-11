# batch_size must larger than 1

import os
import clip
import json
import logging
import shutil
import dtlpy as dl
import numpy as np
from glob import glob
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


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


class ClipAdapter(dl.BaseModelAdapter):
    """
    Model Adapter for CLIP text and image embedding model from OpenAI
    """

    def load(self, local_path, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.arch_name = self.model_entity.configuration.get("model_name", "ViT-B/32")
        if self.arch_name not in clip.available_models():
            raise ValueError(f"Model {self.arch_name} is not an available architecture for CLIP.")
        self.model_name = "CLIP " + self.arch_name

        self.weights_filename = self.configuration.get('weights_filename', 'best.pt')
        model_filepath = os.path.join(local_path, self.weights_filename) if Path(
            self.weights_filename).stem not in clip.available_models() \
            else self.weights_filename

        self.model, self.preprocess = clip.load(name=self.arch_name, device=self.device)
        if os.path.isfile(model_filepath) is True:
            # self.model, self.preprocess = clip.load(name=model_filepath, device=self.device)
            checkpoint = torch.load(model_filepath)
            # Use these 3 lines if you use default model setting (not training setting) of the clip.
            checkpoint["input_resolution"] = self.model.input_resolution  # default is 224
            checkpoint["context_length"] = self.model.context_length  # default is 77
            checkpoint["vocab_size"] = self.model.vocab_size
            self.model.load_state_dict(checkpoint)
        else:
            logger.info("No previously saved model found, loading from default pre-trained weights.")
        self.model.eval()
        logger.info("Loaded model {} successfully".format(self.model_name))

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally
            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        model_path = os.path.join(local_path, self.weights_filename)
        torch.save({'model_state_dict': self.model.state_dict()}, model_path)
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
        not_improving_epochs = 0
        early_stop = self.configuration.get('early_stopping', True)
        early_stopping_epochs = self.configuration.get('early_stopping_epochs', 5)

        self.model.to(device=self.device)
        logger.info("Model set to train mode.")

        ################
        # prepare data #
        ################
        train_items, train_captions = self.move_and_download_images(os.path.join(data_path, 'train'))
        val_items, val_captions = self.move_and_download_images(os.path.join(data_path, 'validation'))
        train_dataset = ImageTextDataset(train_items, train_captions, self.preprocess)
        val_dataset = ImageTextDataset(val_items, val_captions, self.preprocess)

        dataloaders = {'train': DataLoader(train_dataset,
                                           batch_size=batch_size),
                       'val': DataLoader(val_dataset,
                                         batch_size=batch_size)}
        logger.info(
            f"Dataloaders created. Train dataset: {len(train_dataset)} items, validation dataset: {len(val_dataset)} items.")

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
                logger.info(
                    f'Validation loss decreased ({best_loss:.4f} --> {val_loss:.4f}). Saving model ...')
                torch.save(self.model.state_dict(), os.path.join(output_path, self.weights_filename))
            else:
                not_improving_epochs += 1
            if not_improving_epochs > early_stopping_epochs and early_stop is True:
                logger.info("Early stop achieved at epoch ", epoch + 1)
                break
        return

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
            pages = self.model_entity.dataset.items.list(filters=filters)
            if pages.items_count == 0:
                raise ValueError(f'Could not find free-text annotations in subset {subset}. '
                                 f'Cannot train without annotations in the data subsets.')

    @staticmethod
    def move_and_download_images(data_path):
        logger.debug(f"Data path: {data_path}")
        path = Path(data_path)

        def download_stream(item_file):
            with open(item_file) as json_data:
                d = json.load(json_data)
            caption_info = d['prompts']['img_caption']

            item_url = None
            for dictionary in caption_info:
                if dictionary.get("mimetype") == "image/*":
                    item_url = dictionary.get("value")
            item_id = item_url.split('/')[-2]
            item = dl.items.get(item_id=item_id)
            download_path = item.download(local_path=Path(item_file).parents[0])
            image_name = Path(item_file).stem + Path(download_path).suffix
            new_path = Path(item_file).parents[0] / image_name
            os.rename(Path(download_path), new_path)
            return new_path

        item_jsons = (path / "items").rglob("*.json")
        # with ThreadPoolExecutor() as executor:
        #     item_images = [result for result in executor.map(download_stream, item_jsons)]
        item_images = [] # DEBUG
        for item_file in item_jsons:
            item_images.append(download_stream(item_file))

        item_captions = []
        json_files = (path / 'json').rglob("*.json")
        for all_files, json_type in zip([json_files, item_images], ['json', 'items']):
            for src_file in all_files:
                if json_type == 'json':
                    with open(src_file, 'r') as f:
                        data = json.load(f)
                    annotations = data['annotations']
                    for annot in annotations:
                        if annot['label'] == 'free-text':
                            item_captions.append(annot.get('coordinates', ''))
                        else:
                            logger.debug(f"No free-text annotation found in json file {src_file}.")
                            item_captions.append('')
        for root, dirs, files in os.walk(data_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
        return item_images, item_captions


if __name__ == "__main__":
    dl.setenv('rc')
    project = dl.projects.get(project_name='smart image search')

    # dataset = project.datasets.get(dataset_name='TACO 100 prompt items')
    dataset = project.datasets.get(dataset_name='TACO 3 prompt items')
    model = project.models.get(model_name='clip-smart-search')

    # dl.setenv('prod')
    # project = dl.projects.get(project_name='Model mgmt demo')
    # dataset = project.datasets.get(dataset_name='TACO 100 prompt items')

    model.metadata['system'] = {}
    model.metadata['system']['subsets'] = {}

    train_filters = dl.Filters(field='metadata.system.tags.train', values=True)
    val_filters = dl.Filters(field='metadata.system.tags.validation', values=True)

    model.metadata['system']['subsets']['train'] = train_filters.prepare()
    model.metadata['system']['subsets']['validation'] = val_filters.prepare()
    model.name = 'CLIP ' + model.configuration['model_name']
    model.configuration = {'num_epochs': 2}

    new_model = model.clone(model_name=model.name + ' SFT', dataset=dataset)
    new_model.output_type = 'text'

    app = ClipAdapter(model_entity=new_model)
    app.train_model(model=new_model)
