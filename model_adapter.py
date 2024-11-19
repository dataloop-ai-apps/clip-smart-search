# batch_size must larger than 1

import os
import clip
import json
import logging
import time
import dtlpy as dl
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('[CLIP-SEARCH]')
ImageFile.LOAD_TRUNCATED_IMAGES = True


# clip available models: ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

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


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


class ClipAdapter(dl.BaseModelAdapter):
    """
    Model Adapter for CLIP text and image embedding model from OpenAI
    """

    def load(self, local_path, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.arch_name = self.configuration.get("model_name", "ViT-B/32")
        if self.arch_name not in clip.available_models():
            raise ValueError(f"Model {self.arch_name} is not an available architecture for CLIP.")
        self.model_name = "CLIP " + self.arch_name
        self.weights_filename = self.configuration.get('weights_filename', 'best.pt')
        model_filepath = os.path.join(local_path, self.weights_filename) if Path(
            self.weights_filename).stem not in clip.available_models() \
            else self.weights_filename

        self.model, self.preprocess = clip.load(name=self.arch_name, device=self.device)
        if os.path.isfile(model_filepath) is True and self.model_entity.status != 'pre-trained':
            checkpoint = torch.load(model_filepath, map_location=self.device)
            # Use these 3 lines if you use default model setting (not training setting) of the clip.
            # checkpoint["input_resolution"] = self.model.input_resolution  # default is 224
            # checkpoint["context_length"] = self.model.context_length  # default is 77
            # checkpoint["vocab_size"] = self.model.vocab_size
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
        end_training = False

        self.model.to(device=self.device)
        logger.info("Model set to train mode.")

        ################
        # prepare data #
        ################
        train_items, train_captions = self.get_images_and_text(os.path.join(data_path, 'train'))
        val_items, val_captions = self.get_images_and_text(os.path.join(data_path, 'validation'))
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
            if end_training:
                break
            logger.info('Epoch {}/{} Start...'.format(epoch, num_epochs))
            tepoch_time = time.time()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                total_imgs = 0

                with tqdm(dataloaders[phase], unit='batch') as tepoch:
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
                        if phase == 'train':
                            total_loss.backward()

                            if self.device == "cpu":
                                optimizer.step()
                            else:
                                convert_models_to_fp32(self.model)
                                optimizer.step()
                                clip.model.convert_weights(self.model)
                            tepoch.set_postfix(Training_loss=f"{total_loss.item():.4f}")

                        # statistics
                        total_imgs += images.size(0)
                        running_loss += (total_loss.item() * images.size(0))
                        epoch_loss = running_loss / total_imgs

                        if phase == "val":
                            val_loss = epoch_loss

                    logger.info(
                        f'Epoch {epoch}/{num_epochs} - {phase} loss: {total_loss.item():.4f}, Duration {(time.time() - tepoch_time):.2f}')

                    self.model_entity.metrics.create(samples=dl.PlotSample(figure='loss',
                                                                           legend=phase,
                                                                           x=epoch,
                                                                           y=epoch_loss),
                                                     dataset_id=self.model_entity.dataset_id)

            if val_loss < best_loss:
                not_improving_epochs = 0
                best_loss = val_loss
                logger.info(
                    f'Best validation loss decreased ({best_loss:.4f} --> {val_loss:.4f}). Saving model ...')
                torch.save({'model_state_dict': self.model.state_dict()},
                           os.path.join(output_path, self.weights_filename))
            else:
                not_improving_epochs += 1
            if not_improving_epochs > early_stopping_epochs and early_stop is True:
                logger.info("Early stop achieved at epoch ", epoch + 1)
                end_training = True
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
                raise ValueError(f'Could not find items with free-text annotations in subset {subset}. '
                                 f'Make sure there are items with annotations in the data subsets.')

    @staticmethod
    def get_images_and_text(data_path, overwrite=False):
        logger.debug(f"Data path: {data_path}")
        path = Path(data_path)

        def download_stream(item_file, overwrite=False):
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
            try:
                os.rename(Path(download_path), new_path)
            except FileExistsError:
                if overwrite is True:
                    logger.debug(f"Overwriting file {new_path}.")
                    os.remove(new_path)
                    os.rename(Path(download_path), new_path)
                else:
                    logger.debug(f"File {new_path} already exists. Skipping.")
            return new_path

        item_jsons = (path / "items").rglob("*.json")
        with ThreadPoolExecutor() as executor:
            image_paths = list(executor.map(lambda item_file: download_stream(item_file, overwrite), item_jsons))
        # image_paths = []  # DEBUG
        # for item_file in item_jsons:
        #     image_paths.append(download_stream(item_file, overwrite))

        item_captions = []
        json_files = (path / 'json').rglob("*.json")
        for all_files, json_type in zip([json_files, image_paths], ['json', 'items']):
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
        return image_paths, item_captions

    @staticmethod
    def upload_items_with_description(dataset, pairs_filepath: str):
        # description will be uploaded from csv w/ image filepath + text
        data_pairs = pd.read_csv(pairs_filepath)
        for index, row in data_pairs.iterrows():
            file_path = row['filepath']
            annots_path = file_path.replace('items', 'json')
            file_name = Path(file_path).name
            item = dataset.items.upload(local_path=file_path,
                                        local_annotations_path=annots_path,
                                        item_metadata=dl.ExportMetadata.FROM_JSON,
                                        overwrite=True)
            item.set_description(text=row['img_description'])
            item.update()
            print(f"Uploaded {file_name} with description: '{row['img_description']}'")
        return True

    @staticmethod
    def convert_dataset_for_clip(dataset_src, filters=None, existing_subsets=True):
        """
        Converts a dataset of images with descriptions to a dataset of prompt items for CLIP
        :param dataset_src: dl.Dataset
        :param filters: dl.Filters for retrieving the relevant items from dataset (optional)
        :param existing_subsets: optional boolean to keep existing subsets from original items (default: True)
        :return: dataset with prompt items
        """
        project = dataset_src.project
        dataset_name = dataset_src.name + "PROMPT ITEMS"
        try:
            new_dataset = project.datasets.get(dataset_name=dataset_name)
        except dl.exceptions.NotFound:
            new_dataset = project.datasets.create(dataset_name=dataset_name)

        items = dataset_src.items.list(filters=filters)
        for item in items.all():
            item = dataset_src.items.get(item_id=item.id)
            prompt_item = _convert_item(item, new_dataset, existing_subsets)
        new_dataset.switch_recipe(dataset_src.get_recipe_ids()[0])
        return new_dataset


def _convert_item(item_src: dl.Item, dataset: dl.Dataset = None, prompt_key=None, existing_subsets=False):
    if dataset is None:
        dataset = item_src.dataset
    try:
        caption = item_src.description
    except TypeError:
        print(f"Item {item_src.id} has no description. Using blank description.")
        caption = ''
    new_name = Path(item_src.name).stem + '.json'

    prompt_item = dl.PromptItem(name=new_name)
    if prompt_key is None:
        prompt_key = "img_caption"
    prompt = dl.Prompt(key=prompt_key)
    prompt.add_element(mimetype=dl.PromptType.IMAGE, value=item_src.stream)
    prompt_item.prompts.append(prompt)

    new_item = dataset.items.upload(prompt_item,
                                    remote_name=new_name,
                                    remote_path=item_src.dir,
                                    overwrite=True)
    annotations = dl.AnnotationCollection()
    annotations.add(annotation_definition=dl.FreeText(text=caption),
                    prompt_id=prompt_key,
                    model_info={'name': 'GT',
                                'confidence': 1})
    new_item.annotations.upload(annotations)
    if existing_subsets is True:
        try:
            new_item.metadata['system']['subsets'] = item_src.metadata['system']['subsets']
        except KeyError:
            new_item.metadata['system'] = {}
            new_item.metadata['system']['subsets'] = item_src.metadata['system']['subsets']
        new_item.update()
    return new_item
