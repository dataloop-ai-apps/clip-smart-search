# batch_size must larger than 1

import os
import clip
import logging
import dtlpy as dl
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from dtlpy.utilities.dataset_generators.dataset_generator_torch import DatasetGeneratorTorch
from dtlpy.utilities.reports import Report, FigOptions, ConfusionMatrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


logger = logging.getLogger('[CLIP-SEARCH]')
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dl.Package.decorators.module(name='model-adapter',
                              description='Model Adapter for CLIP text and image embedding model from OpenAI')
class ClipAdapter(dl.BaseModelAdapter):
    """
    Model Adapter for CLIP text and image embedding model from OpenAI
    """
    def __init__(self, project: dl.Project = None, model_entity: dl.Model = None):
        if project is None:
            project = self.service_entity.project
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.feature_set = None
        self.feature_set_name = 'clip-feature-set'
        # self.feature_vector_entities = list()
        self.create_feature_set(project=project)

    def load(self, local_path, **kwargs):
        weights_filename = self.model_entity.configuration.get('weights_filename', 'model.pth')

        # load model arch and state
        model_path = os.path.join(local_path, weights_filename)
        logger.info("Loading a model from {}".format(local_path))

        if os.path.exists(model_path):
            self.model = torch.load(model_path, map_location=self.device)
            _, self.preprocess = clip.load("ViT-B/32", device=self.device)
        else:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.model.to(self.device)
        # self.model.eval()
        logger.info("Loaded model from {} successfully".format(model_path))

    def save(self, local_path, **kwargs):
        weights_filename = self.model_entity.configuration.get('weights_filename', 'model.pth')
        model_path = os.path.join(local_path, weights_filename)
        torch.save(self.model, model_path)
        logger.info("Model saved to {}".format(model_path))

    def train(self, data_path, output_path, **kwargs):
        self.model.train()

        batch_size = self.configuration.get('batch_size', 32)
        num_epochs = self.configuration.get('num_epochs', 20)
        on_epoch_end_callback = kwargs.get('on_epoch_end_callback')

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, 'weights'), exist_ok=True)
        logger.info("Model set to train mode.")

        ################
        # prepare data #
        ################
        class image_title_dataset(Dataset):
            def __init__(self, list_image_path, list_txt):
                self.image_path = list_image_path
                # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
                self.title = clip.tokenize(list_txt)

            def __len__(self):
                return len(self.title)

            def __getitem__(self, idx):
                image = self.preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
                title = self.title[idx]
                return image, title

        # load data
        data_pairs = pd.read_csv(r"C:\Users\Yaya Tang\Documents\DATASETS\TACO 100\taco_100_INPUTS.csv")
        list_image_path = data_pairs['filepath']
        list_txt = data_pairs['img_description']
        dataset = image_title_dataset(list_image_path, list_txt)

        dataloader = DataLoader(dataset, batch_size=batch_size)

        # # set and split train/val dataset
        # train_ratio = 0.8
        # val_ratio = 0.2
        # train_size = int(train_ratio * len(dataset))
        # val_size = len(dataset) - train_size
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        # train_dataloader = DataLoader(train_dataset, batch_size=batch_size)  # Define your own dataloader
        # val_dataloader = DataLoader(val_dataset, batch_size=batch_size)  # Define your own dataloader

        train_filter = self.model_entity.metadata['system']['subsets']['train']['filter']
        val_filter = self.model_entity.metadata['system']['subsets']['validation']['filter']

        train_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'train'),
                                              filters=dl.Filters(custom_filter=train_filter),
                                              dataset_entity=self.model_entity.dataset,
                                              id_to_label_map=self.model_entity.id_to_label_map,
                                              label_to_id_map=self.model_entity.label_to_id_map,
                                              overwrite=False,
                                              to_mask=False,
                                              annotation_type=dl.AnnotationType.POLYGON,
                                              transforms=self.preprocess
                                              )

        val_dataset = DatasetGeneratorTorch(data_path=os.path.join(data_path, 'validation'),
                                            filters=dl.Filters(custom_filter=val_filter),
                                            dataset_entity=self.model_entity.dataset,
                                            id_to_label_map=self.model_entity.id_to_label_map,
                                            label_to_id_map=self.model_entity.label_to_id_map,
                                            overwrite=False,
                                            to_mask=False,
                                            annotation_type=dl.AnnotationType.POLYGON,
                                            transforms=self.preprocess
                                            )

        dataloaders = {'train': DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=self.dl_collate),
                       'val': DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn=self.dl_collate,
                                         )}
        logger.debug("Train and Val data loaders created")





        #################
        # prepare model #
        #################

        if self.device == "cpu":
            self.model.float()

        loss_img = nn.CrossEntropyLoss()
        loss_txt = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=5e-5,
                                     betas=(0.9, 0.98),
                                     eps=1e-6,
                                     weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, total=len(dataloader))
            for batch in dataloader:
                optimizer.zero_grad()

                images, texts = batch
                images = images.to(self.device)
                texts = texts.to(self.device)

                # forward pass
                logits_per_image, logits_per_text = self.model(images, texts)

                # calc loss + backprop
                ground_truth = torch.arange(len(images), dtype=torch.long, device=self.device)
                total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_loss.backward()
                if self.device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(self.model)
                    optimizer.step()
                    clip.model.convert_weights(self.model)

                pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item():.4f}")

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, f"checkpoints/CLIP_epoch_{epoch + 1}.pt")


if __name__ == "__main__":
    dl.setenv('prod')
    project = dl.projects.get(project_name='Model mgmt demo')  # id 1d2ffb6d-fe1c-4f8e-b2da-d05778ef7f03
    app = ClipAdapter(project=project)
    dataset = dl.datasets.get(dataset_name='taco for clip')
    app.extract_dataset(dataset=dataset)
