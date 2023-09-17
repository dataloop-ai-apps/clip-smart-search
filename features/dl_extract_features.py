import dtlpy as dl
import torch
import clip
import os
from PIL import Image
from PIL import ImageFile
import pathlib
import pandas as pd
import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def extract_dataset(dataset, feature_set: dl.FeatureSet):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # directory = r'E:\Applications\clip-app\data\items\raw'
    # directory = r'E:\Datasets\COCO\2017\images\val2017'
    # output = list()
    # files = list(pathlib.Path(directory).glob('*.jpg'))
    pages = dataset.items.list()
    pbar = tqdm.tqdm(total=pages.items_count)
    with torch.no_grad():
        for item in pages.all():
            item: dl.Item
            # assert False
            orig_image = Image.fromarray(item.download(save_locally=False, to_array=True))
            # orig_image = Image.open(filepath)
            image = preprocess(orig_image).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            output = image_features[0].cpu().detach().numpy().tolist()
            feature_set.features.create(value=output, entity_id=item.id)
            # print(i_filename)
            pbar.update()
            # if i_filename == 100:
            #     break

    # pd.DataFrame(output).to_csv(os.path.join(directory, 'clip_vit_32_embeddings.tsv'),
    #                             header=False,
    #                             index_label=False,
    #                             index=False,
    #                             sep='\t')


def search():
    custom_filter = {'filter': {'$and': [{'hidden': False},
                                         {'type': 'file'}]},
                     'page': 0,
                     'pageSize': 100,
                     'resource': 'items',
                     'join': {'resource': 'feature_vectors',
                              # 'joinBy': {'id': 'entityId'},
                              'filter': {'$and': [{'featureSetId': '642ecb860eb44c6a2f4544df'},
                                                  {'$euclid': {'id': '123123123', '$gt': 123}}]},
                              # 'sort': {'distance': 'euclidean',
                              #          'value': [1, 1, 1, 1, 1]}
                              },
                     }
    # filters = dl.Filters(custom_filter=custom_filter)

    dl.setenv('rc')
    feature_set = dl.feature_sets.get(feature_set_id='642ecb860eb44c6a2f4544df')
    filters = dl.Filters(use_defaults=False,
                         custom_filter={'$and': [{'featureSetId': '642ecb860eb44c6a2f4544df'},
                                                 {'$euclid': {'value': [1, 2, 3, 4],
                                                              'eq': 1}}
                                                 ]},
                         resource='feature_vectors')
    test = dl.features.list(filters=filters)
    print(test.items_count)
    # pages = dataset.items.list(filters=filters)
    feature_set.features.list(filters=filters)


if __name__ == "__main__":
    dl.setenv('rc')
    dataset = dl.datasets.get(dataset_id='642d25b27c05caf3239111a5')
    # TACO trash ID = 64c27e74615b1c5d7d576776

    # feature_set = dataset.project.feature_sets.create(name='clip_vit_32_embeddings',
    #                                                   size=512,
    #                                                   set_type='embedding',
    #                                                   data_type=None,
    #                                                   entity_type=dl.FeatureEntityType.ITEM)
    feature_set = dl.feature_sets.get(feature_set_id='642ecb860eb44c6a2f4544df')

    # extract_dataset(dataset=dataset, feature_set=feature_set)

# interviews_df = pd.read_csv('GeekforGeeks.tsv', sep='\t')
