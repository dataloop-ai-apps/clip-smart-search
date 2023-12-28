import dtlpy as dl

dl.setenv('rc')
project = dl.projects.get(project_name='COCO ors')
dataset = project.datasets.get(dataset_name='features')
project.feature_sets.list().print()

feature_set = project.feature_sets.get(feature_set_id='650da7deba266d44057229b4')

for x in range(10):
    for y in range(10):
        remote_name = f'x-{x:0>2}_y-{y:0>2}.jpg'
        item = dataset.items.upload(local_path=r"E:\TypesExamples\troy_and_abed.jpeg",
                                    remote_name=remote_name)
        feature_set.features.create(value=[x, y],
                                    entity_id=item.id)
        # assert False


def filters_vectors():
    custom_filter = {
        "value": {
            "$euclid": {
                "input": [5, 5],  # string || number[], // feature vector ID || actual vectors value
                "$euclidSort": {
                    "eu_dist": 'ascending'
                },
            }
        },
        "featureSetId": feature_set.id
    }

    filters = dl.Filters(custom_filter=custom_filter,
                         resource=dl.FiltersResource.FEATURE)

    res = feature_set.features.list(filters=filters)
    print(res.items_count)


def filter_items():

    from features.extract_features import ClipExtractor
    clipp = ClipExtractor(dl.Context(project=project))

    import clip
    model, preprocess = clip.load("ViT-B/32", device='cuda')
    text = clip.tokenize(["image of a business man with a tie"]).to('cuda')
    dataset = dl.datasets.get(dataset_id='64e46f8d70b4f336c3717630')
    text_features = model.encode_text(text)

    custom_filter = {
        'filter': {'$and': [{'hidden': False}, {'type': 'file'}]},
        'page': 0,
        'pageSize': 1000,
        'resource': 'items',
        'join': {
            'on': {
                'resource': 'feature_vectors',
                'local': 'entityId',
                'forigen': 'id'
            },
            'filter': {
                'value': {
                    '$euclid': {
                        'input': text_features[0].tolist(),
                        '$euclidSort': {'eu_dist': 'ascending'}
                    }
                },
                'featureSetId': feature_set.id
            },
        }
    }
    filters = dl.Filters(custom_filter=custom_filter,
                         resource=dl.FiltersResource.ITEM)

    res = dataset.items.list(filters=filters)
    print(res.items_count)

    for i, f in enumerate(res.items):
        filt = dl.Filters(resource=dl.FiltersResource.FEATURE,field='entityId', values=f.id)
        p = list(feature_set.features.list(filters=filt).all())
        print(p[0].value)
        if i == 10:
            break

# success, response = self._client_api.gen_request(req_type="POST",
#                                                  path="/features/vectors/query",
#                                                  json_req=filters.prepare(),
#                                                  headers={'user_query': filters._user_query}
#                                                  )
