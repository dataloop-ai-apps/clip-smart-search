import dtlpy as dl

# dl.setenv('rc')
project = dl.projects.get(project_name='COCO ors')
dataset = project.datasets.get(dataset_name='features')

filters = dl.Filters(resource=dl.FiltersResource.FEATURE_SET)
filters.add(field='name', values='clip-image-search-shadi')
feature_sets_test = project.feature_sets.list(filters=filters)
project.datasets.list().print()
project.feature_sets.list().print()

# feature_set = project.feature_sets.get(feature_set_id='65912420824d66514cabc52a')
try:
    feature_set = project.feature_sets.get(feature_set_name='clip-single-store-shadi')
except dl.exceptions.NotFound:
    feature_set = project.feature_sets.create(name='clip-single-store-shadi',
                                              entity_type=dl.FeatureEntityType.ITEM,
                                              project_id=project.id,
                                              set_type='clip',
                                              size=2)  #

for x in range(10):
    for y in range(10):
        remote_name = f'x-{x:0>2}_y-{y:0>2}.jpg'
        item = dataset.items.upload(local_path=r"/assets/000000002532.jpg",
                                    remote_name=remote_name)
        # item = dl.items.get(item_id='65127951f0947f1f432b7f10')
        feature_set.features.create(value=[x, y],
                                    entity=item)
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
    # clipp = ClipExtractor(dl.Context(project=project))

    from dataloop_scripts.test_query import query
    import dtlpy as dl
    project = dl.projects.get(project_name='COCO ors')
    feature_set = project.feature_sets.get(feature_set_name='clip-singlestore-shadi2')
    dataset = dl.datasets.get(dataset_id='659473f46877cb9a7c29aa68')
    # model, preprocess = clip.load("ViT-B/32", device='cuda')
    # text = clip.tokenize(["image of a horse"]).to('cuda')
    # text_features = model.encode_text(text)
    query['join']['filter']['value']['$euclid']['input'] = [value for key, value in
                                                            query['join']['filter']['value']['$euclid'][
                                                                'input'].items()]
    custom_filter = {
        'filter': {'$and': [{'hidden': False}, {'type': 'file'}]},
        'page': 0,
        'pageSize': 0,
        'resource': 'items',
        'join': {
            'on': {
                'resource': 'feature_vectors',
                'local': 'entityId',
                'forigen': 'id'
            },
            'filter': {
                'featureSetId': feature_set.id
            },
        }
    }
    # custom_filter = {
    #     "value": {
    #         "$euclid": {
    #             "input": text_features[0].tolist(),  # string || number[], // feature vector ID || actual vectors value
    #             "$euclidSort": {
    #                 "eu_dist": 'ascending'
    #             },
    #         }
    #     },
    #     "featureSetId": feature_set.id
    # }

    filters = dl.Filters(custom_filter=custom_filter,
                         resource=dl.FiltersResource.ITEM)

    res = dataset.items.list(filters=filters)
    print(res.items_count)

    for i, f in enumerate(res.items):
        filt = dl.Filters(resource=dl.FiltersResource.FEATURE, field='entityId', values=f.id)
        p = list(feature_set.features.list(filters=filt).all())
        print(p[0].value)
        if i == 10:
            break

# success, response = self._client_api.gen_request(req_type="POST",
#                                                  path="/features/vectors/query",
#                                                  json_req=filters.prepare(),
#                                                  headers={'user_query': filters._user_query}
#                                                  )
