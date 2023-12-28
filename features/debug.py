import dtlpy as dl

dl.setenv('rc')
project = dl.projects.get('COCO ors')


def try_create():
    # project.bots.create(name='dummy', return_credentials=True)

    # dl.login_m2m('.dataloop.ai', '@')
    filters = dl.Filters(resource=dl.FiltersResource.FEATURE_SET, use_defaults=False)
    project.feature_sets.list(filters=filters).print()
    feature_set = project.feature_sets.create(name='clip-image-search',
                                              entity_type=dl.FeatureEntityType.ITEM,
                                              project_id=project.id,
                                              set_type='clip',
                                              size=2)
    # feature_set = project.feature_sets.get(feature_set_name='clip-image-search')
    feature = feature_set.features.create(project_id='2cb9ae90-b6e8-4d15-9016-17bacc9b7bdf',
                                          entity_id='650fd3dcaffd5c9fb3214c88',
                                          value=[0.10843600332736969, 0.23928828537464142],
                                          )

    dl.client_api.print_response(dl.client_api.last_response)
    dl.client_api.last_response.headers


def try_featres():
    from features.extract_features import ClipExtractor

    dataset = dl.datasets.get(dataset_id='5f4d13ba4a958a49a7747cd9')
    extractor = ClipExtractor(context={'project': dataset.project})

    items = dataset.items.list()

    for item in items.all():
        _ = extractor.extract_item(item=item)
