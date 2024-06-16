import unittest
import dtlpy as dl
import os
import json

BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = "CLIP-Semantic-Tests"


class MyTestCase(unittest.TestCase):
    project: dl.Project = None
    dataset: dl.Dataset = None
    root_path: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tests_data_path: str = os.path.join(root_path, 'tests', 'example_data')
    prepare_item_function = dict()

    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        os.chdir(cls.root_path)
        if dl.token_expired():
            dl.login_m2m(email=BOT_EMAIL, password=BOT_PWD)
        cls.project = dl.projects.get(project_id=PROJECT_ID)
        try:
            cls.dataset = cls.project.datasets.get(dataset_name=DATASET_NAME)
        except dl.exceptions.NotFound:
            cls.dataset = cls.project.datasets.create(dataset_name=DATASET_NAME)
        cls.feature_set_name = 'clip-feature-set'

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all apps
        for app in cls.project.apps.list().all():
            if app.project.id == cls.project.id:
                app.uninstall()

        # Delete all dpks
        filters = dl.Filters(resource=dl.FiltersResource.DPK)
        filters.add(field="scope", values="project")
        for dpk in cls.project.dpks.list(filters=filters).all():
            if dpk.project.id == cls.project.id and dpk.creator == BOT_EMAIL:
                dpk.delete()
        dl.logout()

    # Item preparation functions
    def _prepare_item(self, item_name: str, remote_path: str = None):
        local_path = os.path.join(self.tests_data_path, item_name)
        if remote_path is None:
            remote_path = "/"
        item = self.dataset.items.upload(
            local_path=local_path,
            remote_path=remote_path,
            overwrite=True
        )
        return item

    # Extract functions
    def _perform_extract_item(self, item_name: str):
        # Upload item
        item = self._prepare_item(item_name=item_name)

        # Open dataloop json
        dataloop_json_filepath = os.path.join(self.root_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r', encoding="utf8") as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dataloop_json["scope"] = "project"
        dataloop_json["name"] = f'{dataloop_json["name"]}-{self.project.id}'
        service_name = dataloop_json.get('components', dict()).get('services', list())[0].get("name", None)

        try:
            service = self.project.services.get(service_name=service_name)
        except dl.exceptions.NotFound:
            # Publish dpk and install app
            dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=self.project)
            dpk = self.project.dpks.publish(dpk=dpk)
            app = self.project.apps.install(dpk=dpk)

            service = app.project.services.get(service_name=service_name)

        execution = service.execute(
            execution_input=[
                dl.FunctionIO(
                    name="item",
                    type=dl.PackageInputType.ITEM,
                    value=item.id
                )
            ],
            project_id=self.project.id,
            function_name="extract_item"
        )
        execution = execution.wait()

        # Execution output format:
        # {"item_id": item_id}
        item = execution.output
        return item

    def _perform_extract_dataset(self, item_names: list):
        # Upload items
        items = list()
        dataset_test_path = "/dataset_test"
        for item_name in item_names:
            item = self._prepare_item(item_name=item_name, remote_path=dataset_test_path)
            items.append(item)

        # Open dataloop json
        dataloop_json_filepath = os.path.join(self.root_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r', encoding="utf8") as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dataloop_json["scope"] = "project"
        dataloop_json["name"] = f'{dataloop_json["name"]}-{self.project.id}'
        service_name = dataloop_json.get('components', dict()).get('services', list())[0].get("name", None)

        try:
            service = self.project.services.get(service_name=service_name)
        except dl.exceptions.NotFound:
            # Publish dpk and install app
            dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=self.project)
            dpk = self.project.dpks.publish(dpk=dpk)
            app = self.project.apps.install(dpk=dpk)

            # Get service
            service = app.project.services.get(service_name=service_name)

        filters = dl.Filters()
        filters.add(field="dir", values=dataset_test_path)
        filters_json = filters.prepare()
        filters_json["dataset"] = {"dataset_id": self.dataset.id}
        execution = service.execute(
            execution_input=[
                dl.FunctionIO(
                    name="item",
                    type=dl.PackageInputType.DATASET,
                    value=self.dataset.id
                ),
                dl.FunctionIO(
                    name="query",
                    type=dl.PackageInputType.JSON,
                    value=filters_json
                )
            ],
            project_id=self.project.id,
            function_name="extract_dataset"
        )
        execution = execution.wait()

        # Execution output format:
        # {"dataset_id": dataset_id}
        dataset = execution.output
        return dataset

    # Test functions
    # def test_extract_image_item(self):
    #     # Delete previous features
    #     feature_set = self.project.feature_sets.get(feature_set_name=self.feature_set_name)
    #     all_features = list(feature_set.features.list().all())
    #     for feature in all_features:
    #         feature_set.features.delete(feature_id=feature.id)
    #
    #     item_name = "car_image.jpeg"
    #     extract_item = self._perform_extract_item(item_name=item_name)
    #
    #     # Validate that the output is an item
    #     self.assertTrue(isinstance(extract_item, dict))
    #     extract_item = self.dataset.items.get(item_id=extract_item.get('item_id', None))
    #     self.assertTrue(isinstance(extract_item, dl.Item))
    #
    #     # Validate that the feature for the item was created
    #     filters = dl.Filters(resource=dl.FiltersResource.FEATURE)
    #     filters.add(field="entityId", values=extract_item.id)
    #     feature_vector_entity = feature_set.features.list(filters=filters)
    #     self.assertTrue(feature_vector_entity.items_count == 1)
    #     # TODO: Added feature vector validation

    # def test_extract_text_item(self):
    #     # Delete previous features
    #     feature_set = self.project.feature_sets.get(feature_set_name=self.feature_set_name)
    #     all_features = list(feature_set.features.list().all())
    #     for feature in all_features:
    #         feature_set.features.delete(feature_id=feature.id)
    #
    #     item_name = "lorem_text.txt"
    #     extract_item = self._perform_extract_item(item_name=item_name)
    #
    #     # Validate that the output is an item
    #     self.assertTrue(isinstance(extract_item, dict))
    #     extract_item = self.dataset.items.get(item_id=extract_item.get('item_id', None))
    #     self.assertTrue(isinstance(extract_item, dl.Item))
    #
    #     # Validate that the feature for the item was created
    #     filters = dl.Filters(resource=dl.FiltersResource.FEATURE)
    #     filters.add(field="entityId", values=extract_item.id)
    #     feature_vector_entity = feature_set.features.list(filters=filters)
    #     self.assertTrue(feature_vector_entity.items_count == 1)
    #     # TODO: Added feature vector validation

    def test_extract_dataset(self):
        # Delete previous features
        feature_set = self.project.feature_sets.get(feature_set_name=self.feature_set_name)
        all_features = list(feature_set.features.list().all())
        for feature in all_features:
            feature_set.features.delete(feature_id=feature.id)

        item_names = ["car_image.jpeg", "lorem_text.txt"]
        extract_dataset = self._perform_extract_dataset(item_names=item_names)

        # Validate that the output is the dataset
        self.assertTrue(isinstance(extract_dataset, dict))
        self.assertTrue(extract_dataset.get('dataset_id', None) == self.dataset.id)


if __name__ == '__main__':
    unittest.main()
