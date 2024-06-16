import unittest
import dtlpy as dl
import os
import json
import enum

BOT_EMAIL = os.environ['BOT_EMAIL']
BOT_PWD = os.environ['BOT_PWD']
PROJECT_ID = os.environ['PROJECT_ID']
DATASET_NAME = "CLIP-Semantic-Tests"


class ItemTypes(enum.Enum):
    IMAGE = "image"


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
        cls.prepare_item_function = {
            ItemTypes.IMAGE.value: cls._prepare_image_item
        }
        cls.feature_set_name = 'clip-feature-set'

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete all models
        for model in cls.project.models.list().all():
            model.delete()

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
    def _prepare_image_item(self, item_name: str):
        local_path = os.path.join(self.tests_data_path, item_name)
        item = self.dataset.items.upload(
            local_path=local_path,
            overwrite=True
        )
        return item

    # Perdict function
    def _perform_extract_item(self, item_type: ItemTypes, item_name: str):
        # Upload item
        item = self.prepare_item_function[item_type.value](self=self, item_name=item_name)

        # Open dataloop json
        dataloop_json_filepath = os.path.join(self.root_path, 'dataloop.json')
        with open(dataloop_json_filepath, 'r', encoding="utf8") as f:
            dataloop_json = json.load(f)
        dataloop_json.pop('codebase')
        dataloop_json["scope"] = "project"
        dataloop_json["name"] = f'{dataloop_json["name"]}-{self.project.id}'
        service_name = dataloop_json.get('components', dict()).get('services', list())[0].get("name", None)

        # Publish dpk and install app
        dpk = dl.Dpk.from_json(_json=dataloop_json, client_api=dl.client_api, project=self.project)
        dpk = self.project.dpks.publish(dpk=dpk)
        app = self.project.apps.install(dpk=dpk)

        # Get model and predict
        service = app.project.services.get(service_name=service_name)

        execution = service.execute(
            execution_input=[
                dl.FunctionIO(
                    name="item",
                    type=dl.PackageInputType.ITEM,
                    value=item.id
                )
            ],
            function_name="extract_item"
        )
        execution = execution.wait()

        # Execution output format:
        # {"item_id": item_id}
        item = execution.output
        return item

    # Test functions
    def test_extract_item(self):
        item_name = "car_image.jpeg"
        item_type = ItemTypes.IMAGE
        # extract_item = self._perform_extract_item(item_type=item_type, item_name=item_name)
        # self.assertTrue(isinstance(extract_item, dict))
        # extract_item = self.dataset.items.get(item_id=extract_item.get('item_id', None))
        # print(extract_item)
        feature_set = self.project.feature_sets.get(feature_set_name=self.feature_set_name)
        print(feature_set)


if __name__ == '__main__':
    unittest.main()
