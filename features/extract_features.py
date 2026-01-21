import json
import logging
import time
from io import BytesIO
import dtlpy as dl


logger = logging.getLogger('[CLIP-SEARCH]')

DPK_NAME = "clip-model-pretrained"
BASE_MODEL_NAME = "openai-clip"
MODEL_NAME = "CLIP model for semantic search"


class ClipExtractor(dl.BaseServiceRunner):
    def __init__(self, project=None):
        if project is None:
            project = self.service_entity.project
        self.project = project
        self.model = None
        self.feature_set = None
        
        # Install/get the CLIP model adapter and set up the model
        self._validate_model()
        
        # Set up the feature set and store its ID in binaries
        self._setup_feature_set()

    def _validate_model(self):
        """Validate the model exists and is deployed."""
        try:
            self.model = self.project.models.get(model_name=MODEL_NAME)
            logger.info(f'Custom model found: {self.model.name}, id: {self.model.id}')
        except dl.exceptions.NotFound:
            logger.info(f'Custom model "{MODEL_NAME}" not found, cloning from base model...')
            self.model = self._create_model()
        return True

    def _create_model(self):
        """Create a new model based on the DPK."""
        model = None
        try:
            dpk = dl.dpks.get(dpk_name=DPK_NAME)
            app = self.project.apps.get(app_name=dpk.display_name)
        except dl.exceptions.NotFound:
            logger.info(f'App "{DPK_NAME}" not found, installing...')
            dpk = dl.dpks.get(dpk_name=DPK_NAME)
            app = self.project.apps.install(
                app_name=dpk.display_name,
                dpk=dpk,
                custom_installation=dpk.to_json()
            )
            logger.info(f'Installed {dpk.display_name} app: {app.name}, ID: {app.id}')
            model = self._add_model_from_app(app, dpk)
            
        return model

    def _add_model_from_app(self, app: dl.App, dpk: dl.Dpk):
        model_from_dpk = dpk.to_json().get("components", dict()).get('models',[{}])[0]

        request = {
            "name": MODEL_NAME,
            "description": "OpenAI CLIP model for search with NLP",
            "scope": "project",
            "configuration": model_from_dpk.get("configuration", {}),
            "outputType": "embedding",
            "moduleName": "clip-module",
            "packageId": dpk.id,
            "status": "pre-trained",
            "projectId": self.project.id,
            "app": {
                "id": app.id,
                "dpkId": dpk.id,
                "componentName": model_from_dpk.get("name", MODEL_NAME),
                "dpkName": dpk.name,
                "dpkVersion": dpk.version
            }
        }
        success, response = dl.client_api.gen_request(req_type='POST', path=f'/ml/models', json_req=request)
        if not success:
            logger.error(f'Failed to create model: {response}')
            raise Exception(f'Failed to create model: {response.content}')
            
        model = dl.Model.from_json(_json=response.json(), client_api=dl.client_api, project=None, package=None)

        return model

    def _setup_feature_set(self):
        """Get the model's feature set and store its ID in binaries."""
        # The model adapter creates a feature set named after the model

        binaries = self.project.datasets._get_binaries_dataset()
        
        # Check if the file already exists with the correct ID
        try:
            existing_item = binaries.items.get(filepath="/clip_feature_set.json")
            existing_feature_set_id = existing_item.metadata.get("system", {}).get("clip_feature_set_id", None)
            self.feature_set = self.project.feature_sets.get(feature_set_id=existing_feature_set_id)
            if self.feature_set.model_id != self.model.id:
                self.feature_set.model_id = self.model.id
                self.feature_set.update()
        except dl.exceptions.NotFound:
            if self.model.feature_set is None:
                self.feature_set = self.project.feature_sets.create(
                    name=self.model.name,
                    entity_type=dl.FeatureEntityType.ITEM,
                    model_id=self.model.id,
                    project_id=self.project.id,
                    set_type=self.model.name,
                    size=self.model.configuration.get('embeddings_size', 256),
                )
            else:
                self.feature_set = self.model.feature_set
                    # Upload/update the metadata file
            buffer = BytesIO()
            buffer.write(json.dumps({}, default=lambda x: None).encode())
            buffer.seek(0)
            buffer.name = "clip_feature_set.json"
            
            binaries.items.upload(
                local_path=buffer,
                item_metadata={
                    "system": {
                        "clip_feature_set_id": self.feature_set.id
                    }
                },
                overwrite=True
            )
            logger.info(f'Updated binaries with feature set ID: {self.feature_set.id}')

    def extract_item(self, item: dl.Item) -> dl.Item:
        """Extract CLIP features for a single item using the model adapter."""
        logger.info(f'Started on item id: {item.id}, filename: {item.filename}')
        tic = time.time()
        
        self.model.embed(item=item)

        logger.info(f'Done. runtime: {(time.time() - tic):.2f}[s]')
        return item

    def extract_dataset(self, dataset: dl.Dataset, query=None, progress=None):
        """Extract CLIP features for a dataset using the model adapter."""
        logger.info(f'Starting dataset extraction for dataset: {dataset.name}')
        tic = time.time()
        
        # Use the model adapter to embed the entire dataset (it handles mimetype filtering)
        execution = self.model.embed_datasets(dataset_ids=[dataset.id])
        execution.wait()
        
        if progress is not None:
            progress.update(progress=100)
        
        logger.info(f'Dataset extraction done. runtime: {(time.time() - tic):.2f}[s]')
        return dataset


if __name__ == "__main__":
    dl.setenv('')
    project = dl.projects.get(project_id='')
    app = ClipExtractor(project=project)
    dataset = project.datasets.get(dataset_id='')
    app.extract_dataset(dataset=dataset)
