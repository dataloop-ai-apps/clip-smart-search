import json
import logging
import time
import dtlpy as dl

from io import BytesIO

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
        self._setup_model_adapter()
        
        # Set up the feature set and store its ID in binaries
        self._setup_feature_set()

    def _setup_model_adapter(self):
        """Install the CLIP model adapter DPK and deploy the model."""
        logger.info(f'Getting DPK: {DPK_NAME}')
        clip_model_dpk = dl.dpks.get(dpk_name=DPK_NAME)
        
        try:
            model_app = self.project.apps.install(
                app_name=clip_model_dpk.display_name,
                dpk=clip_model_dpk,
                custom_installation=clip_model_dpk.to_json()
            )
            logger.info(f"Installed {clip_model_dpk.display_name} app: {model_app.name}, ID: {model_app.id}")
        except dl.exceptions.BadRequest:
            logger.info(f"{clip_model_dpk.display_name} app already installed, getting existing app")
            model_app = self.project.apps.get(app_name=clip_model_dpk.display_name)
            logger.info(f"Retrieved existing app: {model_app.name}, ID: {model_app.id}")
        
        # Try to get the custom model by name
        try:
            self.model = self.project.models.get(model_name=MODEL_NAME)
            logger.info(f'Custom model found: {self.model.name}, id: {self.model.id}')
        except dl.exceptions.NotFound:
            # Custom model doesn't exist, clone from base model
            logger.info(f'Custom model "{MODEL_NAME}" not found, cloning from base model...')
            
            base_model = self.project.models.get(model_name=BASE_MODEL_NAME)
            logger.info(f'Base model found: {base_model.name}, id: {base_model.id}')
            
            self.model = self.project.models.clone(
                from_model=base_model,
                model_name=MODEL_NAME,
                description="CLIP model for semantic search in clip-smart-search"
            )
            logger.info(f'Cloned model created: {self.model.name}, id: {self.model.id}')
        
        if self.model.status != 'deployed':
            logger.info(f'Deploying model: {self.model.name}')
            self.model.deploy()
            self.model.wait_for_model_ready()
            self.model = self.project.models.get(model_name=MODEL_NAME)
            logger.info(f'Model deployed: {self.model.name}')

    def _setup_feature_set(self):
        """Get the model's feature set and store its ID in binaries."""
        # The model adapter creates a feature set named after the model
        feature_set_name = self.model.name
        
        try:
            self.feature_set = self.project.feature_sets.get(feature_set_name=feature_set_name)
            logger.info(f'Feature Set found! name: {self.feature_set.name}, id: {self.feature_set.id}')
            # Update the binaries with the feature set info (for entire.js frontend)
            self._update_binaries_metadata()
        except dl.exceptions.NotFound:
            # Feature set will be created by the model adapter on first embed
            logger.info(f'Feature Set "{feature_set_name}" not found yet, will be created on first embed')
            self.feature_set = None

    def _get_model_feature_set(self):
        """Get the feature set created by the model adapter and update binaries."""
        feature_set_name = self.model.name
        try:
            self.feature_set = self.project.feature_sets.get(feature_set_name=feature_set_name)
            logger.info(f'Feature Set found! name: {self.feature_set.name}, id: {self.feature_set.id}')
            self._update_binaries_metadata()
        except dl.exceptions.NotFound:
            logger.warning(f'Feature Set "{feature_set_name}" still not found after embed')

    def _update_binaries_metadata(self):
        """Update the clip_feature_set.json in binaries with the feature set ID."""
        if self.feature_set is None:
            logger.info('Feature set not yet available, skipping binaries update')
            return
            
        binaries = self.project.datasets._get_binaries_dataset()
        
        # Check if the file already exists with the correct ID
        try:
            existing_item = binaries.items.get(filepath="/clip_feature_set.json")
            existing_metadata = existing_item.metadata.get("system", {})
            if existing_metadata.get("clip_feature_set_id") == self.feature_set.id:
                logger.info('Binaries metadata already up to date')
                return
        except dl.exceptions.NotFound:
            pass
        
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
        
        # After embed, get the feature set and update binaries if needed
        if self.feature_set is None:
            self._get_model_feature_set()
        
        logger.info(f'Done. runtime: {(time.time() - tic):.2f}[s]')
        return item

    def extract_dataset(self, dataset: dl.Dataset, query=None, progress=None):
        """Extract CLIP features for a dataset using the model adapter."""
        logger.info(f'Starting dataset extraction for dataset: {dataset.name}')
        tic = time.time()
        
        # Use the model adapter to embed the entire dataset (it handles mimetype filtering)
        execution = self.model.embed_datasets(dataset_ids=[dataset.id])
        execution.wait()
        
        # After embed, get the feature set and update binaries if needed
        if self.feature_set is None:
            self._get_model_feature_set()
        
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
