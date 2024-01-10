async function run(textInput, itemsQuery) {
    try {
        let tokenizer
        let textModel
        let transformers
        console.log('loading dependencies')
        console.log(textInput)
        debugger
        async function loadDependencies() {
            try {
                let quantized = false; // change to `true` for a much smaller model (e.g. 87mb vs 345mb for image model), but lower accuracy
                transformers = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.1')
                transformers.env.allowLocalModels = false
                transformers.env.remoteHost = "https://huggingface.co/"
                transformers.env.remotePathTemplate = "Xenova/clip-vit-base-patch32/resolve/main"

                tokenizer = await transformers.AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32')
                textModel = await transformers.CLIPTextModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch32', { quantized });

            }
            catch (e) {
                console.log(e)
            }

        }
        await loadDependencies()
        let binariesFileName = 'clip_feature_set.json'
        textInput = textInput['Text Box']
        let texts = [textInput];
        let textInputs = tokenizer(texts, { padding: true, truncation: true });
        let { text_embeds } = await textModel(textInputs);
        let vector = text_embeds.data
        console.log(vector);

        let query = {
            filter: { $and: [{ hidden: false }, { type: 'file' }] },
            page: 0,
            pageSize: 1000,
            resource: 'items',
            join: {
                on: {
                    resource: 'feature_vectors',
                    local: 'entityId',
                    forigen: 'id'
                },
                filter: {
                    value: {
                        $euclid: {
                            input: Array.from(vector),
                            $euclidSort: { eu_dist: 'ascending' }
                        }
                    },
                    featureSetId: '659475b38f8f9e8bcedb717a'
                }
            }
        }
        console.log(query)
        return query
    }
    catch (e) {
        console.log(e)
    }
}




datasets = await dl.datasets.query()
datasets.items.find(object => object.name === 'Binaries')

try{
    const item = await dl.items.getByName("/clip_feature_set.json", { binaries: true })
    let featureSetId = item.metadata.system.clip_feature_set_id
    }
catch(e){
    console.log(e)
    dl.sendEvent({ name: "app:toastMessage",
                    payload: {
                    message: "CLIP FeatureSet does not exist for this project, please run pre-process",
                    type: "error"}
                    })
        }