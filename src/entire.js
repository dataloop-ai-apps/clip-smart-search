async function run(textInput, itemsQuery) {
    try {
        let tokenizer
        let textModel
        let transformers

        async function loadDependencies() {
            try {
                let quantized = false; // change to `true` for a much smaller model (e.g. 87mb vs 345mb for image model), but lower accuracy
                transformers = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.1')
                // transformers.env.allowLocalModels = false
                // transformers.env.remoteHost = "https://huggingface.co/"
                // transformers.env.remotePathTemplate = "Xenova/clip-vit-base-patch32/resolve/main"

                tokenizer = await transformers.AutoTokenizer.from_pretrained('Xenova/clip-vit-base-patch32')
                textModel = await transformers.CLIPTextModelWithProjection.from_pretrained('Xenova/clip-vit-base-patch32', { quantized });

            }
            catch (e) {
                console.log(e)
            }

        }
        await loadDependencies()

        // console.log('done')
        // get text embedding:
        let texts = ['cigarettes on the sidewalk'];
        let textInputs = tokenizer(texts, { padding: true, truncation: true });
        let { text_embeds } = await textModel(textInputs);
        let vector = text_embeds.data
        // console.log(textInputs);
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
                            input: vector,
                            $euclidSort: { eu_dist: 'ascending' }
                        }
                    },
                    featureSetId: 'feature_set.id'
                }
            }
        }

        return query
    }
    catch (e) {
        console.log(e)
    }
}





