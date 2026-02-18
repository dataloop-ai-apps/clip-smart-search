async function run(textInput, itemsQuery) {
    let tokenizer;
    let textModel;
    let transformers;
    let featureSetId;
    const defaultFilter = { $and: [{ hidden: false }, { type: "file" }] };
    const filter =
        itemsQuery && itemsQuery.filter ? itemsQuery.filter : defaultFilter;
    const defaultQuery = { filter: filter, resource: "items" };
    console.log(
        `input itemsQuery: ${
            itemsQuery ? JSON.stringify(itemsQuery, null, 2) : "null"
        }`
    );
    const dataset = await dl.datasets.get();
    try {
        textInput = textInput["Text Box"];
    } catch (e) {
        dl.sendEvent({
            name: "app:toastMessage",
            payload: {
                message: "For CLIP FeatureSet input text is required",
                type: "error",
            },
        });
        return defaultQuery;
    }

    try {
        const item = await dl.items.getByName("/clip_feature_set.json", {
            binaries: true,
        });
        featureSetId = item.metadata.system.clip_feature_set_id;
    } catch (e) {
        // Feature set missing: start extraction if none has been run, else ask user to wait or retry
        const p = await dl.projects.get();
        const executions = await dl.executions.query({
            functionName: "extract_dataset",
            serviceName: "clip-extraction",
            projectId: p.id,
        });
        if (executions.totalItemsCount === 0) {
            dl.sendEvent({
                name: "app:toastMessage",
                payload: {
                    message:
                        "CLIP FeatureSet does not exist for this project, running extraction",
                    type: "warning",
                },
            });
            await dl.executions.create({
                functionName: "extract_dataset",
                serviceName: "clip-extraction",
                input: { dataset: { dataset_id: dataset.id }, query: null },
            });
        } else {
            dl.sendEvent({
                name: "app:toastMessage",
                payload: {
                    message:
                        "CLIP FeatureSet not ready. Extraction was already run for this project - please wait for it to complete or check execution status.",
                    type: "warning",
                },
            });
        }
        return defaultQuery;
    }
    const query_feature = {
        filter: {
            $and: [
                {
                    $or: [
                        {
                            "metadata.system.mimetype": "image/*",
                        },
                        {
                            "metadata.system.mimetype": "text/*",
                        },
                    ],
                },
                {
                    hidden: false,
                },
                {
                    type: "file",
                },
            ],
        },
        resource: "items",
        join: {
            on: {
                resource: "feature_vectors",
                local: "entityId",
                forigen: "id",
            },
            filter: {
                featureSetId: featureSetId,
            },

        },
    };
    const query_item = {
        filter: {
            $and: [
                {
                    $or: [
                        {
                            "metadata.system.mimetype": "image/*",
                        },
                        {
                            "metadata.system.mimetype": "text/*",
                        },
                    ],
                },
                {
                    hidden: false,
                },
                {
                    type: "file",
                },
            ],
        },
        resource: "items",
    };

    const items_count = await dl.items.countByQuery(query_item);
    const items_with_feature_count = await dl.items.countByQuery(query_feature);
    if (items_count !== items_with_feature_count) {
        dl.sendEvent({
            name: "app:toastMessage",
            payload: {
                message:
                    "Feature extraction was not run on entire dataset, please run again!",
                type: "warning",
            },
        });
    }
    console.log("loading dependencies");
    async function loadDependencies() {
        try {
            let quantized = false; // change to `true` for a much smaller model (e.g. 87mb vs 345mb for image model), but lower accuracy
            transformers = await import(
                "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.1"
            );
            transformers.env.allowLocalModels = false;
            transformers.env.remoteHost = "https://huggingface.co/";
            transformers.env.remotePathTemplate =
                "Xenova/clip-vit-base-patch32/resolve/main";

            tokenizer = await transformers.AutoTokenizer.from_pretrained(
                "Xenova/clip-vit-base-patch32"
            );
            textModel =
                await transformers.CLIPTextModelWithProjection.from_pretrained(
                    "Xenova/clip-vit-base-patch32",
                    { quantized }
                );
        } catch (e) {
            console.log(e);
            return defaultQuery;
        }
    }
    await loadDependencies();
    let texts = [textInput];
    let textInputs = tokenizer(texts, { padding: true, truncation: true });
    let { text_embeds } = await textModel(textInputs);
    let vector = text_embeds.data;
    console.log(`vector: ${vector}`);

    let query = {
        filter: filter,
        page: 0,
        pageSize: 1000,
        resource: "items",
        join: {
            on: {
                resource: "feature_vectors",
                local: "entityId",
                forigen: "id",
            },
            filter: {
                value: {
                    $euclid: {
                        input: Array.from(vector),
                        $euclidSort: { eu_dist: "ascending" },
                    },
                },
                featureSetId: featureSetId,
                datasetId: dataset.id,
            },
        },
    };
    console.log(`query: ${JSON.stringify(query, null, 2)}`);
    return query;
}
