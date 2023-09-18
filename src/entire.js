import Tokenizer from 'https://deno.land/x/clip_bpe@v0.0.6/mod.js'
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.12.0/dist/ort.js"

let tokenizer = new Tokenizer()
let session = null
let url = `https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-text-vit-32-float32-int32.onnx`

export const downloadBlobWithProgress = (url, onProgress) => {
    return new Promise((res, rej) => {
        let blob;
        const xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.responseType = 'arraybuffer';
        xhr.onload = function (e) {
            blob = new Blob([this.response]);
        };
        if (onProgress) {
            xhr.onprogress = onProgress;
        }
        xhr.onloadend = function (e) {
            res(blob);
        };
        xhr.onerror = rej;
        xhr.send();
    });
};

async function init(options = {}) {
    const { onProgress, warmup } = options
    const blob = await downloadBlobWithProgress(url, onProgress)
    const textModelUrl = URL.createObjectURL(blob)
    console.log('Model Loaded Successfully')
    session = await ort.InferenceSession.create(textModelUrl)
    console.log('Session Initialized Successfully')

    if (warmup === true) {
        await runModel('Say hi to the nice model')
    }
    return session
}

async function runModel(text) {
    let textTokens = tokenizer.encodeForCLIP(text)
    textTokens = Int32Array.from(textTokens)
    const feeds = { input: new ort.Tensor('int32', textTokens, [1, 77]) }
    const results = await session.run(feeds)
    return [...results['output'].data]
}

async function run(textInput, itemsQuery) {
    await init()
    let vector = await runModel(textInput)
    let query = {
        'filter': { '$and': [{ 'hidden': false }, { 'type': 'file' }] },
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
                        'input': vector,
                        '$euclidSort': { 'eu_dist': 'ascending' }
                    }
                },
                'featureSetId': 'feature_set.id'
            },
        }
    }

    return query
}

