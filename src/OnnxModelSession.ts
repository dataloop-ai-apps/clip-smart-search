// @ts-ignore
import Tokenizer from 'https://deno.land/x/clip_bpe@v0.0.6/mod.js'
import { downloadBlobWithProgress } from './utils'

const { InferenceSession, Tensor } = window.ort

export class OnnxModelSession {
    private static _instance: OnnxModelSession | null = null

    public static get instance() {
        if (!this._instance) {
            this._instance = new OnnxModelSession()
        }
        return this._instance
    }

    private _tokenizer: Tokenizer | null = null
    private _session: typeof InferenceSession | null = null
    private _url: string = `https://huggingface.co/rocca/openai-clip-js/resolve/main/clip-text-vit-32-float32-int32.onnx`

    public get tokenizer() {
        if (!this._tokenizer) {
            this._tokenizer = new Tokenizer()
        }
        return this._tokenizer
    }

    public async init(
        options: {
            onProgress?: (ev: ProgressEvent<EventTarget>) => any
            warmup?: boolean
        } = {}
    ) {
        const { onProgress, warmup } = options
        const blob = await downloadBlobWithProgress(this._url, onProgress)
        const textModelUrl = URL.createObjectURL(blob)
        console.log('Model Loaded Successfully')
        this._session = await InferenceSession.create(textModelUrl)
        console.log('Session Initialized Successfully')

        if (warmup) {
            await Promise.all([this.run('A'), this.run('B'), this.run('C')])
        }
        return this._session
    }

    public async run(text: string) {
        let textTokens = this.tokenizer.encodeForCLIP(text)
        textTokens = Int32Array.from(textTokens)
        const feeds = { input: new Tensor('int32', textTokens, [1, 77]) }
        const results = await this._session.run(feeds)
        return [...results['output'].data]
    }
}
