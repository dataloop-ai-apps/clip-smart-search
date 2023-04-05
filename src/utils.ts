export const sleep = (ms: number) => {
    return new Promise((r) => setTimeout(r, ms))
}

export const downloadBlobWithProgress = (
    url: string,
    onProgress?: (ev: ProgressEvent<EventTarget>) => any
) => {
    return new Promise<Blob>((res, rej) => {
        var blob
        var xhr = new XMLHttpRequest()
        xhr.open('GET', url, true)
        xhr.responseType = 'arraybuffer'
        xhr.onload = function (e) {
            blob = new Blob([this.response])
        }
        if (onProgress) {
            xhr.onprogress = onProgress
        }
        xhr.onloadend = function (e) {
            res(blob)
        }
        xhr.onerror = rej
        xhr.send()
    })
}

export const cosineSimilarity = (A: Float32Array, B: Float32Array) => {
    if (A.length !== B.length) throw new Error('A.length !== B.length')
    let dotProduct = 0,
        mA = 0,
        mB = 0
    for (let i = 0; i < A.length; i++) {
        dotProduct += A[i] * B[i]
        mA += A[i] * A[i]
        mB += B[i] * B[i]
    }
    mA = Math.sqrt(mA)
    mB = Math.sqrt(mB)
    let similarity = dotProduct / (mA * mB)
    return similarity
}

// Tweaked version of example from here: https://developer.mozilla.org/en-US/docs/Web/API/ReadableStreamDefaultReader/read
export const makeTextFileLineIterator = async function* (
    blob: Blob,
) {
    const utf8Decoder = new TextDecoder('utf-8')
    const stream = await blob.stream()
    const reader = stream.getReader()

    let { value: read, done: readerDone } = await reader.read()
    let chunk: string | Uint8Array = read
    chunk = chunk ? utf8Decoder.decode(chunk, { stream: true }) : ''

    let re = /\r\n|\n|\r/gm
    let startIndex = 0

    while (true) {
        let result = re.exec(chunk)
        if (!result) {
            if (readerDone) {
                break
            }
            let remainder = chunk.substr(startIndex)
            ;({ value: chunk, done: readerDone } = await reader.read())
            chunk =
                remainder +
                (chunk ? utf8Decoder.decode(chunk, { stream: true }) : '')
            startIndex = re.lastIndex = 0
            continue
        }
        yield chunk.substring(startIndex, result.index)
        startIndex = re.lastIndex
    }
    if (startIndex < chunk.length) {
        // last line didn't end in a newline char
        yield chunk.substr(startIndex)
    }
}

export const generateEmbeddingMap = async (fileBlob: Blob) => {
    const embeddings: { [key: string]: any } = {}
    let i = 0
    for await (let line of makeTextFileLineIterator(fileBlob)) {
        if (!line || !line.trim()) continue // <-- to skip final new line (not sure if this is needed)
        let [filePath, embeddingVec] = line.split('\t')
        embeddings[filePath] = JSON.parse(embeddingVec)
        i++
        if (i % 1000 === 0) {
            await sleep(10)
        }
    }
    return embeddings
}

// export const  getRgbData = async (blob: Blob) =>  {
//     // let blob = await fetch(imgUrl, {referrer:""}).then(r => r.blob());

//     let resizedBlob = await bicubicResizeAndCenterCrop(blob)
//     let img = await createImageBitmap(resizedBlob)

//     let canvas = new OffscreenCanvas(224, 224)
//     let ctx = canvas.getContext('2d')
//     ctx.drawImage(img, 0, 0)
//     let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)

//     let rgbData = [[], [], []] // [r, g, b]
//     // remove alpha and put into correct shape:
//     let d = imageData.data
//     for (let i = 0; i < d.length; i += 4) {
//         let x = (i / 4) % canvas.width
//         let y = Math.floor(i / 4 / canvas.width)
//         if (!rgbData[0][y]) rgbData[0][y] = []
//         if (!rgbData[1][y]) rgbData[1][y] = []
//         if (!rgbData[2][y]) rgbData[2][y] = []
//         rgbData[0][y][x] = d[i + 0] / 255
//         rgbData[1][y][x] = d[i + 1] / 255
//         rgbData[2][y][x] = d[i + 2] / 255
//         // From CLIP repo: Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
//         rgbData[0][y][x] = (rgbData[0][y][x] - 0.48145466) / 0.26862954
//         rgbData[1][y][x] = (rgbData[1][y][x] - 0.4578275) / 0.26130258
//         rgbData[2][y][x] = (rgbData[2][y][x] - 0.40821073) / 0.27577711
//     }
//     const flattened = Float32Array.from(rgbData.flat().flat())
//     return flattened
// }

// async function bicubicResizeAndCenterCrop(blob) {
//     let im1 = vips.Image.newFromBuffer(await blob.arrayBuffer())

//     // Resize so smallest side is 224px:
//     const scale = 224 / Math.min(im1.height, im1.width)
//     let im2 = im1.resize(scale, { kernel: vips.Kernel.cubic })

//     // crop to 224x224:
//     let left = (im2.width - 224) / 2
//     let top = (im2.height - 224) / 2
//     let im3 = im2.crop(left, top, 224, 224)

//     let outBuffer = new Uint8Array(im3.writeToBuffer('.png'))
//     im1.delete(), im2.delete(), im3.delete()
//     return new Blob([outBuffer], { type: 'image/png' })
// }
