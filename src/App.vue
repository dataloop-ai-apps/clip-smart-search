<template>
    <dl-theme-provider :is-dark="isDark">
        <DlProgressBar
            :label="label"
            :value="progress"
            :indeterminate="indeterminate"
            width="200px"
            show-value
            show-percentage
            :summary="summary"
        />

        <DlInput
            v-model="searchText"
            placeholder="Search item embeddings"
            type="text"
        />
        <DlButton label="Click" @click="performSearch" />

        <DlTable
            :columns="columns"
            :rows="rows"
            dense
            :loading="loading"
            row-key="name"
            title="Similarity With Search Text"
        />
    </dl-theme-provider>
</template>

<script setup lang="ts">
import { DlEvent } from '@dataloop-ai/jssdk'
import { computed, onMounted, ref } from 'vue'
import {
    DlThemeProvider,
    DlProgressBar,
    DlInput,
    DlButton,
    DlTable
} from '@dataloop-ai/components'
import {
    cosineSimilarity,
    downloadBlobWithProgress,
    generateEmbeddingMap
} from './utils'
import { OnnxModelSession } from './OnnxModelSession'

const columns = [
    {
        name: 'name',
        required: true,
        label: 'Item name',
        align: 'left',
        field: 'name'
    }
    {
        name: 'value',
        required: true,
        label: 'Cosine Value',
        align: 'left',
        field: 'name',
        sortable: true
    }
]

const theme = ref('light')
const progress = ref(0)
const label = ref(null)
const modelSession = ref(null)
const embeddings = ref(null)
const searchText = ref('')
const summary = ref('')
const rows = ref([])
const loading = ref(false)
const indeterminate = ref(false)

const THRESHOLD = 0.2093

const performSearch = async () => {
    loading.value = true
    const result: { name: string; value: any }[] = []
    const embedding = await modelSession.value.run(searchText.value)
    for (const [key, value] of Object.entries(embeddings.value)) {
        if (cosineSimilarity(embedding, value as any) < THRESHOLD) {
            result.push({ name: key, value })
        }
    }

    rows.value = result.sort((a, b) => a.value - b.value )
    loading.value = false
}

const isDark = computed(() => theme.value === 'dark')

const initEvents = () => {
    window.dl.on(DlEvent.READY, async () => {
        try {
            const settings = await window.dl.settings.get()
            theme.value = settings.theme
            initializeDependencies()
        } catch (e) {
            throw new Error('Error getting settings', e)
        }
    })
    window.dl.on(DlEvent.THEME, (mode: string) => {
        theme.value = mode
    })
}

const loadEmbeddingsMap = async () => {
    const item = await window.dl.items.get('642d27857708821f1c4ac972')
    const url = await window.dl.items.stream(item.stream)
    const blob = await downloadBlobWithProgress(url)
    const map = await generateEmbeddingMap(blob)
    return map
}

const initializeDependencies = async () => {
    try {
        modelSession.value = new OnnxModelSession()
        label.value = 'Loading...'
        summary.value = 'Loading model binaries'
        await modelSession.value.init({
            onProgress: onModelInitProgress,
            warmup: true
        })
        label.value = 'Ready!'
        summary.value = ''
    } catch (e) {
        console.error('Error initializing OnnxModelSession', e)
        label.value = 'Error!'
    }
    try {
        label.value = 'Loading...'
        summary.value = 'Loading Embeddings'
        indeterminate.value = true
        embeddings.value = await loadEmbeddingsMap()
        label.value = 'Ready!'
        summary.value = ''
        indeterminate.value = false
    } catch (e) {
        console.error('Failed to initialize EmbeddingsMap', e)
        label.value = 'Error!'
    }

    summary.value = ''
}

onMounted(async () => {
    try {
        await window.dl.init()
        initEvents()
    } catch (e) {
        console.error('Error initializing xFrameDriver', e)
    }
})

const onModelInitProgress = (event: ProgressEvent<EventTarget>) => {
    const ratio = event.loaded / event.total
    progress.value = ratio
}
</script>

<style scoped></style>
