<template>
    <dl-theme-provider :is-dark="isDark">
        <img
            v-if="itemStream"
            :src="itemStream"
            :width="itemWidth"
            :height="itemHeight"
        />
        <div style="display: flex; gap: 5px">
            <dl-chip
                v-for="annotation in annotations"
                :key="annotation.id"
                :label="annotation.label"
                :color="annotation.labelColor"
            />
        </div>
    </dl-theme-provider>
</template>

<script setup lang="ts">
import {DlAnnotationEvent, DlEvent, SDKAnnotation} from '@dataloop-ai/jssdk'
import { computed, onMounted, ref } from 'vue'
import { DlChip, DlThemeProvider } from '@dataloop-ai/components'

const theme = ref('light')
const item = ref(null)
const itemStream = ref('')
const annotations = ref<SDKAnnotation[]>([])

const isDark = computed(() => theme.value === 'dark')

const getAnnotations = async () => {
    const pagedResponse = await window.dl.annotations.query()
    annotations.value = pagedResponse.items
    debugger
}

const initEvents = () => {
    window.dl.on(DlEvent.READY, async () => {
        try {
            const settings = await window.dl.settings.get()
            theme.value = settings.theme
            await getItem()
            await getAnnotations()
        } catch (e) {
            throw new Error('Error getting settings', e)
        }
    })
    window.dl.on(DlEvent.THEME, (mode: string) => {
        debugger
        theme.value = mode
    })
    window.dl.on(DlAnnotationEvent.CREATED, getAnnotations)
    window.dl.on(DlAnnotationEvent.BULK_DELETED, getAnnotations)
    window.dl.on(DlAnnotationEvent.DELETED, getAnnotations)
}

const getItem = async () => {
    item.value = await window.dl.items.get()
    itemStream.value = await window.dl.items.stream(item.value.stream)
}

const itemWidth = computed(() => {
    return item.value?.metadata?.system?.width
})

const itemHeight = computed(() => {
    return item.value?.metadata?.system?.height
})

onMounted(async () => {
    try {
        await window.dl.init()
        initEvents()
    } catch (e) {
        console.error('Error initializing xFrameDriver', e)
    }
})
</script>

<style scoped>
</style>
