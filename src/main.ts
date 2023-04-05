import "./style.css";
import { createApp } from "vue"
import App from "./App.vue"
import { xFrameDriver } from '@dataloop-ai/jssdk'

declare global {
    interface Window {
        dl: xFrameDriver,
        ort: any
    }
}

createApp(App).mount("#app")
