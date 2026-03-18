const { ipcRenderer } = require('electron');
const path = require('path');

// Expose the absolute path of the plugin directory
window.__RESOLVE_PLUGIN_PATH__ = __dirname;

// Exposure of the WorkflowIntegration module
// This allows the renderer (even when served from localhost) to access Resolve's API
try {
    const WorkflowIntegration = require(path.join(__dirname, 'WorkflowIntegration.node'));
    window.WorkflowIntegration = WorkflowIntegration;
    console.log('[Preload] WorkflowIntegration module loaded and exposed');

    // Overload require to handle the relative path often used in Resolve scripts
    // This fixes the "Cannot find module './WorkflowIntegration.node'" error
    const oldRequire = window.require;
    window.require = function (moduleName) {
        if (moduleName === './WorkflowIntegration.node' || moduleName === 'WorkflowIntegration.node') {
            return WorkflowIntegration;
        }
        return oldRequire.apply(this, arguments);
    };
} catch (e) {
    console.error('[Preload] Failed to load WorkflowIntegration:', e);
}

window.addEventListener('DOMContentLoaded', () => {
    const replaceText = (selector, text) => {
        const element = document.getElementById(selector)
        if (element) element.innerText = text
    }

    for (const type of ['chrome', 'node', 'electron']) {
        replaceText(`${type}-version`, process.versions[type])
    }
})
