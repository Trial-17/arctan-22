const { ipcRenderer } = require('electron');
const path = require('path');
const fs = require('fs');

// Expose the absolute path of the plugin directory
window.__RESOLVE_PLUGIN_PATH__ = __dirname;

// Plugin id — MUST match <Id> in manifest.xml. library_davinci.js calls
// WorkflowIntegration.Initialize(window.__RESOLVE_PLUGIN_ID__); a mismatch makes
// the Resolve API fail to bind ("Resolve API indisponible") and every tool breaks.
window.__RESOLVE_PLUGIN_ID__ = 'com.premierecopilot.davinciclaude';

// Localise WorkflowIntegration.node — le pont natif vers l'API de Resolve.
// On NE redistribue PAS ce binaire (il est spécifique à l'OS) : on charge la copie
// de DaVinci Resolve lui-même, depuis ses exemples SDK. Avantages : bon OS garanti
// (corrige Windows, où un .node macOS empaqueté ne se chargerait jamais) et version
// alignée sur le Resolve installé. Les Workflow Integration Plugins exigent Resolve
// Studio → ce dossier Developer/Examples est toujours présent. Les noms des sous-
// dossiers d'exemples variant selon la version, on scanne Examples/*/.
// Ordre : install Resolve d'abord (bon OS), puis une éventuelle copie locale en
// ultime secours (jamais le binaire mauvais-OS empaqueté par erreur).
function locateWorkflowIntegrationNode() {
    const isWin = process.platform === 'win32';
    const examplesBases = isWin
        ? [path.join(process.env.PROGRAMDATA || 'C:\\ProgramData', 'Blackmagic Design', 'DaVinci Resolve', 'Support', 'Developer', 'Workflow Integrations', 'Examples')]
        : ['/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Workflow Integrations/Examples'];

    for (const base of examplesBases) {
        try {
            const subs = fs.readdirSync(base);
            for (const sub of subs) {
                const candidate = path.join(base, sub, 'WorkflowIntegration.node');
                if (fs.existsSync(candidate)) return candidate;
            }
        } catch (e) { /* dossier absent → base suivante */ }
    }

    // Ultime secours : une copie livrée à côté du plugin (normalement ABSENTE).
    const local = path.join(__dirname, 'WorkflowIntegration.node');
    try { if (fs.existsSync(local)) return local; } catch (e) {}
    return null;
}

// Exposure of the WorkflowIntegration module
// This allows the renderer (even when served from localhost) to access Resolve's API
try {
    const nodePath = locateWorkflowIntegrationNode();
    if (!nodePath) throw new Error("WorkflowIntegration.node introuvable (Resolve Studio installé ?)");
    const WorkflowIntegration = require(nodePath);
    window.WorkflowIntegration = WorkflowIntegration;
    console.log('[Preload] WorkflowIntegration chargé depuis: ' + nodePath);

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
