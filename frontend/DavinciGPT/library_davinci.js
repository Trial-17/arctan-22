window.$ = window.$ || {};
$._MYFUNCTIONS = $._MYFUNCTIONS || {};

$._MYFUNCTIONS.rewindPointsStack = [];

/**
 * Affiche une boîte de dialogue (alerte) dans DaVinci Resolve.
 * Comme le conteneur DaVinci est une fenêtre Electron (Chromium), la fonction native alert() fonctionne.
 * @param {string} message - Le message à afficher
 */
$._MYFUNCTIONS.showAlert = function (message) {
    alert(message);
};


