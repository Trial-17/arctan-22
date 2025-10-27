#include "json2.js"

$._MYFUNCTIONS ={}

$._MYFUNCTIONS.createPremiereCopilotTempFolder = function() {
    var userDocsPath = Folder.myDocuments.fsName;

    var folderPath = userDocsPath + "/Adobe/Premiere Pro/Premiere Copilot/temp";
    var targetFolder = new Folder(folderPath);

    // Vérifie si le dossier existe déjà
    if (!targetFolder.exists) {
        var created = targetFolder.create();
        if (!created) {
            alert("Erreur : impossible de créer le dossier : " + folderPath);
            return null;
        }
    }

    return targetFolder;
}


// Podcast    -------------------------------------------------------------------------------
$._MYFUNCTIONS.podcast_getTrackNumber = function() {
    // Récupère la séquence active
    var sequence = app.project.activeSequence;
    if (!sequence) {
        $.writeln("Aucune séquence active trouvée.");
        return;
    }
    
    // Initialisation des compteurs pour les pistes vidéo et audio
    var videoCount = 0;
    var audioCount = 0;
    
    // Parcours des pistes vidéo
    for (var i = 0; i < sequence.videoTracks.numTracks; i++) {
        var track = sequence.videoTracks[i];
        if (track.clips.numItems > 0) {
            videoCount++;
        }
    }
    
    // Parcours des pistes audio
    for (var j = 0; j < sequence.audioTracks.numTracks; j++) {
        var track = sequence.audioTracks[j];
        if (track.clips.numItems > 0) {
            audioCount++;
        }
    }
    
    $.writeln("Nombre de pistes vidéo avec éléments : " + videoCount);
    $.writeln("Nombre de pistes audio avec éléments : " + audioCount);

    // Renvoie un tableau [nombrePistesVideo, nombrePistesAudio]
    return [videoCount, audioCount];
}

$._MYFUNCTIONS.exportAudioTracksDirect = function() {
    // var presetFile = new File(presetPath);
    // var truePresetPath = presetFile.fsName;

    var os = $.os.toLowerCase();
    var basePath = "";

    // Récupère la séquence active
    var sequence = app.project.activeSequence;

    var exportedPaths = [];
    // Parcourt chaque piste audio
    for (var i = 0; i < sequence.audioTracks.numTracks; i++) {
        var track = sequence.audioTracks[i];
        if (track.clips.numItems > 0) {
            // Mute toutes les pistes audio sauf la piste i
            for (var j = 0; j < sequence.audioTracks.numTracks; j++) {
                if (j === i) {
                    sequence.audioTracks[j].setMute(0);
                }
                else {
                    sequence.audioTracks[j].setMute(1);
                }
            }
            
            var tempFolder = $._MYFUNCTIONS.createPremiereCopilotTempFolder();
            
            if (os.indexOf("mac") !== -1) {
                var truePresetPath = "/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForJumpCuts.epr";
                var outputFile = new File(tempFolder.fsName + "/Track " + (i + 1) + ".mp3");
            } else if (os.indexOf("windows") !== -1) {
                var presetFile = new File("C:/Program Files/Common Files/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForJumpCuts.epr");
                var truePresetPath = presetFile.fsName; // te donne automatiquement un chemin natif : C:\...
                var outputFile = new File(tempFolder.fsName + "/Track " + (i + 1) + ".mp3");
            }

            var success = sequence.exportAsMediaDirect(outputFile.fsName, truePresetPath, 1);
            exportedPaths.push(outputFile.fsName);
        }
    }
    
    for (var i = 0; i < sequence.audioTracks.numTracks; i++) {
        sequence.audioTracks[i].setMute(0);
    }
    
    return JSON.stringify(exportedPaths);
}

$._MYFUNCTIONS.supprimerClipsDesactives_PODCAST = function() {
    var sequence = app.project.activeSequence;


    var clipsASupprimer = []; // Tableau pour stocker les clips à supprimer

    // --- ÉTAPE 1 : Collecter tous les clips désactivés ---
    // On parcourt les pistes vidéo
    for (var i = 0; i < sequence.videoTracks.numTracks; i++) {
        var pisteVideo = sequence.videoTracks[i];
        for (var j = 0; j < pisteVideo.clips.numItems; j++) {
            var clip = pisteVideo.clips[j];
            if (clip.disabled) {
                clipsASupprimer.push(clip);
            }
        }
    }


    if (clipsASupprimer.length === 0) {
        alert("No disabled clips found.");
        return;
    }

    // --- ÉTAPE 2 : Trier les clips de la fin vers le début ---
    // On trie en fonction du temps de début (start time) de chaque clip, par ordre décroissant.
    clipsASupprimer.sort(function(a, b) {
        return b.start.seconds - a.start.seconds;
    });

    // --- ÉTAPE 3 : Supprimer les clips en une seule passe ---
    for (var i = 0; i < clipsASupprimer.length; i++) {
        // Le premier 'true' active la suppression avec raccord (ripple delete)
        clipsASupprimer[i].remove(false, false);
    }
    
}

$._MYFUNCTIONS.perform_cut = function(timecodes) {
    timecodes = JSON.parse(timecodes);
    app.enableQE();
    var seq = app.project.activeSequence;
    var clipsToProcess = [];


    for (var k = 0; k < seq.videoTracks.numTracks; k++) {
        var track = seq.videoTracks[k];
        if (track && track.clips.numItems > 0) {
            for (var c = 0; c < track.clips.numItems; c++) {
                var clip = track.clips[c];
                clip.disabled = 1;
            }
        }
    }



    if (seq) {
        var ticksPerSecond = 254016000000;
        var sqe = qe.project.getActiveSequence();

        if (sqe) {
            for (var i = 1; i < timecodes.length; i++) {
                var timeObj = timecodes[i];
                var timeInSeconds = timeObj.time;
                var timeInTicks = Math.floor(timeInSeconds * ticksPerSecond);

                var time = new Time();
                time.ticks = timeInTicks.toString();

                seq.setPlayerPosition(time.ticks);

                var currentTicks = parseInt(time.ticks);

                // Exception pour time = 0 : pas de razor, juste désactiver/activer
                if (timeInSeconds === 0) {
                    for (var k = 0; k < seq.videoTracks.numTracks; k++) {
                        var track = seq.videoTracks[k];
                        if (track && track.clips.numItems > 0) {
                            for (var c = 0; c < track.clips.numItems; c++) {
                                var clip = track.clips[c];
                                var clipStart = parseInt(clip.start.ticks);
                                var clipEnd = clipStart + parseInt(clip.duration.ticks);

                                if (clipStart <= currentTicks && currentTicks < clipEnd) {
                                    // if (clip.hasOwnProperty("disabled")) {
                                    //     clip.disabled = (k !== timeObj.camera);
                                    // }
                                    if (k !== timeObj.camera){
                                        clipsToProcess.push(clip);
                                    }
                                }
                            }
                        }
                    }
                    continue; // ne pas faire de razor à t = 0
                }

                // Razor all video tracks
                for (var j = 0; j < sqe.numVideoTracks; j++) {
                    var track = sqe.getVideoTrackAt(j);
                    if (track) {
                        track.razor(sqe.CTI.timecode);
                    }
                }

                var track = seq.videoTracks[timeObj.camera];

                for (var c = track.clips.numItems-1 ; c >= 0; c--) {
                    var clip = track.clips[c];
                    var clipStart = parseInt(clip.start.ticks);
                    var clipEnd = clipStart + parseInt(clip.duration.ticks);

                    if (clipStart <= currentTicks && currentTicks < clipEnd) {
                        clipsToProcess.push(clip);
                        break;
                    }
                }
            }
        } 
    } 


    for(var k = 0; k < clipsToProcess.length; k++){
        var clip = clipsToProcess[k];
        clip.disabled = 0;
    }

}



// Tool    -------------------------------------------------------------------------------
$._MYFUNCTIONS.test = function() {
    alert('Le test JSX fonctionne');
}

$._MYFUNCTIONS.backupActiveSequence = function() {
    var project = app.project;
    var rootBin = project.rootItem;
    var backupBinName = "Copilot Backup";
    var backupBin = null;

    // --- Ce qui a été ajouté : une petite fonction pour le formatage ---
    // Ajoute un zéro devant un nombre s'il est inférieur à 10 (ex: 7 -> "07")
    function padZero(num) {
        return num < 10 ? '0' + num : num;
    }

    // Vérifier si le bin "Copilot Backup" existe déjà
    for (var i = 0; i < rootBin.children.numItems; i++) {
        var item = rootBin.children[i];
        if (item.type === 2 && item.name === backupBinName) { // Type 2 = Bin
            backupBin = item;
            break;
        }
    }

    // Si le bin n'existe pas, le créer
    if (backupBin === null) {
        backupBin = rootBin.createBin(backupBinName);
    }

    var activeSequence = app.project.activeSequence;
    if (!activeSequence) {
        return false;
    }
    var originalSequenceID = activeSequence.sequenceID;

    var originalSequenceName = activeSequence.name;
    var initialNumItems = rootBin.children.numItems;
    var seqDuplicate = activeSequence.clone();


    var newSequenceProjectItem = null;

    // Identifier la nouvelle séquence ajoutée
    for (var i = initialNumItems; i < rootBin.children.numItems; i++) {
        var item = rootBin.children[i];
        if (item.type === 1) { // Type 1 = Séquence
            newSequenceProjectItem = item;
            break;
        }
    }

    if (newSequenceProjectItem) {
        // --- MODIFICATION PRINCIPALE : Renommer la séquence ---

        // 1. Obtenir la date et l'heure actuelles
        var now = new Date();
        var timestamp = now.getFullYear() + '-' + 
                        padZero(now.getMonth() + 1) + '-' + // Les mois sont de 0 à 11
                        padZero(now.getDate()) + ' ' + 
                        padZero(now.getHours()) + ':' + 
                        padZero(now.getMinutes()) + ':' + 
                        padZero(now.getSeconds());

        // 2. Construire le nouveau nom en gérant les backups existants
        var baseName = originalSequenceName;
        var backupSuffixRegex = / Copilot Backup \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/;

        if (backupSuffixRegex.test(originalSequenceName)) {
            baseName = originalSequenceName.replace(backupSuffixRegex, "");
        }

        var newName = baseName + " Copilot Backup " + timestamp;

        // 3. Appliquer le nouveau nom à la séquence clonée
        newSequenceProjectItem.name = newName;
        
        // 4. Déplacer la séquence renommée dans le bin de backup
        newSequenceProjectItem.moveBin(backupBin);
    }

    // Réactiver la séquence d'origine après clonage
    for (var i = 0; i < app.project.sequences.length; i++) {
        var seq = app.project.sequences[i];
        if (seq.sequenceID === originalSequenceID) {
            app.project.activeSequence = seq;
            break;
        }
    }
    $.sleep(1); 
    // Fermer la séquence clonée si elle est ouverte
    // if (clonedSequenceName) {
    //     for (var i = 0; i < app.project.sequences.length; i++) {
    //         var seq = app.project.sequences[i];
    //         if (seq.name === clonedSequenceName) {
    //             seq.close(); // Ferme la séquence clonée
    //             break;
    //         }
    //     }
    // }
}

$._MYFUNCTIONS.getImportPath = function() {

    var project = app.project;
    var rootBin = project.rootItem;
    var videoExtensions = [".mp4", ".mov", ".avi"];
    var videoPaths = [];

    function searchBin(bin) {
        for (var i = 0; i < bin.children.numItems; i++) {
            var item = bin.children[i];
            if (item.type === 2) { // Type 2 = Bin (Dossier dans le projet)
                searchBin(item); // Recherche récursive dans les sous-dossiers
            } else if (item.type === 1) { // Type 1 = Media
                var filePath = item.getMediaPath(); // Récupère le chemin absolu du fichier

                if (filePath && typeof filePath === "string") {
                    var lowerPath = filePath.toLowerCase();
                    for (var j = 0; j < videoExtensions.length; j++) {
                        if (lowerPath.indexOf(videoExtensions[j]) !== -1) {
                            videoPaths.push(filePath);
                            break;
                        }
                    }
                }
            }
        }
    }

    searchBin(rootBin);

    return JSON.stringify(videoPaths);

}

// DeepResearch    -------------------------------------------------------------------------------

// $._MYFUNCTIONS.importIntoDeepResearch = function(input) {

//     var folderName = "DeepResearch Import";
//     var root = app.project.rootItem;
//     var targetBin = null;

//     // 1. Cherche ou crée le chutier "DeepResearch Import"
//     for (var i = 0; i < root.children.numItems; i++) {
//         var child = root.children[i];
//         if (child && child.name === folderName && child.type === ProjectItemType.BIN) {
//             targetBin = child;
//             break;
//         }
//     }
//     if (targetBin === null) {
//         targetBin = root.createBin(folderName);
//     }

//     var filePaths = JSON.parse(input);

//     // 3. Importation dans le chutier cible
//     var importedItems = [];

//     for (var j = 0; j < filePaths.length; j++) {
//         var filePath = filePaths[j];
//         var file = new File(filePath);

//         if (file.exists) {
//             var beforeCount = targetBin.children.numItems;

//             app.project.importFiles(
//                 [file.fsName],
//                 false, // suppressUI
//                 targetBin,
//                 false  // importAsNumberedStills
//             );

//             var afterCount = targetBin.children.numItems;

//             if (afterCount > beforeCount) {
//                 var newItem = targetBin.children[afterCount - 1];
//                 importedItems.push(newItem);
//             }
//         } else {
//             $.writeln("Fichier introuvable : " + filePath);
//         }
//     }

//     // 4. Ajout à la fin de la séquence active
//     if (importedItems.length > 0 && app.project.activeSequence) {
//         var seq = app.project.activeSequence;
//         var videoTrack = seq.videoTracks[0]; // on place sur V1

//         // On récupère le temps de fin max actuel
//         var endTime = 0.0;
//         for (var t = 0; t < videoTrack.clips.numItems; t++) {
//             var clip = videoTrack.clips[t];
//             var thisEnd = clip.start.seconds + clip.duration.seconds;
//             if (thisEnd > endTime) {
//                 endTime = thisEnd;
//             }
//         }

//         for (var k = 0; k < importedItems.length; k++) {
//             var projectItem = importedItems[k];
//             if (projectItem && projectItem.type === ProjectItemType.CLIP) {
//                 videoTrack.insertClip(projectItem, endTime);
//                 endTime += projectItem.getOutPoint().seconds; // on empile proprement
//             }
//         }
//     } else {
//         Alert("Aucune séquence active ou aucun rush importé.");
//     }
// }


// Video Generation    ----------------------------------------------------------------------------

$._MYFUNCTIONS.openFinderAndGetPath = function() {
    var prompt = "Select an image";
    var filter = "*.jpg;*.jpeg;*.png"; // Restriction aux images
    var multiSelect = false;

    var selectedFile = File.openDialog(prompt, filter, multiSelect);

    if (selectedFile !== null) {
        var filePath = selectedFile.fsName;

        // Vérifie que c'est une image PNG ou JPEG
        var extension = filePath.toLowerCase().split(".").pop();
        if (!(extension === "jpg" || extension === "jpeg" || extension === "png")) {
            return "undefined";
        }



        return filePath;
    }

    return "undefined";
}

$._MYFUNCTIONS.openFinderAndGetAudioPath = function() {
    var prompt = "Sélectionnez un fichier audio";
    var filter = "*.mp3;*.wav";
    var multiSelect = false;

    var selectedFile = File.openDialog(prompt, filter, multiSelect);

    if (selectedFile !== null) {
        var filePath = selectedFile.fsName;
        var extension = filePath.toLowerCase().split(".").pop();

        if (extension === "mp3" || extension === "wav") {
            return filePath;
        }
    }

    return "undefined";
};

$._MYFUNCTIONS.openFinderAndGetVideoPath = function() {
    var prompt = "Sélectionnez un fichier vidéo";
    var filter = "*.mp4;*.mov";
    var multiSelect = false;

    var selectedFile = File.openDialog(prompt, filter, multiSelect);

    if (selectedFile !== null) {
        var filePath = selectedFile.fsName;
        var extension = filePath.toLowerCase().split(".").pop();

        if (extension === "mp4" || extension === "mov") {
            return filePath;
        }
    }

    return "undefined";
};

$._MYFUNCTIONS.exportCurrentFrameToTempPNG = function() {
    app.enableQE();

    var activeSequence = qe.project.getActiveSequence();

    var timecode = activeSequence.CTI.timecode; // ex: "00:00:01:05"
    var safeTime = timecode.replace(/:|;/ig, '_');

    var documentsFolder = Folder.myDocuments;
    var outputFolder = new Folder(documentsFolder.fullName + "/Adobe/Premiere Pro/Premiere Copilot/temp");

    if (!outputFolder.exists) {
        outputFolder.create();
        $.writeln("📁 Created folder: " + outputFolder.fsName);
    }

    var fileName = activeSequence.name + '__' + safeTime ;

    var os = $.os.toLowerCase();

    if (os.indexOf("mac") !== -1) {
        var outputPath = outputFolder.fsName + '/' + fileName;
    } else if (os.indexOf("windows") !== -1) {
        var outputPath = outputFolder.fsName + "'\'" + fileName;
    }

    // Export frame at current CTI timecode
    activeSequence.exportFramePNG(timecode, outputPath);


    return outputPath + '.png';
}

$._MYFUNCTIONS.importVideoToVideoGenerationBin = function(filePath) {
    if (!filePath || !File(filePath).exists) {
        $.writeln("❌ Fichier introuvable : " + filePath);
        return;
    }

    var projectRoot = app.project.rootItem;
    var binName = "Video Generation";
    var targetBin = null;

    // Chercher le bin "Video Generation" à la racine
    for (var i = 0; i < projectRoot.children.numItems; i++) {
        var child = projectRoot.children[i];
        if (child && child.type === ProjectItemType.BIN && child.name === binName) {
            targetBin = child;
            break;
        }
    }

    // Le créer s'il n'existe pas
    if (!targetBin) {
        targetBin = projectRoot.createBin(binName);
        $.writeln("📁 Bin créé : " + binName);
    }

    // Importer la vidéo dans le bin
    var imported = app.project.importFiles(
        [filePath],
        false,     // suppress UI
        targetBin, // target bin
        false      // import recursively
    );

    if (imported && imported[0]) {
        $.writeln("✅ Fichier importé : " + imported[0].name + " → " + binName);
    } else {
        $.writeln("❌ Échec de l'importation.");
    }
}

$._MYFUNCTIONS.export5SecondVideoFromPlayhead = function() {
    var userDocsPath = Folder.myDocuments.fsName;
    var exportFolder1 = userDocsPath + "/Adobe/Premiere Pro/Premiere Copilot/temp";
    var exportFolder = new Folder(exportFolder1);

    var targetSequence = app.project.activeSequence;
    if (!targetSequence) {
        alert("No active sequence found.");
        return null;
    }

    var playheadTime = targetSequence.getPlayerPosition(); // This is a Time object
    targetSequence.setInPoint(playheadTime.seconds);

    var thirtySecondsInTicks = 5 * 254016000000;
    var endTimeTicks = parseFloat(playheadTime.ticks) + thirtySecondsInTicks;

    var sequenceDurationTicks = parseFloat(targetSequence.end.ticks);
    if (endTimeTicks > sequenceDurationTicks) {
        endTimeTicks = sequenceDurationTicks;
    }

    var tempTime = new Time();
    tempTime.ticks = endTimeTicks.toString();
    targetSequence.setOutPoint(tempTime.seconds);

    var os = $.os.toLowerCase();
    var basePath = "";

    // Génère un timestamp unique
    function getTimestamp() {
        var now = new Date();
        var pad = function(num) { return (num < 10 ? "0" : "") + num; };
        return now.getFullYear().toString() +
            pad(now.getMonth() + 1) +
            pad(now.getDate()) + "_" +
            pad(now.getHours()) +
            pad(now.getMinutes()) +
            pad(now.getSeconds());
    }

    var timestamp = getTimestamp();
    if (os.indexOf("mac") !== -1) {
        var truePresetPath = "/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/VideoForGeneration.epr";
        var outputFilePath = exportFolder.fsName + "/" + targetSequence.name + "_" + timestamp + ".mp4";
    } else if (os.indexOf("windows") !== -1) {
       var presetFile = new File("C:/Program Files/Common Files/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/VideoForGeneration.epr");
        truePresetPath = presetFile.fsName;
        var outputFilePath = File(exportFolder.absoluteURI + "/" + targetSequence.name + "_" + timestamp + ".mp4").fsName;
    }

    // Export audio for the defined range
    var success = targetSequence.exportAsMediaDirect(outputFilePath, truePresetPath, 1); // 1 = in to out


    if (!success) {
        alert("Erreur lors de l'export audio de la séquence : " + targetSequence.name);
        return null;
    }

    $.writeln("Export audio réussi vers : " + outputFilePath);
    return outputFilePath;
}

$._MYFUNCTIONS.exportAudioSelection = function() {
    var activeSequence = app.project.activeSequence;
    if (!activeSequence) {
        alert("Aucune séquence active trouvée.");
        return null;
    }

    var MAX_DURATION_SECONDS = 300; // 5 minutes
    var FALLBACK_DURATION_SECONDS = 30; // 30 secondes

    var inPoint = activeSequence.getInPointAsTime();
    var outPoint = activeSequence.getOutPointAsTime();

    // Vérifie si une plage significative est définie par l'utilisateur
    var hasUserDefinedRange = outPoint.seconds > inPoint.seconds;
    if (hasUserDefinedRange) {
        // CAS 1: L'utilisateur a défini des points d'entrée/sortie
        var currentDuration = outPoint.seconds - inPoint.seconds;
        if (currentDuration > MAX_DURATION_SECONDS) {
            // La durée dépasse 5 minutes, on raccourcit le point de sortie
            var newOutTime = new Time();
            newOutTime.seconds = inPoint.seconds + MAX_DURATION_SECONDS;

            activeSequence.setOutPoint(newOutTime.seconds);
            $.writeln("La sélection dépassait 5 minutes. La plage d'export a été raccourcie.");
        }
        // Si la durée est inférieure ou égale à 5 minutes, on ne change rien.
    } else {
        // CAS 2: Pas de points d'entrée/sortie définis, on utilise l'ancien comportement
        var playheadTime = activeSequence.getPlayerPosition();
        activeSequence.setInPoint(playheadTime.seconds);

        var newOutTime = new Time();
        newOutTime.seconds = playheadTime.seconds + FALLBACK_DURATION_SECONDS;

        activeSequence.setOutPoint(newOutTime.seconds);
    }

    var userDocsPath = Folder.myDocuments.fsName;
    var exportFolder1 = userDocsPath + "/Adobe/Premiere Pro/Premiere Copilot/temp";
    var exportFolder = new Folder(exportFolder1);
    var os = $.os.toLowerCase();
    var basePath = "";

    // Génère un timestamp unique
    function getTimestamp() {
        var now = new Date();
        var pad = function(num) { return (num < 10 ? "0" : "") + num; };
        return now.getFullYear().toString() +
            pad(now.getMonth() + 1) +
            pad(now.getDate()) + "_" +
            pad(now.getHours()) +
            pad(now.getMinutes()) +
            pad(now.getSeconds());
    }

    var timestamp = getTimestamp();
    if (os.indexOf("mac") !== -1) {
        var truePresetPath = "/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForTranscriptionV2.epr";
        var outputFilePath = exportFolder.fsName + "/" + activeSequence.name + "_" + timestamp + ".mp3";
    } else if (os.indexOf("windows") !== -1) {
       var presetFile = new File("C:/Program Files/Common Files/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForTranscriptionV2.epr");
        truePresetPath = presetFile.fsName; // te donne automatiquement un chemin natif : C:\...
        var outputFilePath = File(exportFolder.absoluteURI + "/" + activeSequence.name + "_" + timestamp + ".mp3").fsName;
    }

    // Export audio for the defined range
    var success = activeSequence.exportAsMediaDirect(outputFilePath, truePresetPath, 1); // 1 = in to out


    if (!success) {
        alert("Erreur lors de l'export audio de la séquence : " + activeSequence.name);
        return null;
    }

    $.writeln("Export audio réussi vers : " + outputFilePath);
    return outputFilePath;
};

// AutoEdit            ----------------------------------------------------------------------------

// AutoEdit --- 0. Récupérer les stems    

$._MYFUNCTIONS.getAllBins = function() {
    var bins = [];

    /**
     * Parcours récursivement les éléments du projet à la recherche de bins.
     *
     * @param {ProjectItem} item Élément courant du projet à explorer.
     */
    function traverseItems(item) {
        if (item.type === ProjectItemType.BIN) {
            bins.push(item.name);

            // Explorer récursivement les enfants du bin
            for (var j = 0; j < item.children.numItems; j++) {
                traverseItems(item.children[j]);
            }
        }
    }

    var rootItems = app.project.rootItem.children;

    // Parcourir chaque élément racine du projet
    for (var i = 0; i < rootItems.numItems; i++) {
        traverseItems(rootItems[i]);
    }

    return JSON.stringify(bins);
}

$._MYFUNCTIONS.getAllSequences = function() {
    var sequences = app.project.sequences;
    var numSequences = sequences.numSequences;
    var sequenceNames = [];

    for (var i = 0; i < numSequences; i++) {
        sequenceNames.push(sequences[i].name);
    }

    return JSON.stringify(sequenceNames);
}

$._MYFUNCTIONS.exportFullAudioFromSequence = function() {
    var userDocsPath = Folder.myDocuments.fsName;
    var exportFolder1 = userDocsPath + "/Adobe/Premiere Pro/Premiere Copilot/temp";
    var exportFolder = new Folder(exportFolder1);

    var targetSequence = app.project.activeSequence;

    var os = $.os.toLowerCase();
    var basePath = "";

    // Génère un timestamp unique
    function getTimestamp() {
        var now = new Date();
        var pad = function(num) { return (num < 10 ? "0" : "") + num; };
        return now.getFullYear().toString() +
            pad(now.getMonth() + 1) +
            pad(now.getDate()) + "_" +
            pad(now.getHours()) +
            pad(now.getMinutes()) +
            pad(now.getSeconds());
    }

    var timestamp = getTimestamp();
    if (os.indexOf("mac") !== -1) {
        var truePresetPath = "/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForDemucs.epr";
        var outputFilePath = exportFolder.fsName + "/" + targetSequence.name + "_" + timestamp + ".wav";
    } else if (os.indexOf("windows") !== -1) {
       var presetFile = new File("C:/Program Files/Common Files/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForDemucs.epr");
        truePresetPath = presetFile.fsName; // te donne automatiquement un chemin natif : C:\...
        var outputFilePath = File(exportFolder.absoluteURI + "/" + targetSequence.name + "_" + timestamp + ".wav").fsName;
    }

    // Export audio complet en une fois
    var success = targetSequence.exportAsMediaDirect(outputFilePath, truePresetPath, 1); // 1 = in to out

    if (!success) {
        alert("Erreur lors de l'export audio de la séquence : " + targetSequence.name);
        return null;
    }

    $.writeln("Export audio réussi vers : " + outputFilePath);
    return outputFilePath;
}

$._MYFUNCTIONS.getRushPathsFromBin = function(binName) {
    var paths = [];

    /**
     * Parcours récursivement les éléments d'un dossier pour collecter les chemins des rushs.
     *
     * @param {ProjectItem} item - L'élément courant à explorer.
     */
    function traverseBin(item) {
        if (item.type === ProjectItemType.CLIP || item.type === ProjectItemType.FILE) {
            var mediaPath = item.getMediaPath();
            if (mediaPath) {
                paths.push(mediaPath);
            }
        } else if (item.type === ProjectItemType.BIN) {
            for (var i = 0; i < item.children.numItems; i++) {
                traverseBin(item.children[i]);
            }
        }
    }

    /**
     * Recherche récursive du bin par son nom.
     *
     * @param {ProjectItem} item - Élément courant à vérifier.
     * @param {String} name - Nom du bin recherché.
     * @returns {ProjectItem|null} Le bin trouvé, sinon null.
     */
    function findBinByName(item, name) {
        if (item.type === ProjectItemType.BIN && item.name === name) {
            return item;
        } else if (item.type === ProjectItemType.BIN) {
            for (var i = 0; i < item.children.numItems; i++) {
                var foundBin = findBinByName(item.children[i], name);
                if (foundBin) {
                    return foundBin;
                }
            }
        }
        return null;
    }

    // Rechercher le bin dans la racine du projet
    var rootItems = app.project.rootItem.children;
    var targetBin = null;

    for (var j = 0; j < rootItems.numItems; j++) {
        targetBin = findBinByName(rootItems[j], binName);
        if (targetBin) {
            break;
        }
    }

    if (!targetBin) {
        alert("Aucun dossier nommé '" + binName + "' trouvé.");
        return [];
    }

    traverseBin(targetBin);

    return JSON.stringify(paths);
}


// AutoEdit --- 1.a Music Analysis

$._MYFUNCTIONS.importAllWavToDemixed = function(folderPath) {
    var project = app.project;
    var root = project.rootItem;

    // Vérifie ou crée le dossier 'demixed'
    var demixedBin = null;
    for (var i = 0; i < root.children.numItems; i++) {
        if (root.children[i].type === ProjectItemType.BIN && root.children[i].name === "demixed") {
            demixedBin = root.children[i];
            break;
        }
    }
    if (!demixedBin) {
        demixedBin = root.createBin("demixed");
    }

    var importFolder = new Folder(folderPath);
    if (!importFolder.exists) {
        // alert("Le dossier spécifié n'existe pas : " + folderPath);
        return [];
    }

    var wavFileObjects = importFolder.getFiles("*.wav");
    if (wavFileObjects.length === 0) {
        // alert("Aucun fichier WAV trouvé dans le dossier spécifié.");
        return [];
    }

    var wavPaths = [];
    for (var j = 0; j < wavFileObjects.length; j++) {
        wavPaths.push(wavFileObjects[j].fsName);
    }

    // 🔒 Créer une vraie liste JavaScript des nodeId existants
    var existingNodeIds = [];
    for (var p = 0; p < demixedBin.children.numItems; p++) {
        var child = demixedBin.children[p];
        existingNodeIds.push(child.nodeId);
    }

    // Importer les nouveaux fichiers
    project.importFiles(wavPaths, true, demixedBin, false);

    // 🔍 Identifier les nouveaux éléments
    var importedItems = [];
    for (var k = 0; k < demixedBin.children.numItems; k++) {
        var item = demixedBin.children[k];
        var isNew = true;
        for (var n = 0; n < existingNodeIds.length; n++) {
            if (item.nodeId === existingNodeIds[n]) {
                isNew = false;
                break;
            }
        }
        if (isNew) {
            importedItems.push(item);
        }
    }

    return importedItems;
}

$._MYFUNCTIONS.ensure_minimum_empty_tracks = function(requiredEmptyTracks, forVideo, forAudio) {
    if (typeof requiredEmptyTracks !== "number" || requiredEmptyTracks < 1) {
        $.writeln("Please provide a valid number of empty tracks (>= 1).");
        return;
    }

    if (!forVideo && !forAudio) {
        $.writeln("Please enable either 'forVideo' or 'forAudio' or both.");
        return;
    }

    // Enable QE DOM (required for using qe.project functions)
    app.enableQE();

    // Get the active sequence
    var qeSequence = qe.project.getActiveSequence();
    if (!qeSequence) {
        alert("No active sequence found!");
        return;
    }

    // Add empty video tracks if requested
    if (forVideo) {
        var totalVideoTracks = qeSequence.numVideoTracks;
        var emptyVideoTracksCount = 0;

        for (var trackIndex = totalVideoTracks - 1; trackIndex >= 0; trackIndex--) {
            var videoTrack = app.project.activeSequence.videoTracks[trackIndex];
            if (videoTrack.clips.length === 0) {
                emptyVideoTracksCount++;
            } else {
                break; // Stop counting when a non-empty track is found
            }
        }

        var videoTracksToAdd = requiredEmptyTracks - emptyVideoTracksCount;
        if (videoTracksToAdd > 0) {
            qeSequence.addTracks(videoTracksToAdd, totalVideoTracks, 0); // Add video tracks
            $.writeln(videoTracksToAdd + " empty video track(s) added to meet the requirement.");
        } else {
            $.writeln("No video tracks added. The sequence already has " + emptyVideoTracksCount + " empty video track(s).");
        }
    }

    // Add empty audio tracks if requested
    if (forAudio) {
        var totalAudioTracks = qeSequence.numAudioTracks;
        var emptyAudioTracksCount = 0;

        for (var trackIndex = totalAudioTracks - 1; trackIndex >= 0; trackIndex--) {
            var audioTrack = app.project.activeSequence.audioTracks[trackIndex];
            if (audioTrack.clips.length === 0) {
                emptyAudioTracksCount++;
            } else {
                break; // Stop counting when a non-empty track is found
            }
        }

        var audioTracksToAdd = requiredEmptyTracks - emptyAudioTracksCount;
        if (audioTracksToAdd > 0) {
            qeSequence.addTracks(0, 0, audioTracksToAdd, 0, totalAudioTracks); // Add audio tracks
            $.writeln(audioTracksToAdd + " empty audio track(s) added to meet the requirement.");
        } else {
            $.writeln("No audio tracks added. The sequence already has " + emptyAudioTracksCount + " empty audio track(s).");
        }
    }
}

$._MYFUNCTIONS.insertWavsOnePerEmptyTrack = function(importedWavs) {
    var seq = app.project.activeSequence;
    if (!seq) {
        alert("Aucune séquence active.");
        return;
    }

    var insertTime = 0; // secondes
    var trackCount = seq.audioTracks.numTracks;
    var wavIndex = 0;

    for (var t = 0; t < trackCount; t++) {
        if (wavIndex >= importedWavs.length) {
            break; // tous les clips ont été insérés
        }

        var track = seq.audioTracks[t];

        // Vérifie si la piste est vide
        if (track.clips.numItems === 0) {
            track.insertClip(importedWavs[wavIndex], insertTime);
            wavIndex++;
        }
    }

    if (wavIndex < importedWavs.length) {
        alert("⚠️ " + wavIndex + " clip(s) inséré(s). " + (importedWavs.length - wavIndex) + " non inséré(s) faute de pistes vides.");
    } else {
        // alert("✅ Tous les WAV ont été insérés, un par piste vide.");
    }
}

$._MYFUNCTIONS.addMarkersOnFirstAudioClipAtTimes = function(secondsList, colorIndex) {
    var seq = app.project.activeSequence;
    if (!seq) {
        alert("Aucune séquence active.");
        return;
    }

    var audioTrack = seq.audioTracks[0];
    if (!audioTrack || audioTrack.clips.numItems === 0) {
        alert("Aucun clip dans la première piste audio.");
        return;
    }

    for (var i = 0; i < secondsList.length; i++) {
        var targetTime = secondsList[i];
        var time = new Time();
        time.seconds = targetTime;

        var targetClip = null;

        for (var c = 0; c < audioTrack.clips.numItems; c++) {
            var clip = audioTrack.clips[c];
            var clipStart = clip.start.seconds;
            var clipEnd = clip.end.seconds;

            if (targetTime >= clipStart && targetTime <= clipEnd) {
                targetClip = clip;
                break;
            }
        }

        if (targetClip) {
            var markers = targetClip.projectItem.getMarkers();
            if (markers) {
                var localTime = new Time();

                // On ajuste en tenant compte de l'inPoint
                var inPointSeconds = targetClip.inPoint.seconds;
                var timelineOffset = targetTime - targetClip.start.seconds;

                localTime.seconds = inPointSeconds + timelineOffset;

                var marker = markers.createMarker(localTime.seconds);
                marker.name = "Marker at " + targetTime.toFixed(2) + "s";
                marker.comments = "Ajouté par script";

                // Applique la couleur
                marker.setColorByIndex(colorIndex, 0); // 0 = premier marqueur
            }
        }
    }
}

$._MYFUNCTIONS.addSequenceMarker = function(startSeconds, endSeconds, title, colorIndex) {
    var seq = app.project.activeSequence;
    if (!seq) {
        alert("Aucune séquence active.");
        return;
    }

    if (endSeconds <= startSeconds) {
        alert("Le temps de fin doit être supérieur au début.");
        return;
    }

    var marker = seq.markers.createMarker(startSeconds); // start : OK
    marker.name = title;


    // ✅ End = valeur absolue en secondes
    marker.end = endSeconds;

    // ✅ Couleur
    marker.setColorByIndex(colorIndex, 0); // 0 = ce marqueur
}

$._MYFUNCTIONS.apply_music_analysis = function(folderPath, beats, downbeats) {


    $._MYFUNCTIONS.addMarkersOnFirstAudioClipAtTimes(beats, 5 );
    $._MYFUNCTIONS.addMarkersOnFirstAudioClipAtTimes(downbeats, 6);

    var importedWavs = $._MYFUNCTIONS.importAllWavToDemixed(folderPath);

    if (importedWavs.length > 0) {
        $._MYFUNCTIONS.ensure_minimum_empty_tracks(importedWavs.length , false, true) 
        $._MYFUNCTIONS.insertWavsOnePerEmptyTrack(importedWavs);
    }

    // for (var i = 0; i < 100000; i++) {
    //     var start = segments[i].start;
    //     var end = segments[i].end;
    //     var label = segments[i].label;
    //     var colorIndex = 0;

    //     // Choix de la couleur en fonction du label
    //     if (label === "start") {
    //         colorIndex = 0;
    //     } else if (label === "intro") {
    //         colorIndex = 1;
    //     } else if (label === "solo") {
    //         colorIndex = 2;
    //     } else if (label === "inst") {
    //         colorIndex = 3;
    //     } else if (label === "break") {
    //         colorIndex = 4;
    //     } else if (label === "outro") {
    //         colorIndex = 5;
    //     }

    //     $._MYFUNCTIONS.addSequenceMarker(start, end, label, colorIndex);
    // }



}


// AutoEdit --- 1.b Transcription Analysis

$._MYFUNCTIONS.movePlayheadToSeconds = function(seconds) {
    if (typeof seconds !== "number") {
        throw new Error("Expected a number in seconds.");
    }

    var time = new Time();
    time.seconds = seconds;
    app.project.activeSequence.setPlayerPosition(time.ticks);
}

$._MYFUNCTIONS.razorAllTracksAtPlayhead = function() {
    var sqe = qe.project.getActiveSequence();
    if (!sqe) {
        alert("No active sequence found.");
        return;
    }

    var playerTime = sqe.CTI.timecode;
    if (!playerTime) {
        alert("Playhead position not available.");
        return;
    }

    // Razor all VIDEO tracks
    for (var i = 0; i < sqe.numVideoTracks; i++) {
        var vTrack = sqe.getVideoTrackAt(i);
        if (vTrack) {
            vTrack.razor(playerTime);
        }
    }

    // Razor all AUDIO tracks
    for (var j = 0; j < sqe.numAudioTracks; j++) {
        var aTrack = sqe.getAudioTrackAt(j);
        if (aTrack) {
            aTrack.razor(playerTime);
        }
    }

    $.writeln("Razor at: " + playerTime);
}

$._MYFUNCTIONS.deleteSelectedTrackItem = function() {
    var sequence = app.project.activeSequence;
    if (!sequence) {
        alert("No active sequence.");
        return;
    }



    // Parcours VIDEO tracks
    for (var v = 0; v < sequence.videoTracks.numTracks; v++) {
        var track = sequence.videoTracks[v];
        for (var i = 0; i < track.clips.numItems; i++) {
            var clip = track.clips[i];
            if (clip.isSelected()) {
                clip.remove(false, false); // false = pas de ripple
                $.writeln("Deleted selected VIDEO clip: " + clip.name);

            }
        }
    }

    // Parcours AUDIO tracks
    for (var a = 0; a < sequence.audioTracks.numTracks; a++) {
        var track = sequence.audioTracks[a];
        for (var i = 0; i < track.clips.numItems; i++) {
            var clip = track.clips[i];
            if (clip.isSelected()) {
                clip.remove(false, false);
                $.writeln("Deleted selected AUDIO clip: " + clip.name);

            }
        }
    }

    $.writeln("No selected clip found to delete.");
    return;
}

$._MYFUNCTIONS.selectAllTrackItemsAtPlayhead = function() {
    var sequence = app.project.activeSequence;
    if (!sequence) {
        alert("No active sequence.");
        return;
    }

    var playhead = sequence.getPlayerPosition(); // Time object
    var playheadTicks = Number(playhead.ticks);  // force number


        // VIDEO tracks
    for (var v = 0; v < sequence.videoTracks.numTracks; v++) {
        var vTrack = sequence.videoTracks[v];
        for (var i = 0; i < vTrack.clips.numItems; i++) {
            vTrack.clips[i].setSelected(false, false);
        }
    }

    // AUDIO tracks
    for (var a = 0; a < sequence.audioTracks.numTracks; a++) {
        var aTrack = sequence.audioTracks[a];
        for (var i = 0; i < aTrack.clips.numItems; i++) {
            aTrack.clips[i].setSelected(false, false);
        }
    }

    var selectedCount = 0;

    // VIDEO tracks
    for (var v = 0; v < sequence.videoTracks.numTracks; v++) {
        var track = sequence.videoTracks[v];
        for (var i = 0; i < track.clips.numItems; i++) {
            var clip = track.clips[i];
            var start = Number(clip.start.ticks);
            var end = Number(clip.end.ticks);

            if (start <= playheadTicks && playheadTicks < end) {
                clip.setSelected(true, true); // updateUI = true
                selectedCount++;
            }
        }
    }

    // AUDIO tracks
    for (var a = 0; a < sequence.audioTracks.numTracks; a++) {
        var track = sequence.audioTracks[a];
        for (var i = 0; i < track.clips.numItems; i++) {
            var clip = track.clips[i];
            var start = Number(clip.start.ticks);
            var end = Number(clip.end.ticks);

            if (start <= playheadTicks && playheadTicks < end) {
                clip.setSelected(true, true);
                selectedCount++;
            }
        }
    }

    $.writeln("✅ Selected " + selectedCount + " clip(s) under playhead.");
}

$._MYFUNCTIONS.closeAllGaps = function() {
    var sequence = app.project.activeSequence;
    if (!sequence) {
        alert("No active sequence.");
        return;
    }

    var TICKS_PER_SECOND = 254016000000;

    function closeGapsOnTrack(track) {
        var lastOut = null;

        for (var i = 0; i < track.clips.numItems; i++) {
            var clip = track.clips[i];
            var clipStart = Number(clip.start.ticks);

            // Cas spécial : gap entre 0 et le 1er clip
            if (i === 0 && clipStart > 0) {
                var gapSec = clipStart / TICKS_PER_SECOND;
                clip.move(-gapSec);
                $.writeln("⏪ Moved first clip (" + clip.name + ") to start (gap: " + gapSec.toFixed(3) + "s)");
            }
            // Cas classique : gap entre deux clips
            else if (lastOut !== null && clipStart > lastOut) {
                var gapTicks = clipStart - lastOut;
                var gapSec = gapTicks / TICKS_PER_SECOND;
                clip.move(-gapSec);
                $.writeln("💥 Moved clip (" + clip.name + ") left by " + gapSec.toFixed(3) + "s");
            }

            lastOut = Number(clip.end.ticks);
        }
    }

    // VIDEO TRACKS
    for (var v = 0; v < sequence.videoTracks.numTracks; v++) {
        closeGapsOnTrack(sequence.videoTracks[v]);
    }

    // AUDIO TRACKS
    for (var a = 0; a < sequence.audioTracks.numTracks; a++) {
        closeGapsOnTrack(sequence.audioTracks[a]);
    }

    $.writeln("✅ All gaps closed.");
}

$._MYFUNCTIONS.perform_cut_transcription = function(timeRanges) {
    app.enableQE();
    // var timeRanges = JSON.parse(timeRangesStr);
    // alert(JSON.stringify(timeRanges));
    for (var i = 0; i < timeRanges.length; i++) {
        var pair = timeRanges[i];

        var inTime = pair[0];
        var outTime = pair[1] ; 
        // alert(inTime + " " +  outTime);
        $._MYFUNCTIONS.movePlayheadToSeconds(inTime);
        $.writeln("Moved to in time: " + inTime);
        $._MYFUNCTIONS.razorAllTracksAtPlayhead();

        $._MYFUNCTIONS.movePlayheadToSeconds(outTime);
        $.writeln("Moved to out time: " + outTime);
        $._MYFUNCTIONS.razorAllTracksAtPlayhead();

        // Retour à inTime
        $._MYFUNCTIONS.movePlayheadToSeconds(inTime);

        // Sélectionne tous les clips présents à inTime
        $._MYFUNCTIONS.selectAllTrackItemsAtPlayhead();

        // Supprime les clips sélectionnés
        $._MYFUNCTIONS.deleteSelectedTrackItem();
    }

    $._MYFUNCTIONS.closeAllGaps();
}

$._MYFUNCTIONS.importSRTAsCaption = function(pathToSRT) {
    var sequence = app.project.activeSequence;
    if (!sequence) {
        alert("No active sequence.");
        return;
    }

    // Vérification de l'extension
    if (!pathToSRT || pathToSRT.slice(-4).toLowerCase() !== ".srt") {
        alert("Please provide a valid .srt file path.");
        return;
    }

    // Recherche ou création du bin "Copilot Transcription"
    var rootItem = app.project.rootItem;
    var targetBin = null;

    for (var i = 0; i < rootItem.children.numItems; i++) {
        var child = rootItem.children[i];
        if (child && child.name === "Copilot Transcription" && child.type === ProjectItemType.BIN) {
            targetBin = child;
            break;
        }
    }

    if (!targetBin) {
        targetBin = rootItem.createBin("Copilot Transcription");
    }

    // Importation dans le bin cible
    var result = app.project.importFiles([pathToSRT], false, targetBin, false);
    if (!result || result.length === 0) {
        alert("❌ Failed to import SRT file into project.");
        return;
    }

    var projectItem = result[0];
    alert("SRT successfully imported in the Copilot Transcription bin");
}


// AutoEdit --- 3. Montage

$._MYFUNCTIONS.resetInOutPointsForBin = function(binName) {
    var rootItem = app.project.rootItem;

    function findBinByName(parentItem, name) {
        for (var i = 0; i < parentItem.children.numItems; i++) {
            var currentItem = parentItem.children[i];
            if (currentItem.type === ProjectItemType.BIN) {
                if (currentItem.name === name) {
                    return currentItem;
                }
                var nested = findBinByName(currentItem, name);
                if (nested) return nested;
            }
        }
        return null;
    }

    function resetInOutRecursively(item) {
        for (var i = 0; i < item.children.numItems; i++) {
            var child = item.children[i];
            if (child.type === ProjectItemType.CLIP) {
                child.clearInPoint();
                child.clearOutPoint();
            } else if (child.type === ProjectItemType.BIN) {
                resetInOutRecursively(child);
            }
        }
    }

    var targetBin = findBinByName(rootItem, binName);
    if (targetBin) {
        resetInOutRecursively(targetBin);

    } 
}

$._MYFUNCTIONS.getFirstFreeTracks = function(startTimeSec, endTimeSec) {
    $._MYFUNCTIONS.ensure_minimum_empty_tracks(1 , false, true) 
    $._MYFUNCTIONS.ensure_minimum_empty_tracks(1 , true, false) 

    var seq = app.project.activeSequence;

    var startTicks = timeToTicks(startTimeSec);
    var endTicks = timeToTicks(endTimeSec);

    var freeVideoIndex = findFreeTrackIndex(seq.videoTracks, startTicks, endTicks);
    var freeAudioIndex = findFreeTrackIndex(seq.audioTracks, startTicks, endTicks);

    return {
        videoTrackIndex: freeVideoIndex,
        audioTrackIndex: freeAudioIndex
    };

    function findFreeTrackIndex(trackCollection, start, end) {
        for (var i = 0; i < trackCollection.numTracks; i++) {
            var track = trackCollection[i];
            var isFree = true;

            for (var j = 0; j < track.clips.numItems; j++) {
                var clip = track.clips[j];
                var clipStart = parseInt(clip.start.ticks);
                var clipEnd = parseInt(clip.end.ticks);

                var overlap = !(clipEnd <= start || clipStart >= end);
                if (overlap) {
                    isFree = false;
                    break;
                }
            }

            if (isFree) {
                return i;
            }
        }
        return -1; // Aucun track libre trouvé
    }

    function timeToTicks(seconds) {
        return Math.floor(seconds * 254016000000);
    }
}

$._MYFUNCTIONS.findProjectItemByPath = function (path) {
    var rootItem = app.project.rootItem;

    function searchBin(bin) {
        for (var i = 0; i < bin.children.numItems; i++) {
            var item = bin.children[i];
            if (item.type === ProjectItemType.BIN) {
                var found = searchBin(item);
                if (found) return found;
            } else {
                if (item.getMediaPath && item.getMediaPath() === path) {
                    return item;
                }
            }
        }
        return null;
    }

    return searchBin(rootItem);
}

$._MYFUNCTIONS.insertClipsFromData = function (clipData) {
    if (!app.project.activeSequence) {
        $.writeln("❌ Aucune séquence active.");
        return;
    }

    var sequence = app.project.activeSequence;

    for (var i = 0; i < clipData.length; i++) {
        var clip = clipData[i];
        var nextClip = clipData[i + 1];
        var duration = nextClip ? (nextClip.time - clip.time) : 2.0;

        var item = $._MYFUNCTIONS.findProjectItemByPath(clip.clip_file);
        if (!item) {
            $.writeln("⚠️ Clip non trouvé : " + clip.clip_file);
            continue;
        }

        var sourceIn = item.getInPoint().seconds;
        var sourceOut = item.getOutPoint().seconds;
        var sourceDuration = sourceOut - sourceIn;
        var sourceMiddle = sourceIn + (sourceDuration / 2);

        var newIn = sourceMiddle - (duration / 2);
        newIn = Math.max(sourceIn, newIn);

        // tentative de +0.5s
        var newOut = sourceMiddle + (duration / 2) + 0.5;
        newOut = Math.min(sourceOut, newOut); // ne pas dépasser la fin réelle du rush

        // appliquer points in/out
        item.setInPoint(newIn, 4);  // 4 = audio + vidéo
        item.setOutPoint(newOut, 4);

        // overwrite à la bonne position
        var result = $._MYFUNCTIONS.getFirstFreeTracks(clip.time, nextClip.time);
        sequence.overwriteClip(item, clip.time, 0, result.audioTrackIndex);
    }
}

$._MYFUNCTIONS.insertClipsFromData_SPEAKER = function (clipData) {
    if (!app.project.activeSequence) {
        $.writeln("❌ Aucune séquence active.");
        return;
    }

    var sequence = app.project.activeSequence;

    for (var i = 0; i < clipData.length; i++) {
        var clip = clipData[i];

        var item = $._MYFUNCTIONS.findProjectItemByPath(clip.clip_file);
        if (!item) {
            $.writeln("⚠️ Clip non trouvé : " + clip.clip_file);
            continue;
        }

        var sourceIn = item.getInPoint().seconds;
        var sourceOut = item.getOutPoint().seconds;
        var sourceDuration = sourceOut - sourceIn;
        var sourceMiddle = sourceIn + (sourceDuration / 2);

        var clipDuration = clip.timeOUT - clip.timeIN;

        var newIn = sourceMiddle - (clipDuration / 2);
        newIn = Math.max(sourceIn, newIn);

        var newOut = sourceMiddle + (clipDuration / 2);
        newOut = Math.min(sourceOut, newOut);

        // Ajustement si la durée est trop courte à cause des bornes
        if ((newOut - newIn) < clipDuration) {
            newIn = newOut - clipDuration;
            if (newIn < sourceIn) {
                newIn = sourceIn;
                newOut = sourceIn + clipDuration;
            }
        }

        item.setInPoint(newIn, 4);  // 4 = audio + vidéo
        item.setOutPoint(newOut, 4);
        var result = $._MYFUNCTIONS.getFirstFreeTracks(clip.timeIN, clip.timeOUT);
        sequence.overwriteClip(item, clip.timeIN, result.videoTrackIndex, result.audioTrackIndex);
    }
}


// AutoEdit --- A. Quick Edit 

$._MYFUNCTIONS.captureSelectedTrackItemMarkers = function () {
    var sequence = app.project.activeSequence;

    if (!sequence) {
        alert("No active sequence found.");
        return [];
    }

    var playheadPositions = [];

    // --- 1. Récupérer les TrackItems sélectionnés ---
    var selectedTrackItems = sequence.getSelection();

    for (var i = 0; i < selectedTrackItems.length; i++) {
        var clip = selectedTrackItems[i];

        if (clip.projectItem) {
            var clipMarkers = clip.projectItem.getMarkers();

            if (clipMarkers) {
                var clipMarker = clipMarkers.getFirstMarker();

                while (clipMarker) {

 

                    // Temps absolu du marqueur dans la séquence :
                    // $.writeln(JSON.stringify(clipMarker.start)); 
                    // $.writeln(JSON.stringify(clip.inPoint)); 
                    // $.writeln(JSON.stringify(clip.start)); 
                    var markerOffsetInSource = parseInt(clipMarker.start.ticks) - parseInt(clip.inPoint.ticks);
                    var absoluteTicks = parseInt(clip.start.ticks) + markerOffsetInSource;

                    var time = new Time();
                    time.ticks = absoluteTicks.toString(); // On remet en string pour éviter erreurs de typage

                    sequence.setPlayerPosition(time.ticks);
                    var currentTime = sequence.getPlayerPosition().seconds;

                    playheadPositions.push(currentTime);

                    clipMarker = clipMarkers.getNextMarker(clipMarker);
                }
            }
        }
    }

    // --- 2. Tri et dédoublonnage ---
    playheadPositions.sort(function(a, b) { return a - b; });

    var uniquePositions = [];
    var seenPositions = {};

    for (var k = 0; k < playheadPositions.length; k++) {
        var pos = playheadPositions[k];
        var key = pos.toFixed(4);
        if (!seenPositions[key]) {
            uniquePositions.push(pos);
            seenPositions[key] = true;
        }
    }

    return uniquePositions;
}

$._MYFUNCTIONS.fast_edit = function() {

    var allMarkerTimes = $._MYFUNCTIONS.captureSelectedTrackItemMarkers (); // Utilise ta fonction personnalisée

    if (!allMarkerTimes || allMarkerTimes.length === 0) {
        alert("❌ Liste de marqueurs vide.");
        return;
    }

    var sequence = app.project.activeSequence;
    if (!sequence) {
        alert("❌ Aucune séquence active.");
        return;
    }

    // 1. Récupérer tous les clips sélectionnés, triés dans l'ordre de leur position dans la timeline
    var selectedClips = [];

    for (var t = 0; t < sequence.videoTracks.numTracks; t++) {
        var track = sequence.videoTracks[t];
        for (var c = 0; c < track.clips.numItems; c++) {
            var clip = track.clips[c];
            if (clip.isSelected()) {
                selectedClips.push({
                    clip: clip,
                    videoTrackIndex: t,
                    start: clip.start.seconds
                });
            }
        }
    }

    selectedClips.sort(function (a, b) {
        return a.start - b.start;
    });

    if (selectedClips.length === 0) {
        alert("❌ Aucun clip sélectionné.");
        return;
    }

    var previousOutTime = allMarkerTimes[0]; // premier marker en base

    for (var i = 0; i < selectedClips.length; i++) {
        var clipData = selectedClips[i];
        var clip = clipData.clip;
        var trackIndex = clipData.videoTrackIndex;

        var projectItem = clip.projectItem;

        var originalIn = clip.inPoint.seconds;
        var originalOut = clip.outPoint.seconds;
        var originalDuration = originalOut - originalIn;

        var targetEnd = previousOutTime + originalDuration;

        // Trouver le marker le plus proche avant la fin cible
        var bestMarkerBeforeEnd = null;
        var bestMarkerAfterEnd = null; 
        for (var j = 0; j < allMarkerTimes.length; j++) {
            var markerTime = allMarkerTimes[j];
            if (markerTime < targetEnd) {
                bestMarkerBeforeEnd = markerTime;
            } else {
                k = Math.min(allMarkerTimes.length - 1 , j);
                bestMarkerAfterEnd = allMarkerTimes[k]; 
                break;
            }
        }

        if (!bestMarkerBeforeEnd) {
            bestMarkerBeforeEnd = targetEnd; // fallback
        }

        // ⚠️ CAS PARTICULIER : le marker trouvé est égal au précédent
        if (bestMarkerBeforeEnd === previousOutTime) {
            bestMarkerBeforeEnd = bestMarkerAfterEnd; 

        }
        else if(Math.abs(targetEnd - bestMarkerBeforeEnd) > Math.abs(targetEnd - bestMarkerAfterEnd) ) {
            bestMarkerBeforeEnd = bestMarkerAfterEnd; 

        }




        // $.writeln(bestMarkerBeforeEnd, " ", previousOutTime)
        var newDuration = bestMarkerBeforeEnd - previousOutTime;
        // var maxDuration = projectItem.getOutPoint().seconds - projectItem.getInPoint().seconds;
        // newDuration = Math.min(newDuration, maxDuration);

        var newIn = originalIn;
        var newOut = newIn + newDuration;

        projectItem.setInPoint(newIn, 4);
        projectItem.setOutPoint(newOut, 4);

        clip.remove(0, 1);
        // $.writeln(bestMarkerBeforeEnd, previousOutTime)


        // $.writeln(JSON.stringify(sequence.getSettings().videoFrameRate.seconds));
        // $.writeln(JSON.stringify(previousOutTime));

        var frameDurationSeconds = sequence.getSettings().videoFrameRate.seconds;
        function snapToFrame(timeInSeconds, frameDuration) {
            // $.writeln(JSON.stringify(timeInSeconds));
            // $.writeln(JSON.stringify(frameDuration));
            return Math.floor(timeInSeconds / frameDuration) * frameDuration;
        }
        previousOutTime = snapToFrame(previousOutTime, frameDurationSeconds);
        // $.writeln(JSON.stringify(previousOutTime));

        sequence.overwriteClip(projectItem, previousOutTime, trackIndex, 0);

        previousOutTime = previousOutTime + newDuration;
    }

    // $.writeln("✅ Clips alignés sur les marqueurs sélectionnés !");
}


// AutoFill      ----------------------------------------------------------------------------

$._MYFUNCTIONS.exportCurrentFrameToHighResPNG = function() {
    app.enableQE();

    var activeSequence = qe.project.getActiveSequence();
    if (!activeSequence) {
        $.writeln("❌ No active sequence.");
        return null;
    }

    var timecode = activeSequence.CTI.timecode; // ex: "00:00:01:05"
    var safeTime = timecode.replace(/:|;/ig, '_');

    var documentsFolder = Folder.myDocuments;
    var outputFolder = new Folder(documentsFolder.fullName + "/Adobe/Premiere Pro/Premiere Copilot/image_generation");

    if (!outputFolder.exists) {
        outputFolder.create();
        $.writeln("📁 Created folder: " + outputFolder.fsName);
    }

    var fileName = activeSequence.name + '__' + safeTime;

    
    var os = $.os.toLowerCase();

    if (os.indexOf("mac") !== -1) {
        var outputPath = outputFolder.fsName + '/' + fileName;
    } else if (os.indexOf("windows") !== -1) {
        var outputPath = outputFolder.fsName + "'\'" + fileName;
    }

    // Export frame at current CTI timecode
    activeSequence.exportFramePNG(timecode, outputPath);

    $.writeln("Exported PNG frame to: " + outputPath);
    return outputPath + '.png';
}

$._MYFUNCTIONS.importImageToImageGenerationBin = function(imagePath) {
    if (!imagePath || !(new File(imagePath)).exists) {
        // alert("Invalid image path.");
        return;
    }

    var project = app.project;
    var rootItem = project.rootItem;
    var targetBin = null;

    // Recherche d'un bin nommé "Image Generation"
    for (var i = 0; i < rootItem.children.numItems; i++) {
        var child = rootItem.children[i];
        if (child && child.type === ProjectItemType.BIN && child.name === "Image Generation") {
            targetBin = child;
            break;
        }
    }

    // Si le bin n'existe pas, on le crée
    if (!targetBin) {
        targetBin = rootItem.createBin("Image Generation");
    }

    // Importation de l'image dans le bin
    var importSucceeded = project.importFiles(
        [imagePath],   // Tableau de chemins
        false,         // suppressUI
        targetBin,     // bin cible
        false          // importAsNumberedStills
    );

}

$._MYFUNCTIONS.isTrackClearAtTime = function(track, startTimeInSeconds, endTimeInSeconds) {
    if (!track) {
        return false;
    }
    
    var clips = track.clips;
    if (clips.numItems === 0) {
        return true; 
    }

    for (var i = 0; i < clips.numItems; i++) {
        var clip = clips[i];
        var overlap = (clip.start.seconds < endTimeInSeconds) && (clip.end.seconds > startTimeInSeconds);

        if (overlap) {
            return false; 
        }
    }

    return true; 
}

// $._MYFUNCTIONS.importImageToImageGenerationBin_V2= function(imagePath) {
//     $._MYFUNCTIONS.ensure_minimum_empty_tracks(1 , false, true) 
//     $._MYFUNCTIONS.ensure_minimum_empty_tracks(1 , true, false) 

//     if (!imagePath || !(new File(imagePath)).exists) {
//         // alert("Le chemin de l'image est invalide ou le fichier n'existe pas.");
//         return null;
//     }

//     var project = app.project;

    
//     var rootItem = project.rootItem;
//     var targetBin = null;

//     // --- 1. RECHERCHE OU CRÉATION DU CHUTIER (BIN) ---
//     for (var i = 0; i < rootItem.children.numItems; i++) {
//         var item = rootItem.children[i];
//         if (item && item.type === ProjectItemType.BIN && item.name === "Generative AI") {
//             targetBin = item;
//             break;
//         }
//     }

//     if (!targetBin) {
//         targetBin = rootItem.createBin("Generative AI");
//     }

//     // --- 2. IMPORTATION DU FICHIER ---
//     var success = project.importFiles(
//         [imagePath],
//         false,
//         targetBin,
//         false
//     );


//     // --- 3. RÉCUPÉRATION DE L'ITEM IMPORTÉ ---
//     var importedItem = targetBin.children[targetBin.children.numItems - 1];


//     // --- 4. INSERTION SUR UNE PISTE VIDÉO VIDE ---
//     var sequence = project.activeSequence;


//     var playheadTime = sequence.getPlayerPosition();
//     var imageDuration = 5.0; // Durée par défaut de 5 secondes si non fournie.
//     var endTimeSeconds = playheadTime.seconds + imageDuration;
    
//     var targetTrack = null;

//     // On parcourt les pistes vidéo pour en trouver une de libre
//     for (var i = 0; i < sequence.videoTracks.numTracks; i++) {
//         var currentTrack = sequence.videoTracks[i];
//         if ($._MYFUNCTIONS.isTrackClearAtTime(currentTrack, playheadTime.seconds, endTimeSeconds)) {
//             targetTrack = currentTrack;
//             break; // On a trouvé une piste libre, on arrête la recherche.
//         }
//     }

//     if (targetTrack) {
//         // On insère le clip en mode "Overwrite" (Écrasement)
//         var newClip = targetTrack.overwriteClip(importedItem, playheadTime.seconds);
        
//         if (newClip) {
//             // On ajuste la durée du clip sur la timeline pour qu'elle corresponde à la durée souhaitée
//             var newEndTime = new Time();
//             newEndTime.seconds = playheadTime.seconds + imageDuration;
//             newClip.end = newEndTime;

//             // alert("Image insérée avec succès !");
//             return newClip;
//         }
//     } 
// }

$._MYFUNCTIONS.importAndPlaceMedia = function(filePath) {
    // S'assure qu'il y a au moins une piste vide de chaque type pour éviter les erreurs
    // Note: vous devez avoir une fonction ensure_minimum_empty_tracks définie ailleurs.
    $._MYFUNCTIONS.ensure_minimum_empty_tracks(1, false, true); // Pour l'audio
    $._MYFUNCTIONS.ensure_minimum_empty_tracks(1, true, false); // Pour la vidéo

    if (!filePath || !(new File(filePath)).exists) {
        return null;
    }

    var project = app.project;
    var sequence = project.activeSequence;


    // --- 1. GESTION DU CHUTIER ---
    var rootItem = project.rootItem;
    var targetBin = null;
    for (var i = 0; i < rootItem.children.numItems; i++) {
        var item = rootItem.children[i];
        if (item && item.type === ProjectItemType.BIN && item.name === "Generative AI") {
            targetBin = item;
            break;
        }
    }
    if (!targetBin) {
        targetBin = rootItem.createBin("Generative AI");
    }

    // --- 2. IMPORTATION ---
    project.importFiles([filePath], false, targetBin, false);
    var importedItem = targetBin.children[targetBin.children.numItems - 1];


    // --- 3. DÉTECTION DU TYPE PAR EXTENSION ---
    var mediaPath = importedItem.getMediaPath();
    var extension = mediaPath ? mediaPath.substr(mediaPath.lastIndexOf('.') + 1).toLowerCase() : '';

    var videoExtensions = ["mov", "mp4", "avi", "mpg", "mpeg", "mxf", "mkv", "webm"];
    var imageExtensions = ["png", "jpg", "jpeg", "tiff", "psd", "gif", "bmp", "tga"];
    var audioExtensions = ["wav", "mp3", "aif", "aiff", "m4a", "aac", "ogg"];
    
    var isVideo = videoExtensions.indexOf(extension) > -1;
    var isImage = imageExtensions.indexOf(extension) > -1;
    var isAudio = audioExtensions.indexOf(extension) > -1;

    // --- 4. LOGIQUE D'INSERTION SPÉCIFIQUE ---
    var playheadTime = sequence.getPlayerPosition();
    var result = { videoClip: null, audioClip: null };

    if (isVideo) {
        // --- CAS VIDÉO : DOIT TROUVER UNE PAIRE DE PISTES LIBRES (VIDÉO + AUDIO) ---
        var sourceIn = importedItem.getInPoint().seconds;
        var sourceOut = importedItem.getOutPoint().seconds;
        var clipDuration = sourceOut - sourceIn;
        var endTimeSeconds = playheadTime.seconds + clipDuration;

        var targetVideoTrack = null;
        var targetAudioTrack = null;

        // On cherche une paire de pistes libres
        for (var v = 0; v < sequence.videoTracks.numTracks; v++) {
            if ($._MYFUNCTIONS.isTrackClearAtTime(sequence.videoTracks[v], playheadTime.seconds, endTimeSeconds)) {
                // Piste vidéo libre trouvée, maintenant on cherche une piste audio libre
                for (var a = 0; a < sequence.audioTracks.numTracks; a++) {
                    if ($._MYFUNCTIONS.isTrackClearAtTime(sequence.audioTracks[a], playheadTime.seconds, endTimeSeconds)) {
                        targetVideoTrack = sequence.videoTracks[v];
                        targetAudioTrack = sequence.audioTracks[a];
                        break; // Sort de la boucle audio
                    }
                }
            }
            if (targetVideoTrack && targetAudioTrack) {
                break; // Sort de la boucle vidéo, on a trouvé notre paire !
            }
        }

        if (targetVideoTrack && targetAudioTrack) {
            // Premiere gère l'insertion de la bonne partie (vidéo/audio) sur la bonne piste
            result.videoClip = targetVideoTrack.overwriteClip(importedItem, playheadTime);
            result.audioClip = targetAudioTrack.overwriteClip(importedItem, playheadTime);
        }

    } else if (isImage) {
        // --- CAS IMAGE : CHERCHE SEULEMENT UNE PISTE VIDÉO LIBRE ---
        var imageDuration = 5.0; // Durée par défaut
        var endTimeSeconds = playheadTime.seconds + imageDuration;
        
        var targetTrack = null;
        for (var i = 0; i < sequence.videoTracks.numTracks; i++) {
            if ($._MYFUNCTIONS.isTrackClearAtTime(sequence.videoTracks[i], playheadTime.seconds, endTimeSeconds)) {
                targetTrack = sequence.videoTracks[i];
                break;
            }
        }
        if (targetTrack) {
            result.videoClip = targetTrack.overwriteClip(importedItem, playheadTime);
            if (result.videoClip) { // Ajuste la durée car c'est une image
                var newEndTime = new Time();
                newEndTime.seconds = playheadTime.seconds + imageDuration;
                result.videoClip.end = newEndTime;
            }
        }
        
    } else if (isAudio) {
        // --- CAS AUDIO : CHERCHE SEULEMENT UNE PISTE AUDIO LIBRE ---
        var sourceIn = importedItem.getInPoint().seconds;
        var sourceOut = importedItem.getOutPoint().seconds;
        var clipDuration = sourceOut - sourceIn;
        var endTimeSeconds = playheadTime.seconds + clipDuration;
        
        var targetTrack = null;
        for (var i = 0; i < sequence.audioTracks.numTracks; i++) {
            if ($._MYFUNCTIONS.isTrackClearAtTime(sequence.audioTracks[i], playheadTime.seconds, endTimeSeconds)) {
                targetTrack = sequence.audioTracks[i];
                break;
            }
        }
        if (targetTrack) {
            result.audioClip = targetTrack.overwriteClip(importedItem, playheadTime);
        }
    } 

    if (result.videoClip || result.audioClip) {
        return result;
    }
    return null;
}


// AudioSync      ----------------------------------------------------------------------------

$._MYFUNCTIONS.exportAudioFromSelectionDirect = function (presetPath) {
    app.enableQE();

    var esSeq = app.project.activeSequence;
    var qeSeq = qe.project.getActiveSequence();

    var audioTracks = esSeq.audioTracks;
    var numTracks = audioTracks.numTracks;

    // Création du dossier avec timestamp
    var now = new Date();
    var folderName = now.getFullYear() + "_" +
        ("0" + (now.getMonth() + 1)).slice(-2) + "_" +
        ("0" + now.getDate()).slice(-2) + "_" +
        ("0" + now.getHours()).slice(-2) + "_" +
        ("0" + now.getMinutes()).slice(-2) + "_" +
        ("0" + now.getSeconds()).slice(-2);

    var exportRoot = Folder(Folder.myDocuments.fullName + "/Adobe/Premiere Pro/Premiere Copilot/audio_sync/" + folderName);
    if (!exportRoot.exists) exportRoot.create();

    for (var t = 0; t < numTracks; t++) {
        var esTrack = audioTracks[t];
        var qeTrack = qeSeq.getAudioTrackAt(t);

        // Mute toutes les pistes sauf la courante
        for (var m = 0; m < numTracks; m++) {
            audioTracks[m].setMute(m === t ? 0 : 1);
        }

        for (var c = 0; c < esTrack.clips.numItems; c++) {
            var esClip = esTrack.clips[c];
            if (!esClip.isSelected()) continue;

            var qeClip = qeTrack.getItemAt(c);
            if (!qeClip) continue;

            // In/Out basés sur ExtendScript clip
            esSeq.setInPoint(esClip.start.seconds);
            esSeq.setOutPoint(esClip.end.seconds);

            var safeName = "track" + t + "_" + esClip.name.replace(/[^\w\-]/g, "_");
            var exportPath = exportRoot.fsName + "/" + safeName + ".wav";

            var success = esSeq.exportAsMediaDirect(
                exportPath,
                presetPath,
                1 // In to Out
            );
        }
    }

    // Réactiver toutes les pistes
    for (var r = 0; r < numTracks; r++) {
        audioTracks[r].setMute(0);
    }
    return exportRoot.fsName; // Retourne le chemin du dossier d'export
}


// AudioSearch     ----------------------------------------------------------------------------

$._MYFUNCTIONS.addMarkersFromAnnotations = function (annotations) {

    var markers = app.project.activeSequence.markers;

    for (var i = 0; i < annotations.length; i++) {
        var entry = annotations[i];
        var timeInSeconds = entry.time_start;
        var commentText = entry.comment;

        var marker = markers.createMarker(timeInSeconds);
        marker.name = "Annotation " + (i + 1);
        marker.comments = commentText;
        marker.setTypeAsComment();
    }

    alert("Markers added from Audio Research !");

}

$._MYFUNCTIONS.exportFullAudioFromSequenceForTranscription = function() {
    var userDocsPath = Folder.myDocuments.fsName;
    var exportFolder1 = userDocsPath + "/Adobe/Premiere Pro/Premiere Copilot/temp";
    var exportFolder = new Folder(exportFolder1);

    var targetSequence = app.project.activeSequence;

    var os = $.os.toLowerCase();
    var basePath = "";

    // Génère un timestamp unique
    function getTimestamp() {
        var now = new Date();
        var pad = function(num) { return (num < 10 ? "0" : "") + num; };
        return now.getFullYear().toString() +
            pad(now.getMonth() + 1) +
            pad(now.getDate()) + "_" +
            pad(now.getHours()) +
            pad(now.getMinutes()) +
            pad(now.getSeconds());
    }

    var timestamp = getTimestamp();
    if (os.indexOf("mac") !== -1) {
        var truePresetPath = "/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForTranscriptionV2.epr";
        var outputFilePath = exportFolder.fsName + "/" + targetSequence.name + "_" + timestamp + ".wav";
    } else if (os.indexOf("windows") !== -1) {
       var presetFile = new File("C:/Program Files/Common Files/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForTranscriptionV2.epr");
        truePresetPath = presetFile.fsName; // te donne automatiquement un chemin natif : C:\...
        var outputFilePath = File(exportFolder.absoluteURI + "/" + targetSequence.name + "_" + timestamp + ".wav").fsName;
    }

    // Export audio complet en une fois
    var success = targetSequence.exportAsMediaDirect(outputFilePath, truePresetPath, 1); // 1 = in to out

    if (!success) {
        alert("Erreur lors de l'export audio de la séquence : " + targetSequence.name);
        return null;
    }

    $.writeln("Export audio réussi vers : " + outputFilePath);
    return outputFilePath;
}

// AGENT FUNCTIONS     ----------------------------------------------------------------------------


$._MYFUNCTIONS.exportProjectStructureToJSON = function() {
    if (!app.project) {
        alert("Aucun projet n'est ouvert.");
        return;
    }

    var sequenceData = {};
    for (var i = 0; i < app.project.sequences.numSequences; i++) {
        var seq = app.project.sequences[i];
        if (seq.projectItem) {
            sequenceData[seq.projectItem.nodeId] = {
                id: seq.sequenceID,
                settings: seq.getSettings()
            };
        }
    }

    function convertVideoDisplayFormat(formatCode) {
        var formatMap = {
            100: '24fps',
            101: '25fps',
            102: '29.97fps',
            103: '29.97fps',
            104: '30fps',
            105: '50fps',
            106: '59.94fps',
            107: '59.94fps',
            108: '60fps',
            109: 'Frames',
            110: '23.976fps',
            111: '16mm Feet + Frames',
            112: '35mm Feet + Frames',
            113: '48fps'
        };
        return formatMap[formatCode] || 'Unknown format code: ' + formatCode;
    }

    function getItemType(projectItem) {
        if (sequenceData[projectItem.nodeId]) {
            return 'Sequence';
        }

        if ((projectItem.type === ProjectItemType.CLIP || projectItem.type === ProjectItemType.FILE) && projectItem.getMediaPath) {
            var mediaPath = projectItem.getMediaPath();
            if (mediaPath && mediaPath.lastIndexOf('.') > -1) {
                var extension = mediaPath.substr(mediaPath.lastIndexOf('.') + 1).toLowerCase();
                var audioExtensions = ['mp3', 'wav', 'aif', 'aiff', 'aac', 'm4a', 'ogg'];
                var videoExtensions = ['mp4', 'mov', 'avi', 'mpg', 'mpeg', 'mxf', 'mkv', 'flv', 'wmv', 'r3d', 'm2ts', 'mts'];
                var imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'tif', 'ico', 'webp'];
                var subtitleExtensions = ['srt', 'sub', 'vtt', 'txt'];
                
                if (audioExtensions.indexOf(extension) > -1) {
                    return 'Audio';
                }
                if (videoExtensions.indexOf(extension) > -1) {
                    return 'Video';
                }
                if (imageExtensions.indexOf(extension) > -1) {
                    return 'Image';
                }
                if (subtitleExtensions.indexOf(extension) > -1) {
                    return 'Subtitle';
                }
            }
        }

        switch (projectItem.type) {
            case ProjectItemType.BIN:
                return 'Bin';
            case ProjectItemType.CLIP:
                return 'Clip';
            case ProjectItemType.FILE:
                return 'File';
            case ProjectItemType.ROOT:
                return 'Root';
            default:
                return 'Unknown';
        }
    }

    function traverseProjectItem(item) {
        var itemInfo = {
            name: item.name,
            type: getItemType(item)
        };

        if (itemInfo.type === 'Sequence') {
            var seqData = sequenceData[item.nodeId];
            itemInfo.nodeId = seqData.id;
            var settings = seqData.settings;
            if (settings) {
                itemInfo.videoFrameHeight = settings.videoFrameHeight;
                itemInfo.videoFrameWidth = settings.videoFrameWidth;
                itemInfo.videoDisplayFormat = convertVideoDisplayFormat(settings.videoDisplayFormat);
                
            }
        } else if (item.type === ProjectItemType.CLIP || item.type === ProjectItemType.FILE) {
            if (item.getMediaPath) {
                itemInfo.nodeId = item.nodeId;
                itemInfo.mediaPath = item.getMediaPath() || "N/A";


                var metadata = item.getProjectMetadata();

                if (itemInfo.type === 'Video') {
                    var resolutionMatch = metadata.match(/<premierePrivateProjectMetaData:Column\.Intrinsic\.VideoInfo>(\d+)\s*x\s*(\d+)/i);
                    if (resolutionMatch && resolutionMatch.length > 2) {
                        itemInfo.width = parseInt(resolutionMatch[1], 10);
                        itemInfo.height = parseInt(resolutionMatch[2], 10);
                    }
                }

                if (itemInfo.type === 'Video' || itemInfo.type === 'Audio') {
                    var durationMatch = metadata.match(/<premierePrivateProjectMetaData:Column\.Intrinsic\.MediaDuration>(.*?)<\/premierePrivateProjectMetaData:Column\.Intrinsic\.MediaDuration>/i);
                    if (durationMatch && durationMatch.length > 1) {
                        itemInfo.duration = durationMatch[1];
                    }
                }


            }
        } else if (itemInfo.type === 'Bin') {
            itemInfo.nodeId = item.nodeId;
        }

        if (item.type === ProjectItemType.BIN && item.children) {
            itemInfo.children = [];
            for (var i = 0; i < item.children.numItems; i++) {
                itemInfo.children.push(traverseProjectItem(item.children[i]));
            }
        }
        
        return itemInfo;
    }

    var rootItem = app.project.rootItem;
    var projectStructure = {
        name: app.project.name,
        projectID: app.project.documentID,
        projectPath: app.project.path,
        type: 'Project',
        children: []
    };

    if (rootItem && rootItem.children) {
        for (var i = 0; i < rootItem.children.numItems; i++) {
            projectStructure.children.push(traverseProjectItem(rootItem.children[i]));
        }
    }

    try {
        var jsonString = JSON.stringify(projectStructure, null, 4);
        $.writeln(jsonString);
       
    } catch (e) {
        alert("Erreur lors de la création du JSON : " + e.toString());
    }
    return jsonString;
}



// Create Sequence
$._MYFUNCTIONS.updateSequencePreset = function(width, height, framerate) {
    // Utiliser le preset dans le dossier Documents
    var documentsFolder = Folder.myDocuments.fsName;
    var presetPath = documentsFolder + "/Adobe/Premiere Pro/Premiere Copilot/PRESET_EDIT.sqpreset";
    var presetFile = new File(presetPath);

    if (!presetFile.exists) {
        alert("Le fichier preset n'a pas été trouvé à l'emplacement: " + presetPath);
        return;
    }

    var framerateMap = {
        '24fps': '10584000000',
        '25fps': '10160640000',
        '30fps': '8467200000',
        '50fps': '5080320000',
        '60fps': '4233600000'
    };

    var frameRateValue = framerateMap[framerate];
    if (!frameRateValue) {
        alert("Invalid or unsupported frame rate: " + framerate);
        return;
    }

    try {
        presetFile.encoding = "UTF-8";
        presetFile.open("r");
        var content = presetFile.read();
        var originalContent = content;
        presetFile.close();

        var newFrameSize = "0,0," + width + "," + height;
        
        // Regex robustes qui ignorent les attributs et les sauts de ligne
        content = content.replace(/<VideoFrameSize[^>]*>[\s\S]*?<\/VideoFrameSize>/, '<VideoFrameSize>' + newFrameSize + '</VideoFrameSize>');
        content = content.replace(/<PreviewVideoFrameSize[^>]*>[\s\S]*?<\/PreviewVideoFrameSize>/, '<PreviewVideoFrameSize>' + newFrameSize + '</PreviewVideoFrameSize>');
        content = content.replace(/<VideoFrameRate[^>]*>[\s\S]*?<\/VideoFrameRate>/, '<VideoFrameRate>' + frameRateValue + '</VideoFrameRate>');


        // if (content === originalContent) {
            // This check is problematic. If the preset already has the correct values,
            // no change will occur, and this will incorrectly throw an error.
            // We can safely proceed even if the content is the same.
            // Therefore, this block is being removed.
        // }

        presetFile.open("w");
        presetFile.write(content);
        presetFile.close();



    } catch (e) {
        alert("An error occurred while modifying the file: " + e.toString());
    }
}

$._MYFUNCTIONS.createSequenceFromPreset = function(presetFileName, seqName) {
    // Utiliser le preset dans le dossier Documents
    var documentsFolder = Folder.myDocuments.fsName;
    var presetPath = documentsFolder + "/Adobe/Premiere Pro/Premiere Copilot/PRESET_EDIT.sqpreset";
    var presetFile = new File(presetPath);
    var proj = app.project;
    var newSeq = proj.newSequence(seqName, presetPath);
    return newSeq.sequenceID;
}

$._MYFUNCTIONS.createSequence = function(args) {
    if (!args || !args.name) {
        alert("The sequence name is required.");
        return null;
    }

    try {
        var width = args.videoFrameWidth || 1920;
        var height = args.videoFrameHeight || 1080;
        var framerate = args.videoDisplayFormat || "25fps";
        var name = args.name;
        var presetFileName = "js/libs/sequence/PRESET_EDIT.sqpreset";

        // 1. Mettre à jour le préréglage de séquence
        $._MYFUNCTIONS.updateSequencePreset(width, height, framerate);

        // 2. Créer la séquence à partir du préréglage mis à jour
        var newSequenceID = $._MYFUNCTIONS.createSequenceFromPreset(presetFileName, name);

        // 3. Déplacer la séquence au bon endroit si parent_path est spécifié
        if (args.parent_path && args.parent_path !== "/" && newSequenceID) {
            var parentBin = $._MYFUNCTIONS.findOrCreateBinByPath(args.parent_path);
            if (parentBin) {
                var sequence = $._MYFUNCTIONS.findItemBySequenceID(newSequenceID);
                if (sequence) {
                    sequence.moveBin(parentBin);
                }
            }
        }

        // 4. Retourner le nouvel ID de la séquence
        return newSequenceID;

    } catch (e) {
        alert("An error occurred while creating the sequence: " + e.toString());
        return null;
    }
}
// Create Bin
$._MYFUNCTIONS.createBin = function(args) {

    if (!args || !args.name) {
        alert("The folder name is required.");
        return null;
    }

    try {
        var binName = args.name;
        if (!binName || binName.length === 0) {
            alert("The folder name cannot be empty.");
            return null;
        }

        var parentBin = app.project.rootItem; // Par défaut, on crée à la racine

        if (args.parent_path) {
            var pathComponents = args.parent_path.split('/');
            
            var findBinByName = function(currentBin, name) {
                if (currentBin.children) {
                    for (var i = 0; i < currentBin.children.numItems; i++) {
                        var child = currentBin.children[i];
                        if (child.type === ProjectItemType.BIN && child.name === name) {
                            return child;
                        }
                    }
                }
                return null;
            };

            // On commence à l'index 1 pour sauter le nom du projet
            for (var i = 1; i < pathComponents.length; i++) {
                var folderName = pathComponents[i];
                if (folderName) {
                    var nextBin = findBinByName(parentBin, folderName);
                    if (nextBin) {
                        parentBin = nextBin;
                    } else {
                        // Si le dossier n'existe pas, on le crée
                        parentBin = parentBin.createBin(folderName);
                        if (!parentBin) {
                            alert("Unable to create the intermediate folder: " + folderName);
                            return null;
                        }
                    }
                }
            }
        }

        var newBin = parentBin.createBin(binName);

        if (newBin) {
            return newBin.nodeId;
        }
        return null;

    } catch (e) {
        alert("An error occurred while creating the folder: " + e.toString());
        return null;
    }
}

$._MYFUNCTIONS.findOrCreateBinByPath = function(path) {
    if (!path) return app.project.rootItem;

    var pathComponents = path.split('/');
    var parentBin = app.project.rootItem;

    var findBinByName = function(currentBin, name) {
        if (currentBin.children) {
            for (var i = 0; i < currentBin.children.numItems; i++) {
                var child = currentBin.children[i];
                if (child.type === ProjectItemType.BIN && child.name === name) {
                    return child;
                }
            }
        }
        return null;
    };

    // Start at index 1 to skip the project name
    for (var i = 1; i < pathComponents.length; i++) {
        var folderName = pathComponents[i];
        if (folderName) {
            var nextBin = findBinByName(parentBin, folderName);
            if (nextBin) {
                parentBin = nextBin;
            } else {
                parentBin = parentBin.createBin(folderName);
                if (!parentBin) {
                    alert("Unable to create the intermediate folder: " + folderName);
                    return null;
                }
            }
        }
    }
    return parentBin;
}

$._MYFUNCTIONS.findItemBySequenceID = function(sequenceID) {
    if (!sequenceID || !app.project) return null;
    
    // Parcourir toutes les séquences du projet
    for (var i = 0; i < app.project.sequences.length; i++) {
        var seq = app.project.sequences[i];
        if (seq.sequenceID === sequenceID) {
            // Retourner le ProjectItem correspondant à la séquence
            return seq.projectItem;
        }
    }
    
    return null;
}

$._MYFUNCTIONS.findItemByPath = function(path) {
    if (!path) return null;
    var pathComponents = path.split('/');
    var currentItem = app.project.rootItem;

    var findChildByName = function(parent, name) {
        if (parent.children) {
            for (var i = 0; i < parent.children.numItems; i++) {
                var child = parent.children[i];
                if (child.name === name) {
                    return child;
                }
            }
        }
        return null;
    };

    // Start at index 1 to skip project name
    for (var i = 1; i < pathComponents.length; i++) {
        var componentName = pathComponents[i];
        if (componentName) {
            currentItem = findChildByName(currentItem, componentName);
            if (!currentItem) {
                return null; // Path component not found
            }
        }
    }
    return currentItem;
}
// Main Move Item
$._MYFUNCTIONS.findItemById = function(itemId) {
        function findItemRecursively(bin, itemId) {
            for (var i = 0; i < bin.children.numItems; i++) {
                var child = bin.children[i];

                if (child.nodeId === itemId) {
                    return child;
                }

            // Check pour les séquences
                for (var j = 0; j < app.project.sequences.numSequences; j++) {
                    var seq = app.project.sequences[j];
                    if (seq.projectItem && seq.projectItem.nodeId === child.nodeId) {
                        if (seq.sequenceID === itemId) {
                        return child;
                        }
                    break;
                    }
                }
                
                if (child.type === ProjectItemType.BIN) {
                    var found = findItemRecursively(child, itemId);
                    if (found) {
                        return found;
                    }
                }
            }
        return null;
    }
    
    return findItemRecursively(app.project.rootItem, itemId);
}

$._MYFUNCTIONS.moveItem = function(args) {
    if (!args || !args.nodeId || !args.new_parent_path) {
        alert("The item ID (nodeId) and the new parent (new_parent_path) are required for moving.");
        return false;
    }

    try {
        var itemToMove = $._MYFUNCTIONS.findItemById(args.nodeId);

        if (!itemToMove) {
            alert("Unable to find the item with ID: " + args.nodeId);
            return false;
        }

        // Trouver ou créer le dossier de destination
        var destinationBin = $._MYFUNCTIONS.findOrCreateBinByPath(args.new_parent_path);

        if (!destinationBin) {
            alert("Unable to find or create the destination folder: " + args.new_parent_path);
            return false;
        }

        // Déplacer l'élément
        itemToMove.moveBin(destinationBin);
        return true;

    } catch (e) {
        alert("An error occurred while moving the item: " + e.toString());
        return false;
    }
}
// Rename Item
$._MYFUNCTIONS.renameItem = function(args) {

    if (!args || !args.nodeId || !args.new_name) {
        alert("The item ID (nodeId) and the new name (new_name) are required for renaming.");
        return false;
    }
    try {
        var itemToRename = $._MYFUNCTIONS.findItemById(args.nodeId);

        if (!itemToRename) {
            alert("Unable to find the item with ID: " + args.nodeId);
            return false;
        }

        var oldName = itemToRename.name;
        itemToRename.name = args.new_name;
        return true;

    } catch (e) {
        alert("An error occurred while renaming the item: " + e.toString());
        return false;
    }
}
// Modify Item 
$._MYFUNCTIONS.modifyItem = function(args) {
    if (!args || !args.nodeId) {
        alert("The item ID (nodeId) is required.");
        return false;
    }

    var success = true;

    // Si new_name est fourni, renommer
    if (args.new_name) {
        success = $._MYFUNCTIONS.renameItem({
            nodeId: args.nodeId,
            new_name: args.new_name
        }) && success;
    }

    // Si new_parent_path est fourni, déplacer
    if (args.new_parent_path) {
        success = $._MYFUNCTIONS.moveItem({
            nodeId: args.nodeId,
            new_parent_path: args.new_parent_path
        }) && success;
    }

    return success;
}
// Delete Bin
$._MYFUNCTIONS.deleteBin = function(args) {

    if (!args || !args.nodeId) {
        alert("The folder ID (nodeId) is required for deletion.");
        return false;
    }
    try {
        var binToDelete = $._MYFUNCTIONS.findItemById(args.nodeId);
        
        if (!binToDelete) {
            alert("Unable to find the bin with ID: " + args.nodeId);
            return false;
        }

        if (binToDelete.type !== ProjectItemType.BIN) {
            alert("The item with ID " + args.nodeId + " is not a bin.");
            return false;
        }

        // Fonction pour vérifier si un bin est vide ou contient uniquement des bins vides (récursif)
        function isBinEmpty(bin) {
            if (bin.children) {
                for (var i = 0; i < bin.children.numItems; i++) {
                    var child = bin.children[i];
                    if (child.type !== ProjectItemType.BIN) {
                        return false; // Contient autre chose qu'un dossier
                    }
                    if (!isBinEmpty(child)) {
                        return false; // Contient un sous-dossier non vide
                    }
                }
            }
            return true; // Est vide ou ne contient que des sous-dossiers vides
        }

        if (isBinEmpty(binToDelete)) {
            var binName = binToDelete.name;
            binToDelete.deleteBin(); // Premiere Pro gère la suppression récursive
        return true;
        } else {
            alert("The folder '" + binToDelete.name + "' cannot be deleted because it contains items (clips, sequences, etc.).");
            return false;
        }

    } catch (e) {
        alert("An error occurred while deleting the folder: " + e.toString());
        return false;
    }
}

$._MYFUNCTIONS.duplicateSequence = function(args) {
    if (!args || !args.nodeId || !args.new_name) {
        alert("The sequence ID (nodeId) and the new name (new_name) are required for duplication.");
        return false;
    }

    try {
        var sourceSequence = null;

        for (var i = 0; i < app.project.sequences.numSequences; i++) {
            var seq = app.project.sequences[i];
            if (seq.sequenceID === args.nodeId) {
                sourceSequence = seq;
                break;
            }
        }

        if (!sourceSequence) {
            alert("Unable to find the source sequence with ID: " + args.nodeId);
            return false;
        }

        var numSequencesBefore = app.project.sequences.numSequences;
        
        sourceSequence.clone();

        var numSequencesAfter = app.project.sequences.numSequences;

        if (numSequencesAfter > numSequencesBefore) {
            var newSequence = app.project.sequences[numSequencesAfter - 1];
            if (newSequence.projectItem) {
                newSequence.projectItem.name = args.new_name;
                return true;
            } else {
                alert("The cloned sequence was not found as a project item.");
                return false;
            }
        } else {
            alert("The sequence duplication failed.");
            return false;
        }

    } catch (e) {
        alert("An error occurred while duplicating the sequence: " + e.toString());
        return false;
    }
}

$._MYFUNCTIONS.updateSequenceSettings = function(args) {
    if (!args || !args.nodeId) {
        alert("The sequence ID (nodeId) is required for modification.");
        return false;
    }

    try {
        var sequenceToUpdate = null;
        
        // Chercher la séquence par sequenceID (nodeId peut être un sequenceID)
        for (var i = 0; i < app.project.sequences.numSequences; i++) {
            var seq = app.project.sequences[i];
            if (seq.sequenceID === args.nodeId) {
                sequenceToUpdate = seq;
                break;
            }
        }

        if (!sequenceToUpdate) {
            alert("Unable to find the sequence to modify with ID: " + args.nodeId);
            return false;
        }

        var settings = sequenceToUpdate.getSettings();
        var modified = false;

        if (args.videoFrameWidth) {
            settings.videoFrameWidth = args.videoFrameWidth;
            modified = true;
        }
        if (args.videoFrameHeight) {
            settings.videoFrameHeight = args.videoFrameHeight;
            modified = true;
        }
        if (args.videoDisplayFormat) {
            var formatMap = {
                '23.976fps': 110,
                '24fps': 100,
                '25fps': 101,
                '29.97fps': 102,
                '30fps': 104,
                '48fps': 113,
                '50fps': 105,
                '59.94fps': 106,
                '60fps': 108
            };
            var formatCode = formatMap[args.videoDisplayFormat];
            if (formatCode !== undefined) {
                settings.videoDisplayFormat = formatCode;
                modified = true;
            } else {
                alert("Warning: Unknown format: " + args.videoDisplayFormat);
            }
        }

        if (modified) {
        sequenceToUpdate.setSettings(settings);
        } else {
            alert("No modification to apply to the sequence.");
        }
        
        return true;

    } catch (e) {
        alert("An error occurred while modifying the sequence: " + e.toString());
        return false;
    }
}

$._MYFUNCTIONS.executeActionList = function(actionList) {

    app.enableQE();

    for (var i = 0; i < actionList.length; i++) {
        var currentAction = actionList[i];
        // alert("Execution of the action " + (i + 1) + "/" + actionList.length + ": " + currentAction.name);

        switch (currentAction.name) {
            case "create_bin":
                $._MYFUNCTIONS.createBin(currentAction.args);
                break;
            case "create_sequence":
                $._MYFUNCTIONS.createSequence(currentAction.args);
                break;
            case "update_sequence":
                $._MYFUNCTIONS.updateSequenceSettings(currentAction.args);
                break;
            case "modify_item":
                $._MYFUNCTIONS.modifyItem(currentAction.args);
                break;
            case "duplicate_sequence":
                $._MYFUNCTIONS.duplicateSequence(currentAction.args);
                break;
            case "delete_bin": // Action de suppression
                $._MYFUNCTIONS.deleteBin(currentAction.args);
                break;
            default:
                // alert("Action not recognized: " + currentAction.name);
                break;
        }
    }
    //  alert("Toutes les actions ont été exécutées.");
}

$._MYFUNCTIONS.AGENT_SPEECH_labelizeAudio = function(nodeId) {

    // créer un dossier propre pour la séquence
    var newBin = app.project.rootItem.createBin("TRASH");

    // récupérer l'item 
    var sourceItem = $._MYFUNCTIONS.findProjectItemByNodeId(nodeId, app.project.rootItem);

    // createNewSequenceFromClips est la méthode la plus simple et robuste pour cela
    var tempSequence = app.project.createNewSequenceFromClips("TEMP_EXPORT", [sourceItem], newBin);
    
    // exporter l'audio, avec le bon setting
    var audioPath = $._MYFUNCTIONS.exportFullAudioFromSequenceForTranscription();

    // supprimer le dossier
    newBin.deleteBin();

    // retourner le path de l'audio 
    return audioPath; 
}

// ============= TIMELINE ACTIONS =============

$._MYFUNCTIONS.executeTimelineActionList = function(actionList) {   
    if (!app.project || !app.project.activeSequence) {
        alert("No active sequence to execute the timeline actions.");
        return;
    }
    app.enableQE();

    for (var i = 0; i < actionList.length; i++) {
        var currentAction = actionList[i];
        // alert("Execution of the timeline action " + (i + 1) + "/" + actionList.length + ": " + currentAction.name);

        switch (currentAction.name) {
            case "insert_item":
                $._MYFUNCTIONS.insertItem(currentAction.args);
                break;
            case "move_item":
                $._MYFUNCTIONS.moveClip(currentAction.args);
                break;
            case "delete_item":
                $._MYFUNCTIONS.deleteTimelineItem(currentAction.args);
                break;
            case "add_marker":
                $._MYFUNCTIONS.addMarker(currentAction.args);
                break;
            default:
                // alert("Timeline action not recognized: " + currentAction.name);
                break;
        }
    }
    // alert("All timeline actions have been executed.");
}

$._MYFUNCTIONS.deleteTimelineItem = function(args) {
    if (!args || !args.ID) {
        alert("The clip ID is required for deletion.");
        return false;
    }

    var sequence = app.project.activeSequence;
    if (!sequence) {
        alert("No active sequence.");
        return false;
    }

    var clipID = args.ID;
    var ripple = args.ripple !== undefined ? args.ripple : false;
    var rippleParam = ripple ? 1 : 0; // 1 = ripple delete (décale les clips suivants), 0 = simple delete
    var deleteType = ripple ? "avec ripple" : "sans ripple";
    
    // Fonction helper pour supprimer un clip et ses liés
    function deleteClipAndLinked(clip, clipType) {
        var clipsToDelete = [clip];
        var clipNames = [clip.name];
        
        // Récupérer les clips liés (audio/vidéo associés)
        try {
            var linkedItems = clip.getLinkedItems();
            if (linkedItems && linkedItems.length > 0) {
                for (var i = 0; i < linkedItems.length; i++) {
                    clipsToDelete.push(linkedItems[i]);
                    clipNames.push(linkedItems[i].name);
                }
            }
        } catch (e) {
            // alert("Unable to retrieve the linked clips: " + e.toString());
        }
        
        // Supprimer tous les clips (principal + liés)
        var allDeleted = true;
        for (var i = 0; i < clipsToDelete.length; i++) {
            try {
                clipsToDelete[i].remove(rippleParam, 0);
            } catch (e) {
                // alert("Error while deleting a linked clip: " + e.toString());
                allDeleted = false;
            }
        }
        
        if (allDeleted) {
            if (clipsToDelete.length > 1) {
                $.writeln("Clip " + clipType + " '" + clipNames[0] + "' et " + (clipsToDelete.length - 1) + " clip(s) lié(s) supprimés " + deleteType);
            } else {
                $.writeln("Clip " + clipType + " '" + clipNames[0] + "' supprimé " + deleteType);
            }
        }
        
        return allDeleted;
    }

    // Itérer sur les pistes vidéo
    if (args.piste_type === "video") {
        var track = sequence.videoTracks[args.piste_number];
    } else {
        var track = sequence.audioTracks[args.piste_number];
    }


    for (var j = 0; j < track.clips.numItems; j++) {
        var clip = track.clips[j];
        if (clip && clip.nodeId === clipID) {
            return deleteClipAndLinked(clip, "vidéo");
        }
    }
    

    // alert("No clip found with ID: " + clipID);
    return false;
}

$._MYFUNCTIONS.addMarker = function(args) {
    if (!args || args.time === undefined || !args.comments) {
        alert("The time (time) and the comment (comments) are required to create a marker.");
        return false;
    }

    var sequence = app.project.activeSequence;
    if (!sequence) {
        alert("No active sequence.");
        return false;
    }

    try {
        // 1. Créer le marker au temps spécifié
        var marker = sequence.markers.createMarker(args.time);
        
        if (!marker) {
            alert("Unable to create the marker.");
            return false;
        }

        // 2. Définir le commentaire (obligatoire)
        marker.name = args.comments;
        marker.comments = args.comments;

        // 3. Définir le type (optionnel, par défaut "Comment")
        var markerType = args.type || "Comment";
        switch (markerType) {
            case "Chapter":
                marker.setTypeAsChapter();
                break;
            case "Segmentation":
                marker.setTypeAsSegmentation();
                break;
            case "WebLink":
                marker.setTypeAsWebLink();
                break;
            case "Comment":
            default:
                marker.setTypeAsComment();
                break;
        }

        // 4. Définir la couleur (optionnel, par défaut Green = 0)
        if (args.color !== undefined) {
            var colorIndex = args.color;
            // Vérifier que l'index est valide (0-7)
            if (colorIndex >= 0 && colorIndex <= 7) {
                marker.setColorByIndex(colorIndex, 0);
            } else {
                $.writeln("Warning: Index de couleur invalide (" + colorIndex + "), doit être entre 0 et 7. Utilisation de la couleur par défaut.");
            }
        }

        // 5. Définir le temps de fin (optionnel, pour les markers de durée)
        if (args.end !== undefined && args.end > args.time) {
            marker.end = marker.end.seconds = args.end;
        }

        var markerInfo = "Marker '" + args.comments + "' ajouté à " + args.time + "s";
        if (args.end !== undefined) {
            markerInfo += " (durée: " + (args.end - args.time) + "s)";
        }
        $.writeln(markerInfo);

        return true;

    } catch (e) {
        alert("An error occurred while creating the marker: " + e.toString());
        return false;
    }
}

// $._MYFUNCTIONS.insertItem = function(args) {
//     if (!args || !args.nodeId) {
//         alert("Le nodeId de l'item est requis pour l'insertion.");
//         return false;
//     }

//     try {
//         var activeSequence = app.project.activeSequence;
//         if (!activeSequence) {
//             alert("Aucune séquence active.");
//             return false;
//         }

//         // 1. Trouver le clip source dans le projet
//         var sourceClip = $._MYFUNCTIONS.findProjectItemByNodeId(args.nodeId, app.project.rootItem);
//         if (!sourceClip) {
//             alert("Impossible de trouver l'item avec nodeId : " + args.nodeId);
//             return false;
//         }

//         // 2. Déterminer la position de départ
//         var startTime = args.start !== undefined ? args.start : activeSequence.end.seconds;

//         // 3. Récupérer les points in/out du média source
//         var sourceIn = sourceClip.getInPoint().seconds;
//         var sourceOut = sourceClip.getOutPoint().seconds;
//         var sourceDuration = sourceOut - sourceIn;

//         // 4. Déterminer la durée cible
//         var targetDuration;
//         if (args.end !== undefined) {
//             // Si end est fourni, calculer la durée
//             targetDuration = args.end - startTime;
//         } else {
//             // Sinon, utiliser toute la durée du média source
//             targetDuration = sourceDuration;
//         }

//         // 5. Calculer les nouveaux points in/out pour respecter la durée cible
//         var newIn, newOut;
        
//         if (targetDuration >= sourceDuration) {
//             // Si la durée demandée >= durée source, utiliser tout le média
//             newIn = sourceIn;
//             newOut = sourceOut;
//         } else {
//             // Sinon, prendre une portion centrée du média
//             var sourceMiddle = sourceIn + (sourceDuration / 2);
//             newIn = sourceMiddle - (targetDuration / 2);
//             newOut = sourceMiddle + (targetDuration / 2);

//             // Ajuster si on dépasse les limites
//             if (newIn < sourceIn) {
//                 newIn = sourceIn;
//                 newOut = sourceIn + targetDuration;
//             }
//             if (newOut > sourceOut) {
//                 newOut = sourceOut;
//                 newIn = sourceOut - targetDuration;
//             }
//         }

//         // 6. Appliquer les points in/out
//         sourceClip.setInPoint(newIn, 4);  // 4 = audio + vidéo
//         sourceClip.setOutPoint(newOut, 4);

//         // 7. Déterminer la piste de destination
//         var videoTrackIndex = 0;
//         var audioTrackIndex = 0;

//         if (args.track) {
//             // Parser le format "video 1" ou "audio 1"
//             var trackParts = args.track.split(' ');
//             if (trackParts.length === 2) {
//                 var trackNumber = parseInt(trackParts[1]) - 1; // Convertir en index (base 0)
//                 if (trackParts[0].toLowerCase() === 'video') {
//                     videoTrackIndex = trackNumber;
//                 } else if (trackParts[0].toLowerCase() === 'audio') {
//                     audioTrackIndex = trackNumber;
//                 }
//             }
//         } else {
//             // Utiliser getFirstFreeTracks pour trouver une piste libre
//             var freeTracks = $._MYFUNCTIONS.getFirstFreeTracks(startTime, startTime + (newOut - newIn));
//             videoTrackIndex = freeTracks.videoTrackIndex !== -1 ? freeTracks.videoTrackIndex : 0;
//             audioTrackIndex = freeTracks.audioTrackIndex !== -1 ? freeTracks.audioTrackIndex : 0;
//         }

//         // 8. Insérer le clip dans la timeline
//         if (args.ripple) {
//             // Insert avec ripple
//             activeSequence.insertClip(sourceClip, startTime, videoTrackIndex, audioTrackIndex);
//             $.writeln("Clip '" + sourceClip.name + "' inséré avec ripple à " + startTime + "s");
//         } else {
//             // Overwrite sans ripple
//             activeSequence.overwriteClip(sourceClip, startTime, videoTrackIndex, audioTrackIndex);
//             $.writeln("Clip '" + sourceClip.name + "' inséré en overwrite à " + startTime + "s");
//         }

//         // 9. Restaurer les points in/out originaux
//         sourceClip.setInPoint(sourceIn, 4);
//         sourceClip.setOutPoint(sourceOut, 4);

//         return true;

//     } catch (e) {
//         alert("Erreur lors de l'insertion du clip : " + e.toString());
//         return false;
//     }
// }

$._MYFUNCTIONS.isTrackClearAtTime = function(track, startTimeSec, endTimeSec) {
    if (!track) { return false; }
    var clips = track.clips;
    if (clips.numItems === 0) { return true; } // La piste est vide, donc libre.

    for (var i = 0; i < clips.numItems; i++) {
        var clip = clips[i];
        // Calcule s'il y a un chevauchement (overlap).
        var overlap = (clip.start.seconds < endTimeSec) && (clip.end.seconds > startTimeSec);
        if (overlap) {
            return false; // Un clip est sur le chemin, la piste n'est pas libre.
        }
    }
    return true; // Aucun chevauchement trouvé.
}

$._MYFUNCTIONS.checkTracksAvailability = function(startTimeSec, endTimeSec, videoTrackIndex, audioTrackIndex) {
    var seq = app.project.activeSequence;
    if (!seq) {
        alert("Aucune séquence active.");
        // Retourne false pour les deux car aucune opération n'est possible.
        return { isVideoTrackFree: false, isAudioTrackFree: false };
    }

    var result = {
        isVideoTrackFree: false, // Par défaut, on considère qu'elle n'est pas libre
        isAudioTrackFree: false
    };

    // --- Vérification de la piste VIDÉO ---
    // On vérifie seulement si un index valide est fourni.
    if (videoTrackIndex !== null && videoTrackIndex >= 0) {
        if (videoTrackIndex < seq.videoTracks.numTracks) {
            var videoTrack = seq.videoTracks[videoTrackIndex];
            result.isVideoTrackFree = $._MYFUNCTIONS.isTrackClearAtTime(videoTrack, startTimeSec, endTimeSec);
        } else {
            // L'index est hors limites, la piste n'existe pas, donc elle ne peut pas être "libre".
            result.isVideoTrackFree = false; 
            $.writeln("Attention : L'index de piste vidéo " + videoTrackIndex + " est invalide.");
        }
    } else {
        // Si aucun index n'est fourni, on considère la vérification comme réussie (car on ne voulait rien vérifier).
        result.isVideoTrackFree = true;
    }

    // --- Vérification de la piste AUDIO ---
    // Même logique que pour la vidéo.
    if (audioTrackIndex !== null && audioTrackIndex >= 0) {
        if (audioTrackIndex < seq.audioTracks.numTracks) {
            var audioTrack = seq.audioTracks[audioTrackIndex];
            result.isAudioTrackFree = $._MYFUNCTIONS.isTrackClearAtTime(audioTrack, startTimeSec, endTimeSec);
        } else {
            result.isAudioTrackFree = false;
            $.writeln("Attention : L'index de piste audio " + audioTrackIndex + " est invalide.");
        }
    } else {
        result.isAudioTrackFree = true;
    }

    return result;
}

$._MYFUNCTIONS.insertItem = function(args) {
    if (!args || !args.nodeId) {
        alert("The item nodeId is required for insertion.");
        return false;
    }

    try {
        var activeSequence = app.project.activeSequence;
        if (!activeSequence) {
            alert("No active sequence.");
            return false;
        }

        // 1. Trouver le clip source dans le projet
        var sourceClip = $._MYFUNCTIONS.findProjectItemByNodeId(args.nodeId, app.project.rootItem);
        if (!sourceClip) {
            alert("Unable to find the item with nodeId: " + args.nodeId);
            return false;
        }

        // 2. Déterminer la position de départ
        var startTime = args.start !== undefined ? args.start : activeSequence.end.seconds;

        // 3. Récupérer les points in/out du média source
        var sourceIn = sourceClip.getInPoint().seconds;
        var sourceOut = sourceClip.getOutPoint().seconds;
        var sourceDuration = sourceOut - sourceIn;

        // 4. Déterminer la durée cible
        var targetDuration;
        if (args.end !== undefined) {
            // Si end est fourni, calculer la durée
            targetDuration = args.end - startTime;
        } else {
            // Sinon, utiliser toute la durée du média source
            targetDuration = sourceDuration;
        }

        // 5. Calculer les nouveaux points in/out pour respecter la durée cible
        var newIn, newOut;
        
        if (targetDuration >= sourceDuration) {
            // Si la durée demandée >= durée source, utiliser tout le média
            newIn = sourceIn;
            newOut = sourceOut;
        } else {
            // Sinon, prendre une portion centrée du média
            var sourceMiddle = sourceIn + (sourceDuration / 2);
            newIn = sourceMiddle - (targetDuration / 2);
            newOut = sourceMiddle + (targetDuration / 2);

            // 6. Appliquer les points in/out
            if (args.inPoint_source) {
                newIn = args.inPoint_source;
            }
            if (args.outPoint_source) {
                newOut = args.outPoint_source;
            }
            else if (args.inPoint_source) {
                newOut = args.inPoint_source + targetDuration;
            }

            // Ajuster si on dépasse les limites
            if (newIn < sourceIn) {
                newIn = sourceIn;
                newOut = sourceIn + targetDuration;
            }
            if (newOut > sourceOut) {
                newOut = sourceOut;
                newIn = sourceOut - targetDuration;
            }
        }


        sourceClip.setInPoint(newIn, 4);  // 4 = audio + vidéo
        sourceClip.setOutPoint(newOut, 4);

        // 7. Déterminer la piste de destination
        var videoTrackIndex = 0;
        var audioTrackIndex = 0;

        // regarder l'extension du path et voir si c'est video ou audio
        var mediaPath = sourceClip.getMediaPath();
        var extension = mediaPath ? mediaPath.substr(mediaPath.lastIndexOf('.') + 1).toLowerCase() : '';

        var videoExtensions = ["mov", "mp4", "avi", "mpg", "mpeg", "mxf", "mkv", "webm"];
        var imageExtensions = ["png", "jpg", "jpeg", "tiff", "psd", "gif", "bmp", "tga"];
        var audioExtensions = ["wav", "mp3", "aif", "aiff", "m4a", "aac", "ogg"];
        
        var isVideo = videoExtensions.indexOf(extension) > -1;
        var isImage = imageExtensions.indexOf(extension) > -1;
        var isAudio = audioExtensions.indexOf(extension) > -1;

        // si une track fourni, regarder si elle est libre 
        var videoIsFree = false;
        var audioIsFree = false;
        var videoTrackIndex = 0;
        var audioTrackIndex = 0;


        if (isVideo) {

            if (args.track_index === undefined || args.track_index === null || args.track_index === '') {
                args.track_index = $._MYFUNCTIONS.getFirstFreeTracks(startTime, startTime + (newOut - newIn)).videoTrackIndex;
                // alert("Track index not provided, using first free track: " + args.track_index);
            }

            videoIsFree = $._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), args.track_index, null).isVideoTrackFree;
            audioIsFree = $._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), null, args.track_index).isAudioTrackFree;
            if (videoIsFree && audioIsFree) {
                videoTrackIndex = args.track_index;
                audioTrackIndex = args.track_index;
            }
            else if (videoIsFree) {
                videoTrackIndex = args.track_index;
                for (i = 0; i < activeSequence.audioTracks.numTracks; i++) {
                    if ($._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), null, i).isAudioTrackFree) {
                        audioTrackIndex = i;
                        break;
                    }
                }
            }
            else if (audioIsFree) {
                audioTrackIndex = args.track_index;
                for (i = 0; i < activeSequence.videoTracks.numTracks; i++) {
                    if ($._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), i, null).isVideoTrackFree) {
                        videoTrackIndex = i;
                        break;
                    }
                }
            }
            else {
                for (i = 0; i < activeSequence.videoTracks.numTracks; i++) {
                    if ($._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), i, null).isVideoTrackFree) {
                        videoTrackIndex = i;
                        break;
                    }
                }
                for (i = 0; i < activeSequence.audioTracks.numTracks; i++) {
                    if ($._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), null, i).isAudioTrackFree) {
                        audioTrackIndex = i;
                        break;
                    }
                }
            }
        } else if (isAudio) {

            if (args.track_index === undefined || args.track_index === null || args.track_index === '') {
                args.track_index = $._MYFUNCTIONS.getFirstFreeTracks(startTime, startTime + (newOut - newIn)).audioTrackIndex;
                // alert("Track index not provided, using first free track: " + args.track_index);
            }



            audioIsFree = $._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), null, args.track_index).isAudioTrackFree;

            if (audioIsFree) {
                audioTrackIndex = args.track_index;
            }
            else {
                for (i = 0; i < activeSequence.audioTracks.numTracks; i++) {
                    if ($._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), null, i).isAudioTrackFree) {
                        audioTrackIndex = i;
                        break;
                    }
                }
            }
        } else if (isImage) {

            if (args.track_index === undefined || args.track_index === null || args.track_index === '') {
                args.track_index = $._MYFUNCTIONS.getFirstFreeTracks(startTime, startTime + (newOut - newIn)).videoTrackIndex;
                // alert("Track index not provided, using first free track: " + args.track_index);
            }

            videoIsFree = $._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), args.track_index, null).isVideoTrackFree;

            if (videoIsFree) {
                videoTrackIndex = args.track_index;
            }
            else {
                for (i = 0; i < activeSequence.videoTracks.numTracks; i++) {
                    if ($._MYFUNCTIONS.checkTracksAvailability(startTime, startTime + (newOut - newIn), i, null).isVideoTrackFree) {
                        videoTrackIndex = i;
                        break;
                    }
                }
            }
        }

        // 8. Insérer le clip dans la timeline
        if (args.ripple) {
            // Insert avec ripple
            activeSequence.insertClip(sourceClip, startTime, videoTrackIndex, audioTrackIndex);
            // alert("Clip '" + sourceClip.name + "' inserted with ripple at " + startTime + "s");
        } else {
            // Overwrite sans ripple
            activeSequence.overwriteClip(sourceClip, startTime, videoTrackIndex, audioTrackIndex);
            // alert("Clip '" + sourceClip.name + "' inserted in overwrite at " + startTime + "s");
        }

        // 9. Restaurer les points in/out originaux
        sourceClip.setInPoint(sourceIn, 4);
        sourceClip.setOutPoint(sourceOut, 4);

        return true;

    } catch (e) {
        alert("An error occurred while inserting the clip: " + e.toString());
        return false;
    }
}

$._MYFUNCTIONS.moveClip = function(args){

    // 1. Supprimer le clip
    $._MYFUNCTIONS.deleteTimelineItem(args);

    // 2. Insérer le clip
    $._MYFUNCTIONS.insertItem(args);


}

$._MYFUNCTIONS.getSequenceStructure = function() {
    var sequence = app.project.activeSequence;
    if (!sequence) {
        alert("No active sequence.");
        return null;
    }

    var sequenceData = [];

    // Itérer sur les pistes vidéo
    for (var i = 0; i < sequence.videoTracks.numTracks; i++) {
        var videoTrack = sequence.videoTracks[i];
        for (var j = 0; j < videoTrack.clips.numItems; j++) {
            var clip = videoTrack.clips[j];

            var effects = [];

            for (var m = 0; m < clip.components.numItems; m++) {
                var effect_data = {};
                var component = clip.components[m];
                effect_data.matchName = component.matchName;

                
                if (i == 0 && j == 0) {
                    var properties = [];
                    for ( var n = 0; n < component.properties.numItems; n++) {

                        properties.push(component.properties[n].displayName + " : " + component.properties[n].getValue());
                    }
                }
                effect_data.properties = properties;
                effects.push(effect_data);

            }

            
            // $.writeln(JSON.stringify(clip));
            if (clip) {
                var clipData = {
                    "ID": clip.nodeId,
                    "nodeId": clip.projectItem.nodeId,
                    "start": clip.start.seconds,
                    "end": clip.end.seconds,
                    "piste": "video " + (i),
                    "effects": effects,
                };
                sequenceData.push(clipData);
            }
        }
    }

    // Itérer sur les pistes audio
    for (var k = 0; k < sequence.audioTracks.numTracks; k++) {
        var audioTrack = sequence.audioTracks[k];
        for (var l = 0; l < audioTrack.clips.numItems; l++) {
            var clip = audioTrack.clips[l];



            var effects = [];

            for (var m = 0; m < clip.components.numItems; m++) {
                var effect_data = {};
                var component = clip.components[m];
                effect_data.matchName = component.matchName;

                
                if (i == 0 && j == 0) {
                    var properties = [];
                    for ( var n = 0; n < component.properties.numItems; n++) {

                        properties.push(component.properties[n].displayName + " : " + component.properties[n].getValue());
                    }
                }
                effect_data.properties = properties;
                effects.push(effect_data);

            }


            if (clip) {

                var clipData = {
                    "ID": clip.nodeId,
                    "nodeId": clip.projectItem.nodeId,
                    "start": clip.start.seconds,
                    "end": clip.end.seconds,
                    "piste": "audio " + (k),
                    "effects": effects,
                };
                sequenceData.push(clipData);
            }
        }
    }

    // Trier les données de la séquence par timecode de départ
    sequenceData.sort(function(a, b) {
        return a.start - b.start;
    });

    var jsonString = JSON.stringify(sequenceData, null, 2);
    
    // Affiche le JSON dans une boîte de dialogue pour le débogage ou la visualisation
    $.writeln(jsonString); 
    
    return jsonString;
}

$._MYFUNCTIONS.findProjectItemByNodeId = function(nodeId, rootItem) {
    // Si aucun rootItem n'est fourni, commencer à la racine du projet
    if (!rootItem) {
        rootItem = app.project.rootItem;
    }
    
    // Vérifier si l'élément actuel a le nodeId recherché
    if (rootItem.nodeId === nodeId) {
        return rootItem;
    }
    
    // Si l'élément actuel a des enfants, les parcourir récursivement
    if (rootItem.children && rootItem.children.numItems > 0) {
        for (var i = 0; i < rootItem.children.numItems; i++) {
            var child = rootItem.children[i];
            var result = $._MYFUNCTIONS.findProjectItemByNodeId(nodeId, child);
            if (result) {
                return result;
            }
        }
    }
    
    // Si aucun élément n'est trouvé, retourner null
    return null;
}

$._MYFUNCTIONS.addEffect = function(args) {
    app.enableQE();

    var activeSequence = app.project.activeSequence;
    var activeSequenceQE = qe.project.getActiveSequence();
    var effect = qe.project.getVideoEffectByName(args.effect_name, true);

    if (args.type_piste === "video") {
        var track = activeSequence.videoTracks[args.no_piste ];
        var trackQE = activeSequenceQE.getVideoTrackAt(args.no_piste );
    } else {
        var track = activeSequence.audioTracks[args.no_piste ];
        var trackQE = activeSequenceQE.getAudioTrackAt(args.no_piste );
    }

    
    for (var j = 0; j < track.clips.numItems; j++) {
        var clip = track.clips[j];
        var clipQE = trackQE.getItemAt(j);

        if (clip.nodeId === args.ID) {

            // parcourir ses composants, si l'effet n'existes pas, l'ajouter, sinon prendre le dernier qui match le nom
            var index_effect = 999;
            for (var k = 0; k < clip.components.length; k++) {
                if (clip.components[k].matchName === args.effect_name) {
                    index_effect = k;
                }
            }

            if (index_effect === 999) {
                clipQE.addVideoEffect(effect);
                index_effect = clip.components.length - 1;
            }



            var component = clip.components[index_effect];
            var properties = [];
            // si la propriété est 999, fin de la fonction
            if (args.property_number === 999) {
                return;
            } else {
                component.properties[args.property_number].setValue(args.property_value, true)

            }
        }
    }
}

$._MYFUNCTIONS.activateSequenceById = function(sequenceId) {


    app.project.openSequence(sequenceId);

    return "Sequence " + sequenceId + " activated with success";

}








// Jumpcut      ----------------------------------------------------------------------------

$._MYFUNCTIONS.exportAudioJumpCuts = function() {
    var userDocsPath = Folder.myDocuments.fsName;
    var exportFolder1 = userDocsPath + "/Adobe/Premiere Pro/Premiere Copilot/temp";
    var exportFolder = new Folder(exportFolder1);

    var targetSequence = app.project.activeSequence;

    var os = $.os.toLowerCase();
    var basePath = "";

    // Génère un timestamp unique
    function getTimestamp() {
        var now = new Date();
        var pad = function(num) { return (num < 10 ? "0" : "") + num; };
        return now.getFullYear().toString() +
            pad(now.getMonth() + 1) +
            pad(now.getDate()) + "_" +
            pad(now.getHours()) +
            pad(now.getMinutes()) +
            pad(now.getSeconds());
    }

    var timestamp = getTimestamp();
    if (os.indexOf("mac") !== -1) {
        var truePresetPath = "/Library/Application Support/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForJumpCuts.epr";
        var outputFilePath = exportFolder.fsName + "/" + targetSequence.name + "_" + timestamp + ".wav";
    } else if (os.indexOf("windows") !== -1) {
       var presetFile = new File("C:/Program Files/Common Files/Adobe/CEP/extensions/PremiereGPTBeta/js/libs/render/Audio_ForJumpCuts.epr");
        truePresetPath = presetFile.fsName; // te donne automatiquement un chemin natif : C:\...
        var outputFilePath = File(exportFolder.absoluteURI + "/" + targetSequence.name + "_" + timestamp + ".wav").fsName;
    }

    // Export audio complet en une fois
    var success = targetSequence.exportAsMediaDirect(outputFilePath, truePresetPath, 1); // 1 = in to out

    if (!success) {
        alert("Erreur lors de l'export audio de la séquence : " + targetSequence.name);
        return null;
    }

    $.writeln("Export audio réussi vers : " + outputFilePath);
    return outputFilePath;
}

$._MYFUNCTIONS.getOffsetJumpCuts = function(){
    try {
        var sequence = app.project.activeSequence;

        if (sequence) {
            var playheadPosition = sequence.getPlayerPosition(); 
            return playheadPosition.seconds; 
        } else {
            alert("No active sequence.");
            return null;
        }
    } catch (error) {
        alert("An error occurred : " + error.toString());
        return null;
    }
}

$._MYFUNCTIONS.disableSegmentsInRange = function(segments) {
    segments = JSON.parse(segments);
    app.enableQE();
    var seq = qe.project.getActiveSequence();
    var sequence = app.project.activeSequence;
    var ticksPerSecond = 254016000000;

    // alert(JSON.stringify(segments));
    var clipsToProcess = [];

    for(var i = 0; i < segments.length; i++){
        var timeInSeconds = segments[i][0];

        var timeInTicks = Math.floor(timeInSeconds * ticksPerSecond);

        var time = new Time();
        time.ticks = timeInTicks.toString();

        sequence.setPlayerPosition(time.ticks);

        var currentTicks = parseInt(time.ticks);

        for (var j = 0; j < seq.numVideoTracks; j++) {
            var track = seq.getVideoTrackAt(j);
            if (track) {
                track.razor(seq.CTI.timecode);
            }
        }
        for(var k = 0; k < seq.numAudioTracks; k++){
            var track = seq.getAudioTrackAt(k);
            if (track) {
                track.razor(seq.CTI.timecode);
            }
        }

        for (var k = 0; k < sequence.videoTracks.numTracks; k++) {
            var track = sequence.videoTracks[k];
            if (track && track.clips.numItems > 0) {
                for (var c = track.clips.numItems-1; c >= 0; c--) {
                    var clip = track.clips[c];
                    var clipStart = parseInt(clip.start.ticks);
                    var clipEnd = clipStart + parseInt(clip.duration.ticks);

                    if (clipStart <= currentTicks && currentTicks < clipEnd) {
                        clipsToProcess.push(clip);
                        break;
                    }
                }
            }
        }

        for (var k = 0; k < sequence.audioTracks.numTracks; k++) {
            var track = sequence.audioTracks[k];
            if (track && track.clips.numItems > 0) {
                for (var c = track.clips.numItems-1; c >= 0; c--) {
                    var clip = track.clips[c];
                    var clipStart = parseInt(clip.start.ticks);
                    var clipEnd = clipStart + parseInt(clip.duration.ticks);

                    if (clipStart <= currentTicks && currentTicks < clipEnd) {
                        clipsToProcess.push(clip);
                        break;
                    }
                }
            }
        }

        // END

        var timeInSeconds = segments[i][1];

        var timeInTicks = Math.floor(timeInSeconds * ticksPerSecond);

        var time = new Time();
        time.ticks = timeInTicks.toString();

        sequence.setPlayerPosition(time.ticks);

        var currentTicks = parseInt(time.ticks);

        for (var j = 0; j < seq.numVideoTracks; j++) {
            var track = seq.getVideoTrackAt(j);
            if (track) {
                track.razor(seq.CTI.timecode);
            }
        }
        for(var k = 0; k < seq.numAudioTracks; k++){
            var track = seq.getAudioTrackAt(k);
            if (track) {
                track.razor(seq.CTI.timecode);
            }
        }

    }



    for(var k = 0; k < clipsToProcess.length; k++){
        var clip = clipsToProcess[k];
        clip.disabled = 1;
    }



}

$._MYFUNCTIONS.supprimerClipsDesactives = function() {
    var sequence = app.project.activeSequence;


    var clipsASupprimer = []; // Tableau pour stocker les clips à supprimer

    // --- ÉTAPE 1 : Collecter tous les clips désactivés ---
    // On parcourt les pistes vidéo
    for (var i = 0; i < sequence.videoTracks.numTracks; i++) {
        var pisteVideo = sequence.videoTracks[i];
        for (var j = 0; j < pisteVideo.clips.numItems; j++) {
            var clip = pisteVideo.clips[j];
            if (clip.disabled) {
                clipsASupprimer.push(clip);
            }
        }
    }

    // On parcourt les pistes audio
    for (var i = 0; i < sequence.audioTracks.numTracks; i++) {
        var pisteAudio = sequence.audioTracks[i];
        for (var j = 0; j < pisteAudio.clips.numItems; j++) {
            var clip = pisteAudio.clips[j];
            if (clip.disabled) {
                clipsASupprimer.push(clip);
            }
        }
    }

    if (clipsASupprimer.length === 0) {
        alert("No disabled clips found.");
        return;
    }

    // --- ÉTAPE 2 : Trier les clips de la fin vers le début ---
    // On trie en fonction du temps de début (start time) de chaque clip, par ordre décroissant.
    clipsASupprimer.sort(function(a, b) {
        return b.start.seconds - a.start.seconds;
    });

    // --- ÉTAPE 3 : Supprimer les clips en une seule passe ---
    for (var i = 0; i < clipsASupprimer.length; i++) {
        // Le premier 'true' active la suppression avec raccord (ripple delete)
        clipsASupprimer[i].remove(true, false);
    }
    
    // Message de confirmation
    alert("Operation finished ✨\n" + clipsASupprimer.length + " clips have been deleted. 🗑️");
}





$._MYFUNCTIONS.deleteFile = function(filePath) {
    var fileToDelete = new File(filePath);
    if (fileToDelete.exists) {
        fileToDelete.remove();
        return "File " + filePath + " deleted.";
    }
    return "File " + filePath + " not found.";
}

// --- Gestion des points de sauvegarde (Rewind Points) ---

// Stocke les index de l'historique d'annulation.
$._MYFUNCTIONS.rewindPointsStack = [];


$._MYFUNCTIONS.createRewindPoint = function() {
    try {
        // Pour la compatibilité avec les anciens moteurs de script
        app.enableQE();
    } catch (e) {}
    
    if (typeof qe === 'undefined' || !app.project) {
        alert("Le projet ou le QE DOM n'est pas disponible pour créer un point de sauvegarde.");
        return -1;
    }

    var rewindIndex = qe.project.undoStackIndex();
    $._MYFUNCTIONS.rewindPointsStack.push(rewindIndex);
    
    var newPointIndex = $._MYFUNCTIONS.rewindPointsStack.length - 1;
    $.writeln("Point de sauvegarde créé avec l'index : " + newPointIndex);
    
    return newPointIndex;
}

$._MYFUNCTIONS.rewindToPoint = function(pointIndex) {
    try {
        app.enableQE();
    } catch (e) {}

    if (typeof qe === 'undefined' || !app.project) {
        alert("Project or QE DOM not available to restore a rewind point.");
        return;
    }

    // Conversion en nombre au cas où il arriverait en chaîne de caractères
    pointIndex = parseInt(pointIndex, 10);

    if (isNaN(pointIndex) || pointIndex < 0 || pointIndex >= $._MYFUNCTIONS.rewindPointsStack.length) {
        alert("Invalid rewind point index: " + pointIndex);
        return;
    }

    var rewindIndex = $._MYFUNCTIONS.rewindPointsStack[pointIndex];
    $.writeln("Retour au point de sauvegarde #" + pointIndex);

    while (qe.project.undoStackIndex() > rewindIndex) {
        qe.project.undo();
    }

    if (qe.project.undoStackIndex() === rewindIndex) {
        // alert("Successfully returned to rewind point #" + pointIndex + ".");
        // Une fois revenu en arrière, tous les points de sauvegarde créés APRÈS ce point deviennent invalides.
        // On nettoie la pile pour ne garder que les points jusqu'à celui restauré.
        $._MYFUNCTIONS.rewindPointsStack.length = pointIndex + 1;
    } else {
        alert("An error occurred while rewinding to point #" + pointIndex + ".");
    }
}




