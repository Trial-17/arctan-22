[Code]
// Fonction pour arrêter les processus sur le port 8000
procedure KillProcessOnPort8000();
var
  ResultCode: Integer;
begin
  // Commande CMD brutale : trouve tous les PIDs sur le port 8000 et les tue
  // /F = Force, /FI = Filter pour cibler le port 8000
  Exec('cmd.exe', 
    '/C "for /f "tokens=5" %a in (''netstat -ano ^| findstr :8000'') do taskkill /F /PID %a 2>nul"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  OldDir1, OldDir2, OldDir3: String;
begin
  if CurStep = ssInstall then
  begin
    // Arrêt forcé des processus sur le port 8000
    KillProcessOnPort8000();
    
    OldDir1 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereGPTBeta');
    OldDir2 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereCopilot');
    OldDir3 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereGPTaescripts');

    if DirExists(OldDir1) then
      DelTree(OldDir1, True, True, True);
    if DirExists(OldDir2) then
      DelTree(OldDir2, True, True, True);
    if DirExists(OldDir3) then
      DelTree(OldDir3, True, True, True);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then
  begin
    KillProcessOnPort8000();
  end;
end;


[Setup]
AppName=PremiereGPTaescripts
AppVersion=1.0
DefaultDirName={commoncf}\Adobe\CEP\extensions\PremiereGPTaescripts
DisableProgramGroupPage=yes
Uninstallable=yes
OutputDir=Output
OutputBaseFilename=PremiereCopilot-Setup-aescripts
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64


[Registry]
Root: HKCU; Subkey: "Software\Adobe\CSXS.12"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: uninsdeletevalue
Root: HKCU; Subkey: "Software\Adobe\CSXS.11"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"; Flags: uninsdeletevalue

[Files]
; Extension CEP
Source: "build\pkg_payload\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Dirs]
; Création des dossiers dans les Documents de l'utilisateur
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\audio_sync"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\image_generation"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\matplotlib_cache"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\music_analysis"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\rush_db"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\script"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\seq_preset"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\sfx"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\temp"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\thumbnails"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\transcription_analysis"
