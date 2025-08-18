[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  OldDir1, OldDir2, OldDir3: String;
begin
  if CurStep = ssInstall then
  begin
    OldDir1 := ExpandConstant('{pf32}\Common Files\Adobe\CEP\extensions\PremiereGPTBeta');
    OldDir2 := ExpandConstant('{pf32}\Common Files\Adobe\CEP\extensions\PremiereCopilot');
    OldDir3 := ExpandConstant('{pf32}\Common Files\Adobe\CEP\extensions\Premierecopilot');

    if DirExists(OldDir1) then
      DelTree(OldDir1, True, True, True);
    if DirExists(OldDir2) then
      DelTree(OldDir2, True, True, True);
    if DirExists(OldDir3) then
      DelTree(OldDir3, True, True, True);
  end;
end;


[Setup]
AppName=PremiereGPTBeta
AppVersion=1.0
DefaultDirName={pf32}\Common Files\Adobe\CEP\extensions\PremiereGPTBeta
DisableProgramGroupPage=yes
Uninstallable=no
OutputDir=.
OutputBaseFilename=PremiereCopilotSetup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Files]
; Extension CEP
Source: "PremiereGPTBeta\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

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
