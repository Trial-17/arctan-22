[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  OldDir1, OldDir2, OldDir3, OldDir4: String;
begin
  if CurStep = ssInstall then
  begin
    OldDir1 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereGPTBeta');
    OldDir2 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereCopilot');
    OldDir3 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereGPTaescripts');
    OldDir4 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereGPTaescripts_genai');

    if DirExists(OldDir1) then
      DelTree(OldDir1, True, True, True);
    if DirExists(OldDir2) then
      DelTree(OldDir2, True, True, True);
    if DirExists(OldDir3) then
      DelTree(OldDir3, True, True, True);
    if DirExists(OldDir4) then
      DelTree(OldDir4, True, True, True);
  end;
end;


[Setup]
AppName=PremiereGPTaescripts_genai
AppVersion=1.0
DefaultDirName={commoncf}\Adobe\CEP\extensions\PremiereGPTaescripts_genai
DisableProgramGroupPage=yes
Uninstallable=yes
OutputDir=Output
OutputBaseFilename=PremiereCopilot-aes-GenAI-Installer
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
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\image_generation"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\matplotlib_cache"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\rush_db"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\script"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\seq_preset"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\temp"
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\thumbnails"
