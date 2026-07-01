[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  PluginsDir: String;
begin
  if CurStep = ssInstall then
  begin
    PluginsDir := ExpandConstant('{commonappdata}\Blackmagic Design\DaVinci Resolve\Support\Workflow Integration Plugins');

    // Supprime la nouvelle version (réinstallation propre)
    if DirExists(PluginsDir + '\davinciClaude') then
      DelTree(PluginsDir + '\davinciClaude', True, True, True);

    // Supprime l'ancien plugin DavinciGPT (migration vers davinciClaude)
    if DirExists(PluginsDir + '\DavinciGPT') then
      DelTree(PluginsDir + '\DavinciGPT', True, True, True);
  end;
end;


[Setup]
AppName=davinciClaude
AppVersion=1.0
DefaultDirName={commonappdata}\Blackmagic Design\DaVinci Resolve\Support\Workflow Integration Plugins\davinciClaude
DisableProgramGroupPage=yes
Uninstallable=no
OutputDir=Output
OutputBaseFilename=davinciClaude-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64


[Files]
Source: "build\pkg_payload\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

; NB : dossiers de données volontairement conservés en "DaVinciGPT" — le bundle
; applicatif servi par l'API écrit en dur dans ~/Documents/DaVinciGPT (getDocumentsBase).
; À renommer en davinciClaude uniquement le jour où le bundle DaVinci dédié change ce chemin.
[Dirs]
Name: "{userdocs}\DaVinciGPT"
Name: "{userdocs}\DaVinciGPT\audio_sync"
Name: "{userdocs}\DaVinciGPT\image_generation"
Name: "{userdocs}\DaVinciGPT\matplotlib_cache"
Name: "{userdocs}\DaVinciGPT\music_analysis"
Name: "{userdocs}\DaVinciGPT\rush_db"
Name: "{userdocs}\DaVinciGPT\script"
Name: "{userdocs}\DaVinciGPT\seq_preset"
Name: "{userdocs}\DaVinciGPT\sfx"
Name: "{userdocs}\DaVinciGPT\temp"
Name: "{userdocs}\DaVinciGPT\thumbnails"
Name: "{userdocs}\DaVinciGPT\transcription_analysis"
