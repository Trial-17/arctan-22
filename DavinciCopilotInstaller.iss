[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  OldDir1: String;
begin
  if CurStep = ssInstall then
  begin
    OldDir1 := ExpandConstant('{commonappdata}\Blackmagic Design\DaVinci Resolve\Support\Workflow Integration Plugins\DavinciGPT');

    if DirExists(OldDir1) then
      DelTree(OldDir1, True, True, True);
  end;
end;


[Setup]
AppName=DavinciGPT
AppVersion=1.0
DefaultDirName={commonappdata}\Blackmagic Design\DaVinci Resolve\Support\Workflow Integration Plugins\DavinciGPT
DisableProgramGroupPage=yes
Uninstallable=no
OutputDir=Output
OutputBaseFilename=DavinciCopilot-Setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64


[Files]
Source: "build\pkg_payload\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

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
