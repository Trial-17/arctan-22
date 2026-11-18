; Installeur Windows DavinciClaude (Inno Setup 6).
; La version est passée par le workflow : ISCC /DAppVer=2.0.N
; Le dossier du plugin reste "davinciClaude" : c'est le nom que Resolve charge.

#ifndef AppVer
  #define AppVer "2.0.0"
#endif
#define AppName "DavinciClaude"
#define SiteUrl "https://www.davinciclaude.com"
#define DocsUrl "https://docs.premierecopilot.com/getting-started/introduction"
#define Assets "installer_assets\davinci"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  PluginsDir: String;
begin
  if CurStep = ssInstall then
  begin
    PluginsDir := ExpandConstant('{commonappdata}\Blackmagic Design\DaVinci Resolve\Support\Workflow Integration Plugins');

    // Supprime la version précédente (réinstallation propre)
    if DirExists(PluginsDir + '\davinciClaude') then
      DelTree(PluginsDir + '\davinciClaude', True, True, True);

    // Supprime l'ancien plugin DavinciGPT (migration vers davinciClaude)
    if DirExists(PluginsDir + '\DavinciGPT') then
      DelTree(PluginsDir + '\DavinciGPT', True, True, True);
  end;
end;


[Setup]
AppId={{46A2D2FF-4D9D-4A44-80CE-357815CB3B57}
AppName={#AppName}
AppVersion={#AppVer}
AppVerName={#AppName} {#AppVer}
AppPublisher={#AppName}
AppPublisherURL={#SiteUrl}
AppSupportURL={#SiteUrl}
AppUpdatesURL={#SiteUrl}/download
VersionInfoVersion={#AppVer}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} Setup
DefaultDirName={commonappdata}\Blackmagic Design\DaVinci Resolve\Support\Workflow Integration Plugins\davinciClaude
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
DisableWelcomePage=no
ShowLanguageDialog=no
LanguageDetectionMethod=uilanguage
Uninstallable=yes
UninstallFilesDir={autopf}\{#AppName}
UninstallDisplayName={#AppName}
UninstallDisplayIcon={uninstallexe}
SetupIconFile={#Assets}\setup.ico
WizardStyle=modern
WizardImageFile={#Assets}\wizard_100.bmp,{#Assets}\wizard_150.bmp,{#Assets}\wizard_200.bmp
WizardSmallImageFile={#Assets}\wizard_small_100.bmp,{#Assets}\wizard_small_150.bmp,{#Assets}\wizard_small_200.bmp
WizardImageStretch=no
OutputDir=Output
OutputBaseFilename=davinciClaude-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible


[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "pt"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "ja"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "ko"; MessagesFile: "compiler:Languages\Korean.isl"


[Messages]
en.WelcomeLabel2=This will install [name] on your computer.%n%nPlease close DaVinci Resolve before continuing.
fr.WelcomeLabel2=[name] va être installé sur votre ordinateur.%n%nFermez DaVinci Resolve avant de continuer.
de.WelcomeLabel2=[name] wird jetzt auf Ihrem Computer installiert.%n%nBitte schließen Sie DaVinci Resolve, bevor Sie fortfahren.
es.WelcomeLabel2=Se instalará [name] en tu ordenador.%n%nCierra DaVinci Resolve antes de continuar.
pt.WelcomeLabel2=O [name] será instalado no seu computador.%n%nFeche o DaVinci Resolve antes de continuar.
ja.WelcomeLabel2=[name] をこのコンピューターにインストールします。%n%n続行する前に DaVinci Resolve を終了してください。
ko.WelcomeLabel2=[name]을(를) 이 컴퓨터에 설치합니다.%n%n계속하기 전에 DaVinci Resolve를 종료하세요.

en.FinishedHeadingLabel=[name] is ready
fr.FinishedHeadingLabel=[name] est prêt
de.FinishedHeadingLabel=[name] ist bereit
es.FinishedHeadingLabel=[name] está listo
pt.FinishedHeadingLabel=O [name] está pronto
ja.FinishedHeadingLabel=[name] の準備ができました
ko.FinishedHeadingLabel=[name] 준비 완료

; Menus Resolve laissés en anglais : ce sont les libellés de l'interface par défaut.
en.FinishedLabel=To open it:%n%n1. Launch DaVinci Resolve (restart it if it was open).%n2. Go to Workspace > Workflow Integrations > davinciClaude.%n3. Sign in, and the guided tour takes it from there.
fr.FinishedLabel=Pour l'ouvrir :%n%n1. Lancez DaVinci Resolve (redémarrez-le s'il était ouvert).%n2. Allez dans Workspace > Workflow Integrations > davinciClaude.%n3. Connectez-vous, la visite guidée prend le relais.
de.FinishedLabel=So öffnen Sie es:%n%n1. Starten Sie DaVinci Resolve (neu starten, falls es geöffnet war).%n2. Öffnen Sie Workspace > Workflow Integrations > davinciClaude.%n3. Melden Sie sich an, die geführte Tour übernimmt den Rest.
es.FinishedLabel=Para abrirlo:%n%n1. Abre DaVinci Resolve (reinícialo si estaba abierto).%n2. Ve a Workspace > Workflow Integrations > davinciClaude.%n3. Inicia sesión y el tour guiado se encarga del resto.
pt.FinishedLabel=Para abrir:%n%n1. Abra o DaVinci Resolve (reinicie se ele estava aberto).%n2. Vá em Workspace > Workflow Integrations > davinciClaude.%n3. Faça login e o tour guiado cuida do resto.
ja.FinishedLabel=開き方:%n%n1. DaVinci Resolve を起動します（起動中の場合は再起動）。%n2. Workspace > Workflow Integrations > davinciClaude を選びます。%n3. サインインすると、ガイドツアーが始まります。
ko.FinishedLabel=여는 방법:%n%n1. DaVinci Resolve를 실행합니다(열려 있었다면 다시 시작).%n2. Workspace > Workflow Integrations > davinciClaude를 선택합니다.%n3. 로그인하면 가이드 투어가 시작됩니다.


[CustomMessages]
en.OpenGuide=Open the getting started guide
fr.OpenGuide=Ouvrir le guide de démarrage
de.OpenGuide=Einstiegsanleitung öffnen
es.OpenGuide=Abrir la guía de inicio
pt.OpenGuide=Abrir o guia de introdução
ja.OpenGuide=はじめにガイドを開く
ko.OpenGuide=시작 가이드 열기


[Files]
Source: "build\pkg_payload\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs


[Run]
Filename: "{#DocsUrl}"; Description: "{cm:OpenGuide}"; Flags: postinstall shellexec nowait skipifsilent unchecked


; NB : dossiers de données volontairement conservés en "DaVinciGPT" : le bundle
; applicatif servi par l'API écrit en dur dans ~/Documents/DaVinciGPT (getDocumentsBase).
; À renommer en davinciClaude uniquement le jour où le bundle DaVinci dédié change ce chemin.
[Dirs]
Name: "{userdocs}\DaVinciGPT"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\audio_sync"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\image_generation"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\matplotlib_cache"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\music_analysis"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\rush_db"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\script"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\seq_preset"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\sfx"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\temp"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\thumbnails"; Flags: uninsneveruninstall
Name: "{userdocs}\DaVinciGPT\transcription_analysis"; Flags: uninsneveruninstall
