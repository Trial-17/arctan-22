; Installeur Windows PremiereCopilot (Inno Setup 6).
; La version est passée par le workflow : ISCC /DAppVer=3.0.N
; Le dossier d'extension reste "PremiereGPTBeta" : c'est l'identifiant CEP du panneau.

#ifndef AppVer
  #define AppVer "3.0.0"
#endif
#define AppName "PremiereCopilot"
#define SiteUrl "https://www.premierecopilot.com"
#define DocsUrl "https://docs.premierecopilot.com/getting-started/introduction"
#define Assets "installer_assets\premiere"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  OldDir1, OldDir2, OldDir3: String;
begin
  if CurStep = ssInstall then
  begin
    OldDir1 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereGPTBeta');
    OldDir2 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\PremiereCopilot');
    OldDir3 := ExpandConstant('{commoncf}\Adobe\CEP\extensions\Premierecopilot');

    if DirExists(OldDir1) then
      DelTree(OldDir1, True, True, True);
    if DirExists(OldDir2) then
      DelTree(OldDir2, True, True, True);
    if DirExists(OldDir3) then
      DelTree(OldDir3, True, True, True);
  end;
end;


[Setup]
AppId={{C88C9B80-923B-4780-B42A-2D444A76BECF}
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
DefaultDirName={commoncf}\Adobe\CEP\extensions\PremiereGPTBeta
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
OutputBaseFilename=PremiereCopilot-Setup
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
en.WelcomeLabel2=This will install [name] on your computer.%n%nPlease close Premiere Pro before continuing.
fr.WelcomeLabel2=[name] va être installé sur votre ordinateur.%n%nFermez Premiere Pro avant de continuer.
de.WelcomeLabel2=[name] wird jetzt auf Ihrem Computer installiert.%n%nBitte schließen Sie Premiere Pro, bevor Sie fortfahren.
es.WelcomeLabel2=Se instalará [name] en tu ordenador.%n%nCierra Premiere Pro antes de continuar.
pt.WelcomeLabel2=O [name] será instalado no seu computador.%n%nFeche o Premiere Pro antes de continuar.
ja.WelcomeLabel2=[name] をこのコンピューターにインストールします。%n%n続行する前に Premiere Pro を終了してください。
ko.WelcomeLabel2=[name]을(를) 이 컴퓨터에 설치합니다.%n%n계속하기 전에 Premiere Pro를 종료하세요.

en.FinishedHeadingLabel=[name] is ready
fr.FinishedHeadingLabel=[name] est prêt
de.FinishedHeadingLabel=[name] ist bereit
es.FinishedHeadingLabel=[name] está listo
pt.FinishedHeadingLabel=O [name] está pronto
ja.FinishedHeadingLabel=[name] の準備ができました
ko.FinishedHeadingLabel=[name] 준비 완료

en.FinishedLabel=To open it:%n%n1. Launch Premiere Pro (restart it if it was open).%n2. Go to Window > Extensions > Premiere GPT.%n3. Sign in, and the guided tour takes it from there.
fr.FinishedLabel=Pour l'ouvrir :%n%n1. Lancez Premiere Pro (redémarrez-le s'il était ouvert).%n2. Allez dans Fenêtre > Extensions > Premiere GPT.%n3. Connectez-vous, la visite guidée prend le relais.
de.FinishedLabel=So öffnen Sie es:%n%n1. Starten Sie Premiere Pro (neu starten, falls es geöffnet war).%n2. Öffnen Sie Fenster > Erweiterungen > Premiere GPT.%n3. Melden Sie sich an, die geführte Tour übernimmt den Rest.
es.FinishedLabel=Para abrirlo:%n%n1. Abre Premiere Pro (reinícialo si estaba abierto).%n2. Ve a Ventana > Extensiones > Premiere GPT.%n3. Inicia sesión y el tour guiado se encarga del resto.
pt.FinishedLabel=Para abrir:%n%n1. Abra o Premiere Pro (reinicie se ele estava aberto).%n2. Vá em Janela > Extensões > Premiere GPT.%n3. Faça login e o tour guiado cuida do resto.
ja.FinishedLabel=開き方:%n%n1. Premiere Pro を起動します（起動中の場合は再起動）。%n2. ウィンドウ > エクステンション > Premiere GPT を選びます。%n3. サインインすると、ガイドツアーが始まります。
ko.FinishedLabel=여는 방법:%n%n1. Premiere Pro를 실행합니다(열려 있었다면 다시 시작).%n2. 창 > 확장 > Premiere GPT를 선택합니다.%n3. 로그인하면 가이드 투어가 시작됩니다.


[CustomMessages]
en.OpenGuide=Open the getting started guide
fr.OpenGuide=Ouvrir le guide de démarrage
de.OpenGuide=Einstiegsanleitung öffnen
es.OpenGuide=Abrir la guía de inicio
pt.OpenGuide=Abrir o guia de introdução
ja.OpenGuide=はじめにガイドを開く
ko.OpenGuide=시작 가이드 열기


[Registry]
Root: HKCU; Subkey: "Software\Adobe\CSXS.12"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"
Root: HKCU; Subkey: "Software\Adobe\CSXS.11"; ValueType: string; ValueName: "PlayerDebugMode"; ValueData: "1"


[Files]
; Extension CEP
Source: "build\pkg_payload\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs


[Run]
Filename: "{#DocsUrl}"; Description: "{cm:OpenGuide}"; Flags: postinstall shellexec nowait skipifsilent unchecked


[Dirs]
; Dossiers de travail dans les Documents de l'utilisateur, conservés à la désinstallation
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\audio_sync"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\image_generation"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\matplotlib_cache"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\music_analysis"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\rush_db"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\script"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\seq_preset"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\sfx"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\temp"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\thumbnails"; Flags: uninsneveruninstall
Name: "{userdocs}\Adobe\Premiere Pro\Premiere Copilot\transcription_analysis"; Flags: uninsneveruninstall
