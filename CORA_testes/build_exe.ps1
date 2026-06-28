param(
    [string]$Python = "python",
    [switch]$Clean
)

# PT: Interrompe o script no primeiro erro para evitar build parcial. | EN: Stops the script on the first error to prevent a partial build.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# PT: Limpeza opcional para gerar build totalmente novo. | EN: Optional cleanup to generate a completely fresh build.
if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "build", "dist", "CORA.spec"
}

# PT: Garante ferramentas atualizadas para reproducibilidade da montagem. | EN: Ensures up-to-date tools for a reproducible build.
& $Python -m pip install --upgrade pip
& $Python -m pip install -r requirements.txt pyinstaller

# PT: Empacota a GUI em modo windowed (sem console) com dependencias extras. | EN: Packages the GUI in windowed mode (without a console) with extra dependencies.
& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name CORA `
    --hidden-import threadpoolctl `
    --hidden-import matplotlib.backends.backend_tkagg `
    --collect-data matplotlib `
    --collect-submodules sklearn `
    cora_projeto/run_cora_gui.py

Write-Host ""
Write-Host "Build concluido: dist\\CORA\\CORA.exe"

