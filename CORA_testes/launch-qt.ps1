param(
    [string]$Folder,
    [string]$Output
)

Set-Location -LiteralPath (Split-Path -Parent $MyInvocation.MyCommand.Path)
. .\.venv\Scripts\Activate.ps1

$args = @("python", "-m", "cora_projeto.qt.init_qt")
if ($Folder) {
    $args += "--folder"
    $args += $Folder
}
if ($Output) {
    $args += "--output"
    $args += $Output
}

& $args
