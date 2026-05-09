param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$InputPath,
    [string]$Weights = "runs\detect\yolo26s_traffic_final\weights\best.pt",
    [string]$Output = "",
    [double]$Conf = 0.25,
    [switch]$Show
)

if ([string]::IsNullOrWhiteSpace($InputPath)) {
    Write-Error 'Input video path is required. Example: .\run_video.ps1 -Input "C:\path\to\video.mp4"'
    exit 1
}

if (-not (Test-Path -LiteralPath $InputPath)) {
    Write-Error "Input video not found: $InputPath"
    exit 1
}

if (-not (Test-Path -LiteralPath $Weights)) {
    Write-Error "Weights not found: $Weights"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $dir = Split-Path -Parent $InputPath
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($InputPath)
    $Output = Join-Path $dir ("{0}_yolo26.mp4" -f $stem)
}

$showArg = ""
if ($Show) {
    $showArg = "--show"
}

python realtime_test.py --weights $Weights --input $InputPath --output $Output --conf $Conf $showArg
