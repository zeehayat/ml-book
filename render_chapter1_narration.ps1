param(
    [Parameter(Mandatory = $true)][string]$NarrationDirectory,
    [Parameter(Mandatory = $true)][string]$AudioDirectory
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $AudioDirectory | Out-Null

$voice = New-Object -ComObject SAPI.SpVoice
$voice.Rate = 1
$voice.Volume = 100

Get-ChildItem -LiteralPath $NarrationDirectory -Filter "*.txt" | Sort-Object Name | ForEach-Object {
    $output = Join-Path $AudioDirectory ($_.BaseName + ".wav")
    $stream = New-Object -ComObject SAPI.SpFileStream
    $stream.Open($output, 3, $false)
    $voice.AudioOutputStream = $stream
    $text = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8
    [void]$voice.Speak($text)
    $stream.Close()
}

$voice.AudioOutputStream = $null
