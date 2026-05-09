Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = 'c:\Users\South Saigon Systems\.gemini\antigravity\scratch\Yolo26_Computer_Vision\labels.zip'
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
$entries = $zip.Entries

Write-Host "Total entries: $($entries.Count)"
Write-Host "---TOP LEVEL STRUCTURE---"
$topDirs = $entries | ForEach-Object { ($_.FullName -split '/')[0] } | Sort-Object -Unique
$topDirs | ForEach-Object { Write-Host $_ }

Write-Host "---DEPTH 2 STRUCTURE---"
$depth2 = $entries | ForEach-Object { $parts = ($_.FullName -split '/'); if($parts.Count -ge 2) { "$($parts[0])/$($parts[1])" } } | Sort-Object -Unique
$depth2 | ForEach-Object { Write-Host $_ }

Write-Host "---FILE EXTENSIONS---"
$exts = $entries | Where-Object { $_.Length -gt 0 } | ForEach-Object { [System.IO.Path]::GetExtension($_.FullName) } | Group-Object | Sort-Object Count -Descending
$exts | ForEach-Object { Write-Host "$($_.Name): $($_.Count)" }

Write-Host "---TOTAL SIZE---"
$totalSize = ($entries | Measure-Object -Property Length -Sum).Sum
Write-Host "Total uncompressed: $([math]::Round($totalSize / 1MB, 2)) MB"

Write-Host "---SAMPLE FILES (first 30)---"
$entries | Select-Object -First 30 | ForEach-Object { Write-Host "$($_.FullName) [$($_.Length)]" }

Write-Host "---SAMPLE FILES (last 20)---"
$entries | Select-Object -Last 20 | ForEach-Object { Write-Host "$($_.FullName) [$($_.Length)]" }

# Read content of a few label files to understand format
Write-Host "---SAMPLE LABEL CONTENTS---"
$txtFiles = $entries | Where-Object { $_.FullName -like "*.txt" -and $_.Length -gt 0 }
$sampled = $txtFiles | Select-Object -First 10
foreach ($entry in $sampled) {
    $stream = $entry.Open()
    $reader = New-Object System.IO.StreamReader($stream)
    $content = $reader.ReadToEnd()
    $reader.Close()
    $stream.Close()
    Write-Host "--- $($entry.FullName) ---"
    Write-Host $content
}

# Count class distribution
Write-Host "---CLASS DISTRIBUTION---"
$classCounts = @{}
foreach ($entry in $txtFiles) {
    $stream = $entry.Open()
    $reader = New-Object System.IO.StreamReader($stream)
    $content = $reader.ReadToEnd()
    $reader.Close()
    $stream.Close()
    $lines = $content.Trim() -split "`n"
    foreach ($line in $lines) {
        $parts = $line.Trim() -split '\s+'
        if ($parts.Count -ge 1) {
            $cls = $parts[0]
            if ($classCounts.ContainsKey($cls)) {
                $classCounts[$cls]++
            } else {
                $classCounts[$cls] = 1
            }
        }
    }
}
foreach ($key in ($classCounts.Keys | Sort-Object)) {
    Write-Host "Class $key : $($classCounts[$key]) instances"
}

# Count files with labels vs empty
$emptyCount = ($entries | Where-Object { $_.FullName -like "*.txt" -and $_.Length -eq 0 }).Count
$nonEmptyCount = ($txtFiles).Count
Write-Host "---LABEL FILE STATS---"
Write-Host "Non-empty label files: $nonEmptyCount"
Write-Host "Empty label files: $emptyCount"

$zip.Dispose()
