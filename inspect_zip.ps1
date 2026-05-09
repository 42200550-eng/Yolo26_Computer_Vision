Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = 'c:\Users\South Saigon Systems\.gemini\antigravity\scratch\Yolo26_Computer_Vision\images.zip'
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
$entries = $zip.Entries
Write-Host "Total entries: $($entries.Count)"
Write-Host "---TOP LEVEL STRUCTURE---"
$topDirs = $entries | ForEach-Object { ($_.FullName -split '/')[0] } | Sort-Object -Unique
$topDirs | ForEach-Object { Write-Host $_ }
Write-Host "---DEPTH 2 STRUCTURE---"
$depth2 = $entries | ForEach-Object { $parts = ($_.FullName -split '/'); if($parts.Count -ge 2) { "$($parts[0])/$($parts[1])" } } | Sort-Object -Unique
$depth2 | ForEach-Object { Write-Host $_ }
Write-Host "---SAMPLE FILES (first 50)---"
$entries | Select-Object -First 50 | ForEach-Object { Write-Host "$($_.FullName) [$($_.Length)]" }
Write-Host "---SAMPLE FILES (last 30)---"
$entries | Select-Object -Last 30 | ForEach-Object { Write-Host "$($_.FullName) [$($_.Length)]" }
Write-Host "---FILE EXTENSIONS---"
$exts = $entries | Where-Object { $_.Length -gt 0 } | ForEach-Object { [System.IO.Path]::GetExtension($_.FullName) } | Group-Object | Sort-Object Count -Descending
$exts | ForEach-Object { Write-Host "$($_.Name): $($_.Count)" }
Write-Host "---TOTAL SIZE---"
$totalSize = ($entries | Measure-Object -Property Length -Sum).Sum
Write-Host "Total uncompressed: $([math]::Round($totalSize / 1GB, 2)) GB"
$zip.Dispose()
