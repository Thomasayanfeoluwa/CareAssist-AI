# list-repo.ps1 — Lists every source file in the repo. Deletes nothing.
$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
Write-Host "Repo root: $root" -ForegroundColor Cyan
Write-Host ""

$exclude = '\\\.git\\|\\\.venv\\|\\node_modules\\|\\__pycache__\\|\\dist\\|\\build\\|\\\.mypy_cache\\|\\\.pytest_cache\\|\\\.ruff_cache\\'

$all = @(
    Get-ChildItem -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch $exclude } |
        ForEach-Object {
            [pscustomobject]@{
                Rel  = $_.FullName.Substring($root.Length + 1)
                Size = $_.Length
            }
        }
) | Sort-Object Rel

$out = Join-Path $root 'repo-listing.txt'
$all | ForEach-Object { "{0,9}  {1}" -f $_.Size, $_.Rel } |
    Set-Content -Path $out -Encoding UTF8

Write-Host "Total files (excluding .git, .venv, node_modules, caches): $($all.Count)" -ForegroundColor Green
Write-Host "Full listing written to: $out" -ForegroundColor Green
Write-Host ""

Write-Host "=== File count by top-level directory ===" -ForegroundColor Cyan
$all |
    Group-Object { ($_.Rel -split '[\\/]')[0] } |
    Sort-Object Name |
    ForEach-Object { "{0,6}  {1}" -f $_.Count, $_.Name }
Write-Host ""

Write-Host "=== Zero-byte files ===" -ForegroundColor Cyan
$zero = $all | Where-Object { $_.Size -eq 0 }
if ($zero.Count -eq 0) { Write-Host "(none)" } else { $zero | ForEach-Object { $_.Rel } }
Write-Host ""

Write-Host "=== Empty directories ===" -ForegroundColor Cyan
$emptyDirs = @(
    Get-ChildItem -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch $exclude } |
        Where-Object { (Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0 }
)
if ($emptyDirs.Count -eq 0) { Write-Host "(none)" } else { $emptyDirs | ForEach-Object { $_.FullName.Substring($root.Length + 1) } }
Write-Host ""

Write-Host "=== Full listing (same content as repo-listing.txt) ===" -ForegroundColor Cyan
$all | ForEach-Object { "{0,9}  {1}" -f $_.Size, $_.Rel }