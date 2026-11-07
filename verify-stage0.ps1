# verify-stage0.ps1 — corrected. Run from repo root.
$ErrorActionPreference = "Stop"
$pass = 0; $fail = 0; $skip = 0

function Check($name, [scriptblock]$body) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    try {
        & $body
        Write-Host "PASS" -ForegroundColor Green; $script:pass++
    } catch {
        Write-Host "FAIL: $_" -ForegroundColor Red; $script:fail++
    }
}
function Skip($name, $reason) {
    Write-Host "`n=== $name === SKIP ($reason)" -ForegroundColor Yellow; $script:skip++
}

function Assert-NoMatch([string[]]$roots, [string]$pattern, [string]$label) {
    $existing = $roots | Where-Object { Test-Path $_ }
    if (-not $existing) { return }
    $files = Get-ChildItem -Path $existing -Recurse -File -ErrorAction SilentlyContinue
    if (-not $files) { return }
    $hits = $files | Select-String -Pattern $pattern -ErrorAction SilentlyContinue
    if ($hits) {
        $hits | ForEach-Object { Write-Host "  $_" }
        throw "$label found"
    }
}

Check "1. No st.secrets in backend/frontend" {
    Assert-NoMatch @("backend","frontend") "st\.secrets" "st.secrets"
}
Check "2. No Flask in backend/frontend" {
    Assert-NoMatch @("backend","frontend") "from flask|Flask\(" "Flask"
}
Check "3. No ChatGroq / langchain_groq in backend/frontend" {
    Assert-NoMatch @("backend","frontend") "ChatGroq|langchain_groq" "Groq SDK"
}
Check "4. No 'OpenAI' claims in source/docs" {
    $roots = @("backend","frontend") | Where-Object { Test-Path $_ }
    if ($roots) {
        $files = Get-ChildItem -Path $roots -Recurse -File `
            -Include *.py,*.ts,*.tsx,*.js,*.jsx,*.md,*.html,*.json,*.yaml,*.yml,*.toml `
            -ErrorAction SilentlyContinue
        if ($files) {
            $hits = $files | Select-String -Pattern "OpenAI" -ErrorAction SilentlyContinue
            if ($hits) {
                $hits | ForEach-Object { Write-Host "  $_" }
                throw "OpenAI claim found in source/docs"
            }
        }
    }
    Write-Host "  no OpenAI claims in source/docs (lockfile and requirements excluded — openai is a transitive SDK, not a model choice)"
}
Check "5. .env not tracked by git" {
    $tracked = git ls-files | Select-String -Pattern "(^|/)\.env$"
    if ($tracked) { $tracked; throw ".env is tracked" }
}
Check "6. .env.example exists" {
    if (-not (Test-Path ".env.example")) { throw ".env.example missing" }
}
Check "7. Lockfile generates cleanly" {
    if (-not (Test-Path "backend\requirements.txt")) { throw "backend\requirements.txt missing" }
    Push-Location backend
    try {
        pip-compile --quiet -o requirements.lock requirements.txt
        if ($LASTEXITCODE -ne 0) { throw "pip-compile failed" }
    } finally { Pop-Location }
}
Check "8. requirements.lock exists" {
    if (-not (Test-Path "backend\requirements.lock")) { throw "requirements.lock missing" }
}
Skip "9. ruff check" "needs Stage 1 code"
Skip "10. mypy" "needs Stage 1 code"
Skip "11. npm ci && npm run build" "needs Stage 1 code"
Check "12. render.yaml exists" {
    if (-not (Test-Path "render.yaml")) { throw "render.yaml missing" }
}
Check "13. Legacy folder populated" {
    if (-not (Test-Path "legacy\README.md")) { throw "legacy\README.md missing" }
    if (-not (Test-Path "legacy\flask\app.py")) { throw "legacy\flask\app.py missing" }
    if (-not (Test-Path "legacy\streamlit\dashboard.py")) { throw "legacy\streamlit\dashboard.py missing" }
}
Check "14. No list_of_files boilerplate" {
    $hits = Get-ChildItem -Recurse -File -Filter "*.py" -ErrorAction SilentlyContinue |
            Select-String -Pattern "list_of_files" -ErrorAction SilentlyContinue
    if ($hits) { $hits | ForEach-Object { Write-Host "  $_" }; throw "list_of_files found" }
}
Check "15. Old entry points removed from root" {
    $bad = @("app.py","dashboard.py","store_index.py","templates","templates.py","src") |
           Where-Object { Test-Path $_ }
    if ($bad) { Write-Host "  still present: $($bad -join ', ')"; throw "old entry points still present" }
}

Write-Host "`n==============================" -ForegroundColor Magenta
Write-Host "PASS: $pass   FAIL: $fail   SKIP: $skip" -ForegroundColor Magenta
Write-Host "==============================" -ForegroundColor Magenta