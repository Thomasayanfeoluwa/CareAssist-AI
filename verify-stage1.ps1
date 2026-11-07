$ErrorActionPreference = "Stop"
$pass = 0; $fail = 0

function Check($name, [scriptblock]$body) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    try { & $body; Write-Host "PASS" -ForegroundColor Green; $script:pass++ }
    catch { Write-Host "FAIL: $_" -ForegroundColor Red; $script:fail++ }
}

Check "1. Backend deps install" {
    Push-Location backend; try {
        python -m pip install -q -r requirements-dev.txt
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    } finally { Pop-Location }
}
Check "2. ruff clean" {
    Push-Location backend; try {
        python -m ruff check app tests
        if ($LASTEXITCODE -ne 0) { throw "ruff failed" }
    } finally { Pop-Location }
}
Check "3. mypy clean" {
    Push-Location backend; try {
        python -m mypy app
        if ($LASTEXITCODE -ne 0) { throw "mypy failed" }
    } finally { Pop-Location }
}
Check "4. pytest green" {
    Push-Location backend; try {
        python -m pytest -q
        if ($LASTEXITCODE -ne 0) { throw "pytest failed" }
    } finally { Pop-Location }
}
Check "5. Frontend npm ci" {
    Push-Location frontend; try {
        npm ci --silent
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
    } finally { Pop-Location }
}
Check "6. Frontend typecheck" {
    Push-Location frontend; try {
        npm run --silent typecheck
        if ($LASTEXITCODE -ne 0) { throw "typecheck failed" }
    } finally { Pop-Location }
}
Check "7. Frontend lint" {
    Push-Location frontend; try {
        npm run --silent lint
        if ($LASTEXITCODE -ne 0) { throw "lint failed" }
    } finally { Pop-Location }
}
Check "8. Frontend build" {
    Push-Location frontend; try {
        npm run --silent build
        if ($LASTEXITCODE -ne 0) { throw "build failed" }
    } finally { Pop-Location }
}
Check "9. No emojis in frontend/src" {
    $hits = Get-ChildItem frontend\src -Recurse -File |
        Select-String -Pattern '[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]' -ErrorAction SilentlyContinue
    if ($hits) { $hits | ForEach-Object { Write-Host "  $_" }; throw "emoji found" }
}
Check "10. No user content in logs (pytest)" {
    Push-Location backend; try {
        python -m pytest -q tests/test_logs.py
        if ($LASTEXITCODE -ne 0) { throw "log-leak test failed" }
    } finally { Pop-Location }
}
Check "11. Healthz responds (in-process)" {
    Push-Location backend; try {
        $env:LLM_MOCK = "true"
        $env:GROQ_API_KEY = "x"; $env:OPENROUTER_API_KEY = "x"
        $env:PINECONE_API_KEY = "x"; $env:GOOGLE_CSE_API_KEY = "x"
        $env:GOOGLE_CSE_ID = "x"; $env:SESSION_SECRET = ("x"*32)
        $env:PILOT_AUTH_TOKEN = "test-pilot-token-1234"
        python -c @"
from fastapi.testclient import TestClient
from app.main import create_app
r = TestClient(create_app()).get('/healthz')
assert r.status_code == 200, r.text
assert r.json()['status'] == 'ok'
print('healthz ok')
"@
        if ($LASTEXITCODE -ne 0) { throw "healthz check failed" }
    } finally { Pop-Location }
}

Write-Host "`n==============================" -ForegroundColor Magenta
Write-Host "PASS: $pass   FAIL: $fail" -ForegroundColor Magenta
Write-Host "==============================" -ForegroundColor Magenta