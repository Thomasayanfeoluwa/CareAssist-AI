# stage0-moves.ps1 — one-time file operations for Stage 0.
# Explaination:
#   D. Move old entry points into legacy/ (kept for reference, not deployed)
#   E. Delete the boilerplate scaffolding script (templates.py)
#   F. Write legacy/README.md so future readers know why these exist

$ErrorActionPreference = "Continue"

# --- D. Move legacy files ---
New-Item -ItemType Directory -Force -Path "legacy\flask","legacy\streamlit" | Out-Null

Move-Item "app.py"          "legacy\flask\app.py"            -ErrorAction SilentlyContinue
Move-Item "dashboard.py"    "legacy\streamlit\dashboard.py"  -ErrorAction SilentlyContinue
Move-Item "store_index.py"  "legacy\flask\store_index.py"    -ErrorAction SilentlyContinue
Move-Item "templates"       "legacy\flask\templates"         -ErrorAction SilentlyContinue
Move-Item "src"             "legacy\flask\src"               -ErrorAction SilentlyContinue

# --- E. Delete boilerplate ---
Remove-Item "templates.py"   -ErrorAction SilentlyContinue
Remove-Item "list_of_files.py" -ErrorAction SilentlyContinue

# --- F. Write legacy/README.md ---
@'
# Legacy implementations — NOT deployed

## flask/app.py
Original single-user Flask app. Known defects (per project verdict):
shared module-level conversation memory, debug=True on 0.0.0.0,
chat history printed to logs, unsanitized marked.parse + innerHTML.

## streamlit/dashboard.py
Original Streamlit prototype. Uses st.cache_resource for a shared memory
object — not safe for multi-user use. Also calls ChatGroq directly.

## store_index.py, src/, templates/
Original indexing script, helper/prompt modules, and Flask HTML template.
Logic is ported into backend/app/ during Stage 1+.

None of this is imported, built, tested, or deployed.
'@ | Set-Content -Path "legacy\README.md" -Encoding UTF8

Write-Host "Stage 0 file operations complete." -ForegroundColor Green
Get-ChildItem -Force | Select-Object Name, Mode | Format-Table -AutoSize