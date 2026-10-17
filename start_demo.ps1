param([switch]$Live, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$demoRuntime = Join-Path $PSScriptRoot 'api/runtime'
New-Item -ItemType Directory -Path $demoRuntime -Force | Out-Null
if (-not (Test-Path -LiteralPath 'api/demo_bundle/manifest.json')) {
    throw 'Saved demo bundle missing. Run: python -m api.build_demo'
}
if (-not (Test-Path -LiteralPath 'web-app/node_modules/next/package.json')) {
    throw 'Frontend dependencies missing. In web-app run: npm install (before the offline rehearsal).'
}
$env:MRONE_DEMO_ONLY = if ($Live) { '0' } else { '1' }
function Test-DemoUrl([string]$Url) {
    try { return (Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 10).StatusCode -eq 200 } catch { return $false }
}
function Test-DemoPort([int]$Port) {
    $demoSocket = New-Object System.Net.Sockets.TcpClient
    try { $demoSocket.Connect('127.0.0.1', $Port); return $true } catch { return $false } finally { $demoSocket.Dispose() }
}
$demoProcesses = @()
try {
    if (Test-DemoUrl 'http://127.0.0.1:8000/health') {
        $demoHealth = Invoke-RestMethod 'http://127.0.0.1:8000/health'
        if ($demoHealth.demo_only -ne (-not $Live.IsPresent)) {
            throw 'An API is already running in a different mode. Stop that API first, then launch again.'
        }
    } else {
        $demoPython = (Get-Command python -ErrorAction Stop).Source
        $demoProcesses += Start-Process -FilePath $demoPython -ArgumentList '-m','uvicorn','api.main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput "$demoRuntime/api-out.log" -RedirectStandardError "$demoRuntime/api-error.log" -PassThru
    }
    if (-not (Test-DemoPort 3000)) {
        $demoNode = (Get-Command node -ErrorAction Stop).Source
        $demoWebMode = if (Test-Path -LiteralPath 'web-app/.next/BUILD_ID') { 'start' } else { 'dev' }
        $demoProcesses += Start-Process -FilePath $demoNode -ArgumentList 'node_modules/next/dist/bin/next',$demoWebMode,'--hostname','127.0.0.1','--port','3000' -WorkingDirectory (Join-Path $PSScriptRoot 'web-app') -WindowStyle Hidden -RedirectStandardOutput "$demoRuntime/web-out.log" -RedirectStandardError "$demoRuntime/web-error.log" -PassThru
    }
    $demoDeadline = (Get-Date).AddSeconds(120)
    while (-not ((Test-DemoUrl 'http://127.0.0.1:8000/health') -and (Test-DemoUrl 'http://127.0.0.1:3000'))) {
        if ((Get-Date) -gt $demoDeadline) { throw "Servers did not become ready. See $demoRuntime logs." }
        foreach ($demoProcess in $demoProcesses) { if ($demoProcess.HasExited) { throw "Server exited. See $demoRuntime logs." } }
        Start-Sleep -Milliseconds 500
    }
    $demoProcesses | Select-Object Id, ProcessName | ConvertTo-Json | Set-Content -LiteralPath "$demoRuntime/launcher-processes.json"
    Write-Host 'MR-one ready: http://localhost:3000'
    Write-Host $(if ($Live) { 'Live mode: uncached requests may use the GPU.' } else { 'Saved replay mode: uncached requests are blocked. No GPU required for Studio demo.' })
    if (-not $NoBrowser) { Start-Process 'http://localhost:3000' }
} catch {
    foreach ($demoProcess in $demoProcesses) { if (-not $demoProcess.HasExited) { Stop-Process -Id $demoProcess.Id } }
    throw
}
