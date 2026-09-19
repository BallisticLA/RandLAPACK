# Asserts that install.ps1 REFUSES the toolchain currently on PATH, with a
# message containing -Expect. Used by core-windows.yaml's guard job to prove
# the architecture check is wired end to end, not just correct in isolation.
#
# The installer is launched exactly as INSTALL_WINDOWS.md prescribes -- via
# cmd with -ExecutionPolicy Bypass -- so the documented invocation stays under
# test alongside the guard itself.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Expect
)

$ErrorActionPreference = "Stop"

$output = & cmd /c "powershell -ExecutionPolicy Bypass -File .\installers\install.ps1 2>&1"
$installerExit = $LASTEXITCODE
$text = $output -join "`n"
Write-Host $text

if ($installerExit -eq 0) {
    throw "install.ps1 succeeded; the architecture guard did not fire."
}
if ($text -notmatch [regex]::Escape($Expect)) {
    throw "The guard fired, but its output contained no '$Expect': the wrong check failed."
}
Write-Host "OK: refused at preflight with the expected explanation ('$Expect')."

# This script succeeds when the installer FAILS, so the non-zero exit code
# that failure left behind has to be cleared: `shell: powershell` exits the
# step with whatever $LASTEXITCODE holds, which would report a pass as a
# failure.
exit 0
