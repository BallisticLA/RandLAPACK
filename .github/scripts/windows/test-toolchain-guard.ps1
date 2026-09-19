# Table test for the toolchain architecture guard.
#
# Why this exists as a separate test rather than relying on the build matrix:
# the guard's whole job is to reject architectures we cannot build for, so
# there is no runner on which "it built fine" demonstrates it works. Driving
# the decision directly covers arm64 and arm on ordinary x64 hardware, and
# covers them in seconds.
#
#
# The integration counterpart lives in core-windows.yaml, which runs the real
# installer under a real x86 and a real cross-compiled arm64 toolchain and
# asserts it refuses. That proves the wiring; this proves the decision.

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Single shared implementation, dot-sourced by both install.ps1 and setup.ps1.
$source = Join-Path $PSScriptRoot "toolchain-arch.ps1"

# Arch, whether it must be accepted, and a phrase the refusal must contain.
# x86 and arm64 must not merely both fail -- they must fail with *different*
# advice, since one is a wrong-shell mistake and the other is an unsupported
# platform. The Expect phrases pin that distinction.
$cases = @(
    @{ Arch = "x64";   ShouldPass = $true;  Expect = "" },
    @{ Arch = "amd64"; ShouldPass = $true;  Expect = "" },
    @{ Arch = "x86";   ShouldPass = $false; Expect = "32-bit developer shell" },
    @{ Arch = "arm64"; ShouldPass = $false; Expect = "x64-only" },
    @{ Arch = "arm";   ShouldPass = $false; Expect = "x64-only" }
)

function Get-GuardVerdicts {
    # Loads the two functions out of $SourceFile without executing the rest
    # of it, then drives them over the case table.
    param([string]$SourceFile, [object[]]$Cases)
    $ast = [System.Management.Automation.Language.Parser]::ParseInput(
        (Get-Content -Raw $SourceFile), [ref]$null, [ref]$null)
    $definitions = foreach ($name in @("Get-ClTargetArchitecture", "Get-ToolchainArchitectureProblem")) {
        $found = $ast.Find({
            param($node)
            $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
        }, $true)
        if (-not $found) { throw "$SourceFile does not define $name." }
        $found.Extent.Text
    }
    $body = @'
param($Cases)
foreach ($case in $Cases) {
    $env:VSCMD_ARG_TGT_ARCH = $case.Arch
    $detected = Get-ClTargetArchitecture
    [pscustomobject]@{
        Arch     = $case.Arch
        Detected = $detected
        Problem  = (Get-ToolchainArchitectureProblem $detected)
    }
}
'@
    $script = "param(`$Cases)`n" + ($definitions -join "`n") + "`n" + ($body -replace '^param\(\$Cases\)\r?\n', '')
    return & ([scriptblock]::Create($script)) $Cases
}

$savedArch = $env:VSCMD_ARG_TGT_ARCH
$failures = 0
try {
    $verdicts = @(Get-GuardVerdicts -SourceFile $source -Cases $cases)
    for ($i = 0; $i -lt $cases.Count; $i++) {
        $case = $cases[$i]
        $verdict = $verdicts[$i]
        $accepted = ($verdict.Problem -eq "")
        $ok = ($accepted -eq $case.ShouldPass)
        if ($ok -and $case.Expect -ne "" -and $verdict.Problem -notmatch [regex]::Escape($case.Expect)) {
            $ok = $false
            Write-Host "    (refused, but the message lacked '$($case.Expect)')"
        }
        if (-not $ok) { $failures++ }
        Write-Host ("{0}  {1,-6} detected={2,-6} {3}" -f
            $(if ($ok) { "OK  " } else { "FAIL" }),
            $case.Arch, $verdict.Detected,
            $(if ($accepted) { "accepted" } else { "refused" }))
    }
} finally {
    $env:VSCMD_ARG_TGT_ARCH = $savedArch
}

Write-Host ""
if ($failures -gt 0) { throw "$failures toolchain-guard assertion(s) failed." }
Write-Host "All toolchain-guard assertions passed."
