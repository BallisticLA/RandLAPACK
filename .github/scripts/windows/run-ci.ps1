# Configures, builds, installs, and tests RandLAPACK natively on Windows with
# MSVC + oneMKL (ILP64, sequential). Shared between GitHub CI and local runs.
#
# Local use, from an MSVC developer prompt in the repository root:
#   .github\scripts\windows\run-ci.ps1 -Task Core -SetupDependencies
#
# CI runs the setup-randlapack-deps-windows action first (for caching) and then
# invokes this script without -SetupDependencies.
#
# The build is serial (no OpenMP) for now: RandLAPACK's rl_rpchol.hh uses an
# OpenMP `collapse(2)` clause, which MSVC only accepts under the /openmp:llvm
# runtime, while RandBLAS's build system currently pins MSVC OpenMP to
# /openmp:experimental. Until that is reconciled, OpenMP stays off on native
# Windows; RandLAPACK guards all OpenMP use, so serial builds are fully
# functional.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Core")]
    [string]$Task,

    [string]$SourceRoot = "",

    [string]$WorkRoot = "",

    [string]$DependencyRoot = "",

    [switch]$SetupDependencies,

    [switch]$SanitizeAddress
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "'$Program $($Arguments -join ' ')' failed with exit code $LASTEXITCODE."
    }
}

function Require-EnvironmentVariable {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if (-not $value) {
        throw "Environment variable $Name is not set. Run setup.ps1 (or pass -SetupDependencies)."
    }
    return $value
}

if ($SourceRoot -eq "") {
    if ($env:GITHUB_WORKSPACE) { $SourceRoot = $env:GITHUB_WORKSPACE }
    else { $SourceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\..")) }
}
if ($WorkRoot -eq "") {
    $WorkRoot = Join-Path (Split-Path $SourceRoot -Parent) "RandLAPACK-windows-ci"
}
if ($DependencyRoot -eq "") {
    $DependencyRoot = Join-Path (Split-Path $SourceRoot -Parent) "windows-deps"
}

if ($SetupDependencies) {
    & (Join-Path $SourceRoot ".github\actions\setup-randlapack-deps-windows\setup.ps1") `
        -DependencyRoot $DependencyRoot -SanitizeAddress:$SanitizeAddress
}

$blasppDir = Require-EnvironmentVariable "blaspp_DIR"
$lapackppDir = Require-EnvironmentVariable "lapackpp_DIR"
$random123Dir = Require-EnvironmentVariable "Random123_DIR"
$gtestPrefix = Require-EnvironmentVariable "googletest_PREFIX"
$mklRoot = Require-EnvironmentVariable "MKLROOT"
$mklBin = [Environment]::GetEnvironmentVariable("MKL_BIN")
if (-not $mklBin) { $mklBin = "$($mklRoot.Replace('/', '\'))\bin" }

# oneMKL enters through raw library paths recorded by BLAS++, so its DLLs are
# not covered by the TARGET_RUNTIME_DLLS staging and must be on PATH for the
# build-time gtest discovery and for ctest.
$env:PATH = "$mklBin;$env:PATH"

$buildDir = Join-Path $WorkRoot "RandLAPACK-build"
$installDir = Join-Path $WorkRoot "RandLAPACK-install"

$configureArgs = @(
    "-S", $SourceRoot, "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$($installDir.Replace('\', '/'))",
    "-Dblaspp_DIR=$blasppDir",
    "-Dlapackpp_DIR=$lapackppDir",
    "-DRandom123_DIR=$random123Dir",
    "-DCMAKE_PREFIX_PATH=$gtestPrefix",
    "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE"
)
if ($SanitizeAddress) { $configureArgs += "-DSANITIZE_ADDRESS=ON" }

Invoke-Checked "cmake" $configureArgs
Invoke-Checked "cmake" @("--build", $buildDir, "--target", "install")

# Same exclusion as the Linux and macOS core jobs.
Invoke-Checked "ctest" @(
    "--test-dir", $buildDir,
    "--exclude-regex", "^TestABRIK\.ABRIK_catch_instability",
    "--output-on-failure")

Write-Host "RandLAPACK Windows $Task validation succeeded."
