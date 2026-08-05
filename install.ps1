# RandLAPACK native Windows installer -- the companion to install.sh.
#
# Run from an MSVC developer prompt (or "Developer PowerShell for VS") in the
# repository root:
#
#   .\install.ps1
#
# What it does, mirroring install.sh's layout in a sibling RandNLA-project
# directory:
#   1. Builds/reuses the dependencies (oneMKL via vcpkg or -MklRoot,
#      GoogleTest, Random123, BLAS++, LAPACK++) under <ProjectDir>\install.
#   2. Configures, builds, installs, and tests RandLAPACK.
#
# Options:
#   -ProjectDir <path>  Where dependencies/builds/installs go
#                       (default: ..\RandNLA-project relative to this script).
#   -MklRoot <path>     Use an existing oneMKL (oneAPI installer layout)
#                       instead of fetching MKL through vcpkg.
#   -Fresh              Reconfigure RandLAPACK from scratch (dependencies are
#                       always reused when present; delete <ProjectDir>\install
#                       subdirectories to force dependency rebuilds).
#   -SkipTests          Do not run the test suite after building.
#
# GPU support is not available on native Windows yet. OpenMP is disabled on
# MSVC for now (see .github/scripts/windows/run-ci.ps1 for why); RandLAPACK
# is fully functional serially.

[CmdletBinding()]
param(
    [string]$ProjectDir = "",
    [string]$MklRoot = "",
    [switch]$Fresh,
    [switch]$SkipTests
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

$sourceRoot = $PSScriptRoot
if (-not (Test-Path (Join-Path $sourceRoot "RandLAPACK\RandLAPACK.hh"))) {
    throw "install.ps1 must sit in the RandLAPACK repository root."
}
if (-not (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    throw "cl.exe is not on PATH. Run from an MSVC developer prompt (Developer PowerShell for VS 2022)."
}
if (-not (Test-Path (Join-Path $sourceRoot "RandBLAS\CMakeLists.txt"))) {
    Write-Host "Initializing the RandBLAS submodule..."
    Invoke-Checked "git" @("-C", $sourceRoot, "submodule", "update", "--init", "--recursive")
}

if ($ProjectDir -eq "") {
    $ProjectDir = Join-Path (Split-Path $sourceRoot -Parent) "RandNLA-project"
}
$ProjectDir = [System.IO.Path]::GetFullPath($ProjectDir)
$dependencyRoot = Join-Path $ProjectDir "install"
$buildDir = Join-Path $ProjectDir "build\RandLAPACK-build"
$installDir = Join-Path $ProjectDir "install\RandLAPACK-install"

Write-Host ""
Write-Host "RandLAPACK Windows install"
Write-Host "  source:       $sourceRoot"
Write-Host "  project dir:  $ProjectDir"
Write-Host ""

# Step 1: dependencies (idempotent; reused when already present).
& (Join-Path $sourceRoot ".github\actions\setup-randlapack-deps-windows\setup.ps1") `
    -DependencyRoot $dependencyRoot -MklRoot $MklRoot

# Step 2: RandLAPACK itself.
if ($Fresh -and (Test-Path $buildDir)) {
    Write-Host "Removing previous build directory (-Fresh)..."
    Remove-Item -Recurse -Force $buildDir
}

Invoke-Checked "cmake" @(
    "-S", $sourceRoot, "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$($installDir.Replace('\', '/'))",
    "-Dblaspp_DIR=$env:blaspp_DIR",
    "-Dlapackpp_DIR=$env:lapackpp_DIR",
    "-DRandom123_DIR=$env:Random123_DIR",
    "-DCMAKE_PREFIX_PATH=$env:googletest_PREFIX",
    "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE")
Invoke-Checked "cmake" @("--build", $buildDir, "--target", "install")

if (-not $SkipTests) {
    Invoke-Checked "ctest" @(
        "--test-dir", $buildDir,
        "--exclude-regex", "^TestABRIK\.ABRIK_catch_instability",
        "--output-on-failure")
}

Write-Host ""
Write-Host "RandLAPACK is installed."
Write-Host "  RandLAPACK_DIR: $installDir\lib\cmake\RandLAPACK"
Write-Host "  blaspp_DIR:     $env:blaspp_DIR"
Write-Host "  lapackpp_DIR:   $env:lapackpp_DIR"
Write-Host "  Random123_DIR:  $env:Random123_DIR"
Write-Host ""
Write-Host "Keep $env:MKL_BIN on PATH when running executables that link MKL."
