# RandLAPACK native Windows installer -- the companion to install.sh.
#
# Run from an MSVC developer prompt (or "Developer PowerShell for VS") in the
# repository root:
#
#   .\install\install.ps1
#
# What it does, mirroring install.sh's layout in a sibling RandNLA-project
# directory:
#   1. Builds/reuses the dependencies (oneMKL from Intel's NuGet packages
#      or -MklRoot, GoogleTest, Random123, BLAS++, LAPACK++) under
#      <ProjectDir>\install.
#   2. Configures, builds, installs, and tests RandLAPACK.
#
# Options:
#   -ProjectDir <path>  Where dependencies/builds/installs go
#                       (default: ..\RandNLA-project relative to this script).
#   -Backend <name>     BLAS/LAPACK backend: mkl (default; auto-discovers an
#                       installed oneAPI, else downloads Intel's pinned NuGet
#                       packages), openblas (official release binaries), or
#                       custom (bring your own via -BlasLibraries).
#   -MklRoot <path>     Use this oneMKL install (oneAPI layout) instead of
#                       auto-discovery/download. Backend mkl only.
#   -BlasLibraries / -LapackLibraries / -BackendBinDir / -BlasInt / -BlasFortran
#                       Backend custom only; see setup.ps1's header.
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
    [ValidateSet("mkl", "openblas", "custom")]
    [string]$Backend = "mkl",
    [string]$MklRoot = "",
    [string]$BlasLibraries = "",
    [string]$LapackLibraries = "",
    [string]$BackendBinDir = "",
    [ValidateSet("lp64", "ilp64")]
    [string]$BlasInt = "lp64",
    [string]$BlasFortran = "",
    # Where the dependency stack lives (default: <ProjectDir>\install). CI
    # points this at its shared, cached dependency directory.
    [string]$DependencyRoot = "",
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

# This script lives in <repo>\install\; the repository root is one level up.
$sourceRoot = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $sourceRoot "RandLAPACK.hh"))) {
    throw "install.ps1 must sit in the install\ directory of a RandLAPACK clone."
}

# ------------------------------------------------------ preflight checks ----
# Catch every missing prerequisite up front, each with its fix, instead of
# failing later with a tool-specific error.
$preflightProblems = @()
if (-not (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    $preflightProblems += ("cl.exe (the MSVC compiler) is not on PATH. Open 'Developer PowerShell " +
        "for VS 2022' from the Start menu and run this script there. If Visual Studio is not " +
        "installed:`n    winget install Microsoft.VisualStudio.2022.Community --override " +
        '"--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"')
}
foreach ($tool in @(
        @{ Name = "cmake.exe"; Hint = "CMake ships with the Visual Studio C++ workload; a Developer PowerShell puts it on PATH." },
        @{ Name = "ninja.exe"; Hint = "Ninja ships with the Visual Studio C++ workload; a Developer PowerShell puts it on PATH." },
        @{ Name = "git.exe";   Hint = "Install Git:`n    winget install Git.Git" },
        @{ Name = "curl.exe";  Hint = "curl.exe ships with Windows 10 (1803+) in System32; check your PATH includes it." })) {
    if (-not (Get-Command $tool.Name -ErrorAction SilentlyContinue)) {
        $preflightProblems += "$($tool.Name) is not on PATH. $($tool.Hint)"
    }
}
if ($preflightProblems.Count -gt 0) {
    $preflightProblems | ForEach-Object { Write-Host "PREREQUISITE MISSING: $_`n" }
    throw "Missing prerequisites ($($preflightProblems.Count)); see the messages above."
}
if ($ProjectDir -ne "" -and $ProjectDir.Length -gt 150) {
    Write-Warning ("-ProjectDir is $($ProjectDir.Length) characters long; deep dependency build " +
        "paths may exceed Windows' 260-character limit. Prefer a shorter location.")
}

if (-not (Test-Path (Join-Path $sourceRoot "RandBLAS\CMakeLists.txt"))) {
    Write-Host "Initializing the RandBLAS submodule..."
    Invoke-Checked "git" @("-C", $sourceRoot, "submodule", "update", "--init", "--recursive")
}

if ($ProjectDir -eq "") {
    $ProjectDir = Join-Path (Split-Path $sourceRoot -Parent) "RandNLA-project"
}
$ProjectDir = [System.IO.Path]::GetFullPath($ProjectDir)
if ($DependencyRoot -eq "") {
    $DependencyRoot = Join-Path $ProjectDir "install"
}
$dependencyRoot = [System.IO.Path]::GetFullPath($DependencyRoot)
$buildDir = Join-Path $ProjectDir "build\RandLAPACK-build"
$installDir = Join-Path $ProjectDir "install\RandLAPACK-install"

Write-Host ""
Write-Host "RandLAPACK Windows install"
Write-Host "  source:       $sourceRoot"
Write-Host "  project dir:  $ProjectDir"
Write-Host ""

# Step 1: dependencies (idempotent; reused when already present).
& (Join-Path $sourceRoot ".github\actions\setup-randlapack-deps-windows\setup.ps1") `
    -DependencyRoot $dependencyRoot -Backend $Backend -MklRoot $MklRoot `
    -BlasLibraries $BlasLibraries -LapackLibraries $LapackLibraries `
    -BackendBinDir $BackendBinDir -BlasInt $BlasInt -BlasFortran $BlasFortran

# Step 2: RandLAPACK itself.
if ($Fresh -and (Test-Path $buildDir)) {
    Write-Host "Removing previous build directory (-Fresh)..."
    Remove-Item -Recurse -Force $buildDir
}

$stageDllDirs = if ($env:RANDNLA_BLAS_BIN) { $env:RANDNLA_BLAS_BIN.Replace('\', '/') } else { "" }
Invoke-Checked "cmake" @(
    "-S", $sourceRoot, "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$($installDir.Replace('\', '/'))",
    "-Dblaspp_DIR=$env:blaspp_DIR",
    "-Dlapackpp_DIR=$env:lapackpp_DIR",
    "-DRandom123_DIR=$env:Random123_DIR",
    "-DCMAKE_PREFIX_PATH=$env:googletest_PREFIX",
    "-DRANDLAPACK_RUNTIME_DLL_DIRS=$stageDllDirs",
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
if ($env:RANDNLA_BLAS_BIN) {
    Write-Host "Runtime DLLs from $env:RANDNLA_BLAS_BIN are staged next to RandLAPACK's"
    Write-Host "test and benchmark executables automatically -- no PATH changes needed."
    Write-Host "For your own executables: find_package(RandLAPACK), then call"
    Write-Host "randlapack_stage_runtime_dlls(<your_target>) in your CMakeLists, or copy"
    Write-Host "the DLLs from that directory beside your .exe."
}
