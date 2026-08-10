# Configures, builds, installs, and tests RandLAPACK natively on Windows with
# MSVC and the selected BLAS/LAPACK backend (-Backend: mkl default, openblas,
# custom). Shared between GitHub CI and local runs.
#
# Local use, from an MSVC developer prompt in the repository root:
#   .github\scripts\windows\run-ci.ps1 -Task Core -SetupDependencies
#
# CI runs the setup-randlapack-deps-windows action first (for caching) and then
# invokes this script without -SetupDependencies.
#
# OpenMP on MSVC uses the /openmp:llvm runtime, selected by RandBLAS's build
# system (RandBLAS #184): it is the only MSVC mode that accepts RandLAPACK's
# 64-bit loop indices and collapse clauses. Pass -OpenMP to enable it;
# without the switch the build is serial (also fully functional --
# RandLAPACK guards all OpenMP use).

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Core")]
    [string]$Task,

    [string]$SourceRoot = "",

    [string]$WorkRoot = "",

    [string]$DependencyRoot = "",

    [ValidateSet("mkl", "openblas", "custom")]
    [string]$Backend = "mkl",

    [switch]$SetupDependencies,

    [switch]$OpenMP,

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
        -DependencyRoot $DependencyRoot -Backend $Backend -SanitizeAddress:$SanitizeAddress
}

$blasppDir = Require-EnvironmentVariable "blaspp_DIR"
$lapackppDir = Require-EnvironmentVariable "lapackpp_DIR"
$random123Dir = Require-EnvironmentVariable "Random123_DIR"
$gtestPrefix = Require-EnvironmentVariable "googletest_PREFIX"
$backendBin = [Environment]::GetEnvironmentVariable("RANDNLA_BLAS_BIN")
if (-not $backendBin) {
    # Legacy fallback for environments set up by an older setup.ps1: derive
    # the DLL directory from MKLROOT, accepting both oneAPI layouts.
    $mklRoot = [Environment]::GetEnvironmentVariable("MKLROOT")
    if ($mklRoot) {
        $backendBin = @("$($mklRoot.Replace('/', '\'))\bin",
                        "$($mklRoot.Replace('/', '\'))\redist\intel64") |
            Where-Object { Test-Path $_ } | Select-Object -First 1
    }
}
if (-not $backendBin) {
    throw "RANDNLA_BLAS_BIN is not set. Run setup.ps1 (or pass -SetupDependencies)."
}

# The BLAS backend enters through raw library paths recorded by BLAS++, so its
# DLLs are not covered by the TARGET_RUNTIME_DLLS staging. RANDLAPACK_RUNTIME_DLL_DIRS
# stages them beside RandLAPACK's executables; the process-PATH prepend below
# additionally covers the RandBLAS submodule's own executables until the
# RandBLAS-side staging lands (planned follow-up).
$env:PATH = "$backendBin;$env:PATH"

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
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    "-DRANDLAPACK_RUNTIME_DLL_DIRS=$($backendBin.Replace('\', '/'))",
    # The RandBLAS submodule's own tests are covered by RandBLAS's CI;
    # building and running them here roughly doubled the job time.
    "-DBUILD_TESTS=OFF"
)
if (-not $OpenMP) { $configureArgs += "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE" }
if ($SanitizeAddress) { $configureArgs += "-DSANITIZE_ADDRESS=ON" }

Invoke-Checked "cmake" $configureArgs
Invoke-Checked "cmake" @("--build", $buildDir, "--target", "install")

# Same exclusion as the Linux and macOS core jobs.
Invoke-Checked "ctest" @(
    "--test-dir", $buildDir,
    "--exclude-regex", "^TestABRIK\.ABRIK_catch_instability",
    "--output-on-failure")

Write-Host "RandLAPACK Windows $Task validation succeeded."
