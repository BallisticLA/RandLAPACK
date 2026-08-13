# RandLAPACK native Windows installer -- the companion to install.sh.
# Full guide, including how to get an x64 toolchain: INSTALL_WINDOWS.md
#
#   powershell -ExecutionPolicy Bypass -File .\install\install.ps1
#
# Builds or reuses the dependencies (a BLAS/LAPACK backend, GoogleTest,
# Random123, BLAS++, LAPACK++) under <ProjectDir>\install, then configures,
# builds, installs and tests RandLAPACK. Mirrors install.sh's layout in a
# sibling RandNLA-project directory. GPU support is not available on native
# Windows yet.
#
# Options:
#   -ProjectDir <path>  Where dependencies/builds/installs go. Defaults to
#                       $env:RANDNLA_PROJECT_DIR when set, otherwise
#                       ..\RandNLA-project next to the clone.
#   -Prefix <path>      Install RandLAPACK itself here instead of
#                       <ProjectDir>\install\RandLAPACK-install.
#   -ModifyEnvironment  Persist RANDNLA_PROJECT_DIR for your user account. The
#                       default touches nothing and prints the setx command.
#   -Backend <name>     mkl (default) | openblas | custom. See setup.ps1.
#   -MklRoot <path>     Use this oneMKL install instead of discovery.
#   -NoDownload         Fail rather than download a backend that was not
#                       found locally. The default fetches one into
#                       <ProjectDir>; nothing is installed system-wide.
#   -Yes                Skip interactive questions, taking each default.
#                       Already skipped when stdin is not a terminal.
#   -NoOpenMP           Build serially. The default enables OpenMP through
#                       MSVC's /openmp:llvm runtime.
#   -Fresh              Reconfigure RandLAPACK from scratch. Dependencies are
#                       always reused; delete <ProjectDir>\install to rebuild
#                       them.
#   -SkipTests          Do not run the test suite after building.
#   -BlasLibraries / -LapackLibraries / -BackendBinDir / -BlasInt / -BlasFortran
#                       Backend custom only; see setup.ps1's header.

[CmdletBinding()]
param(
    [string]$ProjectDir = "",
    [ValidateSet("mkl", "openblas", "custom")]
    [string]$Backend = "mkl",
    [string]$MklRoot = "",
    [switch]$NoDownload,
    [switch]$Yes,
    [switch]$NoOpenMP,
    [string]$BlasLibraries = "",
    [string]$LapackLibraries = "",
    [string]$BackendBinDir = "",
    [ValidateSet("lp64", "ilp64")]
    [string]$BlasInt = "lp64",
    [string]$BlasFortran = "",
    # Where the dependency stack lives (default: <ProjectDir>\install). CI
    # points this at its shared, cached dependency directory.
    [string]$DependencyRoot = "",
    # Install RandLAPACK itself here instead of <ProjectDir>\install\RandLAPACK-install.
    # Dependencies still go in the project directory. For an HPC module tree or
    # any other prefix a site wants to own.
    [string]$Prefix = "",
    # Persist RANDNLA_PROJECT_DIR for this user. Opt-in, mirroring install.sh's
    # --modify-rc: the default touches nothing and prints the setx line instead.
    [switch]$ModifyEnvironment,
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

# Architecture detection is shared with setup.ps1 rather than duplicated: the
# two scripts run independently (CI calls setup.ps1 alone, users call this),
# and two copies of a safety check drift.
$archHelper = Join-Path $sourceRoot ".github\scripts\windows\toolchain-arch.ps1"
if (-not (Test-Path $archHelper)) {
    throw "Missing $archHelper. This clone looks incomplete."
}
. $archHelper

# ------------------------------------------------------ preflight checks ----
# Catch every missing prerequisite up front, each with its fix, instead of
# failing later with a tool-specific error.
$preflightProblems = @()
if (-not (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    $preflightProblems += ("cl.exe (the MSVC compiler) is not on PATH. Open 'x64 Native Tools " +
        "Command Prompt for VS 2022' from the Start menu and run this script there. If Visual " +
        "Studio is not installed:`n    winget install Microsoft.VisualStudio.2022.Community " +
        '--override "--add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"')
} else {
    # RandLAPACK and every BLAS backend the installer provisions are 64-bit.
    # The plain 'Developer PowerShell/Command Prompt for VS 2022' entries
    # default to an x86 toolchain, whose linker silently rejects the x64
    # import libraries; the failure then surfaces much later as BLAS++
    # reporting "BLAS library not found", which points at the wrong thing.
    $archProblem = Get-ToolchainArchitectureProblem (Get-ClTargetArchitecture)
    if ($archProblem -ne "") { $preflightProblems += $archProblem }
}
foreach ($tool in @(
        @{ Name = "cmake.exe"; Hint = "CMake ships with the Visual Studio C++ workload; an x64 Native Tools Command Prompt puts it on PATH." },
        @{ Name = "ninja.exe"; Hint = "Ninja ships with the Visual Studio C++ workload; an x64 Native Tools Command Prompt puts it on PATH." },
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
if (-not (Test-Path (Join-Path $sourceRoot "RandBLAS\CMakeLists.txt"))) {
    Write-Host "Initializing the RandBLAS submodule..."
    Invoke-Checked "git" @("-C", $sourceRoot, "submodule", "update", "--init", "--recursive")
}

# Precedence matches install.sh exactly: the flag, then RANDNLA_PROJECT_DIR,
# then a sibling of this clone. Honouring the environment variable is what lets
# this installer and RandBLAS's use the same project directory, so a machine
# that has already installed one does not scatter a second tree elsewhere.
if ($ProjectDir -eq "") {
    if ($env:RANDNLA_PROJECT_DIR) {
        $ProjectDir = $env:RANDNLA_PROJECT_DIR
    } else {
        $ProjectDir = Join-Path (Split-Path $sourceRoot -Parent) "RandNLA-project"
    }
}
$ProjectDir = [System.IO.Path]::GetFullPath($ProjectDir)
if ($DependencyRoot -eq "") {
    $DependencyRoot = Join-Path $ProjectDir "install"
}
# Checked after resolution rather than before: previously this only fired for an
# explicitly-passed -ProjectDir, so a long *default* path -- the common case,
# since it is derived from wherever the clone happens to sit -- went unwarned.
if ($ProjectDir.Length -gt 150) {
    Write-Warning ("The project directory path is $($ProjectDir.Length) characters long; deep " +
        "dependency build paths may exceed Windows' 260-character limit. Prefer a shorter " +
        "location, such as C:\RandNLA, via -ProjectDir.")
}
$dependencyRoot = [System.IO.Path]::GetFullPath($DependencyRoot)
$buildDir = Join-Path $ProjectDir "build\RandLAPACK-build"
$installDir = if ($Prefix) {
    [System.IO.Path]::GetFullPath($Prefix)
} else {
    Join-Path $ProjectDir "install\RandLAPACK-install"
}

Write-Host ""
Write-Host "RandLAPACK Windows install"
Write-Host "  source:       $sourceRoot"
Write-Host "  project dir:  $ProjectDir"
Write-Host ""

# Step 1: dependencies (idempotent; reused when already present).
& (Join-Path $sourceRoot ".github\actions\setup-randlapack-deps-windows\setup.ps1") `
    -DependencyRoot $dependencyRoot -Backend $Backend -MklRoot $MklRoot `
    -NoDownload:$NoDownload -Yes:$Yes `
    -BlasLibraries $BlasLibraries -LapackLibraries $LapackLibraries `
    -BackendBinDir $BackendBinDir -BlasInt $BlasInt -BlasFortran $BlasFortran

# Step 2: RandLAPACK itself.
if ($Fresh -and (Test-Path $buildDir)) {
    Write-Host "Removing previous build directory (-Fresh)..."
    Remove-Item -Recurse -Force $buildDir
}

$stageDllDirs = if ($env:RANDNLA_BLAS_BIN) { $env:RANDNLA_BLAS_BIN.Replace('\', '/') } else { "" }
$configureArgs = @(
    "-S", $sourceRoot, "-B", $buildDir,
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_INSTALL_PREFIX=$($installDir.Replace('\', '/'))",
    "-Dblaspp_DIR=$env:blaspp_DIR",
    "-Dlapackpp_DIR=$env:lapackpp_DIR",
    "-DRandom123_DIR=$env:Random123_DIR",
    "-DCMAKE_PREFIX_PATH=$env:googletest_PREFIX",
    "-DRANDLAPACK_RUNTIME_DLL_DIRS=$stageDllDirs")
# OpenMP is ON by default. RandLAPACK's CMake selects MSVC's /openmp:llvm
# runtime, the only mode that accepts its 64-bit loop indices and collapse
# clauses; core-windows exercises that configuration on every run. -NoOpenMP
# builds serially, which is also fully functional.
if ($NoOpenMP) { $configureArgs += "-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=TRUE" }
Invoke-Checked "cmake" $configureArgs
Invoke-Checked "cmake" @("--build", $buildDir, "--target", "install")

if (-not $SkipTests) {
    Invoke-Checked "ctest" @(
        "--test-dir", $buildDir,
        "--output-on-failure")
}

# Opt-in, mirroring install.sh's --modify-rc. SetEnvironmentVariable at User
# scope is the Windows equivalent of appending to a shell profile, and the only
# mechanism that survives opening a new shell -- setting $env: alone would last
# only for this process.
if ($ModifyEnvironment) {
    [Environment]::SetEnvironmentVariable("RANDNLA_PROJECT_DIR", $ProjectDir, "User")
    Write-Host ""
    Write-Host "Set RANDNLA_PROJECT_DIR=$ProjectDir for your user account."
    Write-Host "Open a new shell to pick it up."
}

Write-Host ""
Write-Host "RandLAPACK is installed."
Write-Host "  RandLAPACK_DIR: $installDir\lib\cmake\RandLAPACK"
Write-Host "  blaspp_DIR:     $env:blaspp_DIR"
Write-Host "  lapackpp_DIR:   $env:lapackpp_DIR"
Write-Host "  Random123_DIR:  $env:Random123_DIR"
Write-Host ""
if (-not $ModifyEnvironment) {
    Write-Host "To have RandNLA installers reuse this project directory by default, set:"
    Write-Host "    setx RANDNLA_PROJECT_DIR `"$ProjectDir`""
    Write-Host "(or re-run with -ModifyEnvironment)"
    Write-Host ""
}
if ($env:RANDNLA_BLAS_BIN) {
    Write-Host "Runtime DLLs from $env:RANDNLA_BLAS_BIN are staged next to RandLAPACK's"
    Write-Host "test and benchmark executables automatically -- no PATH changes needed."
    Write-Host "For your own executables: find_package(RandLAPACK), then call"
    Write-Host "randlapack_stage_runtime_dlls(<your_target>) in your CMakeLists, or copy"
    Write-Host "the DLLs from that directory beside your .exe."
}
