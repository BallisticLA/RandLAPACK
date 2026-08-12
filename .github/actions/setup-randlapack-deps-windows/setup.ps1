# Builds and installs RandLAPACK's native Windows dependencies:
#   - a BLAS/LAPACK backend (-Backend): oneMKL (default; discovered from an
#     installed oneAPI or fetched from Intel's NuGet packages, ILP64 +
#     sequential), OpenBLAS (official release binaries, LP64), or
#     custom/bring-your-own libraries
#   - GoogleTest v1.17.0
#   - Random123 (headers only)
#   - BLAS++  from BallisticLA/blaspp,  branch remove-symv-debug-print
#   - LAPACK++ from BallisticLA/lapackpp, branch msvc-direct-includes
#
# The BallisticLA branches carry the two one-line MSVC fixes that upstream
# icl-utk-edu has not merged yet (blaspp PR #132, lapackpp PR #87). Once those
# merge, both clones below can move back to upstream master.
#
# Every step is idempotent: work already present under -DependencyRoot (for
# example, restored from a CI cache) is left alone. Run from an MSVC developer
# environment (cl.exe on PATH).

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DependencyRoot,

    # BLAS/LAPACK backend. "mkl" (default): oneMKL, ILP64 + sequential,
    # discovered from an installed oneAPI, else downloaded (see -NoDownload).
    # "openblas": official OpenBLAS release binaries, LP64.
    # "custom": bring your own libraries via -BlasLibraries (anything
    # BLAS++/LAPACK++ can link, e.g. AMD AOCL).
    [ValidateSet("mkl", "openblas", "custom")]
    [string]$Backend = "mkl",

    # Refuse to download a backend that was not found locally, and fail
    # instead. The default is to download, which is the ordinary Windows
    # practice (no system prefix exists for third-party libraries, so
    # per-project acquisition via vcpkg/NuGet/release archives is the norm).
    # This switch is for users who want the stricter Linux/macOS behaviour,
    # where install.sh expects a system BLAS and errors without one.
    # Whatever is downloaded lands under <DependencyRoot>: project-local,
    # nothing installed system-wide, removed when that directory is deleted.
    # With -Backend openblas this always fails: OpenBLAS has no canonical
    # Windows location to discover, so there is nothing to fall back to and
    # the honest answer is to direct the user at -Backend custom.
    [switch]$NoDownload,

    # Skip interactive questions and take the documented default for each,
    # mirroring install.sh's -y/--yes. Prompts are already skipped whenever
    # stdin is not a terminal (CI, piped input), so this is only needed to
    # silence them in an interactive session.
    [switch]$Yes,

    # Use an existing oneMKL install (e.g. from the oneAPI installer) instead
    # of auto-discovery/download. Must contain the ILP64 DLL import libs.
    # Only meaningful with -Backend mkl.
    [string]$MklRoot = "",

    # -Backend custom only: semicolon-separated .lib paths handed verbatim
    # to BLAS++ (-BlasLibraries, required), optionally LAPACK++
    # (-LapackLibraries), the DLL directory for runtime staging
    # (-BackendBinDir), the BLAS integer size, and blaspp's blas_fortran
    # name-mangling hint (e.g. "add").
    [string]$BlasLibraries = "",
    [string]$LapackLibraries = "",
    [string]$BackendBinDir = "",
    [ValidateSet("lp64", "ilp64")]
    [string]$BlasInt = "lp64",
    [string]$BlasFortran = "",

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

# Architecture detection is shared with install/install.ps1 rather than
# duplicated; see that file's note. $PSScriptRoot is
# <repo>\.github\actions\setup-randlapack-deps-windows.
$archHelper = Join-Path $PSScriptRoot "..\..\scripts\windows\toolchain-arch.ps1"
if (-not (Test-Path $archHelper)) {
    throw "Missing $archHelper. This clone looks incomplete."
}
. $archHelper

function Convert-ToCMakePath {
    param([string]$Path)
    return $Path.Replace('\', '/')
}

# Prompts happen only on a terminal and only without -Yes, mirroring
# install.sh's INTERACTIVE flag. Without this gate, a question asked under
# GitHub Actions (or any piped stdin) blocks until the job times out.
$script:Interactive = -not $Yes -and -not [Console]::IsInputRedirected `
    -and [Environment]::UserInteractive

function Read-YesNo {
    # Returns $true/$false. Non-interactive callers get $Default without a
    # prompt, so every question must have a defensible unattended answer.
    param([string]$Question, [bool]$Default)
    if (-not $script:Interactive) { return $Default }
    $suffix = if ($Default) { "[Y/n]" } else { "[y/N]" }
    while ($true) {
        $reply = (Read-Host "$Question $suffix").Trim().ToLowerInvariant()
        if ($reply -eq "") { return $Default }
        if ($reply -in @("y", "yes")) { return $true }
        if ($reply -in @("n", "no")) { return $false }
        Write-Host "Please answer y or n."
    }
}

function Assert-SupportedToolchain {
    # Left unchecked, a non-x64 toolchain surfaces much later as BLAS++
    # reporting "BLAS library not found", which points at the wrong thing
    # entirely: the libraries are present and correct, the linker simply
    # cannot use them at that architecture.
    $problem = Get-ToolchainArchitectureProblem (Get-ClTargetArchitecture)
    if ($problem -ne "") { throw $problem }
}

function Find-PackageConfigDirectory {
    # Install layouts of blaspp/lapackpp vary between revisions; search for the
    # package config file instead of hardcoding lib/cmake/<name>.
    param([string]$InstallRoot, [string]$PackageName)
    $config = Get-ChildItem -Path $InstallRoot -Recurse -Filter "${PackageName}Config.cmake" |
        Select-Object -First 1
    if (-not $config) {
        throw "Could not find ${PackageName}Config.cmake under $InstallRoot."
    }
    return $config.DirectoryName
}

function Clone-Head {
    param([string]$Url, [string]$Destination, [string]$Branch = "")
    if (Test-Path $Destination) {
        Write-Host "Reusing existing clone at $Destination"
        return
    }
    $cloneArgs = @("clone", "--depth", "1")
    if ($Branch -ne "") { $cloneArgs += @("--branch", $Branch) }
    $cloneArgs += @($Url, $Destination)
    Invoke-Checked "git" $cloneArgs
}

function Export-GitHubValue {
    # Publishes a value as a process env var, and to GITHUB_ENV/GITHUB_OUTPUT
    # when running under GitHub Actions (harmless locally).
    param([string]$Name, [string]$Value)
    Set-Item -Path "Env:$Name" -Value $Value
    if ($env:GITHUB_ENV) { Add-Content -Path $env:GITHUB_ENV -Value "$Name=$Value" }
    if ($env:GITHUB_OUTPUT) {
        $outputName = $Name.ToLowerInvariant().Replace('_', '-')
        Add-Content -Path $env:GITHUB_OUTPUT -Value "$outputName=$Value"
    }
    Write-Host "$Name = $Value"
}

function Find-OneMklLayout {
    # Probes a oneMKL root for the ILP64 import libs (oneAPI layout: lib\ on
    # current releases, lib\intel64 on older ones) and a DLL directory
    # (bin\ since oneAPI 2024, redist\intel64 before).
    # Returns @{Root; LibDir; BinDir} or $null.
    param([string]$Root)
    if (-not $Root) { return $null }
    $resolved = [System.IO.Path]::GetFullPath($Root)
    foreach ($libDir in @((Join-Path $resolved "lib"), (Join-Path $resolved "lib\intel64"))) {
        if (-not (Test-Path (Join-Path $libDir "mkl_intel_ilp64_dll.lib"))) { continue }
        $binDir = @((Join-Path $resolved "bin"), (Join-Path $resolved "redist\intel64")) |
            Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($binDir) { return @{ Root = $resolved; LibDir = $libDir; BinDir = $binDir } }
    }
    return $null
}

function Test-BlasLinkage {
    # Compiles and runs a minimal dgemm_/dgesv_ caller against the given
    # import libraries: one clear pass/fail up front instead of a BLAS++
    # probe cascade later. Int64 selects the integer width (ILP64 vs LP64).
    param([string[]]$Libraries, [string]$DllDir, [bool]$Int64, [string]$ScratchDir)
    New-Item -ItemType Directory -Force -Path $ScratchDir | Out-Null
    $src = Join-Path $ScratchDir "blas_conftest.c"
    $intType = if ($Int64) { "long long" } else { "int" }
    @(
        '#include <stdio.h>',
        "typedef $intType blas_int;",
        'extern void dgemm_(const char*, const char*, const blas_int*, const blas_int*,',
        '                   const blas_int*, const double*, const double*, const blas_int*,',
        '                   const double*, const blas_int*, const double*, double*,',
        '                   const blas_int*);',
        'extern void dgesv_(const blas_int*, const blas_int*, double*, const blas_int*,',
        '                   blas_int*, double*, const blas_int*, blas_int*);',
        'int main(void) {',
        '    blas_int n = 2, one = 1, info = -1, ipiv[2];',
        '    double A[4] = {3, 1, 1, 2}, b[2] = {9, 8}, C[4], alpha = 1.0, beta = 0.0;',
        '    dgemm_("N", "N", &n, &n, &n, &alpha, A, &n, A, &n, &beta, C, &n);',
        '    dgesv_(&n, &one, A, &n, ipiv, b, &n, &info);',
        '    if (info != 0) { printf("dgesv_ info=%lld\n", (long long)info); return 1; }',
        '    if (b[0] < 1.9 || b[0] > 2.1 || b[1] < 2.9 || b[1] > 3.1) {',
        '        printf("wrong dgesv_ solution: %f %f\n", b[0], b[1]); return 2;',
        '    }',
        '    printf("BLAS/LAPACK link check OK\n");',
        '    return 0;',
        '}') | Set-Content -Path $src -Encoding ascii
    $exe = Join-Path $ScratchDir "blas_conftest.exe"
    Push-Location $ScratchDir
    try {
        & cl.exe /nologo $src "/Fe:$exe" /link @($Libraries) | Out-Host
        if ($LASTEXITCODE -ne 0) { return $false }
        $savedPath = $env:PATH
        if ($DllDir) { $env:PATH = "$DllDir;$env:PATH" }
        try {
            & $exe | Out-Host
            return ($LASTEXITCODE -eq 0)
        } finally {
            $env:PATH = $savedPath
        }
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------- guards ----

$resolvedRoot = [System.IO.Path]::GetFullPath($DependencyRoot)
if ($resolvedRoot -eq [System.IO.Path]::GetPathRoot($resolvedRoot)) {
    throw "DependencyRoot must not be a filesystem root."
}
New-Item -ItemType Directory -Force -Path $resolvedRoot | Out-Null

if (-not (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    throw "cl.exe is not on PATH. Run from an MSVC developer environment (or ilammy/msvc-dev-cmd in CI)."
}
Assert-SupportedToolchain

# ---------------------------------------------------- argument checks ----

if ($Backend -ne "mkl" -and $MklRoot -ne "") {
    throw "-MklRoot is only meaningful with -Backend mkl."
}
if ($Backend -eq "custom" -and $BlasLibraries -eq "") {
    throw "-Backend custom requires -BlasLibraries (semicolon-separated full paths to .lib files)."
}
if ($Backend -ne "custom") {
    foreach ($customOnly in @(
            @{ Name = "-BlasLibraries";   Value = $BlasLibraries },
            @{ Name = "-LapackLibraries"; Value = $LapackLibraries },
            @{ Name = "-BackendBinDir";   Value = $BackendBinDir },
            @{ Name = "-BlasFortran";     Value = $BlasFortran })) {
        if ($customOnly.Value -ne "") {
            throw "$($customOnly.Name) is only meaningful with -Backend custom."
        }
    }
}

# ----------------------------------------------- BLAS/LAPACK backend ----
# Every branch below must define: $backendLibraries (array of cmake-style
# .lib paths for BLAS++), $backendLapackLibraries ("" = let LAPACK++ resolve
# from the BLAS libs), $backendBlasInt, $backendBlasFortran ("" = unset),
# $backendBlasThreaded ("" = unset), and $backendBin (DLL directory; "" only
# for -Backend custom without -BackendBinDir).

if ($Backend -eq "mkl") {
    $mklLayout = $null
    if ($MklRoot -ne "") {
        $mklLayout = Find-OneMklLayout $MklRoot
        if (-not $mklLayout) {
            throw ("-MklRoot $MklRoot does not contain mkl_intel_ilp64_dll.lib under lib\ or " +
                "lib\intel64\ alongside a bin\ (or redist\intel64\) DLL directory.")
        }
        Write-Host "Using existing oneMKL at $($mklLayout.Root)"
    } else {
        # Discovery before download: a setvars.bat session exports MKLROOT;
        # some oneAPI installers set ONEAPI_ROOT persistently; and the
        # installer's default location is stable even when neither survives
        # into a fresh shell (modern installers set no persistent env vars).
        foreach ($candidate in @(
                $env:MKLROOT,
                $(if ($env:ONEAPI_ROOT) { Join-Path $env:ONEAPI_ROOT "mkl\latest" }),
                "C:\Program Files (x86)\Intel\oneAPI\mkl\latest")) {
            $mklLayout = Find-OneMklLayout $candidate
            if ($mklLayout) {
                Write-Host "Discovered existing oneMKL at $($mklLayout.Root)"
                break
            }
        }
    }
    # Not found: offer to provision it. Auto-provisioning is the default
    # answer because Windows has no system prefix for third-party libraries,
    # so per-project acquisition is normal practice rather than a workaround,
    # and it is what lets a bare machine install in one command. Asking first
    # keeps a 155 MB download from being a surprise, and states plainly where
    # it goes and that nothing touches the system.
    $provisionMkl = $false
    if (-not $mklLayout -and -not $NoDownload) {
        Write-Host ""
        Write-Host "No existing oneMKL found (checked -MklRoot, `$env:MKLROOT, `$env:ONEAPI_ROOT,"
        Write-Host "and C:\Program Files (x86)\Intel\oneAPI\mkl\latest)."
        Write-Host ""
        Write-Host "A pinned, checksum-verified copy (~155 MB) can be downloaded into"
        Write-Host "  $resolvedRoot"
        Write-Host "It is used only by this project: nothing is installed system-wide, no PATH"
        Write-Host "or registry changes, and deleting that directory removes it completely."
        Write-Host ""
        # Default yes: the unattended answer must keep CI and a one-command
        # install on a bare machine working, and this is the same choice
        # install.sh's ask() makes for its own prompts.
        $provisionMkl = Read-YesNo "Download oneMKL now?" $true
        if (-not $provisionMkl) { Write-Host "" }
    }
    if (-not $mklLayout -and -not $provisionMkl) {
        # Reached two ways: -NoDownload, or the question above answered no.
        # Both mean "no backend is available", so both get the same full list
        # of ways to supply one; only the reason differs.
        #
        # Probing the literal oneAPI path (rather than requiring MKLROOT) is
        # the polished Windows behaviour: that path IS canonical for oneMKL,
        # even though Windows has no general system prefix for third-party
        # libraries. See randnla/reference/windows-software-distribution.md.
        #
        # Details to the console, short throw -- the same shape install.ps1's
        # preflight uses. A long multi-line throw message gets echoed twice by
        # PowerShell (message, then FullyQualifiedErrorId) and buried in a
        # stack trace, which makes actionable guidance harder to read, not
        # easier.
        $mklRootShown = if ($env:MKLROOT) { $env:MKLROOT } else { "(not set)" }
        $oneApiShown = if ($env:ONEAPI_ROOT) { Join-Path $env:ONEAPI_ROOT "mkl\latest" } else { "(not set)" }
        Write-Host ""
        $why = if ($NoDownload) { "-NoDownload was given" } else { "the download was declined" }
        Write-Host "No oneMKL installation found, and $why."
        Write-Host ""
        Write-Host "  Looked in (in order):"
        Write-Host "    -MklRoot                                        (not given)"
        Write-Host "    `$env:MKLROOT                                    $mklRootShown"
        Write-Host "    `$env:ONEAPI_ROOT\mkl\latest                     $oneApiShown"
        Write-Host "    C:\Program Files (x86)\Intel\oneAPI\mkl\latest  (default oneAPI location)"
        Write-Host ""
        Write-Host "  A directory only counts if it holds mkl_intel_ilp64_dll.lib under lib\ or"
        Write-Host "  lib\intel64\ alongside a bin\ (or redist\intel64\) DLL directory, so a"
        Write-Host "  partial install is rejected rather than half-used."
        Write-Host ""
        Write-Host "  Pick one:"
        Write-Host "    1. Install oneMKL:            winget install --id Intel.oneMKL --exact"
        Write-Host "    2. Use a copy you already have:  -MklRoot `"<path to ...\mkl\latest>`""
        Write-Host "    3. Use OpenBLAS instead:      -Backend openblas"
        Write-Host "    4. Let the installer fetch a pinned oneMKL (~155 MB into"
        Write-Host "       $resolvedRoot; nothing installed system-wide):"
        Write-Host "       re-run and answer yes, or pass -Yes to skip the question."
        Write-Host ""
        throw "No BLAS/LAPACK backend available; see the options above."
    }
    if (-not $mklLayout) {
        # oneMKL comes straight from Intel's official NuGet packages -- plain
        # zip archives on nuget.org, pinned by version and SHA256. The devel
        # package carries the ILP64/sequential import libs and headers, the
        # redist package the runtime DLLs. This deliberately avoids vcpkg: its
        # Visual Studio-bundled distribution is manifest-only (no classic-mode
        # instance), and nothing here needed vcpkg beyond this one download.
        # The OpenMP/TBB packages the devel nuspec references are skipped on
        # purpose -- RandLAPACK links the sequential MKL DLL set.
        $mklVersion = "2025.2.0.627"
        $mklPackages = @(
            @{ Id = "intelmkl.devel.win-x64"
               Sha256 = "988816fb3cdfc5dcfdd42036c28314dcfda22fe47a29056ae455e360a8833ee5" },
            @{ Id = "intelmkl.redist.win-x64"
               Sha256 = "42bf35a13581aa03ecbee62e83e2c6397a45f13ae8aa657c1727fd0335e52c9e" })
        $mklRoot = Join-Path $resolvedRoot "onemkl-$mklVersion"
        $mklLibDir = Join-Path $mklRoot "lib"
        $mklBin = Join-Path $mklRoot "bin"
        if (Test-Path (Join-Path $mklLibDir "mkl_intel_ilp64_dll.lib")) {
            Write-Host "Reusing oneMKL at $mklRoot"
        } else {
            if (Test-Path $mklRoot) { Remove-Item -Recurse -Force $mklRoot }
            $extractRoot = Join-Path $mklRoot "extract"
            foreach ($package in $mklPackages) {
                $archive = Join-Path $resolvedRoot "$($package.Id).$mklVersion.zip"
                Invoke-Checked "curl.exe" @("-fsSL", "--retry", "5", "--retry-all-errors",
                    "--retry-delay", "3", "-o", $archive,
                    "https://api.nuget.org/v3-flatcontainer/$($package.Id)/$mklVersion/$($package.Id).$mklVersion.nupkg")
                $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
                if ($actual -ne $package.Sha256) {
                    throw "$($package.Id) $mklVersion hash mismatch: expected $($package.Sha256), got $actual."
                }
                Expand-Archive -Path $archive -DestinationPath (Join-Path $extractRoot $package.Id) -Force
                Remove-Item $archive
            }
            # Arrange the pieces into the oneAPI directory shape (lib\,
            # include\, bin\) that Find-OneMklLayout, the checks below, and
            # RandBLAS's MKL_sparse.cmake (MKLROOT/include) already expect.
            Move-Item (Join-Path $extractRoot "intelmkl.devel.win-x64\build\native\win-x64") $mklLibDir
            Move-Item (Join-Path $extractRoot "intelmkl.devel.win-x64\build\native\include") (Join-Path $mklRoot "include")
            Move-Item (Join-Path $extractRoot "intelmkl.redist.win-x64\runtimes\win-x64\native") $mklBin
            Remove-Item -Recurse -Force $extractRoot
        }
        $mklLayout = @{ Root = $mklRoot; LibDir = $mklLibDir; BinDir = $mklBin }
    }
    $mklRoot = $mklLayout.Root
    $mklLibDir = $mklLayout.LibDir
    $mklBin = $mklLayout.BinDir
    foreach ($required in @(
            (Join-Path $mklLibDir "mkl_intel_ilp64_dll.lib"),
            (Join-Path $mklLibDir "mkl_sequential_dll.lib"),
            (Join-Path $mklLibDir "mkl_core_dll.lib"))) {
        if (-not (Test-Path $required)) { throw "oneMKL install is missing $required." }
    }
    # Same link-and-run check the other two backends get: one clear pass/fail
    # here beats a BLAS++ probe cascade three layers down. It catches an
    # incomplete or mismatched oneMKL (and, belt-and-braces after
    # Assert-SupportedToolchain, any remaining bitness mismatch) before anything is
    # built.
    $mklLibs = @(
        (Join-Path $mklLibDir "mkl_intel_ilp64_dll.lib"),
        (Join-Path $mklLibDir "mkl_sequential_dll.lib"),
        (Join-Path $mklLibDir "mkl_core_dll.lib"))
    if (-not (Test-BlasLinkage -Libraries $mklLibs -DllDir $mklBin `
            -Int64 $true -ScratchDir (Join-Path $resolvedRoot "conftest-mkl"))) {
        throw ("oneMKL at $mklRoot failed a minimal ILP64 dgemm_/dgesv_ link-and-run check. " +
            "Verify that the install is complete and that its DLL directory ($mklBin) holds " +
            "the matching runtime DLLs.")
    }
    $backendLibraries = @($mklLibs | ForEach-Object { Convert-ToCMakePath $_ })
    $backendLapackLibraries = ""
    # ILP64 because RandBLAS's MKL sparse backend static-asserts that its
    # int64_t sparse indices match sizeof(MKL_INT).
    $backendBlasInt = "ilp64"
    $backendBlasFortran = ""
    $backendBlasThreaded = "false"
    $backendBin = $mklBin
} elseif ($Backend -eq "openblas") {
    # Official OpenBLAS release binaries: MinGW-built but self-contained
    # (the DLL imports only kernel32/msvcrt -- the MinGW runtimes are linked
    # in statically), MSVC-linkable through the shipped import library, and
    # full LAPACK is included. LP64: no ILP64 OpenBLAS binaries are
    # published for Windows. Without MKL, RandBLAS's MKL sparse
    # acceleration stays off and its portable fallbacks take over.
    #
    # Unlike oneMKL there is nothing to auto-discover: OpenBLAS has no
    # canonical Windows install location (GitHub release zips, vcpkg, conda
    # and MSYS2 all differ, and the release zips even ship CMake config
    # files with wrong hardcoded paths). So rather than probe and guess, ask
    # -- and only when someone is there to answer.
    if ($NoDownload) {
        # -NoDownload must mean what it says for every backend. There is no
        # OpenBLAS to discover, so the only honest outcome is to stop and
        # point at the backend that takes user-supplied libraries.
        Write-Host ""
        Write-Host "-Backend openblas needs to download OpenBLAS, and -NoDownload forbids that."
        Write-Host ""
        Write-Host "  OpenBLAS has no canonical install location on Windows, so unlike oneMKL"
        Write-Host "  there is nothing to discover. To use a copy you already have:"
        Write-Host ""
        Write-Host "    -Backend custom -BlasLibraries `"<path>\libopenblas.lib`" ``"
        Write-Host "      -BackendBinDir `"<directory holding libopenblas.dll>`" ``"
        Write-Host "      -BlasInt lp64 -BlasFortran add"
        Write-Host ""
        throw "-Backend openblas requires a download; use -Backend custom or drop -NoDownload."
    }
    if ($script:Interactive) {
        if (Read-YesNo "Do you already have OpenBLAS installed?" $false) {
            Write-Host ""
            Write-Host "OpenBLAS has no standard layout on Windows, so point at the pieces"
            Write-Host "directly rather than at a root directory. Re-run with:"
            Write-Host ""
            Write-Host "  -Backend custom ``"
            Write-Host "    -BlasLibraries `"<path>\libopenblas.lib`" ``"
            Write-Host "    -BackendBinDir `"<directory holding libopenblas.dll>`" ``"
            Write-Host "    -BlasInt lp64 -BlasFortran add"
            Write-Host ""
            Write-Host "Your libraries are checked with a real dgemm_/dgesv_ link-and-run"
            Write-Host "test before anything is built, so a wrong path fails immediately."
            Write-Host ""
            throw "Re-run with -Backend custom to use your own OpenBLAS (see above)."
        }
    }
    $openblasVersion = "0.3.34"
    $openblasSha256 = "e9cb6134541f36c27346d5fc5995652f060fba227cebbbabcbda5a5a44d7c76b"
    $openblasRoot = Join-Path $resolvedRoot "openblas-$openblasVersion"
    $openblasLib = Join-Path $openblasRoot "lib\libopenblas.lib"
    $openblasBin = Join-Path $openblasRoot "bin"
    if (Test-Path $openblasLib) {
        Write-Host "Reusing OpenBLAS at $openblasRoot"
    } else {
        if (Test-Path $openblasRoot) { Remove-Item -Recurse -Force $openblasRoot }
        $archive = Join-Path $resolvedRoot "OpenBLAS-$openblasVersion-x64.zip"
        Invoke-Checked "curl.exe" @("-fsSL", "--retry", "5", "--retry-all-errors",
            "--retry-delay", "3", "-o", $archive,
            "https://github.com/OpenMathLib/OpenBLAS/releases/download/v$openblasVersion/OpenBLAS-$openblasVersion-x64.zip")
        $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
        if ($actual -ne $openblasSha256) {
            throw "OpenBLAS $openblasVersion hash mismatch: expected $openblasSha256, got $actual."
        }
        Expand-Archive -Path $archive -DestinationPath $openblasRoot -Force
        Remove-Item $archive
    }
    $openblasConftest = Join-Path $resolvedRoot "conftest-openblas"
    if (-not (Test-BlasLinkage -Libraries @($openblasLib) -DllDir $openblasBin `
            -Int64 $false -ScratchDir $openblasConftest)) {
        # The shipped import library is occasionally unusable from MSVC;
        # regenerate it from the .def and retry once (self-healing when the
        # version pin moves).
        Write-Host "Shipped libopenblas.lib failed the link check; regenerating from libopenblas.def..."
        Invoke-Checked "lib.exe" @("/nologo", "/machine:x64",
            "/def:$(Join-Path $openblasRoot 'lib\libopenblas.def')", "/out:$openblasLib")
        if (-not (Test-BlasLinkage -Libraries @($openblasLib) -DllDir $openblasBin `
                -Int64 $false -ScratchDir $openblasConftest)) {
            throw "OpenBLAS link check failed even after import-library regeneration."
        }
    }
    $backendLibraries = @(Convert-ToCMakePath $openblasLib)
    # OpenBLAS bundles LAPACK in the same library.
    $backendLapackLibraries = Convert-ToCMakePath $openblasLib
    $backendBlasInt = "int32"
    $backendBlasFortran = "add"
    $backendBlasThreaded = ""
    $backendBin = $openblasBin
} else {
    # Bring-your-own backend (e.g. AMD AOCL, whose downloads are
    # click-through-gated and cannot be fetched here): the libraries are
    # handed to BLAS++/LAPACK++ verbatim after one clear preflight check.
    # The check calls dgemm_/dgesv_, i.e. it assumes the common
    # lowercase-underscore Fortran mangling (OpenBLAS, AOCL, MKL all
    # export it).
    $customLibs = @($BlasLibraries -split ";" | Where-Object { $_ -ne "" })
    foreach ($lib in $customLibs) {
        if (-not (Test-Path $lib)) { throw "-BlasLibraries entry not found: $lib" }
    }
    if ($BackendBinDir -ne "" -and -not (Test-Path $BackendBinDir)) {
        throw "-BackendBinDir not found: $BackendBinDir"
    }
    if (-not (Test-BlasLinkage -Libraries $customLibs -DllDir $BackendBinDir `
            -Int64 ($BlasInt -eq "ilp64") -ScratchDir (Join-Path $resolvedRoot "conftest-custom"))) {
        throw ("The libraries in -BlasLibraries failed a minimal dgemm_/dgesv_ link-and-run " +
            "check. Verify the paths, the integer size (-BlasInt), and that their runtime " +
            "DLLs are in -BackendBinDir.")
    }
    $backendLibraries = @($customLibs | ForEach-Object { Convert-ToCMakePath $_ })
    $backendLapackLibraries = (@($LapackLibraries -split ";" | Where-Object { $_ -ne "" } |
        ForEach-Object { Convert-ToCMakePath $_ })) -join ";"
    $backendBlasInt = if ($BlasInt -eq "ilp64") { "ilp64" } else { "int32" }
    $backendBlasFortran = $BlasFortran
    $backendBlasThreaded = ""
    $backendBin = $BackendBinDir
    if ($backendBin -eq "") {
        Write-Warning ("No -BackendBinDir given: the custom backend's runtime DLLs will not be " +
            "staged next to executables; making them findable at run time is up to you.")
    }
}

# Distinct BLAS++/LAPACK++ installs per backend; a custom backend is keyed
# by a hash of its library list so switching -BlasLibraries rebuilds.
$backendId = $Backend
if ($Backend -eq "custom") {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $hashHex = ($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($BlasLibraries)) |
        ForEach-Object { $_.ToString("x2") }) -join ""
    $backendId = "custom-$($hashHex.Substring(0, 8))"
}

if ($backendBin -ne "") { $env:PATH = "$backendBin;$env:PATH" }

# ------------------------------------------------------------- GoogleTest ----

$gtestVariant = if ($SanitizeAddress) { "googletest-asan" } else { "googletest" }
$gtestInstall = Join-Path $resolvedRoot "$gtestVariant-install"
if (Test-Path (Join-Path $gtestInstall "include\gtest\gtest.h")) {
    Write-Host "Reusing GoogleTest at $gtestInstall"
} else {
    $gtestSrc = Join-Path $resolvedRoot "$gtestVariant-src"
    Clone-Head "https://github.com/google/googletest.git" $gtestSrc "v1.17.0"
    $gtestBuild = Join-Path $resolvedRoot "$gtestVariant-build"
    $gtestArgs = @(
        "-S", $gtestSrc, "-B", $gtestBuild, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $gtestInstall)",
        "-DBUILD_GMOCK=OFF", "-DINSTALL_GTEST=ON")
    if ($SanitizeAddress) {
        # MSVC container-annotation state must agree across all linked objects,
        # so an ASan build needs an ASan-instrumented GoogleTest.
        $gtestArgs += "-DCMAKE_CXX_FLAGS=/fsanitize=address /Zi"
    }
    Invoke-Checked "cmake" $gtestArgs
    Invoke-Checked "cmake" @("--build", $gtestBuild, "--target", "install")
}

# -------------------------------------------------------------- Random123 ----

$random123Install = Join-Path $resolvedRoot "Random123-install"
if (Test-Path (Join-Path $random123Install "include\Random123\philox.h")) {
    Write-Host "Reusing Random123 at $random123Install"
} else {
    $random123Src = Join-Path $resolvedRoot "Random123-src"
    Clone-Head "https://github.com/DEShawResearch/Random123.git" $random123Src "v1.14.0"
    New-Item -ItemType Directory -Force -Path (Join-Path $random123Install "include") | Out-Null
    Copy-Item -Recurse -Force (Join-Path $random123Src "include\Random123") `
        (Join-Path $random123Install "include\Random123")
}

# ----------------------------------------------------------------- BLAS++ ----

$blasppInstall = Join-Path $resolvedRoot "blaspp-$backendId-install"
# Reuse only on an intact install: a partially restored cache directory must
# trigger a rebuild, not silently skip it.
$blasppReusable = (Test-Path $blasppInstall) -and (Get-ChildItem -Path $blasppInstall -Recurse `
    -Filter "blasppConfig.cmake" -ErrorAction SilentlyContinue | Select-Object -First 1)
if ($blasppReusable) {
    Write-Host "Reusing BLAS++ at $blasppInstall"
} else {
    $blasppSrc = Join-Path $resolvedRoot "blaspp-src"
    Clone-Head "https://github.com/BallisticLA/blaspp.git" $blasppSrc "remove-symv-debug-print"
    # Never re-configure an existing blaspp build in place: that regenerates
    # blas/defines.h without the backend defines. Fresh build tree per run.
    $blasppBuild = Join-Path $resolvedRoot "blaspp-$backendId-build"
    if (Test-Path $blasppBuild) { Remove-Item -Recurse -Force $blasppBuild }
    $blasppArgs = @(
        "-S", $blasppSrc, "-B", $blasppBuild, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $blasppInstall)",
        "-DBUILD_SHARED_LIBS=ON",
        "-Duse_cmake_find_blas=false",
        "-DBLAS_LIBRARIES=$($backendLibraries -join ';')",
        "-Dblas_int=$backendBlasInt",
        "-Duse_openmp=false",
        "-Dgpu_backend=none",
        "-Dbuild_tests=OFF")
    if ($backendBlasThreaded -ne "") { $blasppArgs += "-Dblas_threaded=$backendBlasThreaded" }
    if ($backendBlasFortran -ne "") { $blasppArgs += "-Dblas_fortran=$backendBlasFortran" }
    Invoke-Checked "cmake" $blasppArgs
    Invoke-Checked "cmake" @("--build", $blasppBuild, "--target", "install")
}
$blasppDir = Find-PackageConfigDirectory $blasppInstall "blaspp"

# --------------------------------------------------------------- LAPACK++ ----

$lapackppInstall = Join-Path $resolvedRoot "lapackpp-$backendId-install"
$lapackppReusable = (Test-Path $lapackppInstall) -and (Get-ChildItem -Path $lapackppInstall -Recurse `
    -Filter "lapackppConfig.cmake" -ErrorAction SilentlyContinue | Select-Object -First 1)
if ($lapackppReusable) {
    Write-Host "Reusing LAPACK++ at $lapackppInstall"
} else {
    $lapackppSrc = Join-Path $resolvedRoot "lapackpp-src"
    Clone-Head "https://github.com/BallisticLA/lapackpp.git" $lapackppSrc "msvc-direct-includes"
    $lapackppBuild = Join-Path $resolvedRoot "lapackpp-$backendId-build"
    if (Test-Path $lapackppBuild) { Remove-Item -Recurse -Force $lapackppBuild }
    $lapackppArgs = @(
        "-S", $lapackppSrc, "-B", $lapackppBuild, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $lapackppInstall)",
        "-Dblaspp_DIR=$(Convert-ToCMakePath $blasppDir)",
        "-DBUILD_SHARED_LIBS=ON",
        "-Dgpu_backend=none",
        "-Dbuild_tests=OFF")
    if ($backendLapackLibraries -ne "") {
        $lapackppArgs += "-DLAPACK_LIBRARIES=$backendLapackLibraries"
    }
    Invoke-Checked "cmake" $lapackppArgs
    Invoke-Checked "cmake" @("--build", $lapackppBuild, "--target", "install")
}
$lapackppDir = Find-PackageConfigDirectory $lapackppInstall "lapackpp"

# ----------------------------------------------------------------- export ----

Export-GitHubValue "RANDNLA_BLAS_BACKEND" $Backend
if ($backendBin -ne "") { Export-GitHubValue "RANDNLA_BLAS_BIN" $backendBin }
if ($Backend -eq "mkl") {
    # MKLROOT is what RandBLAS's MKL_sparse.cmake probes for mkl_spblas.h;
    # its absence on other backends is what turns the MKL sparse path off.
    Export-GitHubValue "MKLROOT" (Convert-ToCMakePath $mklRoot)
    Export-GitHubValue "MKL_BIN" $mklBin
}
Export-GitHubValue "googletest_PREFIX" (Convert-ToCMakePath $gtestInstall)
Export-GitHubValue "Random123_DIR" (Convert-ToCMakePath (Join-Path $random123Install "include"))
Export-GitHubValue "blaspp_DIR" (Convert-ToCMakePath $blasppDir)
Export-GitHubValue "lapackpp_DIR" (Convert-ToCMakePath $lapackppDir)
if ($env:GITHUB_PATH -and $backendBin -ne "") {
    Add-Content -Path $env:GITHUB_PATH -Value $backendBin
}

Write-Host "All native Windows dependencies are ready under $resolvedRoot"
