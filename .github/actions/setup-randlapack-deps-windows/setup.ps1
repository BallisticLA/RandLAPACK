# Builds and installs RandLAPACK's native Windows dependencies:
#   - oneMKL (Intel's NuGet packages, ILP64 + sequential DLL set)
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

    # Use an existing oneMKL install (e.g. from the oneAPI installer) instead
    # of downloading Intel's NuGet packages. Must contain the ILP64 DLL
    # import libs.
    [string]$MklRoot = "",

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

function Convert-ToCMakePath {
    param([string]$Path)
    return $Path.Replace('\', '/')
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

# ---------------------------------------------------------------- guards ----

$resolvedRoot = [System.IO.Path]::GetFullPath($DependencyRoot)
if ($resolvedRoot -eq [System.IO.Path]::GetPathRoot($resolvedRoot)) {
    throw "DependencyRoot must not be a filesystem root."
}
New-Item -ItemType Directory -Force -Path $resolvedRoot | Out-Null

if (-not (Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    throw "cl.exe is not on PATH. Run from an MSVC developer environment (or ilammy/msvc-dev-cmd in CI)."
}

# ---------------------------------------------------------------- oneMKL ----

if ($MklRoot -ne "") {
    # Bring-your-own MKL (oneAPI installer layout: import libs in lib\ on
    # current releases, lib\intel64 on older ones).
    $mklRoot = [System.IO.Path]::GetFullPath($MklRoot)
    $mklLibDir = ""
    foreach ($candidate in @((Join-Path $mklRoot "lib"), (Join-Path $mklRoot "lib\intel64"))) {
        if (Test-Path (Join-Path $candidate "mkl_intel_ilp64_dll.lib")) {
            $mklLibDir = $candidate
            break
        }
    }
    if ($mklLibDir -eq "") {
        throw "-MklRoot $MklRoot does not contain mkl_intel_ilp64_dll.lib under lib\ or lib\intel64\."
    }
    $mklBin = @((Join-Path $mklRoot "bin"), (Join-Path $mklRoot "redist\intel64")) |
        Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $mklBin) { throw "-MklRoot $MklRoot has no bin\ (or redist\intel64\) DLL directory." }
    Write-Host "Using existing oneMKL at $mklRoot"
} else {
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
            Invoke-Checked "curl.exe" @("-fsSL", "--retry", "3", "-o", $archive,
                "https://api.nuget.org/v3-flatcontainer/$($package.Id)/$mklVersion/$($package.Id).$mklVersion.nupkg")
            $actual = (Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant()
            if ($actual -ne $package.Sha256) {
                throw "$($package.Id) $mklVersion hash mismatch: expected $($package.Sha256), got $actual."
            }
            Expand-Archive -Path $archive -DestinationPath (Join-Path $extractRoot $package.Id) -Force
            Remove-Item $archive
        }
        # Arrange the pieces into the oneAPI directory shape (lib\, include\,
        # bin\) that the -MklRoot path, the checks below, and RandBLAS's
        # MKL_sparse.cmake (MKLROOT/include) already expect.
        Move-Item (Join-Path $extractRoot "intelmkl.devel.win-x64\build\native\win-x64") $mklLibDir
        Move-Item (Join-Path $extractRoot "intelmkl.devel.win-x64\build\native\include") (Join-Path $mklRoot "include")
        Move-Item (Join-Path $extractRoot "intelmkl.redist.win-x64\runtimes\win-x64\native") $mklBin
        Remove-Item -Recurse -Force $extractRoot
    }
}

foreach ($required in @(
        $mklBin,
        (Join-Path $mklLibDir "mkl_intel_ilp64_dll.lib"),
        (Join-Path $mklLibDir "mkl_sequential_dll.lib"),
        (Join-Path $mklLibDir "mkl_core_dll.lib"))) {
    if (-not (Test-Path $required)) { throw "oneMKL install is missing $required." }
}
$env:PATH = "$mklBin;$env:PATH"

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
        "-S", $gtestSrc, "-B", $gtestBuild, "-G", "NMake Makefiles",
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
    Clone-Head "https://github.com/DEShawResearch/Random123.git" $random123Src
    New-Item -ItemType Directory -Force -Path (Join-Path $random123Install "include") | Out-Null
    Copy-Item -Recurse -Force (Join-Path $random123Src "include\Random123") `
        (Join-Path $random123Install "include\Random123")
}

# ----------------------------------------------------------------- BLAS++ ----

$blasppInstall = Join-Path $resolvedRoot "blaspp-install"
if (Test-Path $blasppInstall) {
    Write-Host "Reusing BLAS++ at $blasppInstall"
} else {
    $blasppSrc = Join-Path $resolvedRoot "blaspp-src"
    Clone-Head "https://github.com/BallisticLA/blaspp.git" $blasppSrc "remove-symv-debug-print"
    $blasppBuild = Join-Path $resolvedRoot "blaspp-build"
    # ILP64 + sequential is required: RandBLAS's MKL sparse backend static-asserts
    # that its int64_t sparse indices match sizeof(MKL_INT).
    $mklLibs = @(
        (Convert-ToCMakePath (Join-Path $mklLibDir "mkl_intel_ilp64_dll.lib")),
        (Convert-ToCMakePath (Join-Path $mklLibDir "mkl_sequential_dll.lib")),
        (Convert-ToCMakePath (Join-Path $mklLibDir "mkl_core_dll.lib"))) -join ';'
    Invoke-Checked "cmake" @(
        "-S", $blasppSrc, "-B", $blasppBuild, "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $blasppInstall)",
        "-DBUILD_SHARED_LIBS=ON",
        "-Duse_cmake_find_blas=false",
        "-DBLAS_LIBRARIES=$mklLibs",
        "-Dblas_int=ilp64",
        "-Dblas_threaded=false",
        "-Duse_openmp=false",
        "-Dgpu_backend=none",
        "-Dbuild_tests=OFF")
    Invoke-Checked "cmake" @("--build", $blasppBuild, "--target", "install")
}
$blasppDir = Find-PackageConfigDirectory $blasppInstall "blaspp"

# --------------------------------------------------------------- LAPACK++ ----

$lapackppInstall = Join-Path $resolvedRoot "lapackpp-install"
if (Test-Path $lapackppInstall) {
    Write-Host "Reusing LAPACK++ at $lapackppInstall"
} else {
    $lapackppSrc = Join-Path $resolvedRoot "lapackpp-src"
    Clone-Head "https://github.com/BallisticLA/lapackpp.git" $lapackppSrc "msvc-direct-includes"
    $lapackppBuild = Join-Path $resolvedRoot "lapackpp-build"
    Invoke-Checked "cmake" @(
        "-S", $lapackppSrc, "-B", $lapackppBuild, "-G", "NMake Makefiles",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=$(Convert-ToCMakePath $lapackppInstall)",
        "-Dblaspp_DIR=$(Convert-ToCMakePath $blasppDir)",
        "-DBUILD_SHARED_LIBS=ON",
        "-Dgpu_backend=none",
        "-Dbuild_tests=OFF")
    Invoke-Checked "cmake" @("--build", $lapackppBuild, "--target", "install")
}
$lapackppDir = Find-PackageConfigDirectory $lapackppInstall "lapackpp"

# ----------------------------------------------------------------- export ----

Export-GitHubValue "MKLROOT" (Convert-ToCMakePath $mklRoot)
Export-GitHubValue "MKL_BIN" $mklBin
Export-GitHubValue "googletest_PREFIX" (Convert-ToCMakePath $gtestInstall)
Export-GitHubValue "Random123_DIR" (Convert-ToCMakePath (Join-Path $random123Install "include"))
Export-GitHubValue "blaspp_DIR" (Convert-ToCMakePath $blasppDir)
Export-GitHubValue "lapackpp_DIR" (Convert-ToCMakePath $lapackppDir)
if ($env:GITHUB_PATH) { Add-Content -Path $env:GITHUB_PATH -Value $mklBin }

Write-Host "All native Windows dependencies are ready under $resolvedRoot"
