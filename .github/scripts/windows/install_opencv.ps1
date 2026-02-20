$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$DepsRoot = Join-Path $RepoRoot ".deps"
$OcvVer   = "4.11.0"

$ZipPath  = Join-Path $DepsRoot "opencv-$OcvVer.zip"
$SrcDir   = Join-Path $DepsRoot "opencv-$OcvVer"
$BuildDir = Join-Path $DepsRoot "opencv-build-$OcvVer"
$InstDir  = Join-Path $DepsRoot "opencv-install-$OcvVer"

New-Item -ItemType Directory -Force -Path $DepsRoot | Out-Null

# Download OpenCV sources
if (-not (Test-Path $ZipPath)) {
  Write-Host "Downloading OpenCV $OcvVer..."
  Invoke-WebRequest -Uri "https://github.com/opencv/opencv/archive/$OcvVer.zip" -OutFile $ZipPath
}

# Extract
if (-not (Test-Path $SrcDir)) {
  Write-Host "Extracting..."
  Expand-Archive -Path $ZipPath -DestinationPath $DepsRoot -Force
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $InstDir  | Out-Null

# Configure (VS2022)
Write-Host "Configuring OpenCV..."
& cmake -S $SrcDir -B $BuildDir -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_INSTALL_PREFIX="$InstDir" `
  -DCMAKE_CONFIGURATION_TYPES="Debug;RelWithDebInfo" `
  -DINSTALL_CREATE_DISTRIB=ON `
  -DBUILD_LIST=features2d,highgui,flann,calib3d,imgcodecs `
  -DWITH_OPENEXR=ON `
  -DBUILD_EXAMPLES=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DBUILD_DOCS=OFF `
  -DWITH_CUDA=OFF

# Build + install both configs (your project uses RelWithDebInfo in CI)
Write-Host "Building+Installing OpenCV RelWithDebInfo..."
& cmake --build $BuildDir --config RelWithDebInfo --target INSTALL

Write-Host "Building+Installing OpenCV Debug..."
& cmake --build $BuildDir --config Debug --target INSTALL

# Find OpenCVConfig.cmake and write OpenCV_DIR
$cfg = Get-ChildItem -Path $InstDir -Recurse -Filter OpenCVConfig.cmake | Select-Object -First 1
if ($null -eq $cfg) { throw "OpenCVConfig.cmake not found under $InstDir" }

$OpenCV_DIR = $cfg.Directory.FullName
Set-Content -Path (Join-Path $DepsRoot "opencv_dir.txt") -Value $OpenCV_DIR -Encoding ASCII

# Also store bin path for runtime (opencv_world*.dll)
$dll = Get-ChildItem -Path $InstDir -Recurse -Filter "opencv_world*.dll" | Select-Object -First 1
if ($dll) {
  Set-Content -Path (Join-Path $DepsRoot "opencv_bin.txt") -Value $dll.Directory.FullName -Encoding ASCII
}

Write-Host "OpenCV_DIR = $OpenCV_DIR"