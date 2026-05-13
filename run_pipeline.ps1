# run_pipeline.ps1
# Semi-automated training pipeline for the Predictive Maintenance project

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host " Predictive Maintenance Training Pipeline"
Write-Host "========================================"
Write-Host ""

# Paths
$RawDataDir = "data/raw"
$RawDataPath = "data/raw/ai4i2020.csv"

$TempDir = "data/raw/temp_ai4i_download"
$ZipPath = "data/raw/ai4i2020.zip"

# UCI download source
$DatasetUrl = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

# Step 1: Check or download raw dataset
Write-Host "Step 1/4: Checking raw dataset..."

if (Test-Path $RawDataPath) {
    Write-Host "Raw dataset already exists at: $RawDataPath"
    Write-Host "Skipping download."
}
else {
    Write-Host "Raw dataset not found."
    Write-Host "Creating raw data folder if needed..."

    New-Item -ItemType Directory -Force -Path $RawDataDir | Out-Null

    Write-Host "Downloading AI4I 2020 dataset ZIP from UCI..."
    Write-Host "Source: $DatasetUrl"

    try {
        Invoke-WebRequest -Uri $DatasetUrl -OutFile $ZipPath
        Write-Host "ZIP downloaded successfully to: $ZipPath"
    }
    catch {
        Write-Host ""
        Write-Host "ERROR: Dataset download failed."
        Write-Host "The UCI download URL may have changed or may be temporarily unavailable."
        Write-Host ""
        Write-Host "Please download the AI4I 2020 dataset manually and place the CSV file here:"
        Write-Host "  $RawDataPath"
        Write-Host ""
        Write-Host "Then run this script again."
        exit 1
    }

    Write-Host "Extracting ZIP file..."

    if (Test-Path $TempDir) {
        Remove-Item $TempDir -Recurse -Force
    }

    New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

    try {
        Expand-Archive -Path $ZipPath -DestinationPath $TempDir -Force
    }
    catch {
        Write-Host ""
        Write-Host "ERROR: Could not extract the downloaded ZIP file."
        Write-Host "Please download the dataset manually and place the CSV file here:"
        Write-Host "  $RawDataPath"
        exit 1
    }

    Write-Host "Searching for CSV file inside the extracted ZIP..."

    $CsvFile = Get-ChildItem -Path $TempDir -Recurse -Filter "*.csv" | Select-Object -First 1

    if ($null -eq $CsvFile) {
        Write-Host ""
        Write-Host "ERROR: No CSV file was found inside the downloaded ZIP."
        Write-Host "Please download the dataset manually and place the CSV file here:"
        Write-Host "  $RawDataPath"
        exit 1
    }

    Write-Host "Found CSV file: $($CsvFile.FullName)"
    Write-Host "Moving CSV to expected project path: $RawDataPath"

    Move-Item -Path $CsvFile.FullName -Destination $RawDataPath -Force

    Write-Host "Cleaning temporary download files..."

    Remove-Item $TempDir -Recurse -Force
    Remove-Item $ZipPath -Force

    Write-Host "Dataset is ready at: $RawDataPath"
}

Write-Host ""

# Step 2: Create SQLite database
Write-Host "Step 2/4: Creating SQLite database..."
python src/create_database.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: src/create_database.py failed."
    exit $LASTEXITCODE
}

Write-Host ""

# Step 3: Create processed datasets and preprocessor artifact
Write-Host "Step 3/4: Creating processed datasets..."
python src/make_dataset.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: src/make_dataset.py failed."
    exit $LASTEXITCODE
}

Write-Host ""

# Step 4: Train and save model
Write-Host "Step 4/4: Training and saving model..."
python src/train_and_save_model.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: src/train_and_save_model.py failed."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "========================================"
Write-Host " Pipeline completed successfully!"
Write-Host "========================================"
Write-Host ""
Write-Host "Created/updated outputs:"
Write-Host "- data/database/predictive_maintenance.db"
Write-Host "- data/processed/X_train_prepared.csv"
Write-Host "- data/processed/X_test_prepared.csv"
Write-Host "- data/processed/y_train.csv"
Write-Host "- data/processed/y_test.csv"
Write-Host "- artifacts/model/preprocessor.joblib"
Write-Host "- artifacts/model/random_forest_model.joblib"
Write-Host "- artifacts/model/feature_names.json"
Write-Host "- artifacts/model/model_metadata.json"
Write-Host ""