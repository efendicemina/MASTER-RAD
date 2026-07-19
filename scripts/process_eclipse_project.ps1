param(
    [Parameter(Mandatory = $true)][int]$Order,
    [Parameter(Mandatory = $true)][string]$Project
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$python = ".\.venv\Scripts\python.exe"
$config = "configs\eclipse_full.yaml"
$auditRoot = "reports\dataset_audit"
$progressPath = Join-Path $auditRoot "processing_progress.csv"
$snapshotPath = Join-Path $auditRoot ("snapshots\{0:D2}_{1}" -f $Order, $Project)

$before = (Get-PSDrive C).Free / 1GB
if ($before -lt 80) {
    throw "Free disk is below 80 GB before $Project processing"
}
if (Test-Path -LiteralPath $snapshotPath) {
    throw "Non-overwriting snapshot already exists: $snapshotPath"
}
if (Test-Path -LiteralPath $progressPath) {
    $existing = Import-Csv $progressPath | Where-Object { $_.project -eq $Project }
    if ($existing) {
        throw "Progress already contains project $Project"
    }
}

& $python -m defect_classifier build-dataset --config $config --project $Project
& $python -m defect_classifier validate-project-output --config $config --project $Project
& $python -m defect_classifier build-dataset --config $config --project $Project
& $python -m defect_classifier validate-training-readiness --config $config

New-Item -ItemType Directory -Path $snapshotPath | Out-Null
$snapshotFiles = @(
    "dataset_manifest.csv",
    "severity_by_project.csv",
    "temporal_coverage.csv",
    "proposed_holdout_distribution.csv",
    "proposed_cv_fold_distribution.csv",
    "duplicate_summary.csv",
    "training_readiness.csv",
    "training_readiness.json",
    "training_readiness.md"
)
foreach ($file in $snapshotFiles) {
    Copy-Item -LiteralPath (Join-Path $auditRoot $file) -Destination $snapshotPath
}

$after = (Get-PSDrive C).Free / 1GB
if ($after -lt 80) {
    throw "Free disk is below 80 GB after $Project processing"
}
$manifest = (Get-Content "data\processed\eclipse_core\manifest.json" -Raw |
    ConvertFrom-Json) | Where-Object source_project -eq $Project
$readiness = Get-Content (Join-Path $auditRoot "training_readiness.json") -Raw |
    ConvertFrom-Json
$projectReadiness = $readiness.projects | Where-Object scope -eq $Project
$row = [pscustomobject]@{
    processing_order = $Order
    project = $Project
    source_size_bytes = $manifest.source_file_size
    raw_rows = $manifest.raw_row_count
    parquet_rows = $manifest.output_row_count
    parquet_size_bytes = $manifest.output_file_size
    parse_errors = $manifest.parse_error_count
    invalid_dates = $manifest.invalid_date_count
    eligible_rows = $projectReadiness.usable_rows
    duration_seconds = $manifest.processing_duration_seconds
    disk_free_before_gb = [math]::Round($before, 2)
    disk_free_after_gb = [math]::Round($after, 2)
    ingestion_status = $manifest.completion_status
    resumability_status = $manifest.resume_status
    readiness_status = $projectReadiness.status
    warnings = ($projectReadiness.reasons -join "; ")
}
if (Test-Path -LiteralPath $progressPath) {
    $row | Export-Csv $progressPath -Append -NoTypeInformation
} else {
    $row | Export-Csv $progressPath -NoTypeInformation
}
$row | Format-List
