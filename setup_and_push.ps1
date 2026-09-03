$ErrorActionPreference = "Continue"

$repoName = "energy-curtailment-forecasting"
$token = "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"
$username = "satarabdus692-bot"
$headers = @{
    Authorization = "Bearer $token"
    Accept        = "application/vnd.github.v3+json"
    "User-Agent"  = "PowerShell"
}

# 1. Create GitHub Repository
Write-Host "Creating GitHub repository: $repoName..."
$repoBody = @{
    name        = $repoName
    description = "Grid Stress Forecaster - Predicting Renewable Energy Curtailment using Prophet, LSTM, and Stacking Ensemble"
    private     = $true
} | ConvertTo-Json

try {
    $repoRes = Invoke-RestMethod -Uri "https://api.github.com/user/repos" -Method Post -Headers $headers -Body $repoBody -ContentType "application/json"
    Write-Host "Repository created successfully: $($repoRes.html_url)"
} catch {
    Write-Host "Repo creation note (may already exist): $($_.Exception.Message)"
}

# 2. Configure Git and push
Write-Host "Configuring git..."
git config --global user.name "$username"
git config --global user.email "$username@users.noreply.github.com"

git init
git branch -M main

Write-Host "Adding all files..."
git add .

Write-Host "Committing files..."
git commit -m "Initial commit: Complete Grid Stress Forecaster pipeline (Prophet, LSTM, Ensemble, Dash app, SQL, EDA)"

$remoteUrl = "https://$($username):$($token)@github.com/$($username)/$($repoName).git"
git remote remove origin 2>$null
git remote add origin $remoteUrl

Write-Host "Pushing to GitHub..."
git push -u origin main --force

Write-Host "Push process finished."
