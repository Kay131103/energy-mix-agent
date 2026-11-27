# Waits for the currently-running benchmark eval to finish, then archives the
# zero-reward GRPO run and relaunches the pipeline (GRPO v2 with the fixed grader).
$log = "runs\watcher.log"
"Watcher started $(Get-Date)" | Out-File $log -Append -Encoding utf8

# 1) wait until no python training/eval process is alive (check every 5 min)
while ($true) {
    $busy = Get-Process python -ErrorAction SilentlyContinue
    if (-not $busy) { break }
    Start-Sleep -Seconds 300
}
"Eval finished or stopped $(Get-Date)" | Out-File $log -Append -Encoding utf8

# 2) archive the no-op GRPO adapter so the bat retrains it
if (Test-Path runs\grpo_local\adapter) {
    Move-Item runs\grpo_local ("runs\grpo_local_zero_reward_" + (Get-Date -Format "MMdd_HHmm")) -Force
    "Archived zero-reward grpo run" | Out-File $log -Append -Encoding utf8
}

# 3) relaunch the full pipeline (skips SFT, reruns GRPO, reruns eval)
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "scripts\run_local_pipeline.bat" `
    -WorkingDirectory (Split-Path $PSScriptRoot -Parent) -WindowStyle Hidden
"Pipeline relaunched $(Get-Date)" | Out-File $log -Append -Encoding utf8
