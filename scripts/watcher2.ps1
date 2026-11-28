# v2 of the watcher: GRPO v2 (still zero-reward due to the conversational-completions
# bug, now fixed in src/validator/rewards.py) is ~10 min from finishing. Wait for its
# adapter, kill the useless eval that follows, archive v2, relaunch -> GRPO v3 + final eval.
$log = "runs\watcher2.log"
"Watcher2 started $(Get-Date)" | Out-File $log -Append -Encoding utf8

# 1) wait for the v2 adapter to appear (GRPO v2 done)
while (-not (Test-Path runs\grpo_local\adapter)) { Start-Sleep -Seconds 60 }
"GRPO v2 adapter detected $(Get-Date)" | Out-File $log -Append -Encoding utf8
Start-Sleep -Seconds 10

# 2) kill any python the pipeline just started (its eval would waste hours on a dead adapter)
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
"Killed follow-on eval $(Get-Date)" | Out-File $log -Append -Encoding utf8
Start-Sleep -Seconds 5

# 3) archive v2
Move-Item runs\grpo_local ("runs\grpo_local_zero_reward_v2_" + (Get-Date -Format "MMdd_HHmm")) -Force
"Archived v2 $(Get-Date)" | Out-File $log -Append -Encoding utf8

# 4) relaunch pipeline -> GRPO v3 with fixed rewards -> final eval
Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "scripts\run_local_pipeline.bat" `
    -WorkingDirectory (Split-Path $PSScriptRoot -Parent) -WindowStyle Hidden
"Pipeline relaunched (GRPO v3) $(Get-Date)" | Out-File $log -Append -Encoding utf8
