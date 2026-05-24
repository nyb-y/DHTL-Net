cd /d %~dp0\..\..

python tools\analysis\aggregate_results.py ^
  --runs_root runs_binary_iqa ^
  --out_dir docs\results\latest ^
  --loss_mode train

python tools\analysis\aggregate_resource_combo.py ^
  --summary_dir docs\results\latest ^
  --out_dir docs\figures\resource_combo
