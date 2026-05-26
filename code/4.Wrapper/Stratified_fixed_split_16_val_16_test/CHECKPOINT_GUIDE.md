# Checkpoint and Resume Guide

## Overview

The experiment script now includes **automatic checkpoint and resume functionality** to handle long-running experiments safely. You can stop and restart the experiment at any time without losing progress.

## How It Works

### Automatic Saving

After **each feature combination** completes (all 13 n_estimators values), the script automatically saves:

1. **`fixed_split_results.csv`** - All results collected so far
2. **`model_registry.csv`** - Model metadata registry
3. **`checkpoint.json`** - Progress tracking information

### Resume on Restart

When you restart the script:

1. It checks for existing result files
2. Loads all previously completed work
3. Identifies which combination-estimator pairs are done
4. **Skips completed work** and continues from where it left off

## Usage

### Starting Fresh

```bash
python run_fixed_split_experiments.py
```

The script will:
- Check for existing progress (none found)
- Start from combination 1
- Save checkpoint after each combination

### Stopping the Experiment

You can stop the experiment at any time:
- Press `Ctrl+C` in the terminal
- Close the terminal window
- System crash/restart

**Your progress is safe!** The last completed combination is already saved.

### Resuming

Simply run the same command again:

```bash
python run_fixed_split_experiments.py
```

The script will:
- Detect existing progress
- Show how many combinations are already done
- Skip completed work
- Continue from the next incomplete combination

### Example Output

**First Run:**
```
Checking for existing progress...
  No existing progress found. Starting fresh.

Feature Combination 1/511
...
  ✓ Checkpoint saved: 13 results, 78 registry entries
Completed combination 1/511

Feature Combination 2/511
...
[User presses Ctrl+C]
```

**Resume Run:**
```
Checking for existing progress...
Found existing results file: fixed_split_results.csv
  Loaded 13 existing results
  Completed 13 combination-estimator pairs

⚠ RESUMING FROM CHECKPOINT
  Already completed: 13 combination-estimator pairs
  Will skip completed work and continue from where we left off

Feature Combination 1/511
    ⏭ Skipping n_estimators = 1 (already completed)
    ⏭ Skipping n_estimators = 10 (already completed)
    ...
  All n_estimators already completed for combination 1

Feature Combination 2/511
  Processing n_estimators = 1...
  [Continues from where it stopped]
```

## Files Created

### checkpoint.json

Tracks progress information:

```json
{
    "last_completed_combination": 5,
    "total_combinations": 511,
    "total_results": 65,
    "timestamp": "2026-02-24 20:30:15"
}
```

### fixed_split_results.csv

Contains all experimental results. Updated after each combination.

Columns:
- Feature Combination
- Combination ID
- Number of Features
- n_estimators
- Train/Validation/Test metrics (R2, MSE, MAE)
- Timestamps

### model_registry.csv

Tracks model metadata (models not saved to disk to save space).

## Safety Features

### Data Integrity

- Results are saved **atomically** after each combination
- If script crashes mid-combination, only that combination is lost
- Previous combinations are safe
- No partial/corrupted data in saved files

### Idempotent Execution

- Running the script multiple times is safe
- Already completed work is automatically skipped
- No duplicate results
- Consistent combination IDs

### Progress Tracking

- Clear console output shows what's being skipped
- Checkpoint file shows exact progress
- Easy to estimate remaining time

## Monitoring Progress

### Check Current Progress

```python
import pandas as pd
import json

# Load results
df = pd.read_csv('fixed_split_results.csv')
print(f"Total results: {len(df)}")
print(f"Unique combinations: {df['Combination ID'].nunique()}")
print(f"Unique n_estimators: {df['n_estimators'].nunique()}")

# Load checkpoint
with open('checkpoint.json', 'r') as f:
    checkpoint = json.load(f)
print(f"Last completed: {checkpoint['last_completed_combination']}/511")
print(f"Progress: {checkpoint['last_completed_combination']/511*100:.1f}%")
```

### Estimate Remaining Time

```python
import pandas as pd
from datetime import datetime

df = pd.read_csv('fixed_split_results.csv')

# Get timestamps
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d_%H%M%S')

# Calculate average time per combination
# (Each combination has 13 n_estimators values)
combinations_done = df['Combination ID'].nunique()
time_span = (df['Timestamp'].max() - df['Timestamp'].min()).total_seconds()
avg_time_per_combo = time_span / combinations_done if combinations_done > 1 else 0

# Estimate remaining
remaining_combos = 511 - combinations_done
estimated_hours = (remaining_combos * avg_time_per_combo) / 3600

print(f"Combinations completed: {combinations_done}/511")
print(f"Average time per combination: {avg_time_per_combo/60:.1f} minutes")
print(f"Estimated remaining time: {estimated_hours:.1f} hours")
```

## Best Practices

### 1. Regular Monitoring

Check progress periodically:
```bash
# View last few lines of results
tail fixed_split_results.csv

# Check checkpoint
cat checkpoint.json
```

### 2. Safe Stopping

- Wait for current combination to finish (shows "Checkpoint saved")
- Then press `Ctrl+C`
- Or just stop anytime - only current combination is lost

### 3. Backup (Optional)

For extra safety, periodically backup:
```bash
cp fixed_split_results.csv fixed_split_results_backup.csv
cp model_registry.csv model_registry_backup.csv
cp checkpoint.json checkpoint_backup.json
```

### 4. Multiple Sessions

You can run the experiment across multiple sessions:
- Day 1: Run for 8 hours, stop
- Day 2: Resume, run for 8 hours, stop
- Day 3: Resume, complete remaining work

## Troubleshooting

### "Results file corrupted"

If CSV file is corrupted:
1. Restore from backup (if available)
2. Or delete the corrupted file
3. Script will start from last good checkpoint

### "Duplicate results detected"

This shouldn't happen, but if it does:
```python
import pandas as pd

df = pd.read_csv('fixed_split_results.csv')
# Remove duplicates based on Combination ID and n_estimators
df_clean = df.drop_duplicates(subset=['Combination ID', 'n_estimators'], keep='last')
df_clean.to_csv('fixed_split_results.csv', index=False)
```

### "Want to restart from scratch"

Delete all checkpoint files:
```bash
rm fixed_split_results.csv
rm model_registry.csv
rm checkpoint.json
```

Then run the script again.

## Performance Tips

### Faster Testing

To test with fewer combinations, edit `run_fixed_split_experiments.py`:

```python
# Test with first 10 combinations only
for combo_idx, feature_combination in enumerate(all_combinations[:10]):
    ...

# Test with fewer n_estimators
estimator_values = [100, 500, 1000]  # Instead of all 13 values
```

### Parallel Execution (Advanced)

You could split the work across multiple machines:
- Machine 1: Combinations 1-170
- Machine 2: Combinations 171-340
- Machine 3: Combinations 341-511

Then merge results:
```python
import pandas as pd

df1 = pd.read_csv('machine1/fixed_split_results.csv')
df2 = pd.read_csv('machine2/fixed_split_results.csv')
df3 = pd.read_csv('machine3/fixed_split_results.csv')

df_combined = pd.concat([df1, df2, df3], ignore_index=True)
df_combined.to_csv('fixed_split_results_combined.csv', index=False)
```

## Summary

✅ **Automatic checkpointing** - No manual intervention needed
✅ **Safe to stop anytime** - Progress is saved
✅ **Automatic resume** - Just run the script again
✅ **No duplicate work** - Completed pairs are skipped
✅ **Progress tracking** - Easy to monitor status
✅ **Data integrity** - Atomic saves prevent corruption

You can now run long experiments with confidence!
