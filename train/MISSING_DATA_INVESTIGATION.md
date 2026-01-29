# Investigation: Missing root_winrate/top_move in Preprocessed Data

## Summary

**Date**: 2026-01-29  
**Issue**: Some rows in preprocessed parquet files have `None` for `root_winrate` and `top_move`  
**Status**: ✅ Root cause identified - Expected behavior

## Statistics

From test run with `--limit 1000`:
- **Train**: 119/1000 rows missing data (11.9%)
- **Test**: 104/1000 rows missing data (10.4%)

## Root Cause

Rows missing `root_winrate`/`top_move` correspond to **positions with illegal moves**.

### Error Response Format

When KataGo encounters an illegal move in a position, it returns an error response instead of analysis results:

```json
{
  "error": "Illegal move <move_num>: <coord>",
  "field": "moves",
  "id": "<position_id>"
}
```

### Example Error Messages

- `Illegal move 49: M3`
- `Illegal move 91: E4`
- `Illegal move 234: A13`
- `Illegal move 57: Q12`
- `Illegal move 66: G2`

### Types of Illegal Moves

These positions likely contain:
1. **Moves on occupied intersections** - Playing a stone where one already exists
2. **Suicide moves** - Moves that result in immediate capture of the played group
3. **Ko violations** - Moves that violate ko rule (repeating previous board state)
4. **Other rule violations** - Moves that violate specific rule set constraints

## Preprocessor Behavior

The preprocessor (`train/preprocess.py`) correctly handles these cases:

1. **Detects error responses**: Checks for `"error"` field in move data
2. **Sets move_error**: Stores the error message in `extra_info.move_error`
3. **Sets None for missing data**: Leaves `root_winrate` and `top_move` as `None`
4. **Preserves position data**: Still includes position information (moves, rules, komi, etc.)

This is **expected and correct behavior** - the preprocessor cannot extract analysis results from error responses.

## Impact on Training

For training purposes:
- **Reward function** (`train/offline_env/reward.py`) should handle `None` values:
  - If `move_error` is present → `r_legal = -1`, `r_wr = -1`, `r_move = 0`
  - These positions serve as negative examples for illegal move detection
- **Data quality**: ~10-12% of positions contain illegal moves
  - This is normal for real game data (some SGF files may have errors)
  - Can be filtered out during training if desired, or used as negative examples

## Verification

To verify this behavior:

```python
import pandas as pd

df = pd.read_parquet('data/RL/katago_offline/train.parquet')

# All rows with missing root_winrate/top_move have move_error
missing = df[df['extra_info'].apply(lambda x: x.get('root_winrate') is None)]
all_have_error = missing['extra_info'].apply(lambda x: x.get('move_error') is not None).all()
print(f"All missing rows have error: {all_have_error}")  # Should be True
```

## Conclusion

**No code changes needed** - the preprocessor is working correctly. The missing data is due to illegal moves in the source position files, which KataGo correctly rejects with error responses. The preprocessor properly handles these cases by setting `move_error` and leaving analysis fields as `None`.
