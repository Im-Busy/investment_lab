# AI Instructions - Command Documentation

## IMPORTANT FOR ALL AI ASSISTANTS

When you complete tasks that involve running commands, scripts, or workflows:

### Required Actions

1. **Document Commands Immediately**
   - If you run a command that successfully executes, add it to `.useful_commands/` directory
   - Create a new `.txt` file or update an existing one based on category
   - Include a brief description explaining what the command does

2. **File Locations**
   - Global commands: `.useful_commands/[category]_commands.txt`
   - Module-specific commands: `src/[module]/AI_COMMANDS.txt`

3. **Documentation Format**
   ```
   ## Category/Feature Name
   
   # Description of what the command does
   actual-command-here
   ```

4. **Check Existing Files First**
   - Before creating a new file, check if relevant commands already exist
   - Browse `.useful_commands/` or the specific `src/[module]/` directory

### Command Categories

- **ML Training**: `ml_training_commands.txt` or `src/ml/AI_COMMANDS.txt`
- **Backtesting**: `backtest_commands.txt` or `src/backtest/AI_COMMANDS.txt`
- **Feature Extraction**: `src/features/AI_COMMANDS.txt`
- **Pattern Detection**: `src/patterns/AI_COMMANDS.txt`
- **Risk Analysis**: `src/risk/AI_COMMANDS.txt`
- **Data Processing**: `src/data_ingestion/AI_COMMANDS.txt`
- **Visualization**: `src/visualization/AI_COMMANDS.txt`
- **Utilities**: `src/utils/AI_COMMANDS.txt`

### Example Entry

```
## Train ML Model

# Train default LightGBM classifier on SPY
uv run scripts/train_ml_model.py --symbol SPY --start 2015-01-01 --end 2024-12-31

# Train with XGBoost instead
uv run scripts/train_ml_model.py --symbol SPY --model-type xgboost
```

---

**Remember**: Future AI sessions and human developers will use these command references. 
Keep them current, accurate, and well-organized.
