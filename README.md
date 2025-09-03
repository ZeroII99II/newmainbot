# Destroyer (Bronze-only)

A minimal RLBot bot focused on Bronze 1v1 basics. No models, no trainers — just clean heuristics:
- Simple kickoff (front flip), shadowing, corner clears, push shots.
- Tiny HUD for intent/boost/controls.

## Run
1. Open **RLBotGUI**.
2. **Load Match Config** → select `rlbot.cfg` in this folder.
3. Click **Run**.

If RLBot prompts for packages, let it install `rlbot` and `numpy`.

## Files
- `bot.py` – main bot (heuristics only)
- `awareness_bronze.py` – context & intent
- `decision_bronze.py` – clamps for intents
- `drills_bronze.py` – simple state-setting spawns (requires `enable_state_setting=True`)
- `overlay_bronze.py` – small HUD overlay
- `destroyer.cfg`, `rlbot.cfg`, `requirements.txt`
