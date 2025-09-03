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

## Training & Resume
- Config: `train.cfg`
- Latest model: `models/destroyer_Bronze_latest.pt` (+ `.opt` and `*_meta.json`)
- Rotating history: `checkpoints/`
- **No torch installed?** The bot runs heuristics and logs metadata; once you install torch, it will start saving `.pt` files and using the trainer. Set `use_model_for_action=true` to let the policy drive instead of heuristics.

To resume after closing RLBot, just start the match again — the trainer auto-loads `*_latest.pt`.
