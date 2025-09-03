# hud_overlay.py — RLBot renderer overlay for stage / intent / progress
from typing import Dict

BAR_W = 220
BAR_H = 10

def draw_hud(renderer, x, y, stage_name: str, intent: str, reasons: str, progress: float, samples: Dict[str, float]):
    try:
        renderer.draw_string_2d(x, y, 1, 1, f"[Destroyer Academy]  Stage: {stage_name}", renderer.white()); y += 16
        renderer.draw_string_2d(x, y, 1, 1, f"Intent: {intent}  |  {reasons}", renderer.yellow()); y += 18
        # Progress bar
        pct = max(0.0, min(1.0, progress))
        renderer.draw_rect_2d(x, y, BAR_W, BAR_H, renderer.grey())
        renderer.draw_rect_2d(x, y, int(BAR_W * pct), BAR_H, renderer.green())
        renderer.draw_string_2d(x + BAR_W + 8, y-2, 1, 1, f"{int(pct*100)}%", renderer.green()); y += BAR_H + 12
        # Show up to 4 sample metrics
        shown = 0
        for k, v in samples.items():
            renderer.draw_string_2d(x, y, 1, 1, f"{k}: {v:.2f}", renderer.cyan()); y += 14
            shown += 1
            if shown >= 4: break
    except Exception:
        pass
