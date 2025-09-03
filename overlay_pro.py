# overlay_pro.py — Streamer-style HUD for Destroyer
# Draws a compact panel with: bot name, stage & progress bar, intent & reasons,
# last attempt label, boost bar, and live controls state.

from typing import Dict, Optional, Iterable

# Safe color helpers with fallbacks to RLBot's color API
def _C(r, g, b):  # opaque RGB
    return (255, r, g, b)  # RLBot renderer.create_color expects (a,r,g,b)

def _bar(renderer, x, y, w, h, frac, fg=None, bg=None, border=None):
    fg = fg or _C(70, 200, 120)
    bg = bg or _C(40, 40, 40)
    border = border or _C(90, 90, 90)
    frac = max(0.0, min(1.0, float(frac)))
    renderer.draw_rect_2d(x, y, w, h, bg)
    renderer.draw_rect_2d(x, y, int(w * frac), h, fg)
    # simple 1px border
    renderer.draw_line_2d(x, y, x+w, y, border)
    renderer.draw_line_2d(x, y+h, x+w, y+h, border)
    renderer.draw_line_2d(x, y, x, y+h, border)
    renderer.draw_line_2d(x+w, y, x+w, y+h, border)

def _pill(renderer, x, y, text, color):
    renderer.draw_rect_2d(x, y, 8 + 7*len(text), 16, _C(30,30,30))
    renderer.draw_string_2d(x+4, y+2, 1, 1, text, color)

def draw_overlay(
    renderer,
    *,
    x: int = 12,
    y: int = 16,
    bot_name: str = "Destroyer",
    stage: str = "Bronze",
    progress: float = 0.0,
    intent: str = "...",
    reasons: str = "",
    last_attempt: str = "",
    boost: float = 33.0,
    action8: Optional[Iterable[float]] = None,
    exploit: bool = False,
    danger: bool = False,
    extra: Optional[Dict] = None,
):
    """Draws the overlay. Call once per tick after you've chosen the action.
    - action8: [steer, throttle, pitch, yaw, roll, jump, boost, handbrake]
    """
    try:
        r = renderer
        if r is None:
            return
        # Begin a named render pass (prevents overlap if others draw too)
        r.begin_rendering("DestroyerOverlay")

        # Panel header
        r.draw_rect_2d(x-6, y-6, 360, 138, _C(10,10,10))
        r.draw_string_2d(x, y, 1, 1, f"{bot_name} • Stage: {stage}", r.white())
        y += 16

        # Stage progress bar
        _bar(r, x, y, 220, 10, progress, fg=_C(88,180,255))
        r.draw_string_2d(x+228, y-2, 1, 1, f"{int(progress*100)}%", r.cyan())
        y += 18

        # Intent + reasons (what we're trying to do)
        r.draw_string_2d(x, y, 1, 1, f"Intent: {intent}", r.yellow()); y += 14
        if reasons:
            r.draw_string_2d(x, y, 1, 1, f"Why: {reasons}", r.grey()); y += 16

        # Status pills
        px = x
        if exploit:
            _pill(r, px, y, "EXPLOIT", r.green()); px += 70
        if danger:
            _pill(r, px, y, "DANGER ZONE", r.red()); px += 110
        if last_attempt:
            _pill(r, px, y, f"LAST: {last_attempt}", r.orange()); px += 16 + 7*len(last_attempt)
        y += 20

        # Boost bar
        r.draw_string_2d(x, y, 1, 1, f"Boost", r.white())
        _bar(r, x+42, y+2, 120, 8, float(boost)/100.0, fg=_C(255,180,0))
        y += 16

        # Live controls (from action8)
        if action8 is not None:
            try:
                a = list(action8) + [0]*8
                steer, throttle, pitch, yaw, roll, jmp, bst, hbk = a[:8]
                # Steer / Throttle bars
                r.draw_string_2d(x, y, 1, 1, "Steer", r.white())
                _bar(r, x+42, y+2, 120, 8, (steer+1)/2, fg=_C(180,255,180)); y += 12
                r.draw_string_2d(x, y, 1, 1, "Throt", r.white())
                _bar(r, x+42, y+2, 120, 8, (throttle+1)/2, fg=_C(180,220,255)); y += 12
                # Buttons
                bx = x; by = y
                def _btn(lbl, on, col_on, col_off):
                    r.draw_rect_2d(bx, by, 56, 14, _C(22,22,22))
                    r.draw_string_2d(bx+4, by+2, 1, 1, lbl, col_on if on else col_off)
                _btn("JUMP", jmp > 0.5, r.green(), r.grey()); bx += 60
                _btn("BOOST", bst > 0.5, r.orange(), r.grey()); bx += 60
                _btn("HBRAKE", hbk > 0.5, r.purple(), r.grey())
            except Exception:
                pass

        r.end_rendering()
    except Exception:
        # Never let HUD crash the bot loop.
        pass
