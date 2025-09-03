# overlay_bronze.py — compact overlay for Bronze
def _C(a,r,g,b): return (a,r,g,b)

def bar(r, x, y, w, h, frac, fg=None, bg=None):
    fg = fg or _C(255, 88,180,255)
    bg = bg or _C(255, 30,30,30)
    frac = max(0.0, min(1.0, float(frac)))
    r.draw_rect_2d(x, y, w, h, bg)
    r.draw_rect_2d(x, y, int(w*frac), h, fg)

def draw_overlay(r, *, x=12, y=16, intent="...", reasons="", boost=33.0, action8=None, dz=False):
    if r is None: return
    try:
        r.begin_rendering("DestroyerBronzeHUD")
        r.draw_rect_2d(x-6, y-6, 340, 86, _C(200, 10,10,10))
        r.draw_string_2d(x, y, 1, 1, "Destroyer • Bronze", r.white()); y += 16
        r.draw_string_2d(x, y, 1, 1, f"Intent: {intent}", r.yellow()); y += 14
        if reasons:
            r.draw_string_2d(x, y, 1, 1, f"Why: {reasons}", r.grey()); y += 14
        if dz:
            r.draw_string_2d(x, y, 1, 1, "DANGER ZONE: clearing to corner", r.red()); y += 14
        r.draw_string_2d(x, y, 1, 1, "Boost", r.white())
        bar(r, x+44, y+2, 120, 8, float(boost)/100.0); y += 14
        if action8 is not None:
            a = list(action8) + [0]*8
            r.draw_string_2d(x, y, 1, 1, f"Steer {a[0]:+.2f}  Throt {a[1]:+.2f}", r.cyan())
        r.end_rendering()
    except Exception:
        pass
