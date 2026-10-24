# Capture app screenshots for the paper with Playwright driving the installed Microsoft Edge.
#
# Start the app first (python app/app.py --no-browser), then:
#   python scripts/capture_screenshots.py [--url http://127.0.0.1:7860] [--example TEXT]
# Needs `pip install playwright`; no browser download is required because Edge is used.
# Paper crops are taken from element positions, so they follow layout changes.

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
LONG = 15 * 60 * 1000


# Bounding boxes are relative to the scrolled viewport, while a full-page clip uses page
# coordinates, so the current scroll offset is added back.
def clip_between(page, top_locator, bottom_locator, pad=12):
    scroll = page.evaluate("window.scrollY")
    top = top_locator.bounding_box()
    bottom = bottom_locator.bounding_box()
    width = page.viewport_size["width"]
    y0 = max(0, top["y"] + scroll - pad)
    return {"x": 0, "y": y0, "width": width, "height": bottom["y"] + scroll + bottom["height"] + pad - y0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--example", default="sigir-20142 · NCT02618655")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5,
                                color_scheme="light")
        page.goto(f"{args.url}/?__theme=light", wait_until="load")
        page.get_by_role("button", name="Run screening").wait_for(timeout=60000)

        # Screening tab: load a validation example and run it.
        page.get_by_label("Load a validation example (optional)").click()
        page.get_by_role("option").filter(has_text=args.example).first.click()
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Run screening").click()
        page.get_by_text("Screened ").wait_for(timeout=LONG)
        page.mouse.move(0, 0)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "app_screening.png"), full_page=True)
        page.screenshot(path=str(OUT / "app_screening_crop.png"), full_page=True,
                        clip=clip_between(page, page.locator(".tl-header"), page.locator(".tl-card").nth(2)))
        print("wrote app_screening.png and app_screening_crop.png")

        page.get_by_role("tab", name="Patient note with evidence").click()
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "app_note_evidence.png"), full_page=True)
        print("wrote app_note_evidence.png")

        # Inference lab. The first comparison loads and warms the draft model, so it is repeated
        # and the second is kept, the same warm-up rule the benchmark scripts follow.
        page.get_by_role("tab", name="Inference lab").click()
        page.wait_for_timeout(800)
        for _ in range(2):
            page.get_by_role("button", name="Run comparison").click()
            page.wait_for_timeout(2000)
            page.get_by_text("Every mode is checked token by token").wait_for(timeout=LONG)
            page.wait_for_function("() => !document.querySelector('.progress-text, .eta-bar')", timeout=LONG)
        page.get_by_role("button", name="Check attention").click()
        page.get_by_text("of float64").wait_for(timeout=LONG)
        page.mouse.move(0, 0)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "app_inference_lab.png"), full_page=True)
        # Gradio draws the table header and body separately, so the crop runs from the status panel
        # down to just above the generated-text box that follows the table.
        status = page.locator(".tl-status").filter(has_text="Every mode is checked").first
        below = page.get_by_text("Generated text (KV cache)").first
        clip = clip_between(page, status, below)
        clip["height"] -= below.bounding_box()["height"] + 24
        page.screenshot(path=str(OUT / "app_lab_crop.png"), full_page=True, clip=clip)
        print("wrote app_inference_lab.png and app_lab_crop.png")
        browser.close()


if __name__ == "__main__":
    main()
