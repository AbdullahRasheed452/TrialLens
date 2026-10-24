# Capture app screenshots for the paper with Playwright driving the installed Microsoft Edge.
#
# Start the app first (python app/app.py --no-browser), then:
#   python scripts/capture_screenshots.py [--url http://127.0.0.1:7860]
# Needs `pip install playwright`; no browser download is required because Edge is used.

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    ap.add_argument("--example", default=None, help="text of the dropdown entry to load (default: first)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5,
                                color_scheme="light")
        page.goto(args.url, wait_until="networkidle")

        # Screening tab: load a validation example and run it.
        page.get_by_label("Load a validation example (optional)").click()
        options = page.get_by_role("option")
        (options.filter(has_text=args.example).first if args.example else options.first).click()
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="Run screening").click()
        page.get_by_text("Finished").wait_for(timeout=15 * 60 * 1000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "app_screening.png"), full_page=True)
        print("wrote app_screening.png")

        page.get_by_role("tab", name="Patient note with evidence").click()
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "app_note_evidence.png"), full_page=True)
        print("wrote app_note_evidence.png")

        # Inference lab: a short live comparison of all decoding modes.
        page.get_by_role("tab", name="Inference lab").click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="Run comparison").click()
        page.get_by_text("Every mode is checked token by token").wait_for(timeout=15 * 60 * 1000)
        page.get_by_role("button", name="Check attention").click()
        page.get_by_text("Errors are against a float64 computation").wait_for(timeout=5 * 60 * 1000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "app_inference_lab.png"), full_page=True)
        print("wrote app_inference_lab.png")
        browser.close()


if __name__ == "__main__":
    main()
