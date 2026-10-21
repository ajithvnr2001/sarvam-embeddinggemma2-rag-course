#!/usr/bin/env python3
"""
Start the FastAPI RAG Studio UI server on port 8090, verify all 4 interactive tabs
with headless Chromium via Playwright, and capture high-resolution screenshots.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = PROJECT_ROOT / "data" / "artifacts" / "screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_server(url: str, timeout_sec: float = 45.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Server at {url} did not become ready within {timeout_sec}s")


def main() -> None:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["HF_HUB_OFFLINE"] = "1"

    uvicorn_bin = str(PROJECT_ROOT / ".venv" / "bin" / "uvicorn")
    server_proc = subprocess.Popen(
        [uvicorn_bin, "ui.app:app", "--host", "127.0.0.1", "--port", "8090"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        wait_for_server("http://127.0.0.1:8090/api/stats", timeout_sec=45.0)
        print("[✓] FastAPI RAG Studio server is live at http://127.0.0.1:8090")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 940}, device_scale_factor=1.5)
            page = context.new_page()
            page.goto("http://127.0.0.1:8090/", wait_until="domcontentloaded")

            # Wait for all 4 live API calls on DOMContentLoaded to populate the UI
            page.wait_for_function(
                "() => !document.getElementById('rag-answer').textContent.includes('Retrieving') && !document.getElementById('rag-answer').textContent.includes('Loading')",
                timeout=90000,
            )
            page.wait_for_function(
                "() => document.getElementById('mrl-cards').children.length === 4",
                timeout=60000,
            )
            page.wait_for_function(
                "() => document.getElementById('mm-results').children.length > 0",
                timeout=60000,
            )
            page.wait_for_function(
                "() => document.getElementById('bq-results').innerHTML.includes('<table')",
                timeout=60000,
            )
            time.sleep(0.5)

            # Tab 1: Grounded RAG & Vertex AI Studio
            p1 = SCREENSHOT_DIR / "ui_tab1_rag_studio.png"
            page.screenshot(path=str(p1), full_page=True)
            print(f"[✓] Captured Tab 1 screenshot: {p1}")

            # Tab 2: Matryoshka (MRL) Live Explorer
            page.locator("button.tab-btn").nth(1).click()
            time.sleep(0.4)
            p2 = SCREENSHOT_DIR / "ui_tab2_mrl_explorer.png"
            page.screenshot(path=str(p2), full_page=True)
            print(f"[✓] Captured Tab 2 screenshot: {p2}")

            # Tab 3: Multimodal Lab (Image + Audio + Interleaved)
            page.locator("button.tab-btn").nth(2).click()
            time.sleep(0.6)
            p3 = SCREENSHOT_DIR / "ui_tab3_multimodal_lab.png"
            page.screenshot(path=str(p3), full_page=True)
            print(f"[✓] Captured Tab 3 screenshot: {p3}")

            # Tab 4: GCP BigQuery Vector Search & Cloud Console
            page.locator("button.tab-btn").nth(3).click()
            time.sleep(0.4)
            p4 = SCREENSHOT_DIR / "ui_tab4_gcp_bigquery.png"
            page.screenshot(path=str(p4), full_page=True)
            print(f"[✓] Captured Tab 4 screenshot: {p4}")

            browser.close()
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()


if __name__ == "__main__":
    main()
