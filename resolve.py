"""
resolver.py
Run: python resolver.py https://arolinks.com/I5gY8n
"""

import sys
import time
import traceback

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# ----------------------------------------------
# Logging helper
# ----------------------------------------------
def log(msg):
    print(msg)


# ----------------------------------------------
# Countdown handling
# ----------------------------------------------
def wait_for_countdown(driver):
    ids = ["ce-time", "timer"]   # common countdown ids

    log("⏳ Checking for countdown timer…")
    start = time.time()

    while time.time() - start < 40:
        found = False
        for cid in ids:
            try:
                el = driver.find_element(By.ID, cid)
                txt = el.text.strip()
                if txt.isdigit():
                    log(f"   Timer = {txt}")
                    if int(txt) <= 0:
                        log("   Countdown finished")
                        return
                found = True
            except:
                pass
        if not found:
            return
        time.sleep(1)


# ----------------------------------------------
# Click verify / continue
# ----------------------------------------------
def click_buttons(driver):
    log("🔍 Looking for Verify / Continue buttons…")

    # IDs first
    CLICK_MAP = [
        (By.ID, "btn6"),  # Verify
        (By.ID, "btn7"),  # Continue
    ]

    for by, value in CLICK_MAP:
        try:
            btn = driver.find_element(by, value)
            btn.click()
            log(f"✔️ Clicked button: {value}")
            time.sleep(1.2)
            return True
        except:
            pass

    # Try by text
    TEXTS = ["Verify", "Continue"]
    for txt in TEXTS:
        try:
            btn = driver.find_element(By.XPATH, f"//*[text()='{txt}']")
            btn.click()
            log(f"✔️ Clicked text button: {txt}")
            time.sleep(1.2)
            return True
        except:
            pass

    return False


# ----------------------------------------------
# Extract final link (Get Link button)
# ----------------------------------------------
def get_final_link(driver):
    log("🔍 Searching for final 'Get Link'…")

    try:
        el = driver.find_element(By.ID, "get-link")
        href = el.get_attribute("href")
        if href:
            return href
    except:
        pass

    # check all anchors
    for a in driver.find_elements(By.TAG_NAME, "a"):
        href = a.get_attribute("href") or ""
        if "telegram" in href or "http" in href:
            if "get" in a.text.lower():
                return href

    return None


# ----------------------------------------------
# Core Flow
# ----------------------------------------------
def resolve(url):
    log("🚀 Starting Selenium Chrome…")

    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    # REMOVE THIS LINE IF YOU WANT VISIBLE CHROME
    chrome_options.add_argument("--headless=new")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    current = url

    try:
        for step in range(12):
            log(f"\n===== STEP {step+1} =====")
            log(f"🌐 Opening: {current}")

            driver.get(current)
            time.sleep(2)

            # countdown waiting
            wait_for_countdown(driver)

            # click verify/continue if appears
            click_buttons(driver)

            time.sleep(1.5)

            # check redirect
            new_url = driver.current_url
            if new_url != current:
                log(f"➡️ Redirected → {new_url}")
                current = new_url

            # check final link
            final = get_final_link(driver)
            if final:
                log("\n🎉 FINAL LINK FOUND:")
                log(final)
                return final

        log("⚠️ Could not extract link.")
        return None

    except Exception as e:
        log("❌ ERROR:")
        log(str(e))
        log(traceback.format_exc())
        return None

    finally:
        driver.quit()


# ----------------------------------------------
# Run from command line
# ----------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python resolver.py <arolinks-url>")
        exit()

    start_url = sys.argv[1]
    resolve(start_url)
