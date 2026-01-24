import requests
import random
import time

FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScravr_a4xEFvbUhO4tK4FXoLKlEzmmmeKkTT26w7QYp1yRtg/formResponse"

options = ["4", "3", "2", "1", "78"]

def send_response():
    choice = random.choice(options)
    
    payload = {
        "entry.40150210": choice,
        "fvv": "1",
        "draftResponse": [],
        "pageHistory": "0"
    }

    r = requests.post(FORM_URL, data=payload)
    return r.status_code, choice

print("Starting auto-submission...\n")

for i in range(1000):
    status, ans = send_response()
    time.sleep(0.15)  # superfast
