# alerts/email_alert.py

from dotenv import load_dotenv
load_dotenv()

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = os.getenv("ALERT_EMAIL")
SENDER_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD")
RECEIVER_EMAIL = os.getenv("ALERT_RECEIVER_EMAIL")


def send_email_alert(aqi, category, alert_level):
    if alert_level == "Safe":
        return

    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECEIVER_EMAIL:
        print("⚠️ Email credentials not set. Skipping email alert.")
        return

    # Severity-based content
    if alert_level == "Caution":
        subject = "⚠️ AQI Caution: Air quality is moderate"
        recommendation = "Limit prolonged or heavy outdoor exertion."
        tone = "This is a precautionary notification."

    elif alert_level == "Warning":
        subject = "⚠️ AQI Warning: Unhealthy for sensitive groups"
        recommendation = "Children, elderly, and sensitive individuals should reduce outdoor activities."
        tone = "Health advisory issued."

    elif alert_level == "Alert":
        subject = "🚨 AQI Alert: Unhealthy air quality detected"
        recommendation = "Avoid outdoor activities. Wear masks if necessary."
        tone = "Immediate attention recommended."

    else:  # Critical
        subject = "🔴 AQI Critical Alert: Hazardous air quality"
        recommendation = "Stay indoors. Use air purifiers and avoid exposure."
        tone = "Emergency-level air quality alert."

    body = f"""
Air Quality Notification

Predicted AQI: {aqi}
Category: {category}
Alert Level: {alert_level}

{tone}

Recommended Action:
- {recommendation}

This alert was generated automatically by the AQI Monitoring System.
"""

    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        print("📧 Email alert sent successfully")

    except Exception as e:
        print(f"❌ Email alert failed: {e}")