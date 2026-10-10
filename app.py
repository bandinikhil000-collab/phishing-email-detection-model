"""
PhishGuard AI — Web Application & REST API Server
Flask-based interactive cybersecurity dashboard and API for
email phishing analysis, URL threat inspection, and model intelligence.
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from detector import get_detector
from url_checker import analyze_url

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
DATASET_PATH = os.path.join(BASE_DIR, "data", "phishing_emails_dataset.csv")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)
detector = get_detector()

# Pre-packaged sample emails representing diverse real-world threat & benign scenarios
SAMPLE_EMAILS = [
    {
        "id": "paypal_suspend",
        "title": "PayPal Account Suspension",
        "category": "Credential Harvesting",
        "type": "phishing",
        "badge_color": "danger",
        "sender": "service-security@paypal-verification-center.xyz",
        "subject": "URGENT: Your PayPal account has been restricted!",
        "text": """Dear Customer,

We detected unauthorized sign-in attempts on your PayPal balance from an unknown IP address in Lagos, Nigeria.

For your protection, your funds have been locked. You must verify your identity within 24 hours to avoid permanent account termination.

Please confirm your details immediately at:
http://paypal-verification-center.xyz/restore-account

Enter your email address, login password, and credit card security code (CVV) to restore full access.

Sincerely,
PayPal Security & Fraud Prevention"""
    },
    {
        "id": "ms365_alert",
        "title": "Microsoft 365 Cloud Alert",
        "category": "Corporate Credential Theft",
        "type": "phishing",
        "badge_color": "danger",
        "sender": "admin@office365-security-auth.top",
        "subject": "Security Alert: Unusual sign-in to Microsoft 365",
        "text": """Microsoft 365 Security Alert

We detected an unrecognized sign-in from Moscow, Russia.
Browser: Firefox on Linux
Time: Today, 03:14 AM UTC

If this was not you, please verify your credentials immediately to avoid mailbox deletion:
http://micros0ft-portal-auth.com/verify

Enter your enterprise password and 6-digit MFA verification code without delay."""
    },
    {
        "id": "fedex_delivery",
        "title": "FedEx Fake Delivery Fee",
        "category": "Delivery Scam / Payment Fraud",
        "type": "phishing",
        "badge_color": "danger",
        "sender": "notification@fedex-express-dispatch.top",
        "subject": "FedEx: Package pending customs fee payment",
        "text": """We attempted delivery of your parcel #FX-9823101 today at 10:15 AM.

An unpaid customs duty and import processing fee of $2.85 is required before our courier can release your parcel.

Click below to pay with credit or debit card:
http://fedex-express-dispatch.top/clearance

If unpaid within 48 hours, the package will be returned to the sender and destroyed."""
    },
    {
        "id": "crypto_scam",
        "title": "Elon Musk Crypto Giveaway",
        "category": "Advance Fee / Financial Scam",
        "type": "phishing",
        "badge_color": "danger",
        "sender": "giveaway-promo@tesla-foundation.uno",
        "subject": "CONGRATULATIONS! You won 5,000 BTC in the official giveaway!",
        "text": """OFFICIAL TESLA & ELON MUSK AIRDROP

To celebrate the new year, Tesla is giving away 5,000 Bitcoin to random email recipients!

Send 0.1 BTC to verify your crypto wallet address and receive 1.0 BTC immediately back!
Visit: http://tesla-crypto-giveaway.club

Hurry! Only 15 reward allocations remain. Act now before the sweepstakes closes!"""
    },
    {
        "id": "work_sprint",
        "title": "Sprint Planning Sync",
        "category": "Authentic Workplace Email",
        "type": "legitimate",
        "badge_color": "safe",
        "sender": "sarah.jenkins@techcorp-enterprises.com",
        "subject": "Project Roadmap & Sprint Planning for Next Tuesday",
        "text": """Hi team,

Hope you are all having a productive week.

Please review the updated sprint board before our planning meeting on Tuesday morning at 10:00 AM PST. We'll be reviewing ticket estimates and assigning ownership for the search feature and authentication middleware.

Let me know if you have any questions or scheduling conflicts.

Best regards,
Sarah Jenkins
Engineering Lead"""
    },
    {
        "id": "code_review",
        "title": "GitHub Pull Request Review",
        "category": "Authentic Engineering Note",
        "type": "legitimate",
        "badge_color": "safe",
        "sender": "alex.m@devstudio.io",
        "subject": "Code Review: PR #342 - Refactor authentication middleware",
        "text": """Hi Alex,

I took a look at your pull request on GitHub. Overall the changes look very clean and well-structured.

I left a few minor comments regarding error handling on line 45 and adding unit tests for the token expiration edge cases. Feel free to ping me once you've pushed the updates so we can merge this before the Thursday release.

Thanks!
Marcus"""
    },
    {
        "id": "flight_receipt",
        "title": "Airline Flight Confirmation",
        "category": "Authentic Transaction Receipt",
        "type": "legitimate",
        "badge_color": "safe",
        "sender": "reservations@delta.com",
        "subject": "Flight Confirmation - Booking Reference #DL88392",
        "text": """Dear Valued Passenger,

Thank you for booking with Delta Air Lines. Your reservation is confirmed.

Booking Reference: DL88392
Flight: DL 1420 - Seattle (SEA) to New York (JFK)
Departure: October 24, 2026 at 08:30 AM
Seat: 14B (Main Cabin)

You can manage your seat selection, add checked baggage, or check in 24 hours prior to departure on our official website at https://www.delta.com or through the Fly Delta mobile app.

We wish you a pleasant journey!"""
    }
]


@app.route("/")
def index():
    """Render the primary PhishGuard AI cybersecurity interface."""
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    API endpoint: Full email phishing and threat analysis.
    Accepts JSON: { text: str, subject: Optional[str], sender: Optional[str] }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON body provided."}), 400

        text = data.get("text", "")
        subject = data.get("subject", "")
        sender = data.get("sender", "")

        if not text or not text.strip():
            return jsonify({"success": False, "error": "Email content cannot be empty. Please enter email body text."}), 400

        result = detector.analyze_email(text=text, subject=subject, sender=sender)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/analyze-url", methods=["POST"])
def api_analyze_url():
    """
    API endpoint: Standalone deep lexical URL analysis.
    Accepts JSON: { url: str }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "Invalid JSON body provided."}), 400

        raw_url = data.get("url", "")
        if not raw_url or not raw_url.strip():
            return jsonify({"success": False, "error": "URL cannot be empty. Please enter a valid URL."}), 400

        result = analyze_url(raw_url.strip())
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500


@app.route("/api/samples", methods=["GET"])
def api_samples():
    """Return realistic pre-loaded email samples for 1-click testing."""
    return jsonify({"success": True, "samples": SAMPLE_EMAILS})


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Return trained model performance statistics and top linguistic features."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            return jsonify({"success": True, "metrics": metrics_data})
        except Exception as e:
            return jsonify({"success": False, "error": f"Error loading metrics: {str(e)}"}), 500
    else:
        return jsonify({"success": False, "error": "Model metrics file not found."}), 404


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "service": "PhishGuard AI Detection Engine",
        "model_loaded": detector.model is not None,
        "is_fallback": detector.is_fallback
    })


if __name__ == "__main__":
    print("=" * 60)
    print("Starting PhishGuard AI Server on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
