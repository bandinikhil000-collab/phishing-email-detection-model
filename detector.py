"""
PhishGuard AI — Email Phishing Detection & Explainability Engine
Combines a trained Scikit-learn TF-IDF + Logistic Regression model with
deep cybersecurity heuristics, sender analysis, and URL inspection.
Provides comprehensive, explainable threat breakdowns and recommended actions.
"""

import os
import re
import json
import joblib
from typing import Dict, Any, List, Optional
from url_checker import analyze_url

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "phishing_classifier.joblib")
VECTORIZER_FILE = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METRICS_FILE = os.path.join(MODELS_DIR, "model_metrics.json")

# Brand names commonly impersonated in phishing
COMMON_BRANDS = {
    "paypal": ["paypal.com"],
    "microsoft": ["microsoft.com", "office.com", "live.com", "outlook.com"],
    "google": ["google.com", "gmail.com"],
    "apple": ["apple.com", "icloud.com"],
    "amazon": ["amazon.com", "amazon.co.uk"],
    "netflix": ["netflix.com"],
    "chase": ["chase.com"],
    "bank of america": ["bankofamerica.com", "bofa.com"],
    "wells fargo": ["wellsfargo.com"],
    "fedex": ["fedex.com"],
    "ups": ["ups.com"],
    "usps": ["usps.com"],
    "dhl": ["dhl.com"],
    "irs": ["irs.gov"],
    "docusign": ["docusign.com"],
    "dropbox": ["dropbox.com"],
    "meta": ["meta.com", "facebook.com", "instagram.com"],
    "facebook": ["facebook.com"],
    "instagram": ["instagram.com"]
}

FREE_MAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "protonmail.com", "mail.com", "zoho.com", "icloud.com", "gmx.com"
}

SUSPICIOUS_ATTACHMENT_EXTENSIONS = [
    r"\.exe\b", r"\.scr\b", r"\.iso\b", r"\.vbs\b", r"\.bat\b",
    r"\.cmd\b", r"\.ps1\b", r"\.xlsm\b", r"\.docm\b", r"\.zip\b",
    r"\.rar\b", r"\.7z\b", r"\.html\b", r"\.htm\b", r"\.hta\b"
]


def clean_text(text: str) -> str:
    """Normalize text matching training pipeline format."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " <EMAIL> ", text)
    text = re.sub(r"[\$£€]\s*\d+([,\.]\d+)?", " <CURRENCY> ", text)
    text = re.sub(r"[!]{2,}", " !! ", text)
    text = re.sub(r"[\?]{2,}", " ?? ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class PhishingDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metrics = None
        self.is_fallback = False
        self.load_model()

    def load_model(self):
        """Loads trained artifacts from disk with graceful fallback mode."""
        try:
            if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
                self.model = joblib.load(MODEL_FILE)
                self.vectorizer = joblib.load(VECTORIZER_FILE)
                self.is_fallback = False
                if os.path.exists(METRICS_FILE):
                    with open(METRICS_FILE, "r", encoding="utf-8") as f:
                        self.metrics = json.load(f)
                print("PhishGuard AI: Successfully loaded trained ML model and vectorizer.")
            else:
                print("PhishGuard AI: Model files not found. Initializing in fallback heuristic mode.")
                self.is_fallback = True
        except Exception as e:
            print(f"PhishGuard AI: Error loading ML model ({e}). Falling back to heuristic mode.")
            self.is_fallback = True

    def extract_urls(self, text: str) -> List[str]:
        """Extract all HTTP/HTTPS and www URLs from plain text or basic HTML."""
        url_pattern = r"(?:https?://|www\.)[^\s\"'<>]+"
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        # Deduplicate while preserving order
        seen = set()
        clean_urls = []
        for u in urls:
            u_clean = u.rstrip(".,;!?:)")
            if u_clean and u_clean not in seen:
                seen.add(u_clean)
                clean_urls.append(u_clean)
        return clean_urls

    def check_sender_address(self, sender: str, full_content: str) -> List[Dict[str, Any]]:
        """Analyze sender email for spoofing, free-mail brand impersonation, and mismatches."""
        warnings = []
        if not sender or not sender.strip():
            return warnings

        sender = sender.strip().lower()
        match = re.search(r"<?([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)>?", sender)
        if not match:
            warnings.append({
                "category": "Sender Identity",
                "severity": "Medium",
                "title": "Invalid Sender Email Format",
                "detail": f"The provided sender '{sender}' does not conform to standard RFC email address format."
            })
            return warnings

        sender_email = match.group(1)
        sender_domain = sender_email.split("@")[-1].lower()

        # Check if email body mentions an authoritative brand, but sender is free mail or unrelated domain
        full_content_lower = full_content.lower()
        for brand, legitimate_domains in COMMON_BRANDS.items():
            if brand in full_content_lower:
                # If brand mentioned, does sender domain match?
                if sender_domain in FREE_MAIL_DOMAINS:
                    warnings.append({
                        "category": "Sender Identity",
                        "severity": "High",
                        "title": f"Public Webmail Used to Impersonate {brand.title()}",
                        "detail": f"The email references {brand.title()}, but was sent from a free public email address (@{sender_domain}). Official organizations never conduct security or account notices from consumer webmail."
                    })
                    break
                elif not any(sender_domain == d or sender_domain.endswith("." + d) for d in legitimate_domains):
                    # Check if sender domain contains brand name as a substring (typosquatting/lookalike)
                    if brand in sender_domain:
                        warnings.append({
                            "category": "Sender Identity",
                            "severity": "High",
                            "title": f"Lookalike Sender Domain ({sender_domain})",
                            "detail": f"The sender domain '{sender_domain}' incorporates the brand name '{brand}' on an unofficial domain."
                        })
                    break

        return warnings

    def detect_heuristics(self, full_text: str, sender: str, urls: List[str]) -> List[Dict[str, Any]]:
        """Deep cybersecurity heuristic checks across 8 primary phishing vectors."""
        warnings = []
        text_lower = full_text.lower()

        # 1. Urgent requests for immediate action
        urgency_patterns = [
            (r"\b(within\s+(?:24|12|48|2|6)\s+hours?|by\s+end\s+of\s+day|immediate(?:ly)?|act\s+now|urgent(?:ly)?|without\s+delay)\b", "Artificial Time Pressure"),
            (r"\b(account\s+(?:will\s+be\s+)?(?:suspended|terminated|closed|disabled|deleted|restricted))\b", "Threat of Account Suspension"),
            (r"\b(final\s+notice|last\s+warning|legal\s+action|arrest\s+warrant|penalties)\b", "Intimidation / Legal Coercion")
        ]
        urgency_matches = []
        for pat, desc in urgency_patterns:
            if re.search(pat, text_lower):
                urgency_matches.append(desc)
        if urgency_matches:
            warnings.append({
                "category": "Psychological Manipulation",
                "severity": "High" if len(urgency_matches) > 1 else "Medium",
                "title": "High Urgency & Coercive Deadline",
                "detail": f"The message attempts to provoke panic by creating artificial urgency: {', '.join(urgency_matches)}. Phishers use urgency to prevent recipients from consulting colleagues or verifying carefully."
            })

        # 2. Requests for passwords, OTPs, or banking info
        credential_patterns = [
            (r"\b(password|passcode|secret\s+phrase|seed\s+phrase|recovery\s+phrase)\b", "Password / Passphrase Request"),
            (r"\b(one-time\s+password|otp|verification\s+code|6-digit\s+code|mfa\s+code|token)\b", "Two-Factor / OTP Token Request"),
            (r"\b(credit\s+card|debit\s+card|cvv|security\s+code|expiration\s+date|pin\s+number)\b", "Payment Card Details"),
            (r"\b(social\s+security(?:\s+number)?|ssn|bank\s+account\s+number|routing\s+number)\b", "Banking / SSN Credentials")
        ]
        credential_matches = []
        for pat, desc in credential_patterns:
            if re.search(pat, text_lower):
                credential_matches.append(desc)
        if credential_matches:
            warnings.append({
                "category": "Credential Harvesting",
                "severity": "High",
                "title": "Request for Sensitive Credentials or Financial Information",
                "detail": f"The message requests private data: {', '.join(credential_matches)}. Legitimate companies never request passwords, full card details, or one-time verification passcodes via email."
            })

        # 3. Unexpected account verification messages
        verification_patterns = [
            r"\b(?:unusual|unauthorized|suspicious)\s+(?:sign-in|activity|attempt|transaction|login)\b",
            r"\bverify\s+your\s+(?:account|identity|billing|profile|credentials)\b",
            r"\bconfirm\s+your\s+(?:identity|details|password|information)\b",
            r"\bclick\s+here\s+to\s+(?:verify|unlock|reactivate|restore)\b"
        ]
        if any(re.search(pat, text_lower) for pat in verification_patterns):
            warnings.append({
                "category": "Account Alert Lure",
                "severity": "Medium",
                "title": "Unsolicited Account Verification / Security Alert",
                "detail": "The message claims unexpected security alerts or asks you to confirm/verify account access. Attackers routinely disguise phishing links as security checkups."
            })

        # 4. Offers that seem too good to be true
        too_good_patterns = [
            r"\b(won|winner|winning)\b.*\b(lottery|sweepstakes|prize|draw)\b",
            r"\b(unclaimed\s+(?:funds|inheritance|millions|estate))\b",
            r"\b(crypto(?:currency)?\s+giveaway|claim\s+\d+(?:\.\d+)?\s*(?:btc|eth|bitcoin))\b",
            r"\b(free\s+\$\d+|\$\d+\s+gift\s+card|\$\d+\s+voucher)\b"
        ]
        if any(re.search(pat, text_lower) for pat in too_good_patterns):
            warnings.append({
                "category": "Financial Scam Lure",
                "severity": "High",
                "title": "Too-Good-To-Be-True Reward or Windfall",
                "detail": "The email promises sudden financial prizes, lotteries, cryptocurrency giveaways, or inheritance funds. Legitimate organizations do not distribute unrequested cash awards."
            })

        # 5. Suspicious attachments mentioned
        attachment_detected = []
        for ext_pat in SUSPICIOUS_ATTACHMENT_EXTENSIONS:
            if re.search(ext_pat, text_lower):
                ext_name = ext_pat.replace(r"\.", ".").replace(r"\b", "")
                attachment_detected.append(ext_name)
        if attachment_detected:
            warnings.append({
                "category": "Malicious Attachment",
                "severity": "High",
                "title": f"High-Risk File Attachment Mentioned ({', '.join(set(attachment_detected))})",
                "detail": "The email references executable, script, or compressed archive file types that are frequently used to deliver malware, trojans, or ransomware."
            })

        # 6. Unusual spelling, capitalization or obfuscation
        all_caps_count = len(re.findall(r"\b[A-Z]{4,}\b", full_text))
        excessive_exclamation = len(re.findall(r"!{2,}", full_text))
        if all_caps_count >= 3 or excessive_exclamation >= 2:
            warnings.append({
                "category": "Message Formatting",
                "severity": "Low",
                "title": "Aggressive Capitalization or Punctuation",
                "detail": f"Detected aggressive styling ({all_caps_count} all-caps words, multiple exclamation marks), characteristic of spam and social engineering."
            })

        # 7. Sender address analysis
        sender_warnings = self.check_sender_address(sender, full_text)
        warnings.extend(sender_warnings)

        return warnings

    def analyze_email(self, text: str, subject: Optional[str] = "", sender: Optional[str] = "") -> Dict[str, Any]:
        """
        Executes full dual-engine email analysis:
        1. TF-IDF + Logistic Regression Machine Learning Classifier.
        2. Deep Cybersecurity Heuristic Threat Engine.
        3. Extracted URL Lexical & Security Analysis.
        Returns unified, explainable assessment.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                "success": False,
                "error": "Email content cannot be empty. Please paste the email body to analyze."
            }

        text = text.strip()
        if len(text) > 100000:
            return {
                "success": False,
                "error": "Email content exceeds the maximum size limit of 100,000 characters."
            }

        subject = (subject or "").strip()
        sender = (sender or "").strip()

        # Combine subject and text for holistic text processing
        full_content = f"Subject: {subject}\n{text}" if subject else text
        cleaned = clean_text(full_content)

        # 1. Machine Learning Inference
        ml_prob_phish = 0.5
        top_contributing_features = []
        model_mode_desc = "Trained Scikit-learn Classifier"

        if not self.is_fallback and self.model is not None and self.vectorizer is not None:
            try:
                X_vec = self.vectorizer.transform([cleaned])
                probs = self.model.predict_proba(X_vec)[0]
                # probs[0] = legit, probs[1] = phish
                ml_prob_phish = float(probs[1])

                # Extract active features in this email and their weights
                feature_names = self.vectorizer.get_feature_names_out()
                coefs = self.model.coef_[0]
                non_zero_indices = X_vec.nonzero()[1]

                active_features = []
                for idx in non_zero_indices:
                    word = feature_names[idx]
                    weight = float(coefs[idx])
                    tfidf_val = float(X_vec[0, idx])
                    impact = weight * tfidf_val
                    active_features.append({
                        "feature": word,
                        "weight": round(weight, 3),
                        "impact": round(impact, 3)
                    })

                # Sort by impact
                active_features.sort(key=lambda x: abs(x["impact"]), reverse=True)
                top_contributing_features = active_features[:8]
            except Exception as e:
                print(f"Inference error: {e}. Using heuristic fallback.")
                self.is_fallback = True

        if self.is_fallback:
            model_mode_desc = "Heuristic Fallback Mode (ML weights unavailable)"
            # Baseline estimate from keyword density
            suspicious_terms = ["verify", "password", "urgent", "suspended", "account", "login", "otp", "bank", "card", "click"]
            hits = sum(1 for term in suspicious_terms if term in cleaned)
            ml_prob_phish = min(0.15 + (hits * 0.12), 0.85)

        # 2. Extract and inspect embedded links
        embedded_urls = self.extract_urls(full_content)
        url_analyses = []
        url_high_risk_count = 0
        url_medium_risk_count = 0

        for url in embedded_urls[:10]:  # Limit to first 10 URLs to avoid overload
            u_res = analyze_url(url)
            if u_res.get("success"):
                url_analyses.append(u_res)
                if u_res.get("risk_level") == "High":
                    url_high_risk_count += 1
                elif u_res.get("risk_level") == "Medium":
                    url_medium_risk_count += 1

        # 3. Rule-based heuristic indicators
        heuristic_warnings = self.detect_heuristics(full_content, sender, embedded_urls)

        # Add warnings from URLs
        if url_high_risk_count > 0:
            heuristic_warnings.append({
                "category": "Embedded Links",
                "severity": "High",
                "title": f"High-Risk Malicious Link Detected ({url_high_risk_count} found)",
                "detail": "At least one URL in the email points to an IP address, lookalike domain, or deceptive subdomain structure."
            })
        elif url_medium_risk_count > 0:
            heuristic_warnings.append({
                "category": "Embedded Links",
                "severity": "Medium",
                "title": "Suspicious or Obfuscated Links Detected",
                "detail": "Embedded links include URL shorteners, unencrypted HTTP protocols, or suspicious domain structures."
            })

        # 4. Fused Multi-factor Scoring
        # Count severity
        high_severity_count = sum(1 for w in heuristic_warnings if w.get("severity") == "High")
        med_severity_count = sum(1 for w in heuristic_warnings if w.get("severity") == "Medium")

        # Base probability from ML (calibrated via Logistic Regression on balanced data)
        # We blend with heuristic signals to prevent one-keyword anomalies
        # while catching multi-factor real-world attacks.
        heuristic_boost = (high_severity_count * 0.18) + (med_severity_count * 0.08)
        fused_score = min(max(ml_prob_phish * 0.65 + heuristic_boost * 0.35, 0.02), 0.98)

        # Classification thresholds into the 3 specified classes:
        # 1. Likely Legitimate
        # 2. Suspicious
        # 3. Likely Phishing
        if fused_score >= 0.65 or high_severity_count >= 2:
            verdict = "Likely Phishing"
            risk_level = "High"
            status_color = "danger"
            calibrated_confidence = round(max(fused_score, 0.72) * 100, 1)
        elif fused_score >= 0.35 or high_severity_count >= 1 or med_severity_count >= 2:
            verdict = "Suspicious"
            risk_level = "Medium"
            status_color = "warning"
            calibrated_confidence = round(max(fused_score, 0.50) * 100, 1)
        else:
            verdict = "Likely Legitimate"
            risk_level = "Low"
            status_color = "safe"
            calibrated_confidence = round((1.0 - fused_score) * 100, 1)

        # 5. Tailored Recommended Actions
        recommended_actions = []
        if risk_level == "High":
            recommended_actions.append("Do NOT click any links, open attachments, or reply to this message.")
            recommended_actions.append("Never share passwords, one-time verification passcodes (OTPs), or bank account details.")
            if sender:
                recommended_actions.append(f"Block sender address '{sender}' and report the message to your security team.")
            else:
                recommended_actions.append("Report this email as phishing in your email client (e.g. Gmail/Outlook report button).")
            recommended_actions.append("If this email claims to be from a service you use, navigate directly to their official portal by typing the known web address into your browser.")
        elif risk_level == "Medium":
            recommended_actions.append("Exercise extreme caution before interacting with this email.")
            recommended_actions.append("Verify the sender's identity through a secondary, out-of-band channel (e.g., direct phone call or known Slack/Teams handle).")
            recommended_actions.append("Inspect any links by hovering over them rather than clicking directly.")
            recommended_actions.append("Do not enter login credentials on any page reached from this email.")
        else:
            recommended_actions.append("This email demonstrates linguistic and structural characteristics of legitimate correspondence.")
            recommended_actions.append("Continue to practice safe cyber hygiene: never share passwords or authentication codes over email.")
            recommended_actions.append("If unexpected attachments or payment requests arrive even from known senders, confirm verbally.")

        return {
            "success": True,
            "verdict": verdict,
            "risk_level": risk_level,
            "status_color": status_color,
            "confidence_score": calibrated_confidence,
            "model_probability": round(ml_prob_phish * 100, 1),
            "model_mode": model_mode_desc,
            "warning_signs": heuristic_warnings,
            "warning_count": len(heuristic_warnings),
            "recommended_actions": recommended_actions,
            "top_features": top_contributing_features,
            "urls_detected": len(embedded_urls),
            "url_analyses": url_analyses,
            "metadata": {
                "input_length_chars": len(text),
                "has_subject": bool(subject),
                "has_sender": bool(sender),
                "sender_analyzed": sender if sender else "None provided"
            }
        }


# Global detector singleton
detector_instance = PhishingDetector()


def get_detector():
    return detector_instance
