"""
PhishGuard AI — Suspicious URL Checker Module
Provides deep, defensive lexical and structural analysis of URLs without making external HTTP requests.
Safe from SSRF (Server-Side Request Forgery).
"""

import re
import ipaddress
from urllib.parse import urlparse, unquote

# Known legitimate popular brands commonly impersonated
POPULAR_BRANDS = [
    "google", "paypal", "microsoft", "apple", "amazon", "netflix",
    "chase", "bankofamerica", "wellsfargo", "citibank", "facebook",
    "instagram", "whatsapp", "linkedin", "twitter", "dropbox",
    "dhl", "fedex", "usps", "ups", "irs", "ebay", "target", "walmart",
    "steam", "spotify", "adobe", "outlook", "office365", "yahoo"
]

# Known URL shorteners
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "buff.ly", "ow.ly",
    "rb.gy", "cutt.ly", "goo.gl", "v.gd", "shorte.st", "trib.al",
    "rebrand.ly", "tiny.cc", "bc.vc", "bl.ink", "lnkd.in"
}

# Suspicious or high-risk TLDs heavily associated with malicious campaigns
HIGH_RISK_TLDS = {
    "top", "xyz", "buzz", "fit", "work", "click", "rest", "cam",
    "gq", "cf", "ml", "tk", "ga", "country", "stream", "gdn",
    "mom", "surf", "monster", "uno", "casa", "kim", "party"
}

# Common character substitutions for typosquatting / lookalikes
LEET_SUBS = {
    '0': 'o', '1': 'l', '3': 'e', '4': 'a', '5': 's',
    '7': 't', '8': 'b', '@': 'a', 'vv': 'w', 'rn': 'm'
}


def normalize_url(raw_url: str) -> str:
    """Ensure URL has a scheme for proper parsing."""
    url = raw_url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def is_ip_address(hostname: str) -> bool:
    """Check if the hostname is a raw IPv4 or IPv6 address."""
    clean_host = hostname.strip("[]")
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False


def get_domain_parts(hostname: str):
    """
    Split hostname into subdomains, second-level domain (SLD), and TLD.
    Handles basic ccTLDs like .co.uk, .com.au, .gov.uk.
    """
    parts = hostname.lower().split('.')
    if len(parts) <= 1:
        return "", hostname, ""

    compound_tlds = {"co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "net.au", "co.nz", "com.br"}
    last_two = ".".join(parts[-2:])

    if last_two in compound_tlds and len(parts) >= 3:
        tld = last_two
        sld = parts[-3]
        subdomains = ".".join(parts[:-3])
    else:
        tld = parts[-1]
        sld = parts[-2]
        subdomains = ".".join(parts[:-2])

    return subdomains, sld, tld


def check_typosquatting(sld: str):
    """
    Detect leetspeak, insertions, and close lookalike brand names in the domain.
    """
    detected_brand_spoofs = []
    
    # 1. Exact brand matches in SLD are fine if it's actually that brand, but
    # if it combines brand with security/login keywords (e.g. paypal-verify), flag it!
    for brand in POPULAR_BRANDS:
        if brand in sld and sld != brand:
            # e.g., paypal-security, login-microsoft
            suspicious_keywords = ["verify", "login", "secure", "update", "service", "account", "support", "billing", "auth"]
            if any(k in sld for k in suspicious_keywords):
                detected_brand_spoofs.append({
                    "brand": brand,
                    "reason": f"Combines trusted brand '{brand}' with credential keyword in '{sld}'"
                })

        # 2. Leet substitutions check (e.g., paypa1 -> paypal, micros0ft -> microsoft)
        normalized_sld = sld
        for char, repl in LEET_SUBS.items():
            normalized_sld = normalized_sld.replace(char, repl)
        
        if normalized_sld != sld:
            for b in POPULAR_BRANDS:
                if b == normalized_sld or (b in normalized_sld and sld != b):
                    detected_brand_spoofs.append({
                        "brand": b,
                        "reason": f"Lookalike domain detected: '{sld}' resembles '{b}' using visual character substitution"
                    })
                    break

        # 3. Simple edit distance check (1 substitution / omission)
        if len(sld) > 4 and brand not in sld:
            # check distance
            if abs(len(sld) - len(brand)) <= 1:
                diffs = sum(1 for a, b in zip(sld, brand) if a != b)
                if diffs == 1 and len(sld) == len(brand):
                    detected_brand_spoofs.append({
                        "brand": brand,
                        "reason": f"Typo-squatted domain '{sld}' differs from legitimate '{brand}' by a single character"
                    })

    return detected_brand_spoofs


def analyze_url(raw_url: str) -> dict:
    """
    Perform comprehensive lexical and security analysis of a URL.
    Returns structured analysis with warning signs, risk level, and explanations.
    """
    if not raw_url or not isinstance(raw_url, str) or not raw_url.strip():
        return {
            "success": False,
            "error": "URL cannot be empty. Please enter a valid URL."
        }

    raw_url = raw_url.strip()
    if len(raw_url) > 2048:
        return {
            "success": False,
            "error": "URL exceeds maximum length limit of 2,048 characters."
        }

    normalized = normalize_url(raw_url)
    try:
        parsed = urlparse(normalized)
    except Exception as e:
        return {
            "success": False,
            "error": f"Invalid URL structure: {str(e)}"
        }

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return {
            "success": False,
            "error": "Invalid URL: Could not extract a valid host or domain name."
        }

    warning_signs = []
    risk_points = 0
    subdomains, sld, tld = get_domain_parts(hostname)

    # 1. IP Address as host check
    if is_ip_address(hostname):
        warning_signs.append({
            "category": "Host Structure",
            "severity": "High",
            "title": "Raw IP Address Used Instead of Domain",
            "detail": f"The URL connects directly to an IP address ({hostname}) rather than a registered domain name. Legitimate organizations virtually never ask users to browse directly to IP addresses."
        })
        risk_points += 40

    # 2. Scheme (HTTP vs HTTPS)
    scheme = parsed.scheme.lower()
    if scheme == "http":
        warning_signs.append({
            "category": "Protocol Security",
            "severity": "Medium",
            "title": "Unencrypted HTTP Protocol",
            "detail": "The URL uses unencrypted HTTP. Any data transmitted (such as passwords, forms, or personal info) can be intercepted on public or unencrypted networks."
        })
        risk_points += 20
    elif scheme == "https":
        # Crucial nuance: HTTPS does not guarantee legitimacy!
        https_notice = "The URL uses HTTPS encryption, but HTTPS only secures the connection in transit. Phishing sites frequently obtain free HTTPS certificates (e.g. Let's Encrypt) to look authentic."
    else:
        warning_signs.append({
            "category": "Protocol Security",
            "severity": "High",
            "title": f"Unusual Protocol ({scheme})",
            "detail": f"The URL uses a non-standard web protocol '{scheme}', which could trigger local applications or file handlers."
        })
        risk_points += 30

    # 3. URL Shortener detection
    if hostname in SHORTENERS or f"www.{hostname}" in SHORTENERS:
        warning_signs.append({
            "category": "Obfuscation",
            "severity": "Medium",
            "title": "Known URL Shortener Detected",
            "detail": f"The domain '{hostname}' is a URL shortener service. Attackers routinely use shorteners to mask malicious destination URLs, bypass basic email filters, and prevent inspection."
        })
        risk_points += 25

    # 4. Misleading Subdomains (Brand in Subdomain)
    for brand in POPULAR_BRANDS:
        if brand in subdomains and brand not in sld:
            warning_signs.append({
                "category": "Deceptive Domain",
                "severity": "High",
                "title": f"Brand Impersonation in Subdomain ({brand})",
                "detail": f"The trusted brand name '{brand}' appears in the subdomain '{subdomains}', but the actual registered destination domain is '{sld}.{tld}'. This is a classic tactic used to trick victims into believing they are visiting an official site."
            })
            risk_points += 45
            break

    # 5. Lookalike / Typosquatting
    spoofs = check_typosquatting(sld)
    for spoof in spoofs:
        warning_signs.append({
            "category": "Deceptive Domain",
            "severity": "High",
            "title": "Potential Lookalike / Typosquatted Domain",
            "detail": spoof["reason"]
        })
        risk_points += 35

    # 6. High-Risk TLD
    if tld in HIGH_RISK_TLDS:
        warning_signs.append({
            "category": "Domain Reputation",
            "severity": "Medium",
            "title": f"High-Risk Top-Level Domain (.{tld})",
            "detail": f"The top-level domain '.{tld}' has historically high rates of malicious spam and phishing registrations due to low registration costs or lax verification."
        })
        risk_points += 20

    # 7. Excessive Subdomains
    subdomain_parts = [p for p in subdomains.split('.') if p]
    if len(subdomain_parts) >= 3:
        warning_signs.append({
            "category": "Obfuscation",
            "severity": "Medium",
            "title": "Excessive Subdomain Nesting",
            "detail": f"The URL contains {len(subdomain_parts)} nested subdomains ({subdomains}). Attackers nest multiple subdomains to push the real domain off-screen on mobile devices."
        })
        risk_points += 20

    # 8. Userinfo '@' symbol trick
    if '@' in raw_url:
        warning_signs.append({
            "category": "URL Deception",
            "severity": "High",
            "title": "Misleading '@' Symbol in URL",
            "detail": "The URL contains an '@' character. In web standards, anything before the '@' is treated as user credentials and ignored by the browser, while the actual site you visit is whatever comes after the '@'."
        })
        risk_points += 45

    # 9. Multiple Hyphens in Domain
    if sld.count('-') >= 2:
        warning_signs.append({
            "category": "Domain Structure",
            "severity": "Medium",
            "title": "Suspicious Hyphen Stuffing",
            "detail": f"The domain '{sld}' contains multiple hyphens ({sld.count('-')}). Phishing domains often chain keywords like 'secure-login-account-update' using hyphens."
        })
        risk_points += 15

    # 10. Hex / Percent encoding obfuscation in path or domain
    if '%' in parsed.path or '%' in parsed.query:
        unquoted_path = unquote(parsed.path)
        if unquoted_path != parsed.path:
            warning_signs.append({
                "category": "Obfuscation",
                "severity": "Low",
                "title": "Percent-Encoded Characters in Path",
                "detail": "The URL path includes percent-encoded characters (%xx), often used to evade keyword filters."
            })
            risk_points += 10

    # 11. Suspicious keywords in path
    path_lower = parsed.path.lower()
    credential_path_keywords = ["login", "signin", "verify", "secure", "update-billing", "otp", "password", "wallet", "recovery"]
    found_path_keywords = [k for k in credential_path_keywords if k in path_lower]
    if found_path_keywords and (risk_points > 0 or tld in HIGH_RISK_TLDS):
        warning_signs.append({
            "category": "Sensitive Action",
            "severity": "Medium",
            "title": "Credential or Authentication Path Keywords",
            "detail": f"The URL path contains sensitive authentication keywords: {', '.join(found_path_keywords)}."
        })
        risk_points += 15

    # Determine Risk Level and Verdict
    if risk_points >= 40:
        risk_level = "High"
        verdict = "Likely Phishing / Malicious Link"
        status_color = "danger"
    elif risk_points >= 20:
        risk_level = "Medium"
        verdict = "Suspicious Link"
        status_color = "warning"
    else:
        risk_level = "Low"
        verdict = "Likely Legitimate / Low Structural Risk"
        status_color = "safe"

    # Contextual Recommended Actions
    recommended_actions = []
    if risk_level in ["High", "Medium"]:
        recommended_actions.append("Do NOT click this link or enter any login credentials or personal details.")
        if any(w['title'].startswith("Known URL Shortener") for w in warning_signs):
            recommended_actions.append("Use a preview tool or check the destination before navigating through shortened links.")
        if any(w['title'].startswith("Brand Impersonation") or "Lookalike" in w['title'] for w in warning_signs):
            recommended_actions.append("Navigate directly to the official website by typing the official web address into your browser.")
        recommended_actions.append("Never enter passwords, two-factor codes, or card details on unfamiliar domains.")
    else:
        recommended_actions.append("No obvious lexical red flags detected in the URL structure.")
        recommended_actions.append("Remember: Even legitimate-looking sites can be compromised, and HTTPS only means the transport is encrypted, not that the site is safe.")
        recommended_actions.append("Always verify the web address matches the service you intended to visit.")

    return {
        "success": True,
        "raw_url": raw_url,
        "normalized_url": normalized,
        "components": {
            "scheme": scheme,
            "hostname": hostname,
            "registered_domain": f"{sld}.{tld}" if sld and tld else hostname,
            "subdomains": subdomains or "(none)",
            "path": parsed.path or "/",
            "query": parsed.query or "(none)",
            "is_ip": is_ip_address(hostname),
            "is_https": scheme == "https"
        },
        "verdict": verdict,
        "risk_level": risk_level,
        "risk_score": min(risk_points, 100),
        "status_color": status_color,
        "warning_signs": warning_signs,
        "warning_count": len(warning_signs),
        "recommended_actions": recommended_actions,
        "https_safety_notice": "CRITICAL: HTTPS ensures that traffic between your browser and the website is encrypted, but cybercriminals routinely use free SSL/TLS certificates. HTTPS does NOT mean the website is legitimate or safe."
    }
