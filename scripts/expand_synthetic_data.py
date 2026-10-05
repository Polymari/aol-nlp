#!/usr/bin/env python3
"""Synthetic Corpus Expansion Script

Expands corpus/synthetic_gotchas.json with:
1. ~1,500 Hard-Negatives: Document headers, date stamps, pagination markers,
   section headings, definitions, standard assent boilerplate, and pro-user rights.
2. ~1,000 Modern 2024-2026 Gotchas: AI training data licenses, mass arbitration batching,
   biometric tracking/brokerage, dark pattern auto-renewals, and digital asset confiscation.
"""

import os
import json
import itertools
import random

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus", "synthetic_gotchas.json")

random.seed(42)

def generate_hard_negatives():
    negatives = []

    # 1. Document Titles & Brand Headers (both Title Case and ALL CAPS)
    doc_types = [
        "Terms of Service", "TERMS OF SERVICE", "Terms of Use", "TERMS OF USE",
        "User Agreement", "USER AGREEMENT", "End User License Agreement", "END USER LICENSE AGREEMENT",
        "Privacy Policy", "PRIVACY POLICY", "Terms and Conditions", "TERMS AND CONDITIONS",
        "Acceptable Use Policy", "Master Subscription Agreement",
        "Developer Terms of Service", "Community Guidelines", "Cookie Policy", "COOKIE POLICY",
        "Privacy Notice", "PRIVACY NOTICE", "Service Level Agreement", "EULA"
    ]
    companies = [
        "", "Acme", "CloudScale", "DataFlow", "NovaTech", "Apex Global",
        "StreamLine", "PulseApp", "Quantum AI", "Nexus Networks", "Vertex Media",
        "Horizon Labs", "Synthetix", "Beacon Health", "Starlight Systems"
    ]
    for dt in doc_types:
        for c in companies:
            title = f"{c} {dt}".strip()
            negatives.append({"text": title, "gotchas": []})

    # 2. Date Metadata & Revision Lines
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    years = ["2022", "2023", "2024", "2025", "2026"]
    date_prefixes = [
        "Last Updated:", "Last updated:", "LAST UPDATED:", "Effective Date:",
        "Effective as of:", "Date of Last Revision:", "Revised:", "Version Date:",
        "Posted:", "Updated:", "Date of issuance:", "Publication date:"
    ]
    for pre in date_prefixes:
        for yr in years:
            for m in months:
                day = random.randint(1, 28)
                negatives.append({"text": f"{pre} {m} {day}, {yr}", "gotchas": []})
                negatives.append({"text": f"{pre} {m} {yr}", "gotchas": []})
                negatives.append({"text": f"{pre} {yr}-{m[:3]}-{day:02d}", "gotchas": []})

    # Version strings
    for maj in [1, 2, 3]:
        for minr in range(12):
            negatives.append({"text": f"Version {maj}.{minr} (Release Date: {random.choice(months)} {random.choice(years)})", "gotchas": []})
            negatives.append({"text": f"Doc Ref: TOS-2025-V{maj}.{minr}", "gotchas": []})
            negatives.append({"text": f"Contract Release Build {maj}.{minr}.0", "gotchas": []})

    # 3. PDF Page Markers & Pagination
    for p in range(1, 60):
        negatives.append({"text": f"Page {p} of 59", "gotchas": []})
        negatives.append({"text": f"Page {p}", "gotchas": []})
        negatives.append({"text": f"— {p} —", "gotchas": []})
        negatives.append({"text": f"- {p} -", "gotchas": []})
        negatives.append({"text": f"[{p}]", "gotchas": []})
        negatives.append({"text": f"Doc #84729-A | Page {p} of 59", "gotchas": []})

    # 4. Standard Numbered Section Headings
    section_titles = [
        "Introduction and Scope", "Eligibility and Registration", "User Account Security",
        "Acceptable Use and Conduct", "Intellectual Property Ownership", "User Feedback and Suggestions",
        "Third-Party Links and Integrations", "Service Availability and Uptime", "Fees and Billing Procedures",
        "Termination and Account Suspension", "Notices and Communications", "Severability and Enforceability",
        "Entire Agreement and Integration", "Governing Law and Jurisdiction", "Contact Information and Customer Support",
        "System Requirements", "Export Controls and Compliance", "Force Majeure", "Relationship of the Parties",
        "Waiver and Cumulative Remedies", "Assignment and Transfer"
    ]
    for i, st in enumerate(section_titles, 1):
        negatives.append({"text": f"{i}. {st}", "gotchas": []})
        negatives.append({"text": f"Section {i}. {st}", "gotchas": []})
        negatives.append({"text": f"Article {i}: {st.upper()}", "gotchas": []})
        negatives.append({"text": f"1.{i} {st}", "gotchas": []})
        negatives.append({"text": f"Clause {i} - {st}", "gotchas": []})

    # 5. Definitions & Legal Construction
    terms = [
        ("Service", "means the web applications, mobile applications, APIs, and associated cloud services provided by Company"),
        ("User", "means any individual or business entity that accesses or utilizes the Service"),
        ("Content", "means text, images, code, audio, video, documents, and other material uploaded or transmitted through the Service"),
        ("Applicable Law", "means all federal, state, local, and international laws, regulations, and statutes applicable to the parties"),
        ("Confidential Information", "means non-public business, technical, or financial information disclosed by either party"),
        ("Customer Data", "refers to data, information, or material submitted by you in the course of using the Service"),
        ("Personal Information", "has the meaning set forth in applicable data protection and privacy regulations"),
        ("Effective Date", "means the date on which you first access the Service or create an account"),
        ("Third-Party Services", "means services, software, feeds, and applications provided by independent entities not affiliated with Company"),
        ("Feedback", "means suggestions, comments, ideas, or recommendations regarding enhancements to the Service")
    ]
    for t_name, t_def in terms:
        negatives.append({"text": f'"{t_name}" {t_def}.', "gotchas": []})
        negatives.append({"text": f'The term "{t_name}" shall mean {t_def.replace("means ", "")}.', "gotchas": []})
        negatives.append({"text": f'As used in this Agreement, "{t_name}" {t_def}.', "gotchas": []})
        negatives.append({"text": f'For purposes of these Terms, "{t_name}" {t_def}.', "gotchas": []})

    # 6. Standard Assent & Preamble
    assent_templates = [
        "By accessing or using the Service, you agree to be bound by these Terms.",
        "Please read these Terms of Service carefully before accessing or using our website.",
        "If you do not agree to all terms and conditions of this agreement, you may not access the Service.",
        "Your access to and use of the Service is conditioned on your acceptance of and compliance with these Terms.",
        "These Terms apply to all visitors, users, and others who access or use the Service.",
        "By creating an account, you represent that you have legal capacity to enter into this agreement.",
        "If you are using the Service on behalf of an entity, you represent that you have authority to bind that entity.",
        "You must be at least 18 years old or the age of legal majority in your jurisdiction to use the Service.",
        "The headings used in this agreement are included for convenience only and will not limit or affect these Terms.",
        "Failure by the Company to enforce any right or provision of these Terms will not be deemed a waiver of such right.",
        "By clicking 'I Agree' or checking the confirmation box, you assent to these terms in full.",
        "This Agreement becomes effective between you and Company on the date of your initial access to the Service."
    ]
    for a in assent_templates:
        negatives.append({"text": a, "gotchas": []})
        negatives.append({"text": f"Section 1. Acceptance. {a}", "gotchas": []})
        negatives.append({"text": f"1. Introduction. {a}", "gotchas": []})

    # 7. Pro-User Privacy & GDPR/CCPA Rights Statements
    privacy_rights = [
        "Under the General Data Protection Regulation (GDPR), you have the right to request access to your personal data.",
        "You have the right to request the rectification of inaccurate personal data concerning you without undue delay.",
        "You have the right to request the erasure of your personal data where one of the statutory grounds applies.",
        "You have the right to receive your personal data in a structured, commonly used, and machine-readable format.",
        "Under the California Consumer Privacy Act (CCPA), you have the right to request deletion of personal information.",
        "You have the right to opt out of the sale or sharing of your personal information at any time.",
        "We do not sell your personal data or share your personal information with third parties for cross-context behavioral advertising.",
        "You may withdraw your consent to data processing at any time by contacting our data protection officer.",
        "You have the right to lodge a complaint with a supervisory authority in your country of residence.",
        "You may browse our public marketing pages anonymously without creating an account or providing personal details.",
        "You may unsubscribe from promotional email communications at any time by clicking the unsubscribe link.",
        "We maintain technical, administrative, and physical safeguards designed to protect personal data from accidental loss."
    ]
    for pr in privacy_rights:
        negatives.append({"text": pr, "gotchas": []})
        negatives.append({"text": f"Your Privacy Rights: {pr}", "gotchas": []})
        negatives.append({"text": f"Article 15. {pr}", "gotchas": []})

    # 8. Contact & Formal Notices
    contact_templates = [
        "If you have any questions about these Terms, please contact us at legal@company.com.",
        "Notices to the Company must be sent by written letter to 100 Main Street, Suite 500, New York, NY 10001.",
        "Formal legal notices to you will be delivered to the email address registered with your user account.",
        "Questions regarding billing, account settings, or technical support should be directed to support@company.com.",
        "These Terms constitute the entire agreement between you and Company regarding the use of the Service.",
        "If any provision of these Terms is held to be invalid or unenforceable, such provision shall be severed.",
        "Neither party shall be liable for any failure or delay in performance resulting from causes beyond its reasonable control."
    ]
    for ct in contact_templates:
        negatives.append({"text": ct, "gotchas": []})
        negatives.append({"text": f"Miscellaneous. {ct}", "gotchas": []})

    return negatives


def generate_modern_gotchas():
    gotchas = []

    # 1. AI Training Data Grabs (Modern 2024-2026)
    ai_actions = [
        ("train our artificial intelligence and machine learning models on any content you upload", "train our artificial intelligence and machine learning models on any content you upload"),
        ("ingest your prompts, queries, and generated outputs into our foundational AI training pipelines without restriction", "ingest your prompts, queries, and generated outputs into our foundational AI training pipelines without restriction"),
        ("use such material to train, refine, and monetize proprietary algorithmic models", "use such material to train, refine, and monetize proprietary algorithmic models"),
        ("used to train and fine-tune commercial generative AI systems", "used to train and fine-tune commercial generative AI systems"),
        ("utilize all user-generated content and conversational telemetry to develop, evaluate, and commercialize machine learning architectures", "utilize all user-generated content and conversational telemetry to develop, evaluate, and commercialize machine learning architectures"),
        ("feed your content into automated neural network training datasets", "feed your content into automated neural network training datasets"),
        ("license your uploaded datasets and user prompts to third-party artificial intelligence developers for model pre-training purposes", "license your uploaded datasets and user prompts to third-party artificial intelligence developers for model pre-training purposes"),
        ("train deep learning representations and algorithmic classifiers on your private correspondence", "train deep learning representations and algorithmic classifiers on your private correspondence"),
        ("incorporate your submitted source code and documentation into foundational code generation models", "incorporate your submitted source code and documentation into foundational code generation models"),
        ("use all uploaded media and artwork to optimize generative visual diffusion models without royalty", "use all uploaded media and artwork to optimize generative visual diffusion models without royalty")
    ]

    prefixes_ai = [
        "You grant Company a perpetual, worldwide, irrevocable license to ",
        "We reserve the right to ",
        "By submitting content to the platform, you agree that we may ",
        "You acknowledge and agree that your workspace data may be ",
        "Company and its corporate affiliates may ",
        "You irrevocably authorize Company to ",
        "We may freely "
    ]

    for pfx in prefixes_ai:
        for phrase, target in ai_actions:
            full_text = f"{pfx}{phrase}."
            if target in full_text:
                gotchas.append({"text": full_text, "gotchas": [target]})
                gotchas.append({"text": f"Section 8. Intellectual Property. {full_text}", "gotchas": [target]})
                gotchas.append({"text": f"{full_text} This authorization survives any termination of your account.", "gotchas": [target]})

    # 2. Mass Arbitration Batching & Bellwether Hurdles (Modern 2024-2026)
    arb_clauses = [
        (
            "If 25 or more claimants submit coordinated arbitration notices, all claims shall be batched in sets of 25 and resolved in phased bellwether proceedings.",
            ["batched in sets of 25 and resolved in phased bellwether proceedings"]
        ),
        (
            "You agree that mass arbitration filings shall be stayed and arbitrated strictly in batches of 50, with each claimant paying individual non-refundable filing fees.",
            ["stayed and arbitrated strictly in batches of 50", "paying individual non-refundable filing fees"]
        ),
        (
            "Before initiating any arbitration proceeding, you must personally attend a mandatory 60-day informal video conference with our legal counsel.",
            ["mandatory 60-day informal video conference with our legal counsel"]
        ),
        (
            "Any claim or dispute arising out of this Agreement shall be resolved through binding individual arbitration, and you expressly waive any right to participate in a class action.",
            ["binding individual arbitration", "waive any right to participate in a class action"]
        ),
        (
            "The parties agree that all disputes must be arbitrated on an individual basis and waive any entitlement to join or consolidate claims with other users.",
            ["arbitrated on an individual basis", "waive any entitlement to join or consolidate claims with other users"]
        ),
        (
            "You hereby irrevocably waive your right to a trial by jury in any action, proceeding, or counterclaim arising out of this contract.",
            ["irrevocably waive your right to a trial by jury"]
        ),
        (
            "All arbitration hearings shall take place exclusively in Dover, Delaware, and each party shall bear its own legal and administrative fees regardless of outcome.",
            ["arbitration hearings shall take place exclusively", "bear its own legal and administrative fees regardless of outcome"]
        ),
        (
            "You agree to resolve all disputes exclusively through confidential binding arbitration and waive any right to bring representative proceedings.",
            ["confidential binding arbitration", "waive any right to bring representative proceedings"]
        ),
        (
            "If one hundred or more arbitration demands are asserted, claims will be staged in increments of 50 with statute of limitations tolled.",
            ["staged in increments of 50 with statute of limitations tolled"]
        ),
        (
            "You forfeit any right to pursue relief in small claims court if an arbitration demand has been initiated by Company.",
            ["forfeit any right to pursue relief in small claims court"]
        )
    ]
    for full_text, targets in arb_clauses:
        gotchas.append({"text": full_text, "gotchas": targets})
        gotchas.append({"text": f"14. Dispute Resolution. {full_text}", "gotchas": targets})
        gotchas.append({"text": f"Section 12. Arbitration Agreement. {full_text}", "gotchas": targets})
        gotchas.append({"text": f"{full_text} Arbitration shall be administered by ADR Services.", "gotchas": targets})

    # 3. Biometric Telemetry & Brokerage (Modern 2024-2026)
    bio_clauses = [
        (
            "We collect, process, and monetize your biometric identifiers, facial geometry scans, and voiceprints with third-party commercial verification partners.",
            ["monetize your biometric identifiers, facial geometry scans, and voiceprints with third-party commercial verification partners"]
        ),
        (
            "You consent to our recording and selling of your continuous keystroke dynamics, mouse trajectory patterns, and eye-tracking telemetry to commercial data brokers.",
            ["selling of your continuous keystroke dynamics, mouse trajectory patterns, and eye-tracking telemetry to commercial data brokers"]
        ),
        (
            "We share your precise geolocation data and device sensory telemetry with advertising exchanges and marketing affiliates without prior notice.",
            ["share your precise geolocation data and device sensory telemetry with advertising exchanges and marketing affiliates"]
        ),
        (
            "By enabling voice features, you authorize us to retain, commercialize, and syndicate your audio recordings and biometric speech patterns.",
            ["commercialize, and syndicate your audio recordings and biometric speech patterns"]
        ),
        (
            "We reserve the right to trade, sell, or disclose your comprehensive browsing history and profiling dossiers to external commercial advertisers.",
            ["trade, sell, or disclose your comprehensive browsing history and profiling dossiers to external commercial advertisers"]
        ),
        (
            "We syndicate your physical location telemetry and Wi-Fi triangulation data to third-party data analytics vendors for behavioral profiling.",
            ["syndicate your physical location telemetry and Wi-Fi triangulation data to third-party data analytics vendors"]
        ),
        (
            "You grant us the right to compile and sell aggregated biometric feature vectors derived from your camera feed to marketing research firms.",
            ["compile and sell aggregated biometric feature vectors derived from your camera feed"]
        )
    ]
    for full_text, targets in bio_clauses:
        gotchas.append({"text": full_text, "gotchas": targets})
        gotchas.append({"text": f"Privacy Disclosures. {full_text}", "gotchas": targets})
        gotchas.append({"text": f"Section 5. Data Collection. {full_text}", "gotchas": targets})

    # 4. Predatory Auto-Renewals, Cancellation Obstacles, Non-Refundability (Modern 2024-2026)
    renewal_clauses = [
        (
            "Your subscription automatically renews at the non-discounted rate unless cancelled via certified postal mail at least 30 days prior to the billing date.",
            ["automatically renews at the non-discounted rate unless cancelled via certified postal mail at least 30 days prior"]
        ),
        (
            "All subscription payments and in-app purchases are strictly non-refundable under any circumstances, and no credits will be granted for partial periods.",
            ["strictly non-refundable under any circumstances", "no credits will be granted for partial periods"]
        ),
        (
            "We may automatically charge your payment method for upgraded tiers and auxiliary features at our sole discretion.",
            ["automatically charge your payment method for upgraded tiers and auxiliary features at our sole discretion"]
        ),
        (
            "To cancel your subscription, you must complete an interactive telephonic retention consultation during weekday business hours.",
            ["must complete an interactive telephonic retention consultation during weekday business hours"]
        ),
        (
            "Upon renewal, fees may increase by up to 25% annually without prior notice or affirmative consent from you.",
            ["increase by up to 25% annually without prior notice or affirmative consent"]
        ),
        (
            "Any promotional discounts expire upon initial term and renewal billing occurs at full rack rate without notification.",
            ["renewal billing occurs at full rack rate without notification"]
        ),
        (
            "You authorize recurring automatic charges to your card on file indefinitely until a written cancellation notice is received and approved by our billing department.",
            ["recurring automatic charges to your card on file indefinitely until a written cancellation notice is received and approved"]
        )
    ]
    for full_text, targets in renewal_clauses:
        gotchas.append({"text": full_text, "gotchas": targets})
        gotchas.append({"text": f"Billing Terms. {full_text}", "gotchas": targets})
        gotchas.append({"text": f"Section 4. Payments and Subscriptions. {full_text}", "gotchas": targets})

    # 5. Unilateral Modifications, Termination, Asset Forfeiture
    unilateral_clauses = [
        (
            "We reserve the right to modify, replace, or discontinue any aspect of these Terms at our sole discretion without notice, effective immediately upon posting.",
            ["modify, replace, or discontinue any aspect of these Terms at our sole discretion without notice"]
        ),
        (
            "We may terminate your account, confiscate accumulated virtual assets, and revoke your software licenses immediately at any time without liability.",
            ["confiscate accumulated virtual assets, and revoke your software licenses immediately at any time without liability"]
        ),
        (
            "You agree to indemnify, defend, and hold harmless Company from any and all claims, liabilities, and expenses arising out of your access or alleged violation.",
            ["indemnify, defend, and hold harmless Company from any and all claims, liabilities, and expenses"]
        ),
        (
            "The services are provided strictly as-is and as-available, and we disclaim all express, implied, and statutory warranties whatsoever.",
            ["provided strictly as-is and as-available", "disclaim all express, implied, and statutory warranties whatsoever"]
        ),
        (
            "In no event shall Company's aggregate liability exceed the total amount paid by you in the preceding three months or fifty dollars ($50.00).",
            ["liability exceed the total amount paid by you in the preceding three months or fifty dollars ($50.00)"]
        ),
        (
            "Your continued access to the platform following any update constitutes irrevocable acceptance of all revised terms.",
            ["continued access to the platform following any update constitutes irrevocable acceptance of all revised terms"]
        ),
        (
            "Company reserves the right to delete, purge, or reclaim user profiles, digital media, and earned credentials without prior notification or compensation.",
            ["delete, purge, or reclaim user profiles, digital media, and earned credentials without prior notification or compensation"]
        )
    ]
    for full_text, targets in unilateral_clauses:
        gotchas.append({"text": full_text, "gotchas": targets})
        gotchas.append({"text": f"Section 12. Terms of Agreement. {full_text}", "gotchas": targets})
        gotchas.append({"text": f"15. Limitation of Liability. {full_text}", "gotchas": targets})

    return gotchas


def main():
    print(f"Loading existing corpus from {CORPUS_PATH}...")
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    print(f"Initial corpus samples: {len(existing)}")
    existing_texts = {item["text"].strip().lower() for item in existing}

    hard_negs = generate_hard_negatives()
    modern_gotchas = generate_modern_gotchas()

    added_negs = 0
    added_gotchas = 0

    for item in hard_negs:
        t = item["text"].strip()
        if t.lower() not in existing_texts:
            existing.append(item)
            existing_texts.add(t.lower())
            added_negs += 1

    for item in modern_gotchas:
        t = item["text"].strip()
        # Verify that all gotcha spans match verbatim
        valid = True
        for g in item["gotchas"]:
            if g not in t:
                print(f"WARNING: Gotcha span '{g}' not found in text '{t}'! Skipping.")
                valid = False
                break
        if valid and t.lower() not in existing_texts:
            existing.append(item)
            existing_texts.add(t.lower())
            added_gotchas += 1

    print(f"Added {added_negs} hard-negative samples.")
    print(f"Added {added_gotchas} modern gotcha samples.")
    print(f"Total corpus size: {len(existing)}")

    # Verify 100% of samples in existing
    for idx, item in enumerate(existing):
        t = item["text"]
        for g in item.get("gotchas", []):
            assert g in t, f"Item {idx} has invalid gotcha '{g}' not in text '{t}'"

    # Write out formatted JSON
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved updated corpus to {CORPUS_PATH}!")


if __name__ == "__main__":
    main()
