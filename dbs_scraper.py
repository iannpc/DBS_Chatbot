"""
DBS iBank Help & Support Scraper
=================================
Scrapes all DBS Help & Support article pages to build a
knowledge base for a RAG-powered chatbot.

All article URLs have been pre-compiled from the 6 category pages:
  Bank, Credit Card, General Help, Borrow, Invest, Insure

Usage:
    pip install requests beautifulsoup4 lxml
    python dbs_scraper.py

Output:
    - dbs_knowledge_base.json   (structured data for RAG ingestion)
    - dbs_scrape_stats.json     (summary statistics)
    - dbs_scrape_failures.json  (failed URLs, if any)
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.dbs.com.sg/personal/support/"
REQUEST_DELAY = 1.5   # seconds between requests 
REQUEST_TIMEOUT = 15

# ═══════════════════════════════════════════════════════════════════════════════
# ALL ARTICLE URLs — compiled from the 6 Help & Support category pages
# ═══════════════════════════════════════════════════════════════════════════════

ARTICLES = {
    # ── BANK: Account Enquiries ──
    "Banking - Account Enquiries": [
        "bank-account-new-opening.html",
        "bank-account-opening-documents-required.html",
        "bank-account-application-status.html",
        "bank-account-activate-ntb.html",
        "bank-account-change-msa-contribution.html",
        "bank-deposit-accounts-account-transactions.html",
        "bank-account-transaction-codes.html",
        "bank-account-view-account-number.html",
        "bank-deposit-accounts-peek-balance.html",
        "bank-deposit-accounts-check-account-balance.html",
        "bank-deposit-accounts-fall-below-fee.html",
        "bank-view-joint-account-info.html",
        "bank-deposit-passbook-conversion.html",
        "bank-deposit-autosave-conversion.html",
        "bank-account-closure.html",
    ],
    # ── BANK: Applications & Facilities ──
    "Banking - Applications & Facilities": [
        "bank-ssb-withdraw-cash-local.html",
        "bank-ssb-deposit-cash.html",
        "bank-ssb-deposit-coins.html",
        "bank-ssb-notes-exchange.html",
        "bank-ssb-exchange-foreign-currencies.html",
        "bank-ssb-phone-banking-authentication.html",
        "bank-ssb-phone-banking-services.html",
    ],
    # ── BANK: Card Enquiries ──
    "Banking - Card Enquiries": [
        "bank-atm-debit-card-application.html",
        "card-activate-new-card.html",
        "card-overseas-enabling-for-overseas-use.html",
        "bank-atm-debit-card-transaction-details.html",
        "bank-atm-debit-card-withdraw-cash-overseas.html",
        "card-issues-card-replacement.html",
        "card-issues-lost-card.html",
        "general-incorrect-transaction.html",
        "bank-atm-debit-card-unable-to-withdraw.html",
        "card-issues-forget-pin.html",
        "bank-atm-debit-card-change-atm-card-limit.html",
        "card-application-link-card-to-deposit-account.html",
        "card-charges-and-fees-overseas-transaction-fees.html",
        "bank-atm-debit-card-charges-overseas-withdrawal.html",
        "card-issues-terminate-card.html",
    ],
    # ── BANK: Cheques ──
    "Banking - Cheques": [
        "bank-cheque-depositing-cheques.html",
        "bank-cheque-clearing.html",
        "bank-cheque-view-cheque-status.html",
        "bank-cheque-returned-cheques-reasons.html",
        "bank-payeasy-purchase-cashiers-order.html",
        "bank-payeasy-purchase-demand-draft.html",
        "bank-cheque-purchase-ibcheque.html",
        "bank-cheque-purchase-ibcheque-add-recipient.html",
        "bank-cheque-purchase-ibcheque-remove-recipient.html",
        "bank-cheque-wrongly-deposited-cheques.html",
        "bank-cheque-cancelling-cheques.html",
        "bank-cheque-multiple-names-in-cheque.html",
        "bank-cheque-request-new-cheque-book.html",
        "bank-cheque-fc-cheque-fees.html",
    ],
    # ── BANK: DBS digibank ──
    "Banking - DBS digibank": [
        "bank-minimum-operating-system-requirements.html",
        "bank-ibanking-digital-token.html",
        "guide-ibanking.html",
        "bank-ibanking-digital-token-setup.html",
        "bank-ibanking-application.html",
        "bank-digibank-reset-uid-pin.html",
        "bank-ibanking-using-ib-secure-device.html",
        "bank-ibanking-notification-alerts.html",
        "bank-switch-simple-full-mode.html",
        "bank-sgfindex-link-insurance.html",
        "bank-sgfindex-unlink-financial-institution.html",
    ],
    # ── BANK: DBS Multiplier ──
    "Banking - DBS Multiplier": [
        "bank-multiplier-eligible-transactions.html",
    ],
    # ── BANK: Fixed Deposit ──
    "Banking - Fixed Deposit": [
        "bank-account-fixed-deposit-opening.html",
        "bank-account-fixed-deposit-placement.html",
        "bank-account-fixed-deposit-withdrawal.html",
        "bank-account-fixed-deposit-maturity-instructions.html",
    ],
    # ── BANK: Local Funds Transfer ──
    "Banking - Local Funds Transfer": [
        "bank-local-funds-transfer-transfer-to-other-bank-accounts.html",
        "bank-local-funds-transfer-add-bank-recipient.html",
        "bank-local-funds-transfer-remove-bank-recipient.html",
        "bank-retrieve-share-transaction-details.html",
        "guide-paylah.html",
        "bank-local-funds-transfer-setup-recurring-funds-transfer.html",
        "bank-local-funds-view-recurring-funds-transfer.html",
        "bank-local-funds-terminate-recurring-funds-transfer.html",
        "bank-local-funds-transfer-change-funds-transfer-limit.html",
        "bank-local-wrong-funds-transfer.html",
        "bank-local-unsuccessful-transfer.html",
        "bank-receive-funds-from-others.html",
    ],
    # ── BANK: Overseas Funds Transfer ──
    "Banking - Overseas Funds Transfer": [
        "guide-remit.html",
        "bank-overseas-funds-transfer-new-remittance.html",
        "bank-overseas-funds-transfer-remittance-add-recipient.html",
        "bank-overseas-funds-transfer-remittance-remove-recipient.html",
        "bank-overseas-dbs-remit.html",
        "bank-overseas-funds-transfer-countries.html",
        "bank-overseas-funds-transfer-fees-and-charges.html",
        "bank-overseas-funds-transfer-service-standards.html",
        "bank-overseas-funds-transfer-delayed.html",
        "bank-overseas-wrong-funds-transfer.html",
        "bank-overseas-funds-transfer-change-funds-transfer-limit.html",
        "bank-overseas-funds-transfer-recurring-funds-transfer-to-overseas-account.html",
        "bank-overseas-funds-transfer-amend-recurring-funds-transfer.html",
        "bank-overseas-funds-transfer-delete-recurring-funds-transfer.html",
        "bank-overseas-funds-transfer-funds-transfer-to-overseas-credit-card.html",
    ],
    # ── BANK: PayNow ──
    "Banking - PayNow": [
        "bank-ssb-paynow-register-profile.html",
        "bank-ssb-paynow-register-child-profile.html",
        "bank-ssb-paynow-check-profile.html",
        "bank-ssb-paynow-amend-profile.html",
        "bank-local-funds-transfer-add-paynow-recipient.html",
        "bank-local-funds-transfer-paynow-transfer.html",
        "bank-local-funds-transfer-remove-paynow-recipient.html",
        "bank-ssb-paynow-deregister-profile.html",
        "bank-ssb-paynow-deregister-child-profile.html",
    ],
    # ── BANK: DBS PayLah! ──
    "Banking - DBS PayLah": [
        "bank-ssb-apply-paylah.html",
        "bank-ssb-paylah-request-funds.html",
        "bank-ssb-paylah-transfer-funds.html",
        "bank-ssb-paylah-responding-transfer-funds.html",
        "bank-ssb-paylah-scan-and-pay.html",
        "bank-ssb-paylah-bill-payment.html",
        "bank-ssb-paylah-merchant-checkout.html",
        "bank-ssb-paylah-transaction-history.html",
        "bank-ssb-paylah-estatement.html",
        "bank-ssb-paylah-change-nickname.html",
        "bank-ssb-paylah-manage-notifications.html",
        "bank-ssb-paylah-manage-wallet.html",
        "bank-ssb-paylah-change-account.html",
        "bank-ssb-paylah-change-wallet-limit.html",
        "bank-ssb-paylah-change-mobile-number.html",
        "bank-ssb-reset-paylah.html",
        "bank-ssb-close-paylah.html",
    ],
    # ── BANK: Payments ──
    "Banking - Payments": [
        "bank-payment-bill-payment.html",
        "bank-payment-add-bill-payment-organisations.html",
        "bank-payment-remove-bill-payment-organisations.html",
        "bank-payment-pay-other-bank-credit-cards.html",
        "bank-payment-add-other-bank-credit-cards-recipient.html",
        "bank-payment-remove-other-bank-credit-cards-recipient.html",
        "bank-payment-enets-d2pay-application.html",
        "bank-payment-enets-d2pay-amend-payment-limit.html",
        "bank-payment-enets-d2pay-deactivation.html",
        "bank-payment-setup-giro-arrangement.html",
        "bank-payment-view-active-giro-arrangements.html",
        "bank-payment-update-giro-limit.html",
        "bank-payment-terminate-giro-arrangement.html",
        "card-payment-update-recurring-bill-payments.html",
        "bank-payment-top-up-mobile-prepaid.html",
        "bank-payment-scan-and-pay-limit.html",
        "bank-payment-issue-edp.html",
        "bank-payment-cashout-edp.html",
        "bank-payment-edp-status.html",
        "bank-payment-cancel-reject-edp.html",
    ],
    # ── BANK: Payment Controls ──
    "Banking - Payment Controls": [
        "card-customise-card-functions.html",
        "card-temp-lock-card.html",
        "card-manage-spending-limit.html",
    ],
    # ── BANK: Statements ──
    "Banking - Statements": [
        "bank-deposit-accounts-calculate-madb.html",
        "bank-statements-consolidated-statements.html",
        "bank-statements-retrieve-printed-statements.html",
        "bank-statements-financial-standing-statement.html",
        "bank-statements-estatements-enrol.html",
        "bank-statements-viewing-estatements.html",
        "bank-statements-manage-eadvicesestatements-notification.html",
        "bank-statements-estatements-deenrol.html",
    ],
    # ── CREDIT CARD: Application & Termination ──
    "Credit Card - Application & Termination": [
        "card-application-new-card.html",
        "card-application-supplementary-card.html",
        "card-application-documents.html",
        "card-application-recommendation.html",
        "card-application-eligibility.html",
        "cards-application-56yo.html",
        "card-application-status.html",
        "card-cancellation-anz-paydown.html",
        "card-payment-mobile-wallet-application.html",
        "card-payment-mobile-wallet-making-payment.html",
        "card-payment-mobile-wallet-remove-card.html",
    ],
    # ── CREDIT CARD: Bill Payment ──
    "Credit Card - Bill Payment": [
        "card-payment-outstanding-balance.html",
        "card-payment-pay-credit-card-bills.html",
        "card-payment-due-date.html",
        "card-statement-change-billing-cycle.html",
        "card-payment-cut-off-times.html",
        "card-payment-giro-application.html",
        "card-payment-ipp.html",
        "card-payment-mp3.html",
        "card-payment-recurring.html",
    ],
    # ── CREDIT CARD: Card Matters ──
    "Credit Card - Card Matters": [
        "card-matters-top-up-card-ezlink-function.html",
        "card-matters-malfunctioned-card-ezlink-function.html",
        "card-transaction-3ds.html",
    ],
    # ── CREDIT CARD: Credit Limit ──
    "Credit Card - Credit Limit": [
        "card-transaction-check-limits.html",
        "card-credit-limit-temp-increase.html",
    ],
    # ── CREDIT CARD: Fees & Charges ──
    "Credit Card - Fees & Charges": [
        "card-charges-and-fees-annual-fee.html",
        "card-charges-and-fees-late-fee.html",
        "card-charges-and-fees-finance-charge.html",
        "card-charges-and-fees-cash-advance-fee.html",
        "card-charges-and-fees-over-limit-fee.html",
        "card-charges-and-fees-returned-cheque-giro-fee.html",
    ],
    # ── CREDIT CARD: Loyalty & Rewards ──
    "Credit Card - Loyalty & Rewards": [
        "card-rewards-checking-your-dbs-points.html",
        "card-rewards-redeeming-dbs-points.html",
        "card-rewards-dbs-points-expiry.html",
        "card-rewards-convert-dbs-points-to-frequent-flyer.html",
    ],
    # ── CREDIT CARD: Transaction ──
    "Credit Card - Transaction": [
        "card-transaction-view-transaction-details.html",
        "card-transaction-declined-transaction.html",
        "card-statement-understanding-statement.html",
    ],
    # ── CREDIT CARD: Account Suspension ──
    "Credit Card - Account Suspension": [
        "card-credit-limit-revised-unsecured-credit-rules-by-mas.html",
    ],
    # ── GENERAL: Bank Details ──
    "General - Bank Details": [
        "bank-general-swift-code-details.html",
        "bank-general-bank-branch-names-codes.html",
        "general-bank-details-crs.html",
        "general-bank-details-fatca.html",
    ],
    # ── GENERAL: Bank Services ──
    "General - Bank Services": [
        "general-digital-assistance.html",
        "general-bank-details-branch-smsq.html",
    ],
    # ── GENERAL: Update Profile ──
    "General - Update Profile": [
        "general-profile-update-address.html",
        "general-profile-update-email-address.html",
        "general-profile-update-mobile-number.html",
        "general-profile-update-personal-details.html",
    ],
    # ── GENERAL: Fraud Prevention ──
    "General - Fraud Prevention": [
        "general-digibank-security-malware-jailbroken-device.html",
        "general-digibank-security-cooling-period.html",
        "general-digibank-security-measures.html",
        "guide-security-on-scams-and-fraud.html",
        "bank-ssb-safety-switch.html",
        "general-card-security-protect-your-card-and-pin.html",
        "general-online-safety.html",
        "general-shared-responsibility-framework.html",
        "general-cra-call-verification.html",
    ],
    # ── BORROW: Banker's Guarantee ──
    "Borrow - Banker's Guarantee": [
        "loans-bankers-guarantee-application.html",
        "loans-bankers-guarantee-application-eligibility.html",
        "loans-bankers-guarantee-features.html",
        "loans-bankers-guarantee-fee.html",
    ],
    # ── BORROW: Cashline & Unsecured Loans ──
    "Borrow - Cashline & Unsecured Loans": [
        "loans-application-new-cashline.html",
        "loans-application-cashline-documents.html",
        "card-application-dbs-personal-loan.html",
        "loans-application-balance-transfer.html",
        "loans-cashline-bill-payment.html",
        "loans-cashline-giro-application.html",
        "loans-cashline-fees-and-charges.html",
        "loans-cashline-understanding-statement.html",
        "loans-cashline-transaction-view-transaction-details.html",
        "loans-view-balancetransfer-personalloan-details.html",
    ],
    # ── BORROW: Car Loan ──
    "Borrow - Car Loan": [
        "loan-carloan-manage-car-loan.html",
    ],
    # ── BORROW: Home Loans ──
    "Borrow - Home Loans": [
        "loans-homeloan-repricing-documents-required.html",
        "loans-homeloan-partial-repayment.html",
        "loans-homeloan-full-repayment.html",
        "loans-homeloan-change-loan-servicing-account.html",
        "loans-homeloan-repay-using-cpf-funds.html",
        "loans-homeloan-revise-cpf-monthly-repayment.html",
        "loans-homeloan-understanding-statement.html",
        "loans-homeloan-check-total-interest.html",
    ],
    # ── BORROW: Renovation Loan ──
    "Borrow - Renovation Loan": [
        "loans-homeloan-reno-loan-application.html",
    ],
    # ── BORROW: Share Financing ──
    "Borrow - Share Financing": [
        "loans-share-financing-deposit.html",
    ],
    # ── INVEST: CPF Investment Scheme ──
    "Invest - CPFIS": [
        "investment-vickers-open-cpfis.html",
        "investment-cpfis-transfer-to-oa.html",
    ],
    # ── INVEST: Shares ──
    "Invest - Shares": [
        "investment-shares-apply-eps.html",
        "investment-shares-esa.html",
        "investment-shares-check-results-esa.html",
        "guide-vickers.html",
    ],
    # ── INVEST: SGS Bonds / T-bills ──
    "Invest - SGS Bonds & T-bills": [
        "investment-sgs-apply.html",
    ],
    # ── INVEST: Singapore Savings Bonds ──
    "Invest - Singapore Savings Bonds": [
        "investment-ssb-apply.html",
        "investment-ssb-check-results.html",
        "investment-ssb-redeem.html",
    ],
    # ── INVEST: SRS ──
    "Invest - SRS": [
        "bank-account-srs-account-opening.html",
        "bank-account-srs-account-contribution.html",
    ],
    # ── INVEST: Unit Trust ──
    "Invest - Unit Trust": [
        "investment-ut-apply-invest-saver.html",
        "investment-ut-update-amount-invest-saver.html",
        "investment-ut-redeem-holdings-invest-saver.html",
        "investment-ut-terminate-invest-saver.html",
    ],
    # ── INVEST: Wealth Management ──
    "Invest - Wealth Management": [
        "wealth-iwealth-portfolio-allocations-view.html",
        "wealth-iwealth-cka.html",
        "wealth-iwealth-invest-funds-online.html",
        "wealth-iwealth-redeem-funds.html",
        "wealth-iwealth-online-fx.html",
        "wealth-iwealth-online-loan-drawdown.html",
        "wealth-iwealth-set-price-alerts.html",
        "wealth-invest-update-investment-profile.html",
        "wealth-iwealth-invest-ecorporate-action.html",
        "wealth-iwealth-track-corporate-account-portfolios.html",
    ],
    # ── INSURE: Travel ──
    "Insure - Travel": [
        "insurance-travel-submit-ts-claim.html",
        "insurance-travel-marketplace-complimentary-travel-insurance.html",
    ],
    # ── INSURE: Protection ──
    "Insure - Protection": [
        "insurance-cashcare-apply.html",
    ],
    # ── GUIDES (from home page) ──
    "Guides": [
        "guide-lny-notes-exchange.html",
        "guide-lny-wealth-notes-exchange.html",
        "guide-online-security.html",
        "guide-card-replacement.html",
        "guide-bill-payment.html",
        "guide-online-equity-trading.html",
        "guide-card-security.html",
        "guide-shopping.html",
        "guide-travel.html",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCRAPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Remove extra whitespace and clean up text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page_content(soup: BeautifulSoup, url: str, category: str) -> dict:
    """Extract structured content from a DBS support article page."""

    # ── Title ──
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text())
    elif soup.find("title"):
        title = clean_text(soup.find("title").get_text()).replace(" | DBS Singapore", "")

    # ── Main content area ──
    main = (
        soup.find("div", class_=re.compile(r"(article|content|main-body|ps3-revamp)", re.I))
        or soup.find("main")
        or soup.find("div", id=re.compile(r"(content|article)", re.I))
        or soup.body
    )

    # ── Extract step-by-step instructions ──
    steps = []
    step_elements = soup.find_all(string=re.compile(r"^Step \d+", re.I))
    for step_el in step_elements:
        parent = step_el.parent if step_el.parent else step_el
        step_text = clean_text(parent.get_text() if hasattr(parent, "get_text") else str(parent))
        if step_text and len(step_text) > 5:
            steps.append(step_text)

    # ── Extract Q&A / FAQ pairs ──
    faq_pairs = []
    for strong in soup.find_all(["strong", "b", "h3", "h4"]):
        text = clean_text(strong.get_text())
        if text.endswith("?") and len(text) > 10:
            answer_parts = []
            sibling = strong.parent.find_next_sibling() if strong.parent else strong.find_next_sibling()
            while sibling and sibling.name not in ["h1", "h2", "h3", "h4"]:
                sib_text = clean_text(sibling.get_text())
                if not sib_text:
                    break
                if sib_text.endswith("?") and len(sib_text) > 10:
                    break
                answer_parts.append(sib_text)
                sibling = sibling.find_next_sibling()
                if len(answer_parts) >= 5:
                    break
            if answer_parts:
                faq_pairs.append({
                    "question": text,
                    "answer": " ".join(answer_parts),
                })

    # ── Extract sections by headings ──
    sections = []
    for heading in soup.find_all(["h2", "h3"]):
        heading_text = clean_text(heading.get_text())
        if not heading_text or heading_text in [
            "Popular Articles", "Popular Guides", "Popular Article",
        ]:
            continue
        content_parts = []
        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in ["h2", "h3"]:
            text = clean_text(sibling.get_text())
            if text and len(text) > 3:
                content_parts.append(text)
            sibling = sibling.find_next_sibling()
        if content_parts:
            sections.append({
                "heading": heading_text,
                "content": " ".join(content_parts),
            })

    # ── Full text extraction (fallback) ──
    full_text = ""
    if main:
        for tag in main.find_all(["nav", "footer", "header", "script", "style", "noscript"]):
            tag.decompose()
        full_text = clean_text(main.get_text())

    # ── Important notes / tips ──
    notes = []
    for note in soup.find_all(class_=re.compile(r"(note|important|tip|info|warning)", re.I)):
        note_text = clean_text(note.get_text())
        if note_text and len(note_text) > 10:
            notes.append(note_text)

    return {
        "url": url,
        "title": title,
        "category": category,
        "full_text": full_text[:15000],
        "steps": steps,
        "faq_pairs": faq_pairs,
        "sections": sections,
        "notes": notes,
        "scraped_at": datetime.now().isoformat(),
    }


def scrape_all():
    """Main scraping function."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })

    # Deduplicate URLs across categories
    all_urls = {}
    for category, urls in ARTICLES.items():
        for url_path in urls:
            full_url = BASE_URL + url_path
            if full_url not in all_urls:
                all_urls[full_url] = category

    total = len(all_urls)
    logger.info(f"Starting scrape of {total} unique article pages...")

    results = []
    failed = []

    for i, (url, category) in enumerate(all_urls.items(), 1):
        logger.info(f"[{i}/{total}] Scraping: {url}")

        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                logger.warning(f"  Skipped (not HTML): {content_type}")
                continue

            soup = BeautifulSoup(response.text, "lxml")
            page_data = extract_page_content(soup, url, category)
            results.append(page_data)
            logger.info(f"  OK: {page_data['title'][:60]}")

        except requests.RequestException as e:
            logger.warning(f"  FAIL: {e}")
            failed.append({"url": url, "category": category, "error": str(e)})

        time.sleep(REQUEST_DELAY)

    # ── Save results ──
    with open("dbs_knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved {len(results)} articles to dbs_knowledge_base.json")

    stats = {
        "total_articles": len(results),
        "failed": len(failed),
        "categories": {},
        "scraped_at": datetime.now().isoformat(),
    }
    for r in results:
        cat = r["category"]
        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

    with open("dbs_scrape_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if failed:
        with open("dbs_scrape_failures.json", "w", encoding="utf-8") as f:
            json.dump(failed, f, indent=2)
        logger.warning(f"{len(failed)} pages failed — see dbs_scrape_failures.json")

    print("\n" + "=" * 60)
    print("SCRAPE SUMMARY")
    print("=" * 60)
    print(f"Total articles scraped: {len(results)}")
    print(f"Failed:                 {len(failed)}")
    print(f"\nBy category:")
    for cat, count in sorted(stats["categories"].items()):
        print(f"  {cat}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    scrape_all()
