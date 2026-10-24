"""Finale Part 1: context rules, consent, redaction and the context veto."""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "service"))
from context_rules import assess, PaymentContext, redact, keyword_baseline, combine, band_for


def test_fixtures_match_round1_table():
    r = subprocess.run([sys.executable, str(ROOT / "ml" / "context_validation.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]


def test_no_consent_means_text_is_ignored_and_not_stored():
    s9 = "The finance office has revised the vendor details. Please remit the balance to the replacement beneficiary ahead of the afternoon cutoff."
    r = assess(s9, PaymentContext(new_beneficiary=True), consent=False)
    assert r.context == 0 and r.redacted_text is None and r.entity_hashes == []
    assert "message context not used: no consent" in r.reasons


def test_redaction_removes_identifiers_and_keeps_hashes():
    txt, ents = redact("Pay 50000 to fraudster99@ybl or call 98765 43210, acct 123456789012, OTP 4821, see https://x.in/a")
    for secret in ("fraudster99@ybl", "98765", "123456789012", "4821", "https://x.in"):
        assert secret not in txt
    assert any(e.startswith("upi:") for e in ents) and any(e.startswith("account:") for e in ents)
    assert redact("a@ybl")[1] == redact("A@YBL")[1]          # same entity -> same hash, so payments can be linked


def test_single_domain_never_escalates():
    assert band_for(70, 1) == "MEDIUM" and band_for(70, 2) == "HIGH" and band_for(85, 2) == "CRITICAL"


def test_keyword_baseline_misses_s09():
    assert keyword_baseline("The finance office has revised the vendor details.")["flag"] is False
    assert keyword_baseline("Share your OTP urgently")["flag"] is True


def test_context_veto_raises_but_never_blocks():
    hi = assess("This is the CEO. Pay into the new account details immediately.", PaymentContext(new_beneficiary=True, amount_deviation=True), True)
    out = combine("Approve", hi)
    assert out["final_decision"] == "Hold for analyst" and out["raised_by_context"]
    assert combine("Block + STR draft", hi)["final_decision"] == "Block + STR draft"      # context never lowers a decision
    low = assess(None, PaymentContext(), True)
    assert combine("Approve", low)["final_decision"] == "Approve"
