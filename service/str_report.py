"""
STR draft from an alert's evidence pack (the "AI investigator drafts an STR ready case report").

Built from a template, not a language model: every sentence is traceable to a field in the evidence pack, nothing
leaves the bank's servers, and the same inputs always give the same draft. The sections follow the parts of a
Suspicious Transaction Report filed with FIU-IND (reporting entity, transactions, linked accounts, grounds of
suspicion, action taken); check them against the current FINnet schema before filing. An analyst must review,
edit and sign before anything is filed.
"""
from __future__ import annotations

PLAIN = {
    "message cue: authority": "the message impersonated an authority (bank, police, government or a company office)",
    "message cue: redirection": "the message redirected payment to a new or 'safe' account",
    "message cue: pressure": "the message used time pressure or threats such as 'digital arrest'",
    "message cue: credential": "the message asked for an OTP, PIN or password",
    "message cue: remote_access": "the message asked for screen sharing or a remote access app",
    "message cue: refund": "the message offered a refund or cashback that required a payment action",
    "message cue: investment": "the message promised guaranteed or very high returns",
    "message cue: identity": "the message threatened KYC, PAN or Aadhaar problems",
    "message cue: secrecy": "the message told the customer to keep the payment secret",
    "payment check: new_beneficiary": "first payment from this customer to this payee",
    "payment check: amount_deviation": "amount far above the customer's usual payments",
    "payment check: payee_named_in_message": "the payee identifier appeared in the scam message",
    "account graph: fan_in_4plus": "the payee received money from four or more different accounts in 24 hours",
    "account graph: rapid_pass_through": "money was forwarded within minutes of arriving",
    "account graph: coordinated_accounts": "the payee was paid by several accounts already flagged for review",
    "account graph: sender_mule_signature": "the payer collected from several accounts and forwarded almost all of it within two minutes",
    "session: remote_access_app": "a remote access app was running during the payment",
    "session: screen_share": "the screen was being shared during the payment",
    "session: on_call_while_paying": "the customer was on a call while paying",
    "session: long_or_video_call": "the call was a long or video call, typical of 'digital arrest' coercion",
    "session: payee_pasted": "the payee details were pasted in rather than typed or chosen",
    "session: long_hesitation": "the customer hesitated for a long time before confirming",
    "session: app_hidden_mid_payment": "the banking app was sent to the background during the payment",
    "session: typing_rhythm_unlike_customer": "typing rhythm did not match the customer's enrolled pattern",
    "session: sim_swap_takeover": "an unenrolled device was used shortly after a SIM swap",
    "session: recent_sim_swap": "the SIM was swapped recently",
    "session: unenrolled_device": "the payment came from a device not enrolled for this customer",
    "session: attack_chain_full": "password reset, new device, new payee and transfer happened in sequence within 24 hours",
    "session: attack_chain_short": "new device, new payee and transfer happened within an hour",
}


def draft(pack: dict, campaign: dict | None = None, reporting_entity: str = "[Bank name, FIU-IND registration]") -> dict:
    a = pack["alert"]; p = a["payment"]; x = a["assessed"]; c = x.get("context", {}); pay = x.get("payment", {})
    grounds = [PLAIN.get(r, r) for r in c.get("reasons", []) if not r.startswith("negated")]
    grounds += [f"payment model: {r['feature']} = {r['display']}" for r in (pay.get("reasons") or [])]
    actions = [f"{e['analyst']}: {e['action']}" + (f" ({e['note']})" if e.get("note") else "") for e in a.get("actions", [])]
    linked = sorted(set(pack.get("linked_alerts", [])) | set((campaign or {}).get("alerts", [])) - {a["alert_id"]})
    sections = {
        "A_reporting_entity": reporting_entity,
        "B_transactions": [{"date_time": p["timestamp"], "from": f'{p["from_bank"]}/{p["from_account"]}', "to": f'{p["to_bank"]}/{p["to_account"]}',
                            "amount": p["amount_paid"], "currency": p["payment_currency"], "mode": p["payment_format"]}],
        "C_linked_accounts_and_alerts": {"alerts": linked, "campaign": (campaign or {}).get("campaign_id"),
                                         "victims_in_campaign": len((campaign or {}).get("victims", [])) or None},
        "D_grounds_of_suspicion": grounds or ["no rule fired; raised by the payment model score alone"],
        "E_action_taken": actions or ["no analyst action yet"],
        "F_attachments": {"evidence_pack_sha256": pack.get("pack_hash"), "ruleset": c.get("ruleset"),
                          "message_text": "redacted copy in the evidence pack" if c.get("redacted_text") else "none or not consented"},
    }
    summary = (f"On {p['timestamp']} account {p['from_account']} paid {p['amount_paid']} {p['payment_currency']} to {p['to_account']}. "
               f"RuseTect raised '{a['shown_decision']}' (context band {c.get('band')}). Grounds: " + "; ".join(grounds[:6]) + ".")
    if campaign:
        summary += f" The payment is part of campaign {campaign['campaign_id']} with {campaign['size']} linked alerts."
    return {"status": "DRAFT, for analyst review and signature", "summary": summary, "sections": sections,
            "disclaimer": "Generated from rules and the evidence pack; risk points are not fraud probabilities. Not filed."}
