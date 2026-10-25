# Person, session and network layer

RuseTect catches scams in the money flow and the message. This layer adds three more levels: the person (is it really
the customer?), the session (is the customer being coached while they pay?) and the network (is this one scam hitting
many people, and has another bank already flagged the payee?). It answers with a pause, not a panic. This document lists
each feature, the design choice behind it, where the code lives and what was measured. Ruleset: **ctx-1.4**.

## 1. Features and design choices

| Feature | In RuseTect | Design choice | Code |
|---|---|---|---|
| Session level: live call, screen share, remote access app, paste, hesitation, app hidden | A fourth evidence domain (up to 40 session points) next to message, payment and account graph | The two domain rule still holds: coaching cues alone can only ask the customer (pause screen), never hold the money; with payment or message evidence they escalate | `service/session_signals.py` |
| Coercion score with weights (screen share 0.30, paste 0.20, hesitation 0.20, SIM swap 0.15, app hidden 0.10, network 0.05), step up at 0.65 | Kept as the coercion index; at 0.65 or more the customer is always asked | Reported next to readable cue points, so an analyst sees which cue fired | same |
| Person level: device hash vs enrolled, SIM swap check before the OTP | Identity gate: unenrolled device within 72 h of a SIM swap means no OTP, re-verify by video KYC or branch | It is an authentication step with a recovery path, not a payment block | `identity_check`, `POST /api/v2/identity/check` |
| Person level: typing rhythm | Typing profile (3 session enrolment) scored by a person-independent verifier | Tested on real keystrokes of people never seen in training (section 3); kept as a weak signal (5 points) | `service/behaviour.py`, `ml/aalto_eval.py` |
| Person level: mouse drift (CNN-LSTM on Balabit) | Mouse dynamics: actions (move, click, drag), 24 shape and speed features, per customer model | **Validated on the real Balabit data** (section 3) | `service/mouse_dynamics.py`, `ml/balabit_eval.py` |
| Attack chain order: password reset, new device, new payee, maximum transfer | Ordered event chains: full (24 h) and short (new device, new payee, transfer within 1 h) | A full chain plus a large new payee transfer reaches an analyst by policy | `attack_chain` |
| One 0 to 1000 score | Display score derived from RuseTect's own decision (0 to 299 approve, 300 to 599 verify, 600+ analyst hold) | It can never disagree with the decision, and it is labelled as a ranking aid, not a probability | `service/risk_score.py` |
| Pause screen: "Is someone on the phone asking you to make this payment?", Hindi and English, continue after 30 min | Pause screen with the question chosen from the evidence (digital arrest, call or screen share, KYC or refund, investment, family emergency) | Uses the message model's scam workflow, so a digital arrest victim gets the digital arrest question | `service/intervention.py` |
| Duress PIN: scammer sees success, funds held, family alerted | Same, with PINs stored only as salted PBKDF2 hashes and compared in constant time | The alert jumps to the top of the analyst queue | `DuressVault`, `POST /api/v2/payments/confirm` |
| Campaign clustering (HDBSCAN on payee VPA, script timing, device) | Campaigns linked by the same payee, the same payee identifier hashed out of messages, or near identical redacted scripts | Deterministic and explainable: every campaign says why its alerts are linked; no density parameter to tune without labelled campaigns | `service/campaigns.py`, `GET /api/v2/campaigns` |
| Cross bank mule flags on Hyperledger Fabric, anchored on Ethereum Sepolia | A signed (Ed25519), hash chained ledger of HMAC tokens; a Merkle root can anchor it anywhere | Same guarantees (tamper evident, no PII shared, members can match accounts) without a blockchain network; a payee is published only after two analysts approve a block | `service/consortium.py` |
| Accounts paid by 5+ flagged senders | Flagged sender count | **Measured and switched off by default** (section 3) | `service/coordination.py`, `ml/coordination_eval.py` |
| AI investigator (LLaMA 3.3 via Groq) drafts an STR | STR draft from a template, sections following an FIU-IND report | Nothing leaves the bank (RBI data localisation); every sentence traces to a field in the evidence pack; an analyst must sign | `service/str_report.py`, `GET /api/v2/alerts/{id}/str` |
| Signed JWTs, rate limits, anti replay nonces, honeypots | Analyst API keys with roles, four eyes enforced on authenticated identities, nonces, per client rate limit | Fixes a real gap: the two analyst block rule used to trust a typed name | `service/security.py` |
| ML-DSA signs every verdict | Ed25519 signs every verdict and evidence pack; public key at `/api/v2/keys/public` | ML-DSA is not in the Python cryptography library yet; swapping it in later changes two functions | same |
| Customer recovery via re-verification | `POST /api/v2/alerts/{id}/reverify` sends the case back to an analyst for release | | `service/server.py` |
| Hindi and English | Pause screens in both; Hindi and Hinglish scam cues (police, CBI, digital arrest, OTP batao, safe khate, kisi ko mat batana) | | `service/context_rules.py` |

Left out on purpose: Isolation Forest in the model blend (0.004 PR-AUC on IBM data in RuseTect's own benchmark), zero
knowledge proofs, and the public testnet anchor.

## 2. Try it

Open `/investigate`, pick a scam (digital arrest, KYC lure with remote access, investment, account takeover chain,
SIM swap login, a genuine family call), press Assess. You see the 0 to 1000 score, the reasons, and the customer's
phone with the pause screen (English or Hindi); stop or continue, or confirm with the duress PIN. Approve a block with
two different demo analysts and the payee is published to the cross bank ledger. Open the STR draft from the alert.

## 3. Measured

### Mouse dynamics on real data (Balabit, 10 users, labelled impostor sessions)

| Mouse actions seen | Median user ROC-AUC | Mean user ROC-AUC | Median equal error rate |
|---|---|---|---|
| 1 | 0.742 | 0.793 | 0.346 |
| 5 | 0.875 | 0.886 | 0.224 |
| 10 | 0.893 | 0.909 | 0.180 |
| 20 | 0.915 | 0.923 | 0.133 |
| 50 | 0.941 | 0.946 | 0.096 |

A bank app sees dozens of mouse or touch actions before a payment is confirmed, so the useful range is 10 to 50
actions. One user is perfectly separable from a single action (probably a different screen setup), which is why the
median is the headline. `results/balabit_eval.json`.

### Scenario replay suite (authored, through the real API, 40 randomised runs each)

* Scam payments stopped before money moved (pause screen, analyst hold, withheld OTP or duress hold): **280 of 280**
  (digital arrest, KYC lure with remote access, investment scam, account takeover chain, typing impostor on a new
  device, SIM swap takeover, duress PIN).
* Genuine payments with a false alarm: **1 of 200**; wrongly held for an analyst: **0 of 200**
  (known payee, older and slower customer, new payee for rent, a small payment during a family call, a first time user).
* Fairness: false alarm rate for the older, slower customer 0% vs 0% for others
  (hesitation alone adds only 5 points by design).
* Time to decision (API timer, median): 10.2 ms for a digital arrest payment with message and session, 3.5 ms for an ordinary payment.
* Mule ring fixture (victims with short history): 9 of 24 payments do not go straight through.

These scenarios were written to exercise each signal; the takeover and impostor rules were set as explicit policy
after the first run showed a full takeover chain only reached verification. They are not real world rates.
`results/scenario_suite.json`.

### Flagged sender network rule ("paid by 5+ flagged senders"), on the full IBM files

Inside the real fusion on the test weeks, the least noisy setting (7 days, senders held for an analyst, 3 of them)
adds 693 flags to catch 1 more laundering payment on HI-Small and
726 flags for 1 on LI-Small. Looser settings add thousands of flags for a dozen catches. So the local
count is off by default; the cross bank version stays on, because it only fires for payees that two analysts already
approved as mules. `results/coordination_eval.json`.

### Retraining with the unused data

See `docs/RETRAINING.md`: the payment models now also learn from LI-Small, and the account opening gate was retrained with the BAF variants.


## 4. Datasets

| Dataset | Used for | Link | Status here |
|---|---|---|---|
| IBM AMLworld HI-Small, LI-Small (synthetic payments, labelled rings) | payment models, rings, context rules | https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml | used in full |
| Feedzai Bank Account Fraud, Base and Variants I to V | account opening gate, bias tests | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | used in full |
| **Balabit Mouse Dynamics Challenge** (real mouse data, 10 users) | mouse dynamics | https://github.com/balabit/Mouse-Dynamics-Challenge | used (section 3) |
| **Aalto 136M Keystrokes** (Dhakal et al., CHI 2018; 1.4 GB zip, research and non commercial use) | typing rhythm | https://userinterfaces.aalto.fi/136Mkeystrokes/ (data: `data/Keystrokes.zip`) | not reachable from this workspace; download it and run the typing evaluation locally |
| **PaySim** (synthetic mobile money, Lopez-Rojas et al.) | mobile money transaction benchmark | https://www.kaggle.com/datasets/ealaxi/paysim1 | needs a Kaggle login; RuseTect's IBM data covers the same role with labelled rings |

## 5. Honest limits

* Session, typing and coercion signals have no public benchmark. The scenario suite is authored, so it checks
  mechanics, not real world rates. A pilot needs consented session data from a bank app.
* Mouse dynamics was validated on desktop mouse data; a phone app needs touch data and its own enrolment.
* Demo analyst keys are published in demo mode (`/api/health` says so); set `RUSETECT_ANALYST_KEYS` and
  `RUSETECT_SIGNING_KEY` for anything real.
