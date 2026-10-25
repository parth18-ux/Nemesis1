# Person, session and network layer

session layer was a separate project of mine: catch the scam while the victim is being coached, at three levels (the person,
the session, the network), and answer with a pause, not a panic. RuseTect caught the scam in the money flow and the
message. This document maps every session layer piece to where it now lives in RuseTect, what changed on the way, and what
was measured. Ruleset: **ctx-1.4**.

## 1. What moved over, and what got better

| session layer idea | In RuseTect | Better than the original because | Code |
|---|---|---|---|
| Session level: live call, screen share, remote access app, paste, hesitation, app hidden | A fourth evidence domain (up to 40 session points) next to message, payment and account graph | The two domain rule still holds: coaching cues alone can only ask the customer (pause screen), never hold the money; with payment or message evidence they escalate | `service/session_signals.py` |
| Coercion score with weights (screen share 0.30, paste 0.20, hesitation 0.20, SIM swap 0.15, app hidden 0.10, network 0.05), step up at 0.65 | Kept as the coercion index; at 0.65 or more the customer is always asked | Reported next to readable cue points, so an analyst sees which cue fired | same |
| Person level: device hash vs enrolled, SIM swap check before the OTP | Identity gate: unenrolled device within 72 h of a SIM swap means no OTP, re-verify by video KYC or branch | It is an authentication step with a recovery path, not a payment block | `identity_check`, `POST /api/v2/identity/check` |
| Person level: typing rhythm (BehaveFormer + Siamese on Aalto) | Typing profile (median z score over key pairs, 3 session enrolment) | Honest scope: Aalto is not reachable from here, so it is a simple statistical profile tested on synthetic timings and kept as a weak signal (5 points) | `service/behaviour.py` |
| Person level: mouse drift (CNN-LSTM on Balabit) | Mouse dynamics: actions (move, click, drag), 24 shape and speed features, per customer model | **Validated on the real Balabit data** (section 3) | `service/mouse_dynamics.py`, `ml/balabit_eval.py` |
| Attack chain order: password reset, new device, new payee, maximum transfer | Ordered event chains: full (24 h) and short (new device, new payee, transfer within 1 h) | A full chain plus a large new payee transfer reaches an analyst by policy | `attack_chain` |
| Unified 0 to 1000 score | Display score derived from RuseTect's own decision (0 to 299 approve, 300 to 599 verify, 600+ analyst hold) | It can never disagree with the decision, and it is labelled as a ranking aid, not a probability | `service/risk_score.py` |
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

Not carried over: Isolation Forest in the model blend (0.004 PR-AUC on IBM data in RuseTect's own benchmark), zero
knowledge proofs, and the public testnet anchor.

## 2. Try it

Open `/investigate`, pick a scam (digital arrest, KYC lure with remote access, investment, account takeover chain,
SIM swap login, a genuine family call), press Assess. You see the 0 to 1000 score, the reasons, and the customer's
phone with the pause screen (English or Hindi); stop or continue, or confirm with the duress PIN. Approve a block with
two different demo analysts and the payee is published to the cross bank ledger. Open the STR draft from the alert.

## 3. Measured

__SESSION_RESULTS__

## 4. Datasets

| Dataset | Used for | Link | Status here |
|---|---|---|---|
| IBM AMLworld HI-Small, LI-Small (synthetic payments, labelled rings) | payment models, rings, context rules | https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml | used in full |
| Feedzai Bank Account Fraud, Base and Variants I to V | account opening gate, bias tests | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | used in full |
| **Balabit Mouse Dynamics Challenge** (real mouse data, 10 users) | mouse dynamics | https://github.com/balabit/Mouse-Dynamics-Challenge | used (section 3) |
| **Aalto 136M Keystrokes** (Dhakal et al., CHI 2018; 1.4 GB zip, research and non commercial use) | typing rhythm | https://userinterfaces.aalto.fi/136Mkeystrokes/ (data: `data/Keystrokes.zip`) | not reachable from this workspace; download it and run the typing evaluation locally |
| **PaySim** (synthetic mobile money, Lopez-Rojas et al.) | the transaction model | https://www.kaggle.com/datasets/ealaxi/paysim1 | needs a Kaggle login; RuseTect's IBM data covers the same role with labelled rings |

## 5. Honest limits

* Session, typing and coercion signals have no public benchmark. The scenario suite is authored, so it checks
  mechanics, not real world rates. A pilot needs consented session data from a bank app.
* Mouse dynamics was validated on desktop mouse data; a phone app needs touch data and its own enrolment.
* Demo analyst keys are published in demo mode (`/api/health` says so); set `RUSETECT_ANALYST_KEYS` and
  `RUSETECT_SIGNING_KEY` for anything real.
