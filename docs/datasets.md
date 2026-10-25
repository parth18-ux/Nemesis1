# Datasets: what we use and what to look for next

Download anything below into `data/` (ignored by git), or point `RUSETECT_DATA` at it. Check each licence before use.

## In use

| Data | Side | Size | Limits |
|---|---|---|---|
| IBM AMLworld HI-Small (Kaggle, Altman et al., NeurIPS 2023) | Payments (L1 to L4) | 5.08M payments, 5,177 laundering, labelled rings | Synthetic; USD-era banking, not UPI |
| IBM AMLworld LI-Small (same source) | Payments (training and out of distribution test) | 6.92M payments, lower laundering rate | Synthetic |
| Feedzai Bank Account Fraud, Base (Kaggle, NeurIPS 2022) | Onboarding (L0) | 1M applications | Synthetic; US-style fields |
| Feedzai Bank Account Fraud, Variants I to V | Onboarding (L0 training, bias tests) | 5M applications | Synthetic, controlled bias shifts |
| Aalto 136M Keystrokes (CHI 2018) | Typing rhythm check | 168,593 people; 7,169 random plus 5,254 older or phone typists used | Real typing test website, not a banking app |
| Balabit Mouse Dynamics Challenge | Mouse dynamics | 10 users, labelled impostor sessions | Real desktop mouse data |
| PaySim (Lopez-Rojas et al., 2016) | Payments benchmark | 6.36M mobile money transactions, 8,213 fraud | Synthetic; see the note below |
| UCI SMS Spam Collection (via GitHub mirror) | Messages (L-1) | 5,574 SMS, 747 spam | Real but ~2011 UK; spam ≠ scam |
| `ml/scam_corpus.py` | Messages (L-1) | 1,791 messages, 144 templates | Written from public advisories; synthetic |

### Note on PaySim

`ml/paysim_eval.py` runs RuseTect's payment recipe on PaySim with the same protocol (time ordered 60/20/20 split,
past only features, XGBoost and LightGBM averaged, thresholds set by alert budget on validation). Results on the test
period (4,254 fraud cases), from `results/paysim_eval.json`:

| Model | Test PR-AUC | Recall at 0.1% alert budget |
|---|---|---|
| Account history only (no balance fields) | 0.52 | 45% at 53% precision |
| Bank visible features including the payer's balance | 0.9997 | 99.9% |
| Every balance column (leaky upper bound) | 1.0 | 99.98% |
| PaySim's own rule (transfers above 200,000) | n/a | 0.3% |
| Rule: the payment empties the account | n/a | 97% at 1% precision |

The near perfect scores come from the simulator: every fraud case empties the payer's balance, so PaySim cannot tell
good models apart. RuseTect keeps the IBM and Feedzai data as its main benchmarks.

## Better candidates

### Messages and calls (biggest gap)

| Candidate | Why | Notes |
|---|---|---|
| **SMS Phishing Dataset** (Mishra & Soni, Mendeley Data, 2022) | ~5,900 SMS labelled ham / spam / **smishing**, with URL/e-mail/phone flags | Closest public match to "scam vs spam"; mostly English |
| **Real Indian reports**: Sanchar Saathi *Chakshu* (DoT), cybercrime.gov.in / 1930 helpline (I4C), bank fraud report inboxes | Real UPI-era scam text, with the scam type | Not public; needs a partnership or data sharing letter. Strongest possible upgrade |
| **Crowd-reported from the app itself** | A "report this message" button in the console/app builds a labelled Indian corpus | Plan consent and PII scrubbing (phone, UPI, names) from day one |
| Hinglish / Hindi spam or phishing sets on Kaggle and GitHub (search "Hinglish SMS spam", "Hindi spam detection") | Code-mixed text, which our corpus has little of | Quality varies; check labels by hand on a sample |
| Fraud-call transcript datasets (search "scam call transcripts", "telecom fraud dialogue dataset", "vishing dataset") | Covers voice scams after speech to text | Most published sets are Chinese or English; useful for method, not Indian phrasing |
| Deepfake voice corpora (ASVspoof 2019/2021, In-the-Wild) | For a voice clone detector upstream of L-1 | Detects synthetic audio, not scam intent; a separate model |

### Payments

| Candidate | Why | Notes |
|---|---|---|
| **SAML-D** (Oztas et al., 2023, Kaggle "Anti Money Laundering Transaction Data") | 9.5M transactions, 28 typologies (11 normal, 17 suspicious), includes high risk corridors and payment types | Synthetic, but richer typologies than AMLworld; good second benchmark |
| **IBM AMLworld HI-Medium / LI-Small** | Same schema, so a drop-in: more scale (HI-Medium ~32M) and a harder low illicit regime (LI) | Tests whether thresholds survive a prevalence shift |
| **PaySim** (Lopez-Rojas, Kaggle) | Mobile-money simulator (TRANSFER → CASH_OUT drains), closest in spirit to wallets/UPI | 6.3M rows; fraud pattern is simple, so expect easy scores |
| **Elliptic / Elliptic++** (Bitcoin) | Graph-labelled illicit flows; tests the GNN and ring linker | Crypto, not UPI |
| **Feedzai BAF variants I to V** | Same as Base with controlled bias shifts | Stress-tests L0 fairness thresholds |
| NPCI / RBI aggregate fraud statistics (RBI annual report, NPCI dispute data) | Real prevalence and loss figures to set alert budgets | Aggregates only, no rows |

## What would change the most

1. Real labelled Indian scam messages, even 2,000 to 3,000 with the scam type. They would replace the synthetic templates and
   make the L-1 thresholds trustworthy.
2. Linked message → payment pairs, i.e. victim reports that include both the lure and the transaction. They are the
   only way to measure the fusion layer, not just unit test it.
3. A second payment dataset (SAML-D) to show the cascade is not tuned to one generator.
