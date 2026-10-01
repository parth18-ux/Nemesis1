"""
Curated Indian scam / legitimate message corpus for the L-1 lure detector.

Why curated: no public, labelled dataset of Indian UPI-era scam messages could be downloaded from this environment
(Kaggle, Hugging Face, Mendeley and Zenodo are blocked). The scam templates paraphrase the modus operandi described in
public advisories (RBI "BE(A)WARE", I4C / cybercrime.gov.in, CERT-In, NPCI, TRAI and DoT alerts, state police
cyber-cell posts). The legitimate templates copy the structure of real bank, OTP, delivery and personal messages,
including the hard cases: banks that write "do not share your OTP", genuine bill reminders and relatives asking
for money.

Every message carries a template id. ml/train_message_model.py holds out whole templates when it evaluates, so a test
message is never a re-filled copy of a training sentence. These numbers are still about synthetic text: replace or
extend this corpus with real reported messages (see docs/datasets.md) before trusting the thresholds.
"""
import random

BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "PNB", "Bank of Baroda", "Kotak", "Canara Bank", "Paytm Payments Bank", "IDFC First"]
APPS = ["PhonePe", "Google Pay", "Paytm", "BHIM", "CRED"]
NAMES = ["Rahul", "Priya", "Amit", "Neha", "Rohit", "Sneha", "Vikas", "Anjali", "Arjun", "Pooja", "Suresh", "Kavita"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Kolkata", "Pune", "Chennai", "Jaipur", "Lucknow"]
UPI = ["refund.desk@ybl", "kyc.help@okaxis", "rbi.verify@paytm", "claimcell@ibl", "taskpay88@axl", "support.care@okicici", "loan.fee@upi", "gift.team@apl"]
BAD_LINKS = ["http://sbi-kyc-update.in/verify", "bit.ly/3kYcUpd", "https://hdfc-netbanking.co/reactivate", "http://pmkisan-gov.online/apply",
             "https://bsesbill-pay.top/pay", "tinyurl.com/rwd-pnts", "http://trade-vip.app/join", "https://yono-sbi.link/app", "http://cust0m-care.site/refund"]
GOOD_LINKS = ["https://www.onlinesbi.sbi", "https://hdfcbank.com", "https://www.icicibank.com", "https://www.myntra.com/my/orders", "https://www.irctc.co.in", "https://bescom.co.in"]
AGENCIES = ["CBI", "ED", "Mumbai Police", "Delhi Police Crime Branch", "Customs", "TRAI", "NCB", "Cyber Cell"]
COURIERS = ["FedEx", "DHL", "Blue Dart", "India Post"]


def _phone(r):
    return r.choice("6789") + "".join(r.choice("0123456789") for _ in range(9))


def _amt(r, lo, hi, step=1):
    return f"{r.randrange(lo, hi, step):,}"


def _fill(t, r):
    return t.format(bank=r.choice(BANKS), app=r.choice(APPS), name=r.choice(NAMES), city=r.choice(CITIES), upi=r.choice(UPI),
                    link=r.choice(BAD_LINKS), glink=r.choice(GOOD_LINKS), agency=r.choice(AGENCIES), courier=r.choice(COURIERS),
                    phone=_phone(r), last4=f"{r.randrange(1000, 9999)}", otp=f"{r.randrange(100000, 999999)}",
                    small=_amt(r, 49, 999), mid=_amt(r, 1000, 25000, 10), big=_amt(r, 25000, 500000, 100),
                    pct=r.choice([30, 40, 60, 100, 200, 300]), day=r.randrange(1, 28), hrs=r.choice([2, 12, 24]),
                    ref=f"{r.randrange(10**11, 10**12)}", time=f"{r.randrange(1, 12)}:{r.choice(['00', '15', '30', '45'])}")


SCAM = {
    "digital_arrest": [
        "This is Inspector {name} from {agency}. Your Aadhaar is linked to a money laundering case. An arrest warrant is issued. Stay on the video call and do not disconnect or tell anyone.",
        "Sir, {agency} has registered an FIR against your mobile number for illegal activity. To avoid digital arrest you must transfer Rs {big} to the RBI verification account for checking. Amount will be refunded in {hrs} hours.",
        "Your number will be blocked by TRAI in 2 hours due to complaints. Press 9 to speak to the {agency} officer. If you ignore, legal action will be taken.",
        "Madam we are from {agency} {city}. Your bank account received black money. Keep this confidential, the case is under the Official Secrets Act. Move your savings to the safe account we share for verification.",
        "Hello, I am calling from {agency}. A case is filed in your name. You are under digital arrest, do not leave the room or call your family until the enquiry completes.",
        "Aapke Aadhaar se juda bank account money laundering mein use hua hai. {agency} officer se baat karo abhi, warna arrest warrant nikal jayega. Kisi ko mat batana.",
    ],
    "parcel_customs": [
        "Your {courier} parcel to Taiwan has been seized by customs. It contains 5 passports and 200g MDMA. Press 1 to connect to the {city} narcotics officer.",
        "{courier}: A package booked with your Aadhaar is held for illegal items. Call customs officer on {phone} immediately to avoid police case.",
        "Your consignment is returned due to illegal drugs found. Your ID was used. Connect with {agency} on Skype now for statement.",
        "Dear customer your {courier} shipment is on hold at {city} customs. Pay clearance charges Rs {small} at {link} within {hrs} hours or it will be destroyed.",
        "India Post: your package could not be delivered due to incomplete address. Update details here {link} within 12 hours.",
    ],
    "kyc_block": [
        "Dear {bank} customer your account will be blocked today. Please update your PAN card immediately click {link}",
        "Your {bank} YONO account has been suspended due to pending KYC. Update KYC now: {link}",
        "{bank}: Dear user, your KYC has expired. Your account will be closed within 24 hrs. Call {phone} to verify.",
        "Dear customer your {app} wallet KYC is pending. Your wallet will be deactivated tonight. Update via {link}",
        "Aapka {bank} account aaj band ho jayega. PAN update karne ke liye link par click karein {link}",
        "Your SIM card will be blocked within 24 hours because KYC documents are not verified. Call customer care {phone} to update.",
        "Alert! Your debit card ending {last4} is blocked. To re-activate share card number and CVV with our executive on {phone}.",
    ],
    "upi_collect_refund": [
        "Your refund of Rs {mid} is ready. Approve the collect request on {app} and enter your UPI PIN to receive the amount in your account.",
        "Sir I sent you the payment for the sofa on OLX. Please scan this QR code and enter your PIN to receive Rs {mid}.",
        "Congratulations! Cashback of Rs {small} is credited. Click accept on the request from {upi} and enter UPI PIN to get it.",
        "I am an army officer, I want to buy your item. I will send money by {app}, just approve the request and put your PIN to receive.",
        "Wrongly sent you Rs {mid} by mistake, please return it to {upi} urgently, my child is in hospital.",
        "Refund processed. To credit Rs {mid} to your account enter UPI PIN on the request sent by {upi}.",
    ],
    "remote_access": [
        "To process your refund please install AnyDesk from Play Store and share the 9 digit code with me.",
        "Download the new {bank} YONO update from {link} (YONO_update.apk) otherwise your net banking stops today.",
        "Sir I am from {bank} customer care, an OTP has come on your phone, please read out the OTP to cancel the fraud transaction.",
        "Please share the 6 digit code you received to verify your account, it is only for security purpose.",
        "Install QuickSupport app so that our technician can fix your {app} issue. Share screen please.",
        "Wedding invitation for you, please download and open: Shaadi_card.apk",
        "Bhai, galti se tumhare number par OTP chala gaya, please forward kar do.",
    ],
    "task_job": [
        "Hi, I am {name} from a recruitment team. Part time job: like YouTube videos and earn Rs 3,000 to Rs 8,000 daily from home. Reply YES on WhatsApp.",
        "Earn Rs {mid} per day by rating hotels on Google. Join our Telegram task group, first 3 tasks free. Contact HR on WhatsApp.",
        "Congratulations, you completed 3 tasks and earned Rs 450. To unlock withdrawal do the prepaid merchant task, deposit Rs {mid} to {upi}.",
        "Work from home opportunity! Simple online tasks, salary Rs 25,000 per week. No experience needed. Join: {link}",
        "Your task commission is frozen because the order failed. Deposit Rs {big} more to recover the full amount with profit.",
        "Product review job, earn daily income Rs 5000. Message on telegram @hr_{name} to start today.",
    ],
    "investment": [
        "Join our VIP stock tips group on WhatsApp. Guaranteed returns of {pct}% in one month. SEBI registered advisor. {link}",
        "Institutional account IPO quota available only for VIP members. Deposit Rs {big} in the trading app to get allotment.",
        "Earn {pct}% profit on crypto trading in 7 days. Minimum investment Rs {mid}. Withdraw anytime. Join now {link}",
        "Our AI trading bot made 3x returns last week. Finance experts recommend it. Invest Rs {mid} today, double your money guaranteed.",
        "Dear investor your profit of Rs {big} is ready but you must pay 20% tax before withdrawal to the account below.",
        "Stock market professor shares free intraday tips daily. Students made Rs {big} profit. Join the WhatsApp group, limited seats.",
    ],
    "utility_disconnect": [
        "Dear consumer your electricity power will be disconnected tonight at 9.30 pm because your previous month bill was not updated. Please contact electricity officer {phone}.",
        "Bijli connection aaj raat kaat diya jayega. Pichle mahine ka bill update nahi hua. Turant {phone} par sampark karein.",
        "Your gas connection will be cut today due to pending KYC. Pay Rs {small} at {link} immediately.",
        "Dear customer your BSES electricity bill is overdue. Power will be disconnected tonight. Pay now to avoid cut {link}",
        "Your mobile number will be deactivated in 2 hours due to unpaid bill. Call {phone} to stop disconnection.",
    ],
    "fake_loan": [
        "Congratulations! Pre-approved personal loan of Rs {big} sanctioned without CIBIL or documents. Pay processing fee Rs {small} to {upi} to release.",
        "Instant loan approved in 5 minutes. No income proof needed. Download app from {link} and pay registration charge.",
        "Your loan file is approved. Pay GST and insurance charges of Rs {mid} today, loan amount will be credited within 1 hour.",
        "You took a loan from our app. Pay Rs {mid} today or we will send your photos to all your contacts.",
        "Mudra loan of Rs {big} approved under government scheme. Pay file charge Rs {small} to {upi}.",
    ],
    "lottery_prize": [
        "Congratulations! Your number has won Rs 25 lakh in KBC lucky draw. To claim call Rana Pratap Singh on WhatsApp {phone} and pay GST.",
        "You have won an iPhone 15 in the {app} anniversary lucky draw. Claim your prize here {link}",
        "Lucky winner! Your SIM card won a Tata Safari in the Jio lottery. Pay registration fee Rs {mid} to receive the car.",
        "WINNER!! As a valued customer you have been selected to receive a cash prize of Rs {big}. Claim at {link}",
        "Dear user you have won a scratch card worth Rs {mid}. Scan the QR code to claim it now.",
    ],
    "family_emergency": [
        "Papa I am in trouble, police caught me with a friend. Please send Rs {big} to this number urgently, don't tell mummy.",
        "Hi mom this is my new number, my phone fell in water. Can you send Rs {mid} for urgent fees? I will return tomorrow.",
        "Uncle, your son had an accident in {city}, he is in hospital. Send Rs {big} immediately for the operation to {upi}.",
        "Beta main police station mein hoon, mujhe pakad liya hai. Jaldi Rs {big} bhejo is number par, kisi ko mat batana.",
        "Your son is caught by police in a rape case. Pay Rs {big} now and we will close the matter, otherwise he goes to jail.",
        "Bhai mera account block hai, yeh mera naya number hai. Urgent Rs {mid} bhej de {upi} par, raat ko lauta dunga.",
    ],
    "sextortion": [
        "I have recorded your video call. Pay Rs {big} in 2 hours or I will upload the video and send it to your family and friends.",
        "Your obscene video is with us. YouTube officer here: pay Rs {mid} to remove it or it goes viral today.",
        "Sir your private video is uploaded. {agency} will arrest you. Pay to delete the clip immediately.",
    ],
    "govt_scheme": [
        "PM Kisan: your installment of Rs 6,000 is pending. Update Aadhaar and bank details at {link} to receive.",
        "Income tax refund of Rs {mid} approved. Verify your account number at {link} to receive the refund.",
        "Free laptop yojana 2026: register now, only 2 days left. Pay registration Rs {small}. {link}",
        "Government scheme: Rs 5,000 subsidy for every family. Fill form at {link} with Aadhaar and OTP.",
        "EPFO: your PF claim is on hold. Share Aadhaar OTP with our officer on {phone} to release.",
    ],
    "reward_points": [
        "Dear {bank} cardholder, your reward points worth Rs {mid} expire today. Redeem now: {link}",
        "{bank} NetBanking reward points of Rs {small} will expire. Download app and redeem: {link}",
        "Your {app} cashback of Rs {small} is expiring tonight. Claim at {link}",
    ],
    "fake_support": [
        "Your complaint for the failed {app} transaction is registered. Our executive will call you, please share the OTP to process refund.",
        "Hello, I am from {app} customer care. For refund of your failed payment, fill your card number and expiry at {link}",
        "Dear user your Myntra order is cancelled. For refund call customer support {phone} on WhatsApp.",
        "{bank} customer care here. Due to server upgrade please confirm your net banking password and the OTP sent to you.",
    ],
}

LEGIT = {
    "bank_debit": [
        "Rs {small}.00 debited from A/c XX{last4} on {day}-10-26 to VPA swiggy@ybl. UPI Ref {ref}. Not you? Call 18001234 - {bank}",
        "Dear Customer, Rs {mid} has been debited from your {bank} A/c XX{last4} on {day}-Oct. Avl Bal Rs {big}. If not done by you, report to the bank.",
        "{bank}: Sent Rs {small} from A/c *{last4} to {name} on {day}/10. UPI Ref No {ref}. Not you? SMS BLOCK to 9223008333",
        "Your A/c XX{last4} is credited with Rs {mid} on {day}-10-2026 by NEFT from {name}. -{bank}",
        "Txn of INR {small} done on {bank} Credit Card XX{last4} at ZOMATO on {day}-Oct. Avl limit INR {big}.",
        "Salary of Rs {big} credited to your A/c XX{last4}. Available balance Rs {big}. - {bank}",
    ],
    "otp": [
        "Dear customer, OTP for your {bank} debit card PIN generation is {otp}. It is valid for 10 minutes. Bank never asks for OTP.",
        "{otp} is your one time password to add beneficiary {name} on {bank} mobile banking. If not initiated by you, call 1800-425-3800.",
        "OTP {otp} for your {app} UPI registration. Do not share it with anyone including {app} employees.",
        "{otp} is your OTP for transaction of Rs {mid} at MYNTRA on {bank} card XX{last4}. Valid for 5 mins. Do not share this OTP with anyone.",
        "{otp} is the OTP to log in to {app}. Never share your OTP with anyone, {app} will never ask for it.",
        "Use OTP {otp} to verify your mobile number on IRCTC. Do not share it with anyone.",
        "Your OTP for {bank} NetBanking is {otp}. Bank never asks for OTP over call. Do not disclose it to anyone.",
        "{otp} is your verification code for WhatsApp. Don't share this code with others.",
    ],
    "bank_notice": [
        "Dear customer, a new device was used to log in to {bank} mobile banking on {day}-Oct at {time}. If this was not you, call the number on the back of your card.",
        "Your {bank} account XX{last4}: cheque no {last4} for Rs {mid} has been cleared.",
        "Your fixed deposit of Rs {big} with {bank} matures on {day}-Nov. It will be auto-renewed unless you instruct otherwise at the branch.",
        "Your request to update your mobile number at {bank} is received. If you did not request this, visit your branch.",
        "{bank} will never ask for your card details, PIN, OTP or password. Do not share them with anyone, even bank staff.",
        "Dear Customer, your KYC is due for periodic update as per RBI. Please visit your nearest {bank} branch with valid ID. Ignore if already done.",
        "Your EMI of Rs {mid} for loan A/c XX{last4} is due on {day}-Oct. Please maintain sufficient balance. - {bank}",
        "Your {bank} credit card statement is generated. Total due Rs {mid}, minimum due Rs {small}, due date {day}-Nov. Pay via {glink}",
        "Your {bank} debit card ending {last4} will expire this month. A new card has been dispatched to your registered address.",
        "Interest of Rs {small} credited to your {bank} savings account XX{last4} for the quarter.",
    ],
    "utility_bill": [
        "Electricity bill payment of Rs {mid} received for consumer no {ref}. Thank you for paying on time.",
        "Your Tata Power bill for Sep is generated. Amount Rs {small}, due {day}-Oct. Pay via the official app or {glink}",
        "Water bill for your property {ref} is Rs {small}. Last date {day}-Oct. Pay at the municipal office or the official portal.",
        "Dear consumer, your electricity bill for Sep is Rs {mid}, due date {day}-Oct-2026. Pay at {glink} or any authorised counter.",
        "Your Jio prepaid plan expires on {day}-Oct. Recharge with Rs 299 to continue unlimited calls and data.",
        "Payment of Rs {small} received for your gas bill. Thank you. Consumer no {ref}.",
        "Your Airtel postpaid bill of Rs {small} is generated, due on {day}-Oct. Pay on the Airtel Thanks app.",
    ],
    "delivery": [
        "Your Myntra order #402-{ref} is out for delivery today. Track: {glink}",
        "{courier}: Shipment {ref} has been delivered to your address in {city}. Thank you.",
        "Your Swiggy order is on the way! {name} is your delivery partner.",
        "Flipkart: your order will be delivered by {day}-Oct. Share OTP {otp} only with the delivery agent at your door.",
        "Your Zomato order has been picked up and will arrive in 25 minutes.",
    ],
    "personal": [
        "Yaar kal ki party ka hisaab: tera share Rs {small} hai, {app} kar dena jab time mile.",
        "Maa, maine fees bhar di hai, receipt WhatsApp kar di. Chinta mat karo.",
        "{name} beta, dadi ki dawai khatam ho gayi, aate waqt le aana. Paise main de dungi.",
        "Running 10 minutes late, traffic near {city} station. Start without me.",
        "Hey {name}, are we still meeting for lunch tomorrow at 1?",
        "Bhai 500 bhej de {app} pe, kal lauta dunga. Petrol khatam ho gaya.",
        "Mummy I reached {city} safely, will call you after dinner.",
        "Papa please send Rs {mid} for the hostel fee, the last date is Friday. I will send the receipt.",
        "Can you pick up milk on the way home? Also call your mom, she was asking.",
        "Happy birthday {name}! Have a great year ahead",
        "Meeting moved to 4 pm today, please inform the team.",
        "Kal college mat aana, holiday declare ho gaya hai.",
        "Sent you Rs {small} for the movie tickets, check your {app}.",
        "Didi, mausi ki tabiyat theek nahi hai, hospital mein hai. Shaam ko aa jaana.",
    ],
    "services": [
        "Your {app} refund of Rs {small} for the cancelled order has been processed and will reach your bank account in 3-5 working days.",
        "Your PF withdrawal claim has been settled. Rs {big} will be credited to your bank account XX{last4}.",
        "Your Aadhaar-PAN link status: linked successfully. No further action is required.",
        "Your appointment with Dr. {name} at Apollo {city} is confirmed for {day}-Oct at {time} am.",
        "Your IRCTC ticket PNR {ref} is confirmed. Train 12951, coach B3, berth 42.",
        "Thank you for shopping at DMart. Your bill amount is Rs {small}. Visit again.",
        "Your Ola ride with {name} is arriving in 3 mins. Vehicle KA01AB{last4}.",
        "Interview scheduled with Infosys HR on {day}-Oct at {time}. Please carry your ID and resume. No fee is charged at any stage.",
        "Your Aadhaar update request has been received at the enrolment centre. Status can be checked on uidai.gov.in",
        "Your income tax return for AY 2026-27 has been processed. Refund of Rs {mid} will be credited to your pre-validated bank account.",
        "Reminder: your car insurance policy expires on {day}-Oct. Renew on the insurer's official app or website.",
    ],
    "marketing": [
        "Big Billion Days sale is live! Up to 80% off on electronics. Shop now on the Flipkart app.",
        "Get flat Rs 100 cashback on your first bill payment with {app}. T&C apply.",
        "New season collection at Westside, 20% off this weekend at your nearest store.",
        "Pre-book the new Galaxy phone and get free earbuds. Visit Samsung.com",
    ],
}


def generate(n_per_template=12, seed=0):
    """List of (text, label, workflow, template_id). label 1 = scam."""
    r = random.Random(seed); out = []
    for label, groups in ((1, SCAM), (0, LEGIT)):
        for wf, temps in groups.items():
            for i, t in enumerate(temps):
                tid = f"{wf}#{i}"
                seen = set()
                for _ in range(n_per_template * 3):
                    s = _fill(t, r)
                    if r.random() < 0.15: s = s.lower()
                    if s not in seen:
                        seen.add(s); out.append((s, label, wf, tid))
                    if len(seen) >= n_per_template:
                        break
    return out
