"""
Cross bank mule intelligence (session layer used Hyperledger Fabric chaincodes and Ethereum anchoring).

The same guarantees with far less machinery: an append only ledger where every entry is signed (Ed25519) by the
bank that publishes it and chained to the previous entry's hash, so any change to history breaks verification.
Account and device identifiers are never shared in clear: each is an HMAC with a consortium secret, so members can
match the same account without learning anything about accounts they have not seen. Three registries, as in the
original design: mule accounts, device hashes linked to fraud, SIM swap takeovers. A Merkle root of the ledger can
be published anywhere (a regulator, a public chain) to anchor it, without putting any data on chain.
"""
from __future__ import annotations
import base64, hashlib, hmac, json, os, time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

REGISTRIES = {"mule_account", "fraud_device", "sim_swap_takeover"}


def _canon(o) -> bytes:
    return json.dumps(o, sort_keys=True, separators=(",", ":")).encode()


class Bank:
    def __init__(self, bank_id: str):
        self.bank_id = bank_id; self.key = Ed25519PrivateKey.generate()
        self.public_b64 = base64.b64encode(self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()


class Consortium:
    def __init__(self, secret: str | None = None, members=("BANK-A", "BANK-B", "BANK-C")):
        self.secret = (secret or os.environ.get("RUSETECT_CONSORTIUM_SECRET") or "demo-consortium-secret").encode()
        self.banks = {b: Bank(b) for b in members}
        self.reset()

    def reset(self):
        self.ledger = []

    def token(self, kind: str, value: str) -> str:
        return hmac.new(self.secret, f"{kind}|{value}".encode(), hashlib.sha256).hexdigest()[:32]

    def publish(self, bank_id: str, registry: str, identifier: str, reason: str, evidence_sha256: str | None = None) -> dict:
        if registry not in REGISTRIES:
            raise ValueError(f"registry must be one of {sorted(REGISTRIES)}")
        bank = self.banks[bank_id]
        body = {"seq": len(self.ledger) + 1, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "bank": bank_id,
                "registry": registry, "token": self.token(registry, identifier), "reason": reason[:120], "evidence_sha256": evidence_sha256,
                "prev": self.ledger[-1]["hash"] if self.ledger else "0" * 64}
        e = {**body, "sig": base64.b64encode(bank.key.sign(_canon(body))).decode()}
        e["hash"] = hashlib.sha256(_canon({**body, "sig": e["sig"]})).hexdigest()
        self.ledger.append(e)
        return e

    def verify(self) -> bool:
        prev = "0" * 64
        for e in self.ledger:
            body = {k: e[k] for k in ("seq", "at", "bank", "registry", "token", "reason", "evidence_sha256", "prev")}
            if e["prev"] != prev or hashlib.sha256(_canon({**body, "sig": e["sig"]})).hexdigest() != e["hash"]:
                return False
            try:
                Ed25519PublicKey.from_public_bytes(base64.b64decode(self.banks[e["bank"]].public_b64)).verify(base64.b64decode(e["sig"]), _canon(body))
            except (InvalidSignature, KeyError):
                return False
            prev = e["hash"]
        return True

    def check(self, registry: str, identifier: str) -> dict:
        tok = self.token(registry, identifier)
        hits = [e for e in self.ledger if e["registry"] == registry and e["token"] == tok]
        return {"flagged": bool(hits), "banks": sorted({e["bank"] for e in hits}), "entries": len(hits)}

    def merkle_root(self) -> str:
        level = [bytes.fromhex(e["hash"]) for e in self.ledger] or [b"\0" * 32]
        while len(level) > 1:
            if len(level) % 2:
                level.append(level[-1])
            level = [hashlib.sha256(level[i] + level[i + 1]).digest() for i in range(0, len(level), 2)]
        return level[0].hex()
