"""
Security for the review desk and the verdicts.

* Analyst identity comes from an API key (header X-Analyst-Key), not from a typed name, so the four eyes rule
  ("two different analysts approve a block") is enforced on authenticated people. Roles: analyst, supervisor, viewer.
  Keys are read from RUSETECT_ANALYST_KEYS (JSON {"key": {"analyst": "...", "role": "..."}}); without it the server
  runs in DEMO mode with three published demo keys, and says so in /api/health.
* Every review action carries a fresh nonce (header X-Nonce); a nonce seen in the last 10 minutes is rejected, so a
  captured request cannot be replayed.
* A small per client rate limit protects the write endpoints.
* Verdicts and evidence packs are signed with Ed25519 (RUSETECT_SIGNING_KEY = base64 32 byte seed; otherwise an
  ephemeral key per process). Anyone can verify with GET /api/v2/keys/public. The original design named
  ML-DSA (FIPS 204); the Python cryptography library used here has no ML-DSA yet, so swapping it in later is a
  change to sign() and verify() only.
"""
from __future__ import annotations
import base64, hashlib, json, os, time
from collections import defaultdict, deque

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

DEMO_KEYS = {"demo-analyst-1": {"analyst": "analyst.asha", "role": "analyst"},
             "demo-analyst-2": {"analyst": "analyst.ravi", "role": "analyst"},
             "demo-supervisor": {"analyst": "supervisor.meera", "role": "supervisor"},
             "demo-viewer": {"analyst": "viewer.ops", "role": "viewer"}}
ROLE_ACTIONS = {"viewer": set(), "analyst": {"release", "verify", "hold", "approve_block"},
                "supervisor": {"release", "verify", "hold", "approve_block"}}


def load_keys():
    raw = os.environ.get("RUSETECT_ANALYST_KEYS")
    if raw:
        return json.loads(raw), False
    return dict(DEMO_KEYS), True


KEYS, DEMO_MODE = load_keys()

PRODUCTION_SECRETS = ("RUSETECT_SIGNING_KEY", "RUSETECT_CONSORTIUM_SECRET", "RUSETECT_HASH_SALT")


def missing_production_secrets(env=os.environ) -> list:
    """Fail closed: with RUSETECT_ANALYST_KEYS set the deployment is not a demo, so the hard-coded demo consortium secret,
    demo hash salt and per-process signing key must not be used."""
    if not env.get("RUSETECT_ANALYST_KEYS"):
        return []
    return [k for k in PRODUCTION_SECRETS if not (env.get(k) or "").strip()]


_missing = missing_production_secrets()
if _missing:
    raise RuntimeError(f"RUSETECT_ANALYST_KEYS is set but {', '.join(_missing)} missing; refusing to start with demo secrets")


def analyst_for(key: str | None) -> dict | None:
    return KEYS.get(key or "")


def allowed(role: str, action: str) -> bool:
    return action in ROLE_ACTIONS.get(role, set())


class NonceCache:
    def __init__(self, ttl_s: int = 600):
        self.ttl, self.seen = ttl_s, {}

    def use(self, nonce: str | None) -> bool:
        """True if the nonce is fresh (and records it); False if missing, too short or reused."""
        now = time.time()
        if len(self.seen) > 50_000:
            self.seen = {n: t for n, t in self.seen.items() if now - t < self.ttl}
        if not nonce or len(nonce) < 8 or len(nonce) > 128:
            return False
        t = self.seen.get(nonce)
        if t is not None and now - t < self.ttl:
            return False
        self.seen[nonce] = now
        return True


class RateLimiter:
    def __init__(self, limit: int = 240, per_s: float = 10.0):
        self.limit, self.per, self.hits = limit, per_s, defaultdict(deque)

    def allow(self, client: str) -> bool:
        now = time.time()
        if len(self.hits) > 50_000:  # bound memory: forget clients with no hit inside the window
            self.hits = defaultdict(deque, {k: q for k, q in self.hits.items() if q and q[-1] > now - self.per})
        q = self.hits[client]
        while q and q[0] <= now - self.per:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


class Signer:
    def __init__(self):
        seed = os.environ.get("RUSETECT_SIGNING_KEY")
        self.ephemeral = not seed
        self.key = Ed25519PrivateKey.from_private_bytes(base64.b64decode(seed)) if seed else Ed25519PrivateKey.generate()
        pub = self.key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_b64 = base64.b64encode(pub).decode()
        self.key_id = hashlib.sha256(pub).hexdigest()[:16]

    def sign(self, obj) -> dict:
        digest = hashlib.sha256(_canon(obj)).hexdigest()
        sig = self.key.sign(_canon(obj))
        return {"alg": "Ed25519", "key_id": self.key_id, "sha256": digest, "signature": base64.b64encode(sig).decode()}

    @staticmethod
    def verify(obj, signature_b64: str, public_b64: str) -> bool:
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(public_b64)).verify(base64.b64decode(signature_b64), _canon(obj))
            return True
        except (InvalidSignature, ValueError):
            return False
