"""Redis backed state equals the in memory engine; partitioned workers score every payment once."""
import json, math, os, shutil, socket, subprocess, sys, threading, time
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service")); sys.path.insert(0, os.path.join(ROOT, "tests"))
redis = pytest.importorskip("redis")
from test_compact_state import _rows, same  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("redis-server") is None, reason="redis-server not installed")


@pytest.fixture(scope="module")
def rurl():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    proc = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL)
    url = f"redis://127.0.0.1:{port}/0"
    for _ in range(100):
        try:
            redis.Redis.from_url(url).ping(); break
        except Exception:
            time.sleep(0.05)
    yield url
    proc.terminate()


def test_redis_state_matches_engine(rurl):
    from rusetect_engine import Engine
    from context_rules import sender_fan_in_24h
    from scale.redis_state import RedisState
    r = redis.Redis.from_url(rurl, decode_responses=True); r.flushdb()
    eng = Engine(load_models=False); eng.reset(); rs = RedisState(r, eng.cfg)
    for i, p in enumerate(_rows(6000)):
        p["eid"] = i
        sf = sender_fan_in_24h(eng, p)
        fa, fb = eng.features(p), rs.features(p)
        assert rs.last_sender_fan_in == sf
        bad = [k for k in fa if not (same(fa[k], fb[k]) or (isinstance(fa[k], float) and math.isclose(fa[k], fb[k], rel_tol=1e-9, abs_tol=1e-9)))]
        assert not bad, (i, {k: (fa[k], fb[k]) for k in bad})


def test_partitioned_workers_score_everything(rurl):
    from scale.producer import publish
    from scale.worker import run
    r = redis.Redis.from_url(rurl, decode_responses=True); r.flushdb()
    pays = []
    for i, p in enumerate(_rows(3000)):
        p["eid"] = i; pays.append(p)
    publish(r, pays, partitions=2)
    th = [threading.Thread(target=run, args=(rurl, k, 200)) for k in range(2)]
    [t.start() for t in th]; [t.join(timeout=240) for t in th]
    out = [json.loads(f["d"]) for _, f in r.xrange("decisions") if "d" in f]
    assert sorted(d["eid"] for d in out) == list(range(3000))
    assert {d["final_decision"] for d in out} <= {"Approve", "Step-up verification", "Hold for analyst", "Block + STR draft"}
