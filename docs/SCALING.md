# Scaling RuseTect

How the scoring path goes from one Python process to a bank's payment volume, what is built and tested in this
repository, and what is still design. Every number below comes from a script in `ml/` and a file in `results/`;
all data is synthetic (IBM AMLworld, Feedzai BAF).

## 1. What has to scale

| Piece | State | Cost per payment | How it scales |
|---|---|---|---|
| Feature computation | per account totals, 24 h windows, "has A paid B" index | a few dictionary and array updates | partition by account; state in a keyed store |
| L1 gatekeeper (30 trees) | none | well under a millisecond, batched | stateless, add processes |
| L2 XGBoost + LightGBM (about 20% of payments) | none | under a millisecond, batched | stateless, add processes |
| Context layer (rules) | none beyond the features | microseconds | stateless |
| Ring linking, owner merge | graph of alerted payments | runs on alerts only | hourly batch job |

The models are cheap. **The account state is the scaling problem.**

## 2. Measured on the full HI-Small file (5,078,345 payments, 515,088 accounts)

`ml/scale_bench.py` streams every payment, in time order, through three state layouts in separate processes.

| Layout | Account state | Payments per second (one core) | Features vs prototype |
|---|---|---|---|
| Prototype objects (`service/rusetect_engine.py`) | 2,136 MB | 42,679 | reference |
| Compact, exact pair index | 1,372 MB | 41,459 | identical on all 5,078,345 payments |
| Compact, Bloom pair index (20M capacity, 0.1% false positives) | 1,361 MB | 31,673 | 0.000% of test payments differ; decisions agree on 100.000% |

Account state is resident memory after streaming minus resident memory right after loading the payments. Inside the
compact layout the per account arrays take 39 MB and the exact pair index about
58 MB (34 MB as a Bloom filter); most of the rest is the
302,073 live 24 h windows and the account id map. So the compact layout cuts state by
36%, and the next saving is in the windows (fixed size ring buffers or sketches), not the pair index.

The compact layout (`service/compact_state.py`):
* accounts get integer ids; their totals live in typed arrays (`array` module), not one Python object each;
* "distinct payees / payers so far" are counters: a pair is new exactly when the payee is new to the payer, so the
  counter moves only when the pair index says "first time" (no per account sets);
* 24 h windows exist only for accounts active in the last day and are evicted when empty;
* the pair index is pluggable: an exact set of 64 bit pair ids (features identical to the prototype, checked in
  `tests/test_compact_state.py`), or a fixed size Bloom filter whose only possible error is to call a truly new
  payee "seen before".

## 3. Scale out: Redis state, stateless workers

Built in `service/scale/`, tested in `tests/test_scale.py` against a real `redis-server`:

```
payments ──► producer ──► Redis Streams  payments:0 … payments:N-1   (partition = crc32(payer) mod N)
                                 │
                 ┌───────────────┴───────────────┐
             worker 0   …   worker N-1            (stateless: features, L1, L2, context rules; batched)
                 │                               │
                 └──────► Redis account state ◄──┘   a:{acct} hash · p:{acct} payee set · wo/wi:{acct} 24 h zsets
                                 │
                           decisions stream ──► analyst review desk, case tool, STR workflow
```

* Every key carries the account in a `{hash tag}`, so Redis Cluster shards the state by account with no code change.
* A worker reads one partition. All of a payer's payments land on one worker, in order, so the payer's own history
  is exact. A payee's receipts from other partitions can lag by milliseconds: the normal trade off of streaming state.
* One read round trip and one write round trip per payment, both pipelined; the models score in batches.
* With a single writer the Redis features equal the in memory engine's (`tests/test_scale.py`, 6,000 payments).

| Workers | Payments | Payments per second | Redis memory |
|---|---|---|---|
| 1 | 200,000 | 934 | 248.2 MB for 185,098 accounts |
| 2 | 200,000 | 1,315 | about 1,406 bytes per account |

Latency at 789 payments per second (two workers, publish to decision, including queueing):
p50 289.4 ms, p95 794.6 ms, p99 970.3 ms.

These are far below the 41,459 per second of the in process state: every payment costs two network
round trips to Redis, and here Redis, the producer and the workers share 2 cores (so the two worker run is
CPU bound, not partition bound). The next steps are known: move the read and update into one server side Lua script
(one round trip), pipeline across a batch where payers do not repeat, and run workers on their own cores or machines.

Run it yourself:

```bash
# local, no Docker
redis-server --save "" --appendonly no &
python service/scale/worker.py --partition 0 &
python service/scale/worker.py --partition 1 &
python service/scale/replay.py --csv path/to/HI-Small_Trans.csv --partitions 2 --limit 200000

# Docker (Redis + API + two workers)
docker compose -f deploy/docker-compose.yml up --build
```

`deploy/docker-compose.yml` was written for a machine with Docker; this workspace has no Docker daemon, so the
compose file itself is untested here (the same processes are tested natively above).

## 4. From here to a bank

1. **Throughput.** UPI runs at roughly 20 billion payments a month, about 7,000 to 8,000 a second on average with
   peaks several times higher (check NPCI's current figures). One bank sees its own share. Workers are stateless, so
   capacity is the number of worker processes; Redis Cluster (or Flink keyed state, Aerospike, or DynamoDB) holds
   the state, sharded by account.
2. **State size.** Keep running totals and 24 h windows only. Replace exact payee sets with a Bloom or cuckoo filter
   per partition once the exact index outgrows memory (section 2 measures the trade off).
3. **Placement.** Inside the bank (RBI data localisation, DPDP Act), on the payment switch before the debit, within
   the switch's timeout budget. Decisions feed the existing case tool; the four eyes block rule and the hash chained
   audit log carry over; STRs go to FIU-IND as today.
4. **Cold start.** Replay recent history into the state store before going live; without history, recall collapses.
5. **Rollout.** Retrain on the bank's labelled fraud, run in shadow mode (score, act on nothing) for weeks, set
   thresholds to the analyst team's capacity, then champion and challenger against the current system with drift
   monitoring.

## 5. Honest limits

* All results are on synthetic data.
* The live payment model holds 0.716 PR-AUC on HI-Small but 0.336 on LI-Small, a lower laundering intensity it never
  trained on (`results/full_eval_*.json`). A bank must retrain on its own data.
* The ctx-1.3 mule signature, tuned on HI-Small validation weeks, does not transfer to LI-Small (no true positives).
* Scale out numbers come from a 2 core container where Redis, the producer and the workers share the cores; they
  show the design works and give a floor, not a production capacity figure.
