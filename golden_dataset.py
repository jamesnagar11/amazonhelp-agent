"""
golden_dataset.py
=================
Iteration 3: Build a 200-group golden evaluation dataset from filtered_amazon_dataset.csv.

Key differences from Iteration 2:
  - Target : 200 conversation groups  (not 850)
  - Sampling: round-robin across sorted EC types (1 → 17, skipping absent ones)
              Each pass picks ONE random unselected group from the current EC bucket.
              Cycle repeats until 200 groups are collected.
  - Output  : golden_dataset.csv  (original 11 columns  +  "rule_number")
              Each group is atomic (all rows in or all rows out).
              Rows sorted by rule_number then conversation_group_id.
"""

import csv
import os
import re
import sys
import random
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE   = os.path.join(BASE_DIR, "filtered_amazon_dataset.csv")
OUTPUT_FILE  = os.path.join(BASE_DIR, "golden_dataset.csv")

BRAND_ID     = "AmazonHelp"
RANDOM_SEED  = 99          # different seed from iter-2 for independent draw
TARGET_GROUPS = 200

TWITTER_TS_FMT = "%a %b %d %H:%M:%S +0000 %Y"

# Heuristic thresholds (identical to Iteration 2 for consistency)
TIME_GAP_HUGE_HOURS   = 12
LONG_CHAIN_MIN_TWEETS = 15
MULTI_ATTEMPT_MIN     = 3

POSITIVE_RE = re.compile(
    r"\b(thank|thanks|thank\s*you|thx|ty\b|great|awesome|perfect|sorted|"
    r"fixed|solved|resolved|worked|brilliant|excellent|amazing|appreciate|"
    r"cheers|helpful|problem\s*solved|issue\s*resolved|all\s*good|good\s*to\s*go|"
    r"it\s*works|working\s*now|that\s*worked|that\s*helped)\b",
    re.IGNORECASE,
)

# Columns present in filtered_amazon_dataset.csv
ORIGINAL_COLS = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
    "root_tweet_id", "root_author_id", "root_created_at",
    "conversation_group_id",
]

# Output columns (original + rule_number)
OUTPUT_COLS = ORIGINAL_COLS + ["rule_number"]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def parse_ts(ts_str: str):
    try:
        return datetime.strptime(ts_str.strip(), TWITTER_TS_FMT)
    except (ValueError, AttributeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load filtered dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_groups(filepath: str):
    """
    Read filtered_amazon_dataset.csv.
    Returns:
      groups      : dict[group_id -> list[row_dict]]  (rows in CSV order)
      group_order : list[group_id]  in first-appearance order
    """
    print(f"[1/5] Loading {filepath} ...")
    groups: dict      = defaultdict(list)
    group_order: list = []
    seen: set         = set()

    with open(filepath, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            gid = row["conversation_group_id"].strip()
            groups[gid].append(row)
            if gid not in seen:
                seen.add(gid)
                group_order.append(gid)

    total_rows = sum(len(v) for v in groups.values())
    print(f"    Loaded {total_rows:,} rows across {len(group_order):,} unique groups.")
    return dict(groups), group_order


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Classify a single group (same logic as Iteration 2)
# ─────────────────────────────────────────────────────────────────────────────

def classify_group(gid: str, rows: list) -> int:
    """
    Returns the primary edge_case_type (1-17) for a conversation group.

    Classification priority (first match wins):
      EC10  Branching conversation            (any tweet has >1 direct reply)
      EC11  Multiple Amazon answers to same tweet
      EC9   Very long chain                   (>= LONG_CHAIN_MIN_TWEETS)
      EC14  Multi-attempt resolution          (>=3 Amazon + >=3 user tweets)
      EC17  Missing middle                    (gap in parent chain, valid root+end)
      EC15  Duplicate replies                 (same author, identical text twice)
      EC4   Self-reply                        (author replies to own tweet)
      EC6   Same-user chain                   (consecutive same non-Amazon user)
      EC7   Huge time gap                     (> TIME_GAP_HUGE_HOURS consecutive)
      EC13  Explicit positive confirmation    (last user tweet has positive keywords)
      EC12  Amazon last, no user confirmation (conversation ended by Amazon)
      EC3   Reply with missing parent         (2-tweet group with absent parent)
      EC1   Minimal exchange                  (tweet_count == 2, no resolution)
      EC2   Default standard Q&A
    """
    tweet_ids = {r["tweet_id"].strip() for r in rows}
    tweet_count = len(rows)

    amazon_rows = [r for r in rows if r["author_id"].strip() == BRAND_ID]
    user_rows   = [r for r in rows if r["author_id"].strip() != BRAND_ID]
    amazon_tweet_count = len(amazon_rows)
    user_tweet_count   = len(user_rows)

    # parent → children + tweet → author maps
    children_of: dict  = defaultdict(list)
    tweet_author: dict = {}
    for r in rows:
        tid = r["tweet_id"].strip()
        pid = r.get("in_response_to_tweet_id", "").strip()
        tweet_author[tid] = r["author_id"].strip()
        if pid:
            children_of[pid].append(tid)

    # timestamps sorted ascending
    timed = []
    for r in rows:
        ts = parse_ts(r.get("created_at", ""))
        if ts:
            timed.append((ts, r))
    timed.sort(key=lambda x: x[0])

    # max consecutive time gap (hours)
    max_gap_hours = 0.0
    for i in range(1, len(timed)):
        diff_h = (timed[i][0] - timed[i - 1][0]).total_seconds() / 3600
        if diff_h > max_gap_hours:
            max_gap_hours = diff_h

    # branching: any tweet has >1 child
    has_branches = any(len(v) > 1 for v in children_of.values())

    # multiple Amazon replies to the SAME parent
    amazon_per_parent: dict = defaultdict(int)
    for r in amazon_rows:
        pid = r.get("in_response_to_tweet_id", "").strip()
        if pid:
            amazon_per_parent[pid] += 1
    multiple_amazon_same_parent = any(v > 1 for v in amazon_per_parent.values())

    # self-reply
    has_self_reply = False
    for r in rows:
        pid = r.get("in_response_to_tweet_id", "").strip()
        if pid and pid in tweet_author:
            if tweet_author[pid] == r["author_id"].strip():
                has_self_reply = True
                break

    # same-user → same-user chain (consecutive non-Amazon)
    has_same_user_chain = False
    if len(timed) >= 2:
        for i in range(1, len(timed)):
            pa = timed[i - 1][1]["author_id"].strip()
            ca = timed[i][1]["author_id"].strip()
            if pa == ca and pa != BRAND_ID:
                has_same_user_chain = True
                break

    # missing parent inside group
    has_missing_parent = any(
        r.get("in_response_to_tweet_id", "").strip()
        and r["in_response_to_tweet_id"].strip() not in tweet_ids
        for r in rows
    )

    # duplicate replies (same author, same text)
    texts_by_author: dict = defaultdict(list)
    for r in rows:
        texts_by_author[r["author_id"].strip()].append(
            r.get("text", "").strip().lower()
        )
    has_duplicates = any(
        len(txts) != len(set(txts)) for txts in texts_by_author.values()
    )

    # last tweet
    last_row       = timed[-1][1] if timed else rows[-1]
    last_author    = last_row["author_id"].strip()
    last_text      = last_row.get("text", "")
    last_is_user   = (last_author != BRAND_ID)
    last_is_amazon = (last_author == BRAND_ID)
    has_positive_end = last_is_user and bool(POSITIVE_RE.search(last_text))

    # multi-attempt indicator
    is_multi_attempt = (amazon_tweet_count >= MULTI_ATTEMPT_MIN
                        and user_tweet_count >= MULTI_ATTEMPT_MIN)

    # ── Classification waterfall ──────────────────────────────────────────────
    if has_branches:
        return 10
    elif multiple_amazon_same_parent:
        return 11
    elif tweet_count >= LONG_CHAIN_MIN_TWEETS:
        return 9
    elif is_multi_attempt:
        return 14
    elif has_missing_parent and tweet_count > 2:
        return 17
    elif has_duplicates:
        return 15
    elif has_self_reply:
        return 4
    elif has_same_user_chain:
        return 6
    elif max_gap_hours > TIME_GAP_HUGE_HOURS:
        return 7
    elif has_positive_end:
        return 13
    elif last_is_amazon:
        return 12
    elif has_missing_parent:
        return 3
    elif tweet_count == 2:
        return 1
    else:
        return 2


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Classify all groups
# ─────────────────────────────────────────────────────────────────────────────

def classify_all(groups: dict, group_order: list) -> tuple:
    """
    Returns:
      ec_to_gids  : dict[ec -> list[group_id]]  (shuffled for fair round-robin)
      group_ec    : dict[group_id -> ec]
    """
    print(f"[2/5] Classifying {len(group_order):,} groups ...")

    ec_to_gids: dict = defaultdict(list)
    group_ec:   dict = {}

    for i, gid in enumerate(group_order, 1):
        if i % 10_000 == 0:
            print(f"    ... {i:,} / {len(group_order):,}")
        ec = classify_group(gid, groups[gid])
        ec_to_gids[ec].append(gid)
        group_ec[gid] = ec

    total = len(group_order)
    print(f"\n    Edge-Case Distribution (used EC types only):")
    for ec in sorted(ec_to_gids):
        pct = len(ec_to_gids[ec]) / total * 100
        bar = "#" * max(1, int(pct / 0.5))
        print(f"      EC{ec:2d}: {len(ec_to_gids[ec]):6,} groups  ({pct:5.2f}%)  {bar}")
    print()

    return dict(ec_to_gids), group_ec


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Round-robin sampling
# ─────────────────────────────────────────────────────────────────────────────

def round_robin_sample(ec_to_gids: dict,
                       target: int = TARGET_GROUPS) -> list:
    """
    Round-robin across sorted EC types (1 → 17, skipping absent ones).

    Each pass:
      For each EC type in sorted order:
        Randomly pick ONE group_id from the EC's remaining (unselected) pool.
        Mark it as selected.
        Move to the next EC type.
      Repeat until `target` groups are collected.

    If a bucket is exhausted during a pass, it is removed from rotation.
    Stops as soon as `target` groups are selected.

    Returns list of (group_id, ec) pairs in selection order.
    """
    print(f"[3/5] Round-robin sampling  (target = {target} groups) ...")
    random.seed(RANDOM_SEED)

    # Shuffle each bucket independently so each pass' single pick is random
    pools: dict = {}
    for ec, gids in ec_to_gids.items():
        pool = list(gids)
        random.shuffle(pool)
        pools[ec] = pool

    sorted_ecs  = sorted(pools.keys())  # EC 1-17, ascending, only present ones
    selected    = []         # (group_id, ec)
    selected_set= set()

    pass_num = 0
    while len(selected) < target:
        pass_num += 1
        made_progress = False

        for ec in list(sorted_ecs):           # iterate sorted EC types
            if len(selected) >= target:
                break
            if ec not in pools or not pools[ec]:
                # Bucket exhausted – remove from rotation
                if ec in pools:
                    del pools[ec]
                sorted_ecs = [e for e in sorted_ecs if e in pools]
                continue

            # Pick a random group from this EC's remaining pool
            idx = random.randrange(len(pools[ec]))
            gid = pools[ec].pop(idx)

            # Defensive check (shouldn't happen but guard anyway)
            if gid in selected_set:
                continue

            selected.append((gid, ec))
            selected_set.add(gid)
            made_progress = True

        if not made_progress:
            # All buckets exhausted before reaching target
            print(f"    WARNING: All EC buckets exhausted after {len(selected)} groups "
                  f"(target was {target}).")
            break

    print(f"    Completed in {pass_num} passes.  Selected {len(selected):,} groups.")

    # Summary table
    ec_sel_count: dict = defaultdict(int)
    for _, ec in selected:
        ec_sel_count[ec] += 1

    total_sel = len(selected)
    print(f"\n    {'EC':>4}  {'Available':>9}  {'Selected':>8}  {'% of sample':>11}")
    print(f"    {'-'*4}  {'-'*9}  {'-'*8}  {'-'*11}")
    for ec in sorted(ec_to_gids.keys()):
        avail = len(ec_to_gids[ec])
        sel   = ec_sel_count.get(ec, 0)
        pct   = sel / total_sel * 100 if total_sel else 0
        print(f"    EC{ec:2d}  {avail:9,}  {sel:8d}  {pct:10.1f}%")
    print()

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Write golden_dataset.csv
# ─────────────────────────────────────────────────────────────────────────────

def write_golden(selected: list,
                 groups: dict,
                 filepath: str) -> None:
    """
    Writes golden_dataset.csv.
    Columns = ORIGINAL_COLS + rule_number.
    Groups sorted by rule_number then conversation_group_id.
    Each group is written atomically (all rows together).
    """
    print(f"[4/5] Writing golden dataset -> {filepath} ...")

    # Sort by (rule_number, group_id) for organised output
    sorted_selected = sorted(selected, key=lambda x: (x[1], x[0]))

    rows_written = 0
    with open(filepath, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLS, extrasaction="ignore")
        writer.writeheader()

        for gid, ec in sorted_selected:
            for row in groups[gid]:
                out = {col: row.get(col, "") for col in ORIGINAL_COLS}
                out["rule_number"] = ec
                writer.writerow(out)
                rows_written += 1

    print(f"    Written {rows_written:,} rows across {len(sorted_selected):,} groups.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Final statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(selected: list, groups: dict) -> None:
    print(f"\n[5/5] Golden Dataset Statistics:")

    ec_dist: dict   = defaultdict(int)
    ec_tweets: dict = defaultdict(int)
    lengths: list   = []

    for gid, ec in selected:
        tc = len(groups[gid])
        ec_dist[ec]   += 1
        ec_tweets[ec] += tc
        lengths.append(tc)

    total_tweets  = sum(lengths)
    total_groups  = len(selected)

    print(f"    Total groups  : {total_groups:,}")
    print(f"    Total tweets  : {total_tweets:,}")
    print(f"    Avg tweets/group : {total_tweets / total_groups:.1f}")
    print(f"    Max tweets/group : {max(lengths)}")
    print(f"    Min tweets/group : {min(lengths)}")
    print(f"\n    {'Rule':>5}  {'Groups':>6}  {'% of 200':>8}  {'Tweets':>7}  {'Avg len':>7}")
    print(f"    {'-'*5}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*7}")
    for ec in sorted(ec_dist.keys()):
        g   = ec_dist[ec]
        tw  = ec_tweets[ec]
        pct = g / total_groups * 100
        avg = tw / g
        print(f"    EC{ec:2d}  {g:6d}  {pct:7.1f}%  {tw:7,}  {avg:7.1f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  AmazonHelp Golden Dataset Builder  -  Iteration 3")
    print("=" * 65)

    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: Input file not found: {INPUT_FILE}")

    # 1. Load
    groups, group_order = load_groups(INPUT_FILE)

    # 2. Classify all groups into EC types
    ec_to_gids, group_ec = classify_all(groups, group_order)

    # 3. Round-robin sample (200 groups)
    selected = round_robin_sample(ec_to_gids, target=TARGET_GROUPS)

    # 4. Write golden_dataset.csv  (original cols + rule_number)
    write_golden(selected, groups, OUTPUT_FILE)

    # 5. Print stats
    print_stats(selected, groups)

    print(f"[DONE]  Golden dataset -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
