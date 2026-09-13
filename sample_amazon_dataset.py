"""
sample_amazon_dataset.py
========================
Iteration 2: Smart stratified sampling of filtered_amazon_dataset.csv
         -> intermediate_amazon_dataset.csv  (all groups, sorted + classified)
         -> sample_amazon_dataset.csv        (~850 groups, original columns only)

Pipeline
--------
1. Load filtered_amazon_dataset.csv  (374 k rows, 82 557 groups)
2. For each group, classify into edge_case_type  1-17 (18+ for unclassified)
3. Sort groups by edge_case_type; write intermediate_amazon_dataset.csv
4. Proportional quota per EC type (850 total across all EC types)
5. Per EC type: split groups into 3 tweet_count tiers, sample evenly
6. Write sample_amazon_dataset.csv  (only original 11 columns, groups atomic)
"""

import csv
import os
import re
import sys
import math
import random
from collections import defaultdict
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE       = os.path.join(BASE_DIR, "filtered_amazon_dataset.csv")
INTERMEDIATE_FILE= os.path.join(BASE_DIR, "intermediate_amazon_dataset.csv")
OUTPUT_FILE      = os.path.join(BASE_DIR, "sample_amazon_dataset.csv")

BRAND_ID         = "AmazonHelp"
RANDOM_SEED      = 42
TARGET_GROUPS    = 850

TWITTER_TS_FMT   = "%a %b %d %H:%M:%S +0000 %Y"

# Heuristic thresholds
TIME_GAP_HUGE_HOURS   = 12   # gap > 12 h between consecutive tweets  -> EC7
LONG_CHAIN_MIN_TWEETS = 15   # tweet_count >= 15  -> EC9
MULTI_ATTEMPT_MIN     = 3    # >= 3 Amazon + >= 3 user tweets  -> EC14

# Positive-sentiment keywords for detecting explicit resolution (EC13)
POSITIVE_RE = re.compile(
    r"\b(thank|thanks|thank\s*you|thx|ty\b|great|awesome|perfect|sorted|"
    r"fixed|solved|resolved|worked|brilliant|excellent|amazing|appreciate|"
    r"cheers|helpful|problem\s*solved|issue\s*resolved|all\s*good|good\s*to\s*go|"
    r"it\s*works|working\s*now|that\s*worked|that\s*helped)\b",
    re.IGNORECASE,
)

# Columns in the source file
ORIGINAL_COLS = [
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
    "root_tweet_id", "root_author_id", "root_created_at",
    "conversation_group_id",
]

# Extra columns present only in the intermediate file
INTERMEDIATE_EXTRA_COLS = [
    "edge_case_type",
    "tweet_count",
    "amazonhelp_tweet_count",
    "user_tweet_count",
]

INTERMEDIATE_COLS = ORIGINAL_COLS + INTERMEDIATE_EXTRA_COLS


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def parse_ts(ts_str: str):
    try:
        return datetime.strptime(ts_str.strip(), TWITTER_TS_FMT)
    except (ValueError, AttributeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load
# ─────────────────────────────────────────────────────────────────────────────

def load_groups(filepath: str):
    """
    Read the filtered CSV.
    Returns:
      groups      : dict[group_id -> list[row_dict]]  (rows in CSV order)
      group_order : list[group_id]  in first-appearance order
    """
    print(f"[1/6] Loading {filepath} ...")
    groups      = defaultdict(list)
    group_order = []
    seen        = set()

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
# STEP 2 – Classify a single group
# ─────────────────────────────────────────────────────────────────────────────

def classify_group(gid: str, rows: list) -> tuple:
    """
    Analyse one conversation group and return
    (edge_case_type: int, metrics: dict).

    Classification priority (first-match wins):
      EC10  Branching conversation            (any tweet has >1 direct reply)
      EC11  Multiple Amazon answers to same tweet
      EC9   Very long chain                   (>= LONG_CHAIN_MIN_TWEETS)
      EC14  Multi-attempt resolution          (>=3 Amazon + >=3 user tweets)
      EC17  Missing middle                    (gap in parent chain, but valid root+end)
      EC15  Duplicate replies                 (same author, identical text twice)
      EC4   Self-reply                        (author replies to own tweet)
      EC6   Same-user chain                   (consecutive same non-Amazon user)
      EC7   Huge time gap                     (> TIME_GAP_HUGE_HOURS between consecutive)
      EC13  Explicit positive confirmation    (last user tweet has positive keywords)
      EC12  Amazon last, no user confirmation (conversation ended by Amazon)
      EC3   Reply with missing parent         (2-tweet group where parent absent)
      EC1   Minimal / no-reply exchange       (tweet_count == 2 with no resolution)
      EC2   Default standard Q&A
    """
    tweet_ids = {r["tweet_id"].strip() for r in rows}

    # ── basic counts ──────────────────────────────────────────────────────────
    tweet_count        = len(rows)
    amazon_rows        = [r for r in rows if r["author_id"].strip() == BRAND_ID]
    user_rows          = [r for r in rows if r["author_id"].strip() != BRAND_ID]
    amazon_tweet_count = len(amazon_rows)
    user_tweet_count   = len(user_rows)

    # ── build parent → children map ────────────────────────────────────────────
    children_of: dict = defaultdict(list)
    tweet_author: dict = {}
    for r in rows:
        tid = r["tweet_id"].strip()
        pid = r.get("in_response_to_tweet_id", "").strip()
        tweet_author[tid] = r["author_id"].strip()
        if pid:
            children_of[pid].append(tid)

    # ── timestamps sorted ascending ────────────────────────────────────────────
    timed = []
    for r in rows:
        ts = parse_ts(r.get("created_at", ""))
        if ts:
            timed.append((ts, r))
    timed.sort(key=lambda x: x[0])

    # ── max consecutive time gap ───────────────────────────────────────────────
    max_gap_hours = 0.0
    for i in range(1, len(timed)):
        diff_h = (timed[i][0] - timed[i - 1][0]).total_seconds() / 3600
        if diff_h > max_gap_hours:
            max_gap_hours = diff_h

    # ── branching: any tweet has >1 child ──────────────────────────────────────
    has_branches = any(len(v) > 1 for v in children_of.values())

    # ── multiple Amazon replies to the SAME parent ─────────────────────────────
    amazon_per_parent: dict = defaultdict(int)
    for r in amazon_rows:
        pid = r.get("in_response_to_tweet_id", "").strip()
        if pid:
            amazon_per_parent[pid] += 1
    multiple_amazon_same_parent = any(v > 1 for v in amazon_per_parent.values())

    # ── self-reply (author responds to own tweet) ──────────────────────────────
    has_self_reply = False
    for r in rows:
        pid = r.get("in_response_to_tweet_id", "").strip()
        if pid and pid in tweet_author:
            if tweet_author[pid] == r["author_id"].strip():
                has_self_reply = True
                break

    # ── same-user → same-user chain (consecutive, non-Amazon) ─────────────────
    has_same_user_chain = False
    if len(timed) >= 2:
        for i in range(1, len(timed)):
            pa = timed[i - 1][1]["author_id"].strip()
            ca = timed[i][1]["author_id"].strip()
            if pa == ca and pa != BRAND_ID:
                has_same_user_chain = True
                break

    # ── missing parent inside the group ───────────────────────────────────────
    has_missing_parent = any(
        r.get("in_response_to_tweet_id", "").strip() and
        r["in_response_to_tweet_id"].strip() not in tweet_ids
        for r in rows
    )

    # ── duplicate replies (same author, identical text) ────────────────────────
    texts_by_author: dict = defaultdict(list)
    for r in rows:
        texts_by_author[r["author_id"].strip()].append(
            r.get("text", "").strip().lower()
        )
    has_duplicates = any(
        len(txts) != len(set(txts))
        for txts in texts_by_author.values()
    )

    # ── last-tweet analysis ────────────────────────────────────────────────────
    if timed:
        last_row = timed[-1][1]
    else:
        last_row = rows[-1]

    last_author    = last_row["author_id"].strip()
    last_text      = last_row.get("text", "")
    last_is_user   = (last_author != BRAND_ID)
    last_is_amazon = (last_author == BRAND_ID)
    has_positive_end = last_is_user and bool(POSITIVE_RE.search(last_text))

    # ── multi-attempt indicator ────────────────────────────────────────────────
    is_multi_attempt = (amazon_tweet_count >= MULTI_ATTEMPT_MIN
                        and user_tweet_count >= MULTI_ATTEMPT_MIN)

    # ── CLASSIFY (priority waterfall) ─────────────────────────────────────────
    if has_branches:
        ec = 10
    elif multiple_amazon_same_parent:
        ec = 11
    elif tweet_count >= LONG_CHAIN_MIN_TWEETS:
        ec = 9
    elif is_multi_attempt:
        ec = 14
    elif has_missing_parent and tweet_count > 2:
        ec = 17
    elif has_duplicates:
        ec = 15
    elif has_self_reply:
        ec = 4
    elif has_same_user_chain:
        ec = 6
    elif max_gap_hours > TIME_GAP_HUGE_HOURS:
        ec = 7
    elif has_positive_end:
        ec = 13
    elif last_is_amazon:
        ec = 12
    elif has_missing_parent:
        ec = 3
    elif tweet_count == 2:
        ec = 1
    else:
        ec = 2

    metrics = {
        "tweet_count"           : tweet_count,
        "amazonhelp_tweet_count": amazon_tweet_count,
        "user_tweet_count"      : user_tweet_count,
        "max_gap_hours"         : round(max_gap_hours, 2),
        "has_branches"          : has_branches,
        "multiple_amazon_same"  : multiple_amazon_same_parent,
        "has_self_reply"        : has_self_reply,
        "has_same_user_chain"   : has_same_user_chain,
        "has_missing_parent"    : has_missing_parent,
        "has_duplicates"        : has_duplicates,
        "has_positive_end"      : has_positive_end,
        "last_is_amazon"        : last_is_amazon,
        "is_multi_attempt"      : is_multi_attempt,
    }

    return ec, metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Classify all groups
# ─────────────────────────────────────────────────────────────────────────────

def classify_all(groups: dict, group_order: list) -> tuple:
    """
    Returns:
      group_meta : dict[gid -> {ec, metrics}]
      ec_counts  : dict[ec  -> int]
    """
    print(f"[2/6] Classifying {len(group_order):,} groups ...")

    group_meta: dict = {}
    ec_counts:  dict = defaultdict(int)

    for i, gid in enumerate(group_order, 1):
        if i % 10_000 == 0:
            print(f"    ... {i:,} / {len(group_order):,}")
        ec, metrics = classify_group(gid, groups[gid])
        group_meta[gid] = {"ec": ec, "metrics": metrics}
        ec_counts[ec]  += 1

    print(f"\n    Edge-Case Distribution:")
    total = len(group_order)
    for ec in sorted(ec_counts):
        pct = ec_counts[ec] / total * 100
        label = f"EC{ec:2d}"
        bar   = "#" * int(pct / 0.5)
        print(f"      {label}: {ec_counts[ec]:6,} groups  ({pct:5.2f}%)  {bar}")
    print()

    return group_meta, dict(ec_counts)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Write intermediate file (ALL groups, sorted by EC type)
# ─────────────────────────────────────────────────────────────────────────────

def write_intermediate(groups: dict, group_order: list,
                       group_meta: dict, filepath: str) -> list:
    """
    Writes intermediate_amazon_dataset.csv.
    Groups are sorted by edge_case_type then by group_id (stable).
    Each row gets extra columns: edge_case_type, tweet_count,
    amazonhelp_tweet_count, user_tweet_count.

    Returns sorted_group_order (list[gid] sorted by EC).
    """
    print(f"[3/6] Writing intermediate file -> {filepath} ...")

    sorted_order = sorted(
        group_order,
        key=lambda gid: (group_meta[gid]["ec"], gid)
    )

    rows_written = 0
    with open(filepath, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INTERMEDIATE_COLS, extrasaction="ignore")
        writer.writeheader()

        for gid in sorted_order:
            meta    = group_meta[gid]
            ec      = meta["ec"]
            metrics = meta["metrics"]
            for row in groups[gid]:
                out = {col: row.get(col, "") for col in ORIGINAL_COLS}
                out["edge_case_type"]       = ec
                out["tweet_count"]          = metrics["tweet_count"]
                out["amazonhelp_tweet_count"] = metrics["amazonhelp_tweet_count"]
                out["user_tweet_count"]     = metrics["user_tweet_count"]
                writer.writerow(out)
                rows_written += 1

    print(f"    Written {rows_written:,} rows across {len(sorted_order):,} groups.")
    return sorted_order


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Stratified sampling (~850 groups)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample(groups: dict, group_order: list,
                       group_meta: dict, ec_counts: dict,
                       target: int = TARGET_GROUPS) -> list:
    """
    For each edge_case_type:
      quota  = round( ec_pct  *  target )
      Split group IDs for that EC into 3 equal-count buckets by tweet_count
        (small = lowest 1/3, medium = middle 1/3, large = top 1/3).
      Sample  ceil(quota / 3)  groups from each bucket (random, seeded).
      Groups above EC17 are ALL included (no sampling).

    Returns list of selected group_ids (unsorted; sort happens at write time).
    """
    print(f"[4/6] Stratified sampling  (target = {target} groups) ...")
    random.seed(RANDOM_SEED)

    total     = len(group_order)
    all_ecs   = sorted(ec_counts)

    # ── groups that are ALWAYS included (EC > 17) ─────────────────────────────
    always_include = [gid for gid in group_order if group_meta[gid]["ec"] > 17]
    sampled_always = always_include  # all of them

    # Remaining budget for EC 1-17
    budget = target - len(sampled_always)
    if budget < 0:
        budget = 0

    # ── EC-1-to-17 groups ─────────────────────────────────────────────────────
    known_ecs = [ec for ec in all_ecs if ec <= 17]

    # Count of groups in EC 1-17 only (denominator for proportions)
    known_total = sum(ec_counts[ec] for ec in known_ecs)

    # ── raw floating-point quotas ──────────────────────────────────────────────
    raw_quotas = {}
    for ec in known_ecs:
        pct            = ec_counts[ec] / known_total
        raw_quotas[ec] = pct * budget

    # ── integer quotas via largest-remainder method ────────────────────────────
    floor_q    = {ec: int(raw_quotas[ec]) for ec in known_ecs}
    remainders = {ec: raw_quotas[ec] - floor_q[ec] for ec in known_ecs}
    deficit    = budget - sum(floor_q.values())
    top_ecs    = sorted(remainders, key=lambda e: remainders[e], reverse=True)
    for i in range(deficit):
        floor_q[top_ecs[i % len(top_ecs)]] += 1

    # ── print per-EC quota table ───────────────────────────────────────────────
    print(f"\n    {'EC':>4}  {'Groups':>7}  {'%':>6}  {'Quota':>6}")
    print(f"    {'-'*4}  {'-'*7}  {'-'*6}  {'-'*6}")
    for ec in known_ecs:
        pct = ec_counts[ec] / total * 100
        print(f"    EC{ec:2d}  {ec_counts[ec]:7,}  {pct:6.2f}%  {floor_q[ec]:6d}")
    if always_include:
        print(f"    EC>17: {len(always_include)} groups  -> ALL included")
    print(f"    {'Total':>4}  {total:7,}  100.00%  {sum(floor_q.values()) + len(sampled_always):6d}")
    print()

    # ── bucket-based sampling for each known EC ────────────────────────────────
    ec_to_gids: dict = defaultdict(list)
    for gid in group_order:
        ec = group_meta[gid]["ec"]
        if ec <= 17:
            ec_to_gids[ec].append(gid)

    sampled_known = []

    for ec in known_ecs:
        gids  = ec_to_gids[ec]
        quota = floor_q[ec]

        if quota <= 0:
            continue

        if len(gids) <= quota:
            # Fewer groups than quota; take all
            sampled_known.extend(gids)
            continue

        # Sort by tweet_count ascending, then split into 3 equal buckets
        gids_sorted = sorted(gids, key=lambda g: group_meta[g]["metrics"]["tweet_count"])
        n  = len(gids_sorted)
        s1 = gids_sorted[: n // 3]          # small
        s2 = gids_sorted[n // 3: 2 * n // 3]  # medium
        s3 = gids_sorted[2 * n // 3:]        # large

        # Distribute quota across 3 buckets: each gets floor(quota/3),
        # remainder goes to small first (fewer tweets = more RAG-friendly)
        base      = quota // 3
        rem       = quota - base * 3
        bq_small  = base + (1 if rem > 0 else 0)
        bq_medium = base + (1 if rem > 1 else 0)
        bq_large  = base

        for bucket, bq in [(s1, bq_small), (s2, bq_medium), (s3, bq_large)]:
            take = min(bq, len(bucket))
            sampled_known.extend(random.sample(bucket, take))

    selected = sampled_known + sampled_always
    print(f"    Selected {len(selected):,} groups total  "
          f"({len(sampled_known)} known EC + {len(sampled_always)} EC>17).")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Write final sample CSV (original columns only)
# ─────────────────────────────────────────────────────────────────────────────

def write_sample(selected_ids: list, groups: dict,
                 group_meta: dict, filepath: str) -> None:
    """
    Writes sample_amazon_dataset.csv.
    • Only ORIGINAL_COLS (no edge_case_type or metric columns).
    • Groups are atomic: all rows of a group are consecutive.
    • Groups sorted by edge_case_type then group_id.
    """
    print(f"[5/6] Writing final sample -> {filepath} ...")

    sorted_ids = sorted(
        selected_ids,
        key=lambda gid: (group_meta[gid]["ec"], gid)
    )

    rows_written = 0
    with open(filepath, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ORIGINAL_COLS, extrasaction="ignore")
        writer.writeheader()

        for gid in sorted_ids:
            for row in groups[gid]:
                writer.writerow({col: row.get(col, "") for col in ORIGINAL_COLS})
                rows_written += 1

    print(f"    Written {rows_written:,} rows across {len(sorted_ids):,} groups.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 – Final statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(selected_ids: list, groups: dict, group_meta: dict) -> None:
    print(f"\n[6/6] Final Sample Statistics:")

    ec_dist: dict         = defaultdict(int)
    ec_tweets: dict       = defaultdict(int)
    tweet_lengths: list   = []

    for gid in selected_ids:
        ec   = group_meta[gid]["ec"]
        tc   = len(groups[gid])
        ec_dist[ec]   += 1
        ec_tweets[ec] += tc
        tweet_lengths.append(tc)

    total_tweets = sum(tweet_lengths)
    total_groups = len(selected_ids)

    print(f"    Total groups  : {total_groups:,}")
    print(f"    Total tweets  : {total_tweets:,}")
    print(f"    Avg tweets/group : {total_tweets / total_groups:.1f}")
    print(f"    Max tweets/group : {max(tweet_lengths)}")
    print(f"    Min tweets/group : {min(tweet_lengths)}")
    print(f"\n    {'EC':>4}  {'Groups':>6}  {'% of sample':>11}  {'Tweets':>7}  {'Avg len':>7}")
    print(f"    {'-'*4}  {'-'*6}  {'-'*11}  {'-'*7}  {'-'*7}")
    for ec in sorted(ec_dist):
        g   = ec_dist[ec]
        tw  = ec_tweets[ec]
        pct = g / total_groups * 100
        avg = tw / g
        print(f"    EC{ec:2d}  {g:6d}  {pct:10.2f}%  {tw:7,}  {avg:7.1f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("  AmazonHelp Dataset Sampler  -  Iteration 2")
    print("=" * 65)

    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: Input file not found: {INPUT_FILE}")

    # 1. Load
    groups, group_order = load_groups(INPUT_FILE)

    # 2. Classify all groups
    group_meta, ec_counts = classify_all(groups, group_order)

    # 3. Write intermediate (sorted, all groups, with extra metadata cols)
    write_intermediate(groups, group_order, group_meta, INTERMEDIATE_FILE)

    # 4. Stratified sample (~850 groups)
    selected_ids = stratified_sample(
        groups, group_order, group_meta, ec_counts, target=TARGET_GROUPS
    )

    # 5. Write final sample (original columns only)
    write_sample(selected_ids, groups, group_meta, OUTPUT_FILE)

    # 6. Print stats
    print_stats(selected_ids, groups, group_meta)

    print(f"[DONE]")
    print(f"  Intermediate -> {INTERMEDIATE_FILE}")
    print(f"  Final sample -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
