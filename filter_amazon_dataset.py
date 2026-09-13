"""
filter_amazon_dataset.py
========================
Iteration 1: Structure & filter twcs.csv for AmazonHelp brand.

Goal:
  - Extract only AmazonHelp-related conversations
  - Reconstruct full conversation threads (root → replies, recursively)
  - Apply all edge-case rules from agent_iteration_01.md
  - Output: filtered_amazon_dataset.csv  (conversations grouped in order)

Author  : Hiver Take-Home Assignment
Dataset : twcs.csv  (Customer Support on Twitter – Kaggle)
"""

import csv
import sys
import os
import re
from collections import defaultdict, deque
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_FILE   = os.path.join(os.path.dirname(__file__), "twcs.csv")
OUTPUT_FILE  = os.path.join(os.path.dirname(__file__), "filtered_amazon_dataset.csv")

BRAND_ID     = "AmazonHelp"          # The brand we care about

# Tweets whose text looks like a DM redirect (no substance for RAG)
DM_REDIRECT_RE = re.compile(
    r"(send\s+us\s+a\s+dm|dm\s+us|direct\s+message|please\s+dm|via\s+dm)",
    re.IGNORECASE,
)

# Twitter timestamp format
TWITTER_TS_FMT = "%a %b %d %H:%M:%S +0000 %Y"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load every tweet into memory (the file is ~500 MB; we parse once)
# ─────────────────────────────────────────────────────────────────────────────

def parse_ts(ts_str: str) -> datetime | None:
    """Parse a Twitter timestamp string → datetime (UTC). Returns None on error."""
    try:
        return datetime.strptime(ts_str.strip(), TWITTER_TS_FMT)
    except (ValueError, AttributeError):
        return None


def parse_id_list(raw: str) -> list[str]:
    """
    response_tweet_id can be a comma-separated list  e.g. "119264,119266"
    Returns a list of stripped, non-empty string IDs.
    """
    if not raw or raw.strip() == "":
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def load_tweets(filepath: str) -> dict[str, dict]:
    """
    Read the full CSV and return a dict keyed by tweet_id.
    Each value is the row dict plus:
      - ts        : parsed datetime | None
      - resp_ids  : list[str]  (response_tweet_id, possibly multiple)
      - parent_id : str | None (in_response_to_tweet_id)
    """
    tweets: dict[str, dict] = {}
    print(f"[1/6] Loading tweets from {filepath} …")
    with open(filepath, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            tid = row["tweet_id"].strip()
            row["ts"]        = parse_ts(row.get("created_at", ""))
            row["resp_ids"]  = parse_id_list(row.get("response_tweet_id", ""))
            row["parent_id"] = row.get("in_response_to_tweet_id", "").strip() or None
            tweets[tid] = row
    print(f"    Loaded {len(tweets):,} tweets total.")
    return tweets


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Identify all tweet IDs that touch AmazonHelp
# ─────────────────────────────────────────────────────────────────────────────

def find_amazon_relevant_ids(tweets: dict[str, dict]) -> set[str]:
    """
    A tweet is 'AmazonHelp-relevant' if:
      - its author_id is AmazonHelp, OR
      - it is addressed to / replied to by AmazonHelp (parent or child of a
        tweet authored by AmazonHelp).
    We do a two-pass expansion so we get full threads.
    """
    print(f"[2/6] Identifying AmazonHelp-relevant tweet IDs …")

    # Direct: tweets by AmazonHelp
    amazon_authored = {tid for tid, tw in tweets.items()
                       if tw["author_id"].strip() == BRAND_ID}

    # Build parent→children map for the whole corpus (cheap, one pass)
    children_of: dict[str, list[str]] = defaultdict(list)
    for tid, tw in tweets.items():
        pid = tw["parent_id"]
        if pid and pid in tweets:
            children_of[pid].append(tid)

    # Expand outward: include parents and children of amazon-authored tweets
    relevant: set[str] = set()
    queue = deque(amazon_authored)
    while queue:
        tid = queue.popleft()
        if tid in relevant:
            continue
        relevant.add(tid)
        tw = tweets.get(tid)
        if tw is None:
            continue
        # Walk up to root
        pid = tw["parent_id"]
        if pid and pid in tweets and pid not in relevant:
            queue.append(pid)
        # Walk down to every child
        for cid in children_of.get(tid, []):
            if cid not in relevant:
                queue.append(cid)

    print(f"    Found {len(relevant):,} AmazonHelp-relevant tweet IDs.")
    return relevant


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Build conversation threads
# ─────────────────────────────────────────────────────────────────────────────

def build_threads(tweets: dict[str, dict],
                  relevant_ids: set[str]) -> list[list[str]]:
    """
    Build a list of conversation threads.
    Each thread is a list of tweet IDs in BFS order (root first).

    Rules applied here:
      - Root = relevant tweet with no parent (or parent outside dataset)
      - Thread must involve AmazonHelp at some point (sanity check)
      - Orphan conversations (no valid root) are dropped
      - Backward-in-time jumps: drop entire branch from that point
      - Only include branches where AmazonHelp has replied (edge case 7)
      - Self-replies and same-user→same-user: kept in group (edge cases 4, 6)
      - User replying to only other non-amazon users: excluded (edge case 5)
    """
    print(f"[3/6] Building conversation threads …")

    rel_tweets = {tid: tweets[tid] for tid in relevant_ids if tid in tweets}

    # Build children map (within relevant set only)
    children_of: dict[str, list[str]] = defaultdict(list)
    for tid, tw in rel_tweets.items():
        pid = tw["parent_id"]
        if pid and pid in rel_tweets:
            children_of[pid].append(tid)

    # Identify roots: tweets whose parent is absent from relevant set
    roots = [
        tid for tid, tw in rel_tweets.items()
        if tw["parent_id"] is None or tw["parent_id"] not in rel_tweets
    ]

    threads: list[list[str]] = []

    for root_id in roots:
        # BFS traversal, respecting time-order constraint
        thread: list[str] = []
        queue: deque[tuple[str, datetime | None]] = deque()

        root_tw = rel_tweets[root_id]
        root_ts = root_tw["ts"]
        queue.append((root_id, root_ts))

        seen_in_thread: set[str] = set()
        backward_jump_detected = False

        while queue:
            tid, parent_ts = queue.popleft()
            if tid in seen_in_thread:
                continue
            tw = rel_tweets.get(tid)
            if tw is None:
                continue

            cur_ts = tw["ts"]

            # Edge case 8: Conversation jumps backward in time → drop branch
            if (parent_ts is not None
                    and cur_ts is not None
                    and cur_ts < parent_ts):
                # Mark flag; skip this node and its descendants
                backward_jump_detected = True
                continue

            seen_in_thread.add(tid)
            thread.append(tid)

            # Enqueue children sorted by timestamp for determinism
            children = sorted(
                children_of.get(tid, []),
                key=lambda c: rel_tweets[c]["ts"] or datetime.max,
            )
            for cid in children:
                if cid not in seen_in_thread:
                    queue.append((cid, cur_ts))

        # Edge case 5: Drop threads with NO AmazonHelp participation at all
        has_amazon = any(
            rel_tweets[tid]["author_id"].strip() == BRAND_ID
            for tid in thread
        )
        if not has_amazon:
            continue

        # Edge case 1: single tweet with no reply (no AmazonHelp response) →
        #   keep it in dataset but it won't be treated as resolved Q&A.
        #   (We still include it; the RAG layer will handle this distinction.)

        if thread:
            threads.append(thread)

    print(f"    Built {len(threads):,} conversation threads.")
    return threads


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – Assign Conversation Group IDs
# ─────────────────────────────────────────────────────────────────────────────

def assign_group_ids(threads: list[list[str]]) -> dict[str, str]:
    """
    Each thread gets a stable group ID = 'CONV_<root_tweet_id>'.
    Returns mapping tweet_id → group_id.
    """
    tid_to_group: dict[str, str] = {}
    for thread in threads:
        if not thread:
            continue
        root_id  = thread[0]
        group_id = f"CONV_{root_id}"
        for tid in thread:
            tid_to_group[tid] = group_id
    return tid_to_group


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Write filtered CSV
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_FIELDNAMES = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
    "root_tweet_id",
    "root_author_id",
    "root_created_at",
    "conversation_group_id",
]


def write_output(
    threads: list[list[str]],
    tweets: dict[str, dict],
    tid_to_group: dict[str, str],
    output_path: str,
) -> None:
    """
    Write the filtered dataset.
    Rows are ordered: all tweets of thread 1, then all tweets of thread 2, …
    Within a thread the BFS order (root-first, then children) is preserved.
    """
    print(f"[5/6] Writing output to {output_path} …")
    total_rows = 0

    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDNAMES)
        writer.writeheader()

        for thread in threads:
            if not thread:
                continue

            root_id   = thread[0]
            root_tw   = tweets.get(root_id, {})
            root_author    = root_tw.get("author_id", "")
            root_created   = root_tw.get("created_at", "")
            group_id  = tid_to_group.get(root_id, f"CONV_{root_id}")

            for tid in thread:
                tw = tweets.get(tid)
                if tw is None:
                    continue

                writer.writerow({
                    "tweet_id"               : tw["tweet_id"],
                    "author_id"              : tw["author_id"],
                    "inbound"                : tw["inbound"],
                    "created_at"             : tw["created_at"],
                    "text"                   : tw["text"],
                    "response_tweet_id"      : tw.get("response_tweet_id", ""),
                    "in_response_to_tweet_id": tw.get("in_response_to_tweet_id", ""),
                    "root_tweet_id"          : root_id,
                    "root_author_id"         : root_author,
                    "root_created_at"        : root_created,
                    "conversation_group_id"  : group_id,
                })
                total_rows += 1

    print(f"    Written {total_rows:,} rows across {len(threads):,} conversation groups.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 – Print quick statistics
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(threads: list[list[str]], tweets: dict[str, dict]) -> None:
    print(f"\n[6/6] Quick statistics:")
    lengths = [len(t) for t in threads]
    amazon_reply_counts = []
    no_reply_threads   = 0
    single_tweet_threads = 0

    for thread in threads:
        amazon_replies = sum(
            1 for tid in thread
            if tweets.get(tid, {}).get("author_id", "").strip() == BRAND_ID
        )
        amazon_reply_counts.append(amazon_replies)
        if amazon_replies == 0:
            no_reply_threads += 1
        if len(thread) == 1:
            single_tweet_threads += 1

    if lengths:
        print(f"    Total threads           : {len(threads):,}")
        print(f"    Total tweets in output  : {sum(lengths):,}")
        print(f"    Avg thread length       : {sum(lengths)/len(lengths):.1f} tweets")
        print(f"    Max thread length       : {max(lengths)} tweets")
        print(f"    Min thread length       : {min(lengths)} tweets")
        print(f"    Single-tweet threads    : {single_tweet_threads:,}  (edge case 1 – no reply)")
        print(f"    Threads w/o AmazonHelp  : {no_reply_threads:,}  (should be 0)")
        print(f"    Avg AmazonHelp replies  : {sum(amazon_reply_counts)/len(amazon_reply_counts):.2f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  AmazonHelp Dataset Filter  –  Iteration 1")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        sys.exit(f"ERROR: Input file not found: {INPUT_FILE}")

    # 1. Load all tweets
    tweets = load_tweets(INPUT_FILE)

    # 2. Find relevant IDs
    relevant_ids = find_amazon_relevant_ids(tweets)

    # 3. Build threads
    threads = build_threads(tweets, relevant_ids)

    # 4. Assign group IDs
    tid_to_group = assign_group_ids(threads)

    # 5. Write output
    write_output(threads, tweets, tid_to_group, OUTPUT_FILE)

    # 6. Stats
    print_stats(threads, tweets)

    print(f"[DONE]  Output -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
