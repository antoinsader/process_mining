
import pandas as pd 
import os
from collections import Counter
import matplotlib.pyplot as plt


# def get_ping_pong_orgs(trace):
#     orgs = trace["org_group"].tolist()
#     ping_pong_orgs = set()
#     ping_pong_pairs = set()

#     last_seen = {}

#     for i, org in enumerate(orgs):
#         if org in last_seen:
#             between = orgs[last_seen[org] + 1 : i]
#             between_unique = set(between) - {org}
#             if between_unique:
#                 ping_pong_orgs.add(org)
#                 for other in between_unique:
#                     if other != org:
#                         ping_pong_pairs.add(tuple(sorted((org,other))))
#                         break
#         last_seen[org] = i
#     return list(ping_pong_orgs), list(ping_pong_pairs)

def get_ping_pong(trace):
    orgs = trace["org_group"].tolist()
    timestamps = trace["timestamp"].tolist()
    orgs = [(x, timestamps[i]) for i, x in enumerate(orgs) if i ==0 or x != orgs[i-1] ]
    pairs = []
    n = len(orgs)
    for ping_idx in range(n):
        ping_item = orgs[ping_idx][0]
        ping_timestamp = orgs[ping_idx][1]
        if ping_item in [x[0] for x in orgs[ping_idx:]]:
            for pong_idx in range(ping_idx+1, n):
                ping_again = orgs[pong_idx][0]
                if ping_item == ping_again:
                    duration_days = (orgs[pong_idx][1] - ping_timestamp).total_seconds() / (24*60*60)
                    pairs.append(((ping_item, orgs[pong_idx - 1][0]), (duration_days)))      
                    break
    return pairs

def ping_pong_trace_serie(trace):
    pairs = get_ping_pong(trace)
    return pd.Series({
        "case_seq_num": trace.name,
        "has_ping_pong": len(pairs) > 0,
        "ping_pong_pairs": pairs
    })





df = pd.read_pickle("./data/1_log_clean.pkl")
ping_pong_df = df.groupby("case_seq_num").apply(ping_pong_trace_serie).reset_index(drop=True)

num_ping_pong = ping_pong_df["has_ping_pong"].sum()
print(f"Number of traces with ping-pong: {num_ping_pong}")
# print(ping_pong_df[ping_pong_df["has_ping_pong"] == True])


all_pairs = [pair[0] for pairs in ping_pong_df["ping_pong_pairs"] for pair in pairs]
pair_counts  = Counter(all_pairs)
tt = sum(pair_counts.values())
print("Most common ping pong pairs")
for (a,b) , count in pair_counts.most_common(5):
    print(f"{a} <-> {b}: {count}/{tt} times, {(count / tt) * 100 :.2f}%")

all_orgs = []
all_orgs =[org for pair in all_pairs for org in pair]
orgs_counts  = Counter(all_orgs)
print(len(orgs_counts))
tt = sum(orgs_counts.values())
print("Most common ping pong org:groups")
for org , count in orgs_counts.most_common(5):
    print(f"{org}: {count}/{tt} traces, {(count / tt) * 100 :.2f}%")


all_durations = [pair[1] for pairs in ping_pong_df["ping_pong_pairs"] for pair in pairs]

plt.hist(all_durations, bins=20)
plt.xlabel("Ping-pong duration (days)")
plt.ylabel("Freq")
plt.title("Distribution of ping-pong durations")
plt.show()


dd = pd.DataFrame(all_durations, columns=["duration"])
dd.describe()

pair_df = pd.DataFrame(
    [
        (row.case_seq_num, p, d)
        for _, row in ping_pong_df.iterrows()
        for (p, d) in row.ping_pong_pairs
    ],
    columns=["case_seq_num", "pair", "duration"]
)
idx_max = pair_df.groupby("pair")["duration"].idxmax()
max_cases = (
    pair_df
    .loc[idx_max, ["case_seq_num", "pair", "duration"]]
    .sort_values("duration", ascending=False)
    .head(10)                       # top 10 pairs, change to .head(5) for top 5
    .reset_index(drop=True)
)

print("Top 5 ping pong pairs by max duration (with case):")
for _, row in max_cases.head(5).iterrows():
    a, b = row["pair"]
    print(f"Case {row['case_seq_num']}: {a} <-> {b} : {row['duration']:.2f} days")
