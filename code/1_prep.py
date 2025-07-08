USE_CACHE = True




from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter

import pandas as pd 
import os



#Convert event log into pd.Dataframe and save it as pickle, restore the df if exists
if not os.path.exists("./data/0_log_raw.pkl") or not USE_CACHE:
    log = xes_importer.apply("./data/ev_log.xes")
    df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    df.columns = ["org_group", "res_country", "org_country", "org_res", "org_involved", "org_role", "concept_name", "impact", "product", "lifecycle_transition", "timestamp", "case_seq_num"]
    df.to_pickle("./data/0_log_raw.pkl")
else:
    df = pd.read_pickle("./data/0_log_raw.pkl")

# filter out cases that does not end with Completed
df = df.sort_values(["case_seq_num", "timestamp"])
df = df.groupby("case_seq_num").filter(lambda trace: trace["concept_name"].iloc[-1] == "Completed").reset_index(drop=True)

# Filter out cases with very short or trivial sequences (>2 events)
df = df.groupby("case_seq_num").filter(lambda x: len(x) > 2)


# Filter out cases that they don't start with 'Accepted' or 'Queued'. 4 cases
df_sorted = df.sort_values(["case_seq_num", "timestamp"])
first_events = df_sorted.groupby("case_seq_num").first().reset_index()
valid_case_ids = first_events[first_events["concept_name"].isin(["Accepted", "Queued"])]["case_seq_num"]
df = df_sorted[df_sorted["case_seq_num"].isin(valid_case_ids)]



# Filter out outlier cases by duration
df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["case_seq_num", "timestamp"]) #sorting other time to make sure
case_durations = df.groupby("case_seq_num")["timestamp"].agg(["first", "last"]).reset_index()
case_durations["duration_days"] = (case_durations["last"] - case_durations["first"]).dt.total_seconds() / (24*60*60)
case_durations = case_durations.sort_values(["duration_days"])
duration_99th = case_durations["duration_days"].quantile(0.99)
valid_case_ids = case_durations[case_durations["duration_days"] < duration_99th ]["case_seq_num"]
df =df[df["case_seq_num"].isin(valid_case_ids)]


#filter out noise 
concept_name_proportions = df["concept_name"].value_counts(normalize=True)
valid_concepts = concept_name_proportions[concept_name_proportions >= 0.01].index
df = df[df["concept_name"].isin(valid_concepts)]

lctransitions_proportions = df["lifecycle_transition"].value_counts(normalize=True)
valid_lstransitions = lctransitions_proportions[lctransitions_proportions >= 0.01].index
df = df[df["lifecycle_transition"].isin(valid_lstransitions)]


df.to_pickle("./data/1_log_clean.pkl")





c_logs = df[df["org_involved"] == "Org line C"]
a2_logs = df[df["org_involved"] == "Org line A2"]
b_logs = df[df["org_involved"] == "Org line B"]

c_logs.to_pickle("./data/1_logs_org_c.pkl")
a2_logs.to_pickle("./data/1_logs_org_a2.pkl")
b_logs.to_pickle("./data/1_logs_org_b.pkl")



