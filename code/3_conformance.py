import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
# from pm4py.objects.conversion.log.variants.to_event_log import Parameters
from pm4py.objects.conversion.process_tree.variants.to_petri_net import apply as pt_to_petri

from pm4py.algo.discovery.inductive.algorithm import apply as inductive_apply
# from pm4py.algo.conformance.alignments.petri_net.algorithm import apply as apply_log
from pm4py.visualization.petri_net.visualizer import view as pn_visualizer
from pm4py.visualization.petri_net.common.visualize import apply as vis_factory
from pm4py.algo.discovery.heuristics.variants.classic import apply as heuristc_miner
from pm4py.algo.conformance.alignments.petri_net.algorithm import apply as alignments_apply

import matplotlib.pyplot as plt


def clean_df(df, status_col="concept_name", completed_fields=["Completed"]):
    """
        Takes df as input, returns a new df cleaned by:
        map lifecycle transitions into new names (to reduce noise)
        drop consecutive duplicates of lifecycle transition in the same case's trace
        Sort by case_seq_num, timestamp
        Change column names
    """
    cases_before = df["case_seq_num"].nunique()
    df = df.sort_values(["case_seq_num", "timestamp"])
    df = df.groupby("case_seq_num").filter(lambda trace: trace[status_col].iloc[-1] in completed_fields).reset_index(drop=True)
    cases_after = df["case_seq_num"].nunique()
    dropped = cases_before - cases_after
    print(f"Dropped {dropped} cases out of {cases_before}  for not having {completed_fields} as last {status_col} in their trace ")

    df = df[df[status_col].ne(df.groupby("case_seq_num")[status_col].shift())].reset_index(drop=True)
    df = df[["case_seq_num", status_col, "timestamp"]]

    df = df.rename(columns={
        "case_seq_num": "case:concept:name",
        status_col: "concept:name",
        "timestamp": "time:timestamp"
    })
    df = df.sort_values(["case:concept:name", "concept:name", "time:timestamp"])
    df = dataframe_utils.convert_timestamp_columns_in_df(df)

    return df


def get_resolution_durations(df,fig_title):
    """
        Return 
            Statistics of duration to resolve a ticket for the dataframe 
            Histogram Plot showing duration in days for resolving 
    """
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    dur = df.groupby("case:concept:name")["time:timestamp"].agg(start="min", end="max").reset_index()
    dur["duration"] = dur["end"] - dur["start"]
    dur["duration_days"] = dur["duration"].dt.total_seconds() /  (24*60*60)
    stats = dur["duration_days"].describe()
    stats_df = stats.to_frame().T

    plt.figure()
    plt.hist(dur['duration_days'], bins=30)
    plt.xlabel("Resolution times (days)")
    plt.ylabel("Number of incidents")
    plt.title(fig_title)

    return stats_df, plt


def get_most_common_trace(df):
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    trace_variants=  (
        df.groupby("case:concept:name")["concept:name"].apply(lambda seq:tuple(seq))
    )
    variants_counts = trace_variants.value_counts(normalize=True)
    top10 = variants_counts.head(10)
    return top10

def get_petri(df):
    df = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp",
        timest_format=None
    )
    lg = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
    net, im, fm = heuristc_miner(lg)
    return net, im, fm, lg

def visualize_petri(net, im, fm):
    gviz = vis_factory(net, im, fm)
    pn_visualizer(gviz)


def get_alignments_fitness(log_1, net_2, im_2, fm_2):
    fit = alignments_apply(log_1, net_2, im_2, fm_2)[0]["fitness"]
    return fit


def get_transitions_net(net):
    return {t.label for t in net.transitions if t.label}



if __name__=="__main__":
    df_org_a2 = pd.read_pickle("./data/1_logs_org_a2.pkl")
    df_org_c = pd.read_pickle("./data/1_logs_org_c.pkl")



    #cleaning
    # cleaned_a2 = clean_df(df_org_a2, "lifecycle_transition",  ["Resolved", "Closed", "Cancelled", "In Call"])
    # cleaned_c = clean_df(df_org_c, "lifecycle_transition" , ["Resolved", "Closed", "Cancelled", "In Call"])

    cleaned_a2 = clean_df(df_org_a2)
    cleaned_c = clean_df(df_org_c)


    #get durations
    dur_stats_a2, dur_fig_a2  = get_resolution_durations(cleaned_a2, "ORG-A2 - Distribution of incident resolving durations")
    dur_stats_c, dur_fig_c  = get_resolution_durations(cleaned_c, "ORG-C - Distribution of incident resolving durations")
    print(f"Duration days statistics of incidents to be resolved in org-A2: ")
    print(dur_stats_a2)
    print(f"Duration days statistics of incidents to be resolved in org-C: ")
    print(dur_stats_c)
    dur_fig_a2.show()
    dur_fig_c.show()



    a2_common_trace = get_most_common_trace(cleaned_a2)
    print("Most common traces in ORG-A2 after filtering")
    print(a2_common_trace)
    
    c_common_trace = get_most_common_trace(cleaned_c)
    print("Most common traces in ORG-C after filtering")
    print(c_common_trace)
    
    
    #get petrinets
    net_a, im_a, fm_a, lg_a = get_petri(cleaned_a2)
    net_c, im_c, fm_c, lg_c = get_petri(cleaned_c)

    #visualize:
    visualize_petri(net_a, im_a, fm_a)
    # visualize_petri(net_c, im_c, fm_c)

    #calculate core fitness
    fitness_a2_on_c = get_alignments_fitness(lg_a,  net_c, im_c, fm_c)
    fitness_c_on_a2 = get_alignments_fitness(lg_c,  net_a, im_a, fm_a)

    print(f"Fitness A2 -> C: {fitness_a2_on_c}")
    print(f"Fitness C -> A2: {fitness_c_on_a2}")
