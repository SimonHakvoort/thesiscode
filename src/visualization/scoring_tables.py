from tabulate import tabulate
import tensorflow as tf





def compute_crps(emos_dict, X, y, variance, samples = 1000):
    crps_scores = {}
    for name, emos in emos_dict.items():
        crps_scores[name] = emos.loss_CRPS_sample_general(X, y, variance, samples).numpy()

    return crps_scores

def compute_twcrps(emos_dict, X, y, variance, t, samples = 1000):
    def chain_function_indicator(y):
        return tf.maximum(y, t)
    
    twcrps_scores = {}
    for name, emos in emos_dict.items():
        twcrps_scores[name] = emos.loss_twCRPS_sample_general(X, y, variance, chain_function_indicator, samples).numpy()

    return twcrps_scores

def make_table(emos_dict, X, y, variance, t_values, samples):
    crps_scores = compute_crps(emos_dict, X, y, variance, samples)
    t_value_scores = {}
    for t in t_values:
        t_value_scores["t = " + str(t)] = compute_twcrps(emos_dict, X, y, variance, t, samples)

    table_data = []
    headers = ["Model"] + ["CRPS"] + ["t = " + str(t) for t in t_values]

    for name, crps_score in crps_scores.items():
        row = [name] + [crps_score] + [t_value_scores["t = " + str(t)][name] for t in t_values]
        table_data.append(row)

    table = tabulate(table_data, headers, tablefmt="grid")
    return table

    
    
    
