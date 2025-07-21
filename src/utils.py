import pickle
import torch
import io
import random
import pandas as pd
import numpy as np
import copy
from collections import defaultdict
import pm4py
from Classes import *
from collections import Counter
from statistics import mean
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
import os 
from graphviz import Digraph
from typing import Any, Dict, List, Tuple, Union
from contextlib import redirect_stderr

from pm4py.objects.petri_net.utils import petri_utils as pm4py_petri_utils
from pm4py.objects.petri_net.obj import PetriNet as pm4py_PetriNet
from pm4py.objects.petri_net.obj import Marking as pm4py_Marking



def generate_probs(n_elements):
    probs = []
    for i in range(n_elements):
        res = random.sample(range(1, 10), 3) 
        total = sum(res)
        norm_res = [round(item / total, 2) for item in res]  
        probs.append(norm_res)
    return probs


def smooth_zero_probs(probabilities_lst):
    zero_indices = []
    non_zero_donating_indices = []
    for index, value in enumerate(probabilities_lst):
        if value == 0:
            zero_indices.append(index)
            
        elif value > 0.1:  
            non_zero_donating_indices.append(index)
    
    assert non_zero_donating_indices == []
    
    for zero_index in zero_indices:
        donating_index = random.choice(non_zero_donating_indices)
        assert probabilities_lst[donating_index] <= 0.01        
        probabilities_lst[zero_index] += 0.01
        probabilities_lst[donating_index] -= 0.01
        
    return probabilities_lst


def generate_argmax_probabilities(argmax_prob_is_right, n_activities):
    '''
    Given probability for argmax being the true activity the function generates probability list
    '''
    
    new_probs_list = []
    for i in range(n_activities):
        new_probs_list.append(random.uniform(1.0, 1000.0))
        
    # Making maximal activity about 0.4 - 0.5 in probability
#     new_probs_list[0] = 500 * (n_activities - 2)
    new_probs_list[0] += 250
    max_value = max(new_probs_list)
    max_value_idx = new_probs_list.index(max_value)
    random_uniform_threshold = random.uniform(0,1)
    
    if argmax_prob_is_right > random_uniform_threshold:
        if max_value_idx != 0:
            new_probs_list[0], new_probs_list[max_value_idx] = new_probs_list[max_value_idx], new_probs_list[0]
            
    else:
        if max_value_idx == 0:                      
            random_idx_in_lst = random.choice([i for i in range(1, n_activities)])
            counter = 0
            while new_probs_list[random_idx_in_lst] == max_value:
                random_idx_in_lst = random.choice([i for i in range(1, n_activities)])
                counter += 1
                if counter > 10:
                    break
                    
            new_probs_list[0], new_probs_list[random_idx_in_lst] = new_probs_list[random_idx_in_lst], new_probs_list[0] 
    
    normalize_factor = sum(new_probs_list)
    new_probs_list_normalized = [round(value / normalize_factor, 2) for value in new_probs_list]
    
    return new_probs_list_normalized


def generate_argmax_probabilities_unique(argmax_prob_is_right, n_activities):
    
    while True:
        new_probs = generate_argmax_probabilities(argmax_prob_is_right, n_activities)
        if len(new_probs) == len(set(new_probs)):
            break
    
    return new_probs
    
    
def generate_probabilities(true_value_prob, n_activities):
    new_probs_list = [true_value_prob]
    noisy_probs = []
    for i in range(n_activities - 1):
        noisy_probs.append(random.uniform(1.0, 10.0))
        
    normalize_factor = sum(noisy_probs) / (1-true_value_prob)

    noisy_probs_normalized = [item / normalize_factor for item in noisy_probs]

    new_probs_list += noisy_probs_normalized
    new_probs_list = [round(item,2) for item in new_probs_list]

    return new_probs_list


def add_noise_by_trace(df, disturbed_trans_frac=0.3, true_trans_prob=0.5, expansion_min=2, expansion_max=4, \
                        randomized_true_trans_prob = False, generate_probs_for_argmax=False, all_activities_unique=None, max_expansion=True):
    
    if all_activities_unique is None:
        all_activities_unique = set([activity[0] for activity in df['concept:name'].tolist()])
#     print(f'This is from add_noise_by_trace and the number of unique activities is {len(all_activities_unique)}')
#     print(f'The length of the dataframe from which activities are collected :{len(df)}')
    unique_cases_ids = df['case:concept:name'].unique()
    
    for case_id in unique_cases_ids:
        case_df = df[df['case:concept:name'] == case_id]
        case_df = case_df.reset_index(drop = True)
        case_df = add_noise(case_df, disturbed_trans_frac=disturbed_trans_frac, true_trans_prob=true_trans_prob,
                            expansion_min=expansion_min, expansion_max=expansion_max,
                            randomized_true_trans_prob=randomized_true_trans_prob, generate_probs_for_argmax=generate_probs_for_argmax,
                            all_activities_unique=all_activities_unique, max_expansion=max_expansion)

        df[df['case:concept:name'] == case_id] = case_df
#     print('this is the df the add noise by trace func returns: /n')
#     display(df)
    return df


def add_noise(df, disturbed_trans_frac=0.3, true_trans_prob=0.5, expansion_min=2, expansion_max=4, \
                              randomized_true_trans_prob = False, generate_probs_for_argmax=False, all_activities_unique=None, max_expansion=True):

    n_rows = int(round(disturbed_trans_frac * len(df)))
    rows_expansion_indexes = random.sample(range(len(df)), n_rows)
    all_activities_unique = set([activity[0] for activity in df['concept:name'].tolist()]) if all_activities_unique is None else all_activities_unique 
#     print('this in in the add_noise function ', all_activities_unique)
    for idx in rows_expansion_indexes:
        if randomized_true_trans_prob:
            true_trans_prob = np.random.uniform(0.01,0.99) 
        curr_activity = {df.at[idx, 'concept:name'][0]}
        
        if max_expansion == True:
            n_expanded_trans = random.randint(len(all_activities_unique), len(all_activities_unique))
        else:
            n_expanded_trans = random.randint(expansion_min, expansion_max)
            
        assert n_expanded_trans <= len(all_activities_unique) 
        noisy_activities = random.sample(all_activities_unique.copy().difference(curr_activity), n_expanded_trans - 1) 

        noisy_activities = [activity for activity in noisy_activities]
        
        for activity in noisy_activities:
            df.at[idx,'concept:name'].append(activity)
            
        if generate_probs_for_argmax:
            df.at[idx,'probs'] = generate_argmax_probabilities_unique(true_trans_prob, n_expanded_trans)

        else:
            df.at[idx,'probs'] = generate_probabilities(true_trans_prob, n_expanded_trans)
    
    return df


def construct_trace_model(trace_df, non_sync_move_penalty=1, add_heuristic=False):   
    trace_df = trace_df.reset_index(drop=True)
    places = [Place(f'place_{i}') for i in range(len(trace_df) + 1)]
    transitions = []
    transition_to_idx_dict = {}
    curr_idx = 0
    
    if not isinstance(trace_df['concept:name'].iloc[0], list):
        trace_df['concept:name'] = trace_df['concept:name'].apply(lambda x: [x])

    
    if add_heuristic:
        for i in trace_df.index:
            for idx, activity in enumerate(trace_df.loc[i,'concept:name']):
                prob = trace_df.loc[i,'probs'][idx]
                weight = non_sync_move_penalty - np.log(prob) / 10**5
                new_transition = Transition(f'{activity}_{i+1}', activity, move_type='trace', prob=prob, weight=weight)
                transitions.append(new_transition)
                transition_to_idx_dict[f'{activity}_{i+1}'] = curr_idx
                curr_idx += 1 
                
    # ---original 08/05/24---
    # else:
    #     for i in trace_df.index:
    #         for idx, activity in enumerate(trace_df.loc[i,'concept:name']):
    #             new_transition = Transition(f'{activity}_{i+1}', move_type='trace', label=activity, prob=trace_df.loc[i,'probs'][idx], weight=non_sync_move_penalty)
    #             transitions.append(new_transition)
    #             transition_to_idx_dict[f'{activity}_{i+1}'] = curr_idx
    #             curr_idx += 1

    else:
        for i in trace_df.index:
            for idx, activity in enumerate(trace_df.loc[i,'concept:name']):
                new_transition = Transition(f'{activity}_ts={i}', move_type='trace', label=activity, prob=trace_df.loc[i,'probs'][idx], weight=non_sync_move_penalty)
                transitions.append(new_transition)
                transition_to_idx_dict[f'{activity}_{i+1}'] = curr_idx
                curr_idx += 1    
    
    trace_model_net = PetriNet('trace_model', places, transitions)    
    
    for i in trace_df.index:
        for activity in trace_df.loc[i, 'concept:name']: 
            trace_model_net.add_arc_from_to(places[i], transitions[transition_to_idx_dict[f'{activity}_{i+1}']])
            trace_model_net.add_arc_from_to(transitions[transition_to_idx_dict[f'{activity}_{i+1}']], places[i+1])
    
    init_mark = tuple([1] + [0] * len(trace_df))
    final_mark = tuple([0] * len(trace_df) + [1])
    
    trace_model_net.init_mark = Marking(init_mark)
    trace_model_net.final_mark = Marking(final_mark)
    
    return trace_model_net
            
    
def pre_process_log(df_log):
    df_log = copy.deepcopy(df_log)
    clean_df_log = df_log[['concept:name', 'case:concept:name']]
    clean_df_log['probs'] = [[1.0]] * len(clean_df_log)
    clean_df_log['probs'].astype('object')
    clean_df_log['concept:name'] = clean_df_log['concept:name'].apply(lambda activity: [activity])
    
    return clean_df_log


def calc_conf_for_log(df_log, model, non_sync_move_penalty=1, add_heuristic=False):
    unique_cases_ids = df_log['case:concept:name'].unique()
    
    for case_id in unique_cases_ids:
        case_df = df_log[df_log['case:concept:name'] == case_id]
        case_df = case_df[['concept:name', 'probs']]
        case_df = case_df.reset_index(drop = True)
#         print('trace len is: ', len(case_df))
#         print('trace case: ', case_id)
        case_trace_model = construct_trace_model(case_df, non_sync_move_penalty, add_heuristic=add_heuristic)

#         print(model.conformance_checking(case_trace_model))
#         print()
#         print()
        
        
def sort_places(places):
    init_mark = [place for place in places if place.name == 'source']
    final_mark = [place for place in places if place.name == 'sink']
    inner_places = [place for place in places if place.name not in {'source', 'sink'}]
    inner_places_sorted = sorted(inner_places, key=lambda x: float(x.name[2:]))
    places_sorted = init_mark + inner_places_sorted + final_mark
    
    return places_sorted


def from_discovered_model_to_PetriNet(discovered_model, non_sync_move_penalty=1, name='discovered_net', 
                                      cost_function=None, conditioned_prob_compute=False, 
                                      quiet_moves_weight=1e-8, sync_moves_weight=1e-6, return_mapping=False,
                                      pm4py_init_marking=None, pm4py_final_marking=None):
    """
    Convert a discovered pm4py model to a PetriNet object.
    Args:
        discovered_model: The pm4py discovered model to convert.
        non_sync_move_penalty: Penalty for non-synchronous moves. Default is 1.
        name: Name of the new PetriNet. Default is 'discovered_net'.
        cost_function: Cost function for the PetriNet. Default is None.
        conditioned_prob_compute: Flag for conditioned probability computation. Default is False.
        quiet_moves_weight: Weight for quiet moves. Default is 0.0000001.
        return_mapping: If True, returns the place_mapping in addition to the PetriNet object. Default is False.
    Returns:
        A PetriNet object with additional pm4py-related fields.
        If return_mapping is True, returns a tuple (PetriNet, place_mapping).
    """
    def create_arc(source, target):
        arc = Arc(source, target)
        petri_new_arcs.append(arc)
        return arc

    # Sort and create new places
    sorted_places = sort_places(discovered_model.places)
    places = [Place(p.name) for p in sorted_places]
    
    # Create place mapping
    place_mapping = {old_p: idx for idx, old_p in enumerate(sorted_places)}
    reverse_place_mapping = {idx: old_p for idx, old_p in enumerate(sorted_places)}
    
    # Create transitions
    transitions = [Transition(t.name, t.label, set(), set(), 'model', 
                              weight=quiet_moves_weight if t.label is None else non_sync_move_penalty) 
                   for t in discovered_model.transitions]
    # Create mappings for efficient lookup
    place_dict = {p.name: p for p in places}
    trans_dict = {t.name: t for t in transitions}
    # Create arcs
    petri_new_arcs = []
    for t in discovered_model.transitions:
        new_t = trans_dict[t.name]
        for arc in t.in_arcs:
            new_arc = create_arc(place_dict[arc.source.name], new_t)
            new_t.in_arcs.add(new_arc)
        for arc in t.out_arcs:
            new_arc = create_arc(new_t, place_dict[arc.target.name])
            new_t.out_arcs.add(new_arc)
    # Set transition names to labels if available
    for t in transitions:
        if t.label is not None:
            t.name = t.label
            
    # Create and setup new PetriNet
    new_PetriNet = PetriNet(name)
    new_PetriNet.add_places(places)
    new_PetriNet.add_transitions(transitions)
    new_PetriNet.init_mark = Marking((1,) + (0,) * (len(places) - 1))
    new_PetriNet.final_mark = Marking((0,) * (len(places) - 1) + (1,))
    new_PetriNet.arcs = petri_new_arcs
    new_PetriNet.cost_function = cost_function
    new_PetriNet.conditioned_prob_compute = conditioned_prob_compute
    new_PetriNet.epsilon = sync_moves_weight
    
    # Add pm4py-related fields
    new_PetriNet.pm4py_net = discovered_model
    new_PetriNet.pm4py_initial_marking = pm4py_init_marking
    new_PetriNet.pm4py_final_marking = pm4py_final_marking
    new_PetriNet.place_mapping = place_mapping
    new_PetriNet.reverse_place_mapping = reverse_place_mapping

    if return_mapping:
        return new_PetriNet, place_mapping
    else:
        return new_PetriNet


def convert_marking(pm4py_marking, place_mapping):
    """
    Convert a pm4py marking to a custom Marking object.

    Args:
        pm4py_marking: A dictionary mapping pm4py Place objects to token counts.
        place_mapping: A dictionary mapping pm4py Place objects to indices in the custom Marking.

    Returns:
        A custom Marking object.
    """
    marking_list = [0] * len(place_mapping)
    for place, tokens in pm4py_marking.items():
        if place in place_mapping:
            marking_list[place_mapping[place]] = tokens
    return Marking(marking_list)


def argmax_stochastic_trace(stochastic_trace_df):
    determ_df = pd.DataFrame(columns=['concept:name', 'case:concept:name', 'probs'])
    # display(stochastic_trace_df.head())
    for i in range(len(stochastic_trace_df)):
        max_val = max(stochastic_trace_df.iloc[i,2])
        max_idx = stochastic_trace_df.iloc[i,2].index(max_val)
        highest_prob_activity = stochastic_trace_df.iloc[i,0][max_idx]
        case_id =  stochastic_trace_df.iloc[i,1]
        new_row = pd.DataFrame({'concept:name': [highest_prob_activity], 'case:concept:name': [case_id], 'probs': [[1.0]]})
        determ_df = pd.concat([determ_df, new_row], ignore_index=True)
        # determ_df = determ_df.append({'concept:name': highest_prob_activity, 'case:concept:name': case_id, 'probs':[1.0]}, ignore_index=True)
        
    return determ_df


def get_non_sync_non_quiet_activities(alignment, quiet_activities):
    non_sync_non_quiet_transitions = []
    for trans in alignment:
        if '>>' in trans:
            if not any(activity for activity in quiet_activities if activity in trans):
                non_sync_non_quiet_transitions.append(trans) 

    
    return non_sync_non_quiet_transitions    


def get_sync_activities(alignment):
    return [align for align in alignment if '>>' not in align]      


# def alignment_accuracy_helper(alignment, true_trace_df):
#     stochastic_align_clean = [item.split(',')[1][1:-1] for item in alignment[0] if item.split(',')[1][1:-1] != '>>']
#     determ_trace_clean = true_trace_df['concept:name'].tolist()

#     similar_activity = 0
#     for idx, item in enumerate(determ_trace_clean):
#         if item in stochastic_align_clean[idx]:
#             similar_activity += 1
        
#     return similar_activity / len(determ_trace_clean), len(determ_trace_clean)


def compare_argmax_and_stochastic_alignments(stochastic_trace_df, true_trace_df, model, non_sync_penalty=1, add_heuristic=False, prob_dict=None, lamda=0.5):

    df_stochastic = stochastic_trace_df[['concept:name', 'probs']]
    df_stochastic = df_stochastic.reset_index(drop = True)
    case_trace_stochastic_model = construct_trace_model(df_stochastic, non_sync_penalty, add_heuristic=add_heuristic)
    stochastic_alignment = model.conformance_checking(case_trace_stochastic_model, prob_dict, lamda=lamda)
#     print(f'the stochastic alignment optimal cost is: {stochastic_alignment[1]} \n')
    argmax_trace = argmax_stochastic_trace(stochastic_trace_df)
#     argmax_trace_preprocessed = pre_process_log(argmax_trace)
#     df_deterministic = argmax_trace_preprocessed[['concept:name', 'probs']]
#     df_deterministic = df_deterministic.reset_index(drop = True)
#     case_trace_deterministic_model = construct_trace_model(df_deterministic, non_sync_penalty, add_heuristic=add_heuristic)    
#     argmax_alignment = model.conformance_checking(case_trace_deterministic_model, prob_dict)

    quiet_activities = {transition.name for transition in model.transitions if transition.label is None}
    non_sync_non_quiet_stochastic_alignment_activities = get_non_sync_non_quiet_activities(stochastic_alignment[0], quiet_activities)
    sync_stochastic_alignment_activities = get_sync_activities(stochastic_alignment[0])  
#     non_sync_non_quiet_argmax_alignment_activities = get_non_sync_non_quiet_activities(argmax_alignment[0], quiet_activities)
#     sync_argmax_alignment_activities = get_sync_activities(argmax_alignment[0])     
    stochastic_acc, trace_len = alignment_accuracy_helper(stochastic_alignment, true_trace_df)
#     print(f'the true trace is: {display(true_trace_df)} \n')
#     print(f'argmax trace is: {display(argmax_trace)}')
    argmax_trace_list = argmax_trace['concept:name'].reset_index(drop=True).tolist()
    true_trace_list = true_trace_df['concept:name'].reset_index(drop=True).tolist()
    argmax_acc = len([i for i, j in zip(argmax_trace_list, true_trace_list) if str(i) == str(j)]) / len(true_trace_list)
#     argmax_acc = sum(argmax_trace['concept:name'].reset_index(drop=True) == true_trace_df['concept:name'].reset_index(drop=True)) / len(argmax_trace)
#     print(stochastic_alignment)
#     print(f'stochastic acc: {stochastic_acc}, argmax acc: {argmax_acc}')
    return stochastic_acc, argmax_acc
#     return non_sync_non_quiet_stochastic_alignment_activities, sync_stochastic_alignment_activities, \
#                             non_sync_non_quiet_argmax_alignment_activities, sync_argmax_alignment_activities, \
#                                                                                    stochastic_acc, argmax_acc, trace_len



def generate_statistics_for_dataset(stochastic_dataset, df_test, model, non_sync_penalty=1, add_heuristic=False, prob_dict=None, lamda=0.5):
    '''Generate 4 arrays for plots. 
       Input: stochastic dataframe with multiple traces
       Output: 4 arrays with amounts of sync and non_syc activities'''
    
    unique_cases_ids = stochastic_dataset['case:concept:name'].unique()
    
    non_sync_non_quiet_stochastic_alignment_len = []
    sync_stochastic_alignment_len = []
    non_sync_non_quiet_argmax_len = []
    sync_argmax_alignment_len = []
    stochastic_acc_lst = []
    argmax_acc_lst = []
    traces_length = []
    
    for case_id in unique_cases_ids:
        case_df = stochastic_dataset[stochastic_dataset['case:concept:name'] == case_id]
        case_df = case_df.reset_index(drop = True)
        true_case_df = df_test[df_test['case:concept:name'] == case_id]

#         non_sync_non_quiet_stochastic_alignment_activities, sync_stochastic_alignment_activities, \
#         non_sync_non_quiet_argmax_alignment_activities, sync_argmax_alignment_activities, stochastic_acc, \
#         argmax_acc, trace_len = compare_argmax_and_stochastic_alignments(case_df, true_case_df, model, non_sync_penalty, add_heuristic=add_heuristic)                               
        trace_len = len(true_case_df)
        stochastic_acc, argmax_acc = compare_argmax_and_stochastic_alignments(case_df, true_case_df, model, non_sync_penalty, add_heuristic=add_heuristic, prob_dict=prob_dict, lamda=lamda) 

    
#         non_sync_non_quiet_stochastic_alignment_len.append(len(non_sync_non_quiet_stochastic_alignment_activities))
#         sync_stochastic_alignment_len.append(len(sync_stochastic_alignment_activities))
#         non_sync_non_quiet_argmax_len.append(len(non_sync_non_quiet_argmax_alignment_activities))
#         sync_argmax_alignment_len.append(len(sync_argmax_alignment_activities))
        stochastic_acc_lst.append(stochastic_acc) 
        argmax_acc_lst.append(argmax_acc) 
        traces_length.append(trace_len)
        
#     return non_sync_non_quiet_stochastic_alignment_len, sync_stochastic_alignment_len, \
#         non_sync_non_quiet_argmax_len, sync_argmax_alignment_len, stochastic_acc_lst, argmax_acc_lst, traces_length

    return stochastic_acc_lst, argmax_acc_lst, traces_length


def calculate_statistics_for_different_uncertainty_levels(df, non_sync_penalty=1, n_traces_for_model_building = 10,
                                                          true_trans_prob=None, expansion_min=2, expansion_max=4,             
                                                          uncertainty_levels=None, generate_probs_for_argmax=False, 
                                                          by_trace=False, add_heuristic=False, 
                                                          change_probs_only=False, cost_functions_lst=None, max_history_len=3,
                                                          precision=3, lamda=0.5, utilize_trace_family=False, random_seed=101,
                                                          all_activities_unique=None, top_activities_for_model_discovery=True,
                                                          frequent_traces_first=True, max_expansion=True):
    
   
    if uncertainty_levels is None:
        uncertainty_levels = np.linspace(0,1,21)
        
    if top_activities_for_model_discovery:
        # selecting the top most frequent n_traces_for_model_building families and sampling one from each
        train_traces = sample_traces(df,n_unique_traces=n_traces_for_model_building, freq_dict=None, frequent_first=frequent_traces_first)
        
    elif utilize_trace_family:
        df_no_loops = remove_loops_in_log_only(df)
        df_train, df_test = train_test_log_split(df_no_loops,n_traces=n_traces_for_model_building,random_selection=False,random_seed=random_seed,
                                         sample_from_each_trace_family=True, n=1)
        train_traces = list(df_train['case:concept:name'].unique())
        
    else:
#         train_traces = list(df['case:concept:name'].unique())[:n_traces_for_model_building]
        random.seed(random_seed)
        train_traces = random.sample(set(df['case:concept:name'].unique()), n_traces_for_model_building)
#         print(f'train traces = {train_traces}')
    train_data = df[df['case:concept:name'].isin(train_traces)]
    test_data = df[~df['case:concept:name'].isin(train_traces)]
    
    prob_dict = build_conditioned_prob_dict(train_data, max_hist_len=max_history_len, precision=precision)

    if not isinstance(cost_functions_lst,  list):
        cost_functions_lst = [cost_functions_lst]
        
    net, initial_marking, final_marking = inductive_miner.apply(train_data)
#     stats_dict = compute_model_statistics(test_data, net, initial_marking, final_marking)
#     print(stats_dict)
#     return stats_dict

    discovered_models_lst = [from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty, cost_function=cost_function, conditioned_prob_compute=True) for cost_function in cost_functions_lst] 
    
#     gviz = pn_visualizer.apply(net, initial_marking, final_marking)
#     pn_visualizer.view(gviz)
    
    if true_trans_prob is None:
        true_trans_prob = np.linspace(0,1,21)
    
    if not isinstance(true_trans_prob, (np.ndarray, list)):
        true_trans_prob = [true_trans_prob]
    
    
    non_sync_stochastic_avgs = []
    sync_stochastic_avgs = []
    non_sync_argmax_avgs = []
    sync_argmax_avgs = []
    stochastic_acc_avgs = [[] for _ in range(len(cost_functions_lst))]
    argmax_acc_avgs = []
    stochastic_trace_resolution = []
    argmax_trace_resolution = []
    
    preprocessed_test_data_noised = None
    
    if all_activities_unique is None:
        all_activities_unique = set(df['concept:name'].tolist())
#     print('all activities unique are: ', all_activities_unique)
    for true_prob in true_trans_prob:
        true_prob = round(true_prob, 2)
        
        for uncert_level in uncertainty_levels:        
            uncert_level = round(uncert_level, 2)
            
            if not change_probs_only:
                preprocessed_test_data = pre_process_log(test_data.copy(deep=True).reset_index(drop=True))

                if by_trace is None:

                    preprocessed_test_data_noised = add_noise(preprocessed_test_data, disturbed_trans_frac=uncert_level,
                                             true_trans_prob=true_prob, expansion_min=expansion_min, expansion_max=expansion_max,
                                             randomized_true_trans_prob = False, generate_probs_for_argmax = generate_probs_for_argmax,
                                             all_activities_unique=all_activities_unique, max_expansion=max_expansion)
                else:

                    preprocessed_test_data_noised = add_noise_by_trace(preprocessed_test_data, disturbed_trans_frac=uncert_level,
                                             true_trans_prob=true_prob, expansion_min=expansion_min, expansion_max=expansion_max,
                                             randomized_true_trans_prob = False, generate_probs_for_argmax = generate_probs_for_argmax,
                                             all_activities_unique=all_activities_unique, max_expansion=max_expansion)         
            
            else:
                if preprocessed_test_data_noised is None:
                    test_data_copy = test_data.copy(deep=True).reset_index(drop=True)
                    preprocessed_test_data_noised = preprocess_and_add_noise(test_data_copy, uncert_level, true_prob, expansion_min, expansion_max, generate_probs_for_argmax,
                                                                             all_activities_unique=all_activities_unique, max_expansion=max_expansion)
                
                else:
                    preprocessed_test_data_noised = reasign_probs_using_argmax(preprocessed_test_data_noised, true_prob)
            
#             non_sync_non_quiet_stochastic_alignment_length, sync_stochastic_alignment_length, \
#             non_sync_non_quiet_argmax_length, sync_argmax_alignment_length, stochastic_acc, \
#             argmax_acc, traces_length = generate_statistics_for_dataset(preprocessed_test_data_noised, df, discovered_net, non_sync_penalty, add_heuristic=add_heuristic)

#             display(preprocessed_test_data_noised.head(30))
#             display(df.head(30))
        
            for i, model in enumerate(discovered_models_lst):
                stochastic_acc, argmax_acc, traces_length = generate_statistics_for_dataset(preprocessed_test_data_noised, df, model, non_sync_penalty, add_heuristic=add_heuristic, prob_dict=prob_dict, lamda=lamda)
                stochastic_acc_avgs[i].append(mean(stochastic_acc)) # [[cost_func_1_avgs], [cost_func_2_avgs]...]
                
            argmax_acc_avgs.append(mean(argmax_acc))
                
#             mean_stochastic_avg = (np.array(stochastic_acc).dot(np.array(traces_length))) / sum(traces_length)
#             mean_argmax_avg = (np.array(argmax_acc).dot(np.array(traces_length))) / sum(traces_length)
            
#             non_sync_stochastic_avgs.append(mean(non_sync_non_quiet_stochastic_alignment_length))
#             sync_stochastic_avgs.append(mean(sync_stochastic_alignment_length))
#             non_sync_argmax_avgs.append(mean(non_sync_non_quiet_argmax_length))
#             sync_argmax_avgs.append(mean(sync_argmax_alignment_length))
            
             
            print(f'true prob: {true_prob},  uncertainty_level: {uncert_level},  mean_stochastic_acc: {[mean(stochastic_acc) for stochastic_acc in stochastic_acc_avgs]},  mean_argmax_acc: {mean(argmax_acc)}')
            print('--------------------------------------------------------------------------------------------------')
            print()
            
        
    return stochastic_acc_avgs, argmax_acc_avgs 



def generate_stats_dict_constant_stochastic_traces_frequency(history):
    '''
    Dictionary where keys are stochastic traces frequency and values are frequency
    of alignments varied by the probability of the true transition
    '''
    
    stats_dict = defaultdict(list)
    stochastic_traces_frequency = 0
    for i in range(21):
        j=i
        while j < 441:
            stats_dict[stochastic_traces_frequency].append(history[j])
            j += 21
        
        stochastic_traces_frequency += 0.05
        stochastic_traces_frequency = round(stochastic_traces_frequency, 2)
            
    return stats_dict


def filter_log(log, max_len=None, n_traces=None, random_selection=False, min_frequency=None, min_representatives_for_variant=5, random_seed=42):
    
    accepted_cases = list(log['case:concept:name'].unique())
  
    if min_frequency is not None:
        freq_dict = generate_trace_freq_dict(log)
        sorted_by_freq_dicts_list = sorted(freq_dict.items(), key=lambda item: item[1]['frequency'], reverse=True)
        accepted_cases = sum([item[1]['cases'] for item in sorted_by_freq_dicts_list if item[1]['frequency'] >= min_frequency], [])
    
    if max_len is not None:   
        accepted_cases = [case for case in accepted_cases if len(log[log['case:concept:name'] == case]) <= max_len] 
            
    if n_traces is None:
        return log[log['case:concept:name'].isin(accepted_cases)]  

    if random_selection:
        accepted_freq_dict = generate_trace_freq_dict(log[log['case:concept:name'].isin(accepted_cases)])
        unique_representative_cases = sum([item['cases'][:min_representatives_for_variant] for item in accepted_freq_dict.values()], [])
        random.seed(random_seed)
        additional_cases = random.sample(list(set(accepted_cases).difference(unique_representative_cases)), n_traces-len(unique_representative_cases))
        accepted_cases = unique_representative_cases + additional_cases
        
    else:
        accepted_cases = accepted_cases[:n_traces]        
    
    return log[log['case:concept:name'].isin(accepted_cases)]  


def get_df_trace_lengths(df):
    return df.groupby(["case:concept:name"])['concept:name'].count().reset_index(name='count')['count'].values


def reasign_probs_using_argmax(df, p_f): 
    probs_length_lst = df['probs'].apply(lambda x: len(x)).tolist()
    new_probs = [generate_argmax_probabilities_unique(p_f, n_t) for n_t in probs_length_lst]
    df['probs'] = new_probs
    
    return df


def preprocess_and_add_noise(df, t_p, p_f, n_t_min, n_t_max, generate_probs_using_argmax, all_activities_unique=None, max_expansion=True):
    preprocessed_data = pre_process_log(df.copy(deep=True).reset_index(drop=True))
    
    return add_noise(preprocessed_data, disturbed_trans_frac=t_p, true_trans_prob=p_f, expansion_min=n_t_min,
                     expansion_max=n_t_max, generate_probs_for_argmax = generate_probs_using_argmax, randomized_true_trans_prob = False,
                     all_activities_unique=all_activities_unique, max_expansion=max_expansion)


def generate_sk_log(dk_log, disturbed_trans_frac=0.3, true_trans_prob=0.5, expansion_min=2, expansion_max=4, \
                       randomized_true_trans_prob = False, generate_probs_for_argmax=False, gradual_noise=True,
                       alter_labels=False, swap_events=False, duplicate_acts=False, fraq=0, determ_noise_only=False,
                       all_activities_unique=None, max_expansion=True):
    
    return add_noise_by_trace(pre_process_log(dk_log.copy(deep=True)), disturbed_trans_frac=disturbed_trans_frac, 
                              true_trans_prob=true_trans_prob, expansion_min=expansion_min, expansion_max=expansion_max, 
                              randomized_true_trans_prob = randomized_true_trans_prob, 
                              generate_probs_for_argmax=generate_probs_for_argmax, all_activities_unique=all_activities_unique, max_expansion=max_expansion)


def train_test_split(df, n_train_traces=15, random_selection=True, random_seed=42):
    
    cases_list = list(df['case:concept:name'].unique()) 
    
    if random_selection:
        random.seed(random_seed)
        train_cases = random.sample(cases_list, n_train_traces)
    
    else:
        train_cases = accepted_cases[:n_train_traces]
    
    train_df = df[df['case:concept:name'].isin(train_cases)] 
    test_df = df[~df['case:concept:name'].isin(train_cases)]
    
    return train_df, test_df


def evaluate_conformance_cost(log, model, non_sync_penalty=1):
    
    cases_list = list(log['case:concept:name'].unique()) 
    conf_costs = []
    for case in cases_list:
        trace_case = log[log['case:concept:name'] == case]
        trace_case = trace_case.drop('case:concept:name', axis=1)
        trace_case = trace_case.reset_index(drop = True)
        trace_case_model = construct_trace_model(trace_case, non_sync_move_penalty=non_sync_penalty)
        trace_alignment, trace_cost = model.conformance_checking(trace_case_model)
        conf_costs.append(trace_cost)
    
    return conf_costs


def reasign_probs(porbs_lst, p_f):
    return [p_f, round(1-p_f,2)]


def evaluate_accuracy_score(stochastic_log, original_log, model, non_sync_penalty=1, prob_dict=None, lamda=0.5):
    
    stochastic_acc_lst = []
    argmax_acc_lst = []
    cases_ids = stochastic_log['case:concept:name'].unique()

    for case in cases_ids:
        stochastic_trace_df = stochastic_log[stochastic_log['case:concept:name'] == case]
        determ_trace_df = original_log[original_log['case:concept:name'] == case]
        
        stochastic_acc, argmax_acc = compare_argmax_and_stochastic_alignments(stochastic_trace_df, determ_trace_df,
                                                                              model, non_sync_penalty=non_sync_penalty, 
                                                                              prob_dict=prob_dict, lamda=lamda)
        stochastic_acc_lst.append(stochastic_acc) 
        argmax_acc_lst.append(argmax_acc)
   
    return mean(stochastic_acc_lst), mean(argmax_acc_lst)


# def evaluate_loss_function(stochastic_log, original_log, model, non_sync_penalty=1, multiple_models=False):
#     ''' model is a single model if multiple_models=False or a list of models otherwise'''
    
#     stochastic_acc_lst = []
#     argmax_acc_lst = []
    
#     for val in np.linspace(0,1,21):
        
#         df_test_stochastic['probs'] = df_test_stochastic['probs'].apply(lambda x: generate_argmax_probabilities_unique(round(val,2),2))
        
#         if multiple _models:
#             evaluate_accuracy_score(stochastic_log, original_log, model, non_sync_penalty=1) for mode
#         mean_stochastic_acc, mean_argmax_acc = evaluate_accuracy_score(stochastic_log, original_log, model, non_sync_penalty=1)
#         stochastic_acc_lst.append(mean_stochastic_acc)
#         argmax_acc_lst.append(mean_argmax_acc)
#         print(f'p_a: {round(val,2)}, stochastic acc: {round(mean_stochastic_acc, 2)}, argmax acc: {round(mean_argmax_acc,2)}')
        
#     return stochastic_acc_lst, argmax_acc_lst 


def evaluate_loss_functions(stochastic_log, original_log, model, cost_functions_lst, non_sync_penalty=1):
   
    models_lst = [from_discovered_model_to_PetriNet(model, non_sync_move_penalty=non_sync_penalty, cost_function=cost_func)
                  for cost_func in cost_functions_lst]
    
    results_lists = [list() for _ in range(len(models_lst))]
    argmax_acc_dict = {}
    
    for val in np.linspace(0,1,21): 
        print(f'argmax prob: {round(val, 2)}')
        stochastic_log['probs'] = stochastic_log['probs'].apply(lambda x: generate_argmax_probabilities_unique(round(val,2),
                                                                                                    len(x)))
        for idx, model in enumerate(models_lst):
            mean_stochastic_acc, mean_argmax_acc = evaluate_accuracy_score(stochastic_log, original_log, model, non_sync_penalty=1)
            results_lists[idx].append(mean_stochastic_acc)
            argmax_acc_dict[round(val,2)] = mean_argmax_acc
    
    return results_lists, argmax_acc_dict


def evaluate_different_model_sizes(stochastic_log, original_log, train_log, cost_functions_lst, non_sync_penalty=1, n_parallel_transitions=2, n_train_traces_lst=None):
  
    if n_train_traces_lst is None:
        n_train_traces_lst = [5, 15, 30, 50, 100, 400, 1000]
    
    res_lsts = []
    res_dicts = []
    
    for n_train_traces in n_train_traces_lst: 
        print(f'Evaluating discovered model from {n_train_traces} traces')
        temp_df = train_log[train_log['case:concept:name'].isin(train_log['case:concept:name'].unique()[:n_train_traces])]
        net, initial_marking, final = inductive_miner.apply(temp_df)
        res_lst, res_dict = evaluate_loss_functions(stochastic_log, original_log, net, cost_functions_lst,
                                                    non_sync_penalty=non_sync_penalty,
                                                    n_parallel_transitions=n_parallel_transitions)
                                                    
        res_lsts.append(res_lst)
        res_dicts.append(res_dict)
    
    return res_lsts, res_dicts


def plot_results(results_lst, argmax_acc_dict):
    
    idx = [round(val,2) for val in np.linspace(0,1,21)]

#     res_df = pd.DataFrame(
#         {'linear': results_lst[0],
#          'exponential': results_lst[1],
#          'logarithmic': results_lst[2],
#          'argmax': list(argmax_acc_dict.values()),
#         }, index = idx) 
    

    plt.figure(figsize=(8,6))
    plt.scatter(idx, results_lst[0], label='linear', marker='+', color='black', s=90)
    plt.scatter(idx, results_lst[1], label='exponential', marker='x', color='black', s=50)
    plt.scatter(idx, results_lst[2], label='logarithmic', marker='o', color='black', s=30) 
    plt.scatter(idx, argmax_acc_dict.values(), label='argmax', marker='^', color='black', s=50) 
    
    plt.xlabel('P_a')
    plt.ylabel('Accuracy')
    plt.legend();
    
#     ax = res_df.plot()
#     ax.set_xlabel("p_a")
#     ax.set_ylabel("accuracy score");


def trace_self_loops_indices(trace_df):
    trace_df_copy = copy.copy(trace_df)
    trace_df_copy['concept:name_shifted'] = trace_df_copy['concept:name'].shift(1)
    trace_df_copy['case:concept:name_shifted'] = trace_df_copy['case:concept:name'].shift(1)
    trace_df_copy['is_duplicate'] = trace_df_copy.apply(check_duplicate_rows, axis=1)
    return trace_df_copy['is_duplicate'].to_numpy().astype(np.bool_)

    
def sample_n_traces(df, n_traces=10, random=True):
    if random is False:
        trace_cases = list(df['case:concept:name'].unique())[:n_traces]
    else:
        trace_cases = sample(list(df['case:concept:name'].unique()),n_traces)
    return df[df['case:concept:name'].isin(trace_cases)]


def remove_self_loops_in_trace(trace_df):
    return trace_df[(trace_df['concept:name'] != trace_df.shift(1)['concept:name']) | (trace_df['case:concept:name'] != trace_df.shift(1)['case:concept:name'])]


def remove_self_loops_in_dataset(log_df, return_self_loops_indices=False):
    trace_cases = list(log_df['case:concept:name'].unique())
    
    new_traces_lst = []
    self_loop_indices_lst = []
    for trace_case in trace_cases:
        trace = log_df[log_df['case:concept:name'] == trace_case]
        curr_trace_self_loop_indices = trace_self_loops_indices(trace)
        trace = remove_self_loops_in_trace(trace)
        new_traces_lst.append(trace)
        self_loop_indices_lst += list(curr_trace_self_loop_indices)
    
    new_log_df = pd.concat(new_traces_lst)
    if return_self_loops_indices:
        return new_log_df, np.array(self_loop_indices_lst)
    return new_log_df


def check_duplicate_rows(row):
    if row['concept:name'] != row['concept:name_shifted'] or row['case:concept:name_shifted'] != row['case:concept:name_shifted']:
        return 0
    return 1


def argmax_sk_trace(trace_df):
    trace_df['argmax_activity_label'] = trace_df.apply(get_max_prob_activity, axis=1)
    return trace_df[['argmax_activity_label', 'case:concept:name']]


def get_max_prob_activity(row):
    max_idx = row['probs'].index(max(row['probs']))
    max_prob_activity = row['concept:name'][max_idx]
    return max_prob_activity


def filter_softmax_matrice(sftm_mat, is_dup_bool_vec):
    np_sftm_mat = sftm_mat.squeeze(0).cpu().numpy()
    return np_sftm_mat[:,np.invert(is_dup_bool_vec)]


def remove_loops_in_trace_and_matrice(trace_df, sftm_mat):
    self_loops_indices = trace_self_loops_indices(trace_df)
    no_loops_trace = remove_self_loops_in_trace(trace_df)
    no_loops_sftm_mat = filter_softmax_matrice(sftm_mat, self_loops_indices)
    return no_loops_trace, no_loops_sftm_mat


def remove_loops_in_log_and_sftm_matrices_lst(log_df, sftm_mat_lst):
    trace_cases = list(log_df['case:concept:name'].unique())
    no_loops_trace_lst = []
    no_loops_sftm_mat_lst = []
    
    for i, case in enumerate(trace_cases):
        trace = log_df[log_df['case:concept:name'] == case]
        loop_indices = trace_self_loops_indices(trace)
        no_loops_trace = remove_self_loops_in_trace(trace)
        no_loop_stmx_mat = filter_softmax_matrice(sftm_mat_lst[i], loop_indices)
        no_loops_trace_lst.append(no_loops_trace)
        no_loops_sftm_mat_lst.append(no_loop_stmx_mat)
    
    return pd.concat(no_loops_trace_lst), no_loops_sftm_mat_lst
    

# def sfmx_mat_to_sk_trace(sftm_mat, case_num, round_precision=20):
#     if type(sftm_mat) is torch.Tensor:
#         sftm_mat = sftm_mat.squeeze(0).cpu().numpy()
    
# #     activities_arr = np.arange(19) valid only for 50 salads
#     activities_arr = np.arange(sftm_mat.shape[0])
#     df_prob_lst = []
#     df_activities_lst = []
# #     di = activity_map_dict() valid only for 50 salads
    
#     for i in range(sftm_mat.shape[1]):
#         probs = np.round(sftm_mat[:,i], round_precision)
#         activities = list(activities_arr[np.nonzero(probs)])
#         activities = [str(act) for act in activities]
# #         activities = [di[str(act)] for act in activities]  valid only for 50 salads
#         df_prob_lst.append(list(probs[np.nonzero(probs)]))
#         df_activities_lst.append(activities)
   
#     case_lst = [case_num] * sftm_mat.shape[1]
    
#     df = pd.DataFrame(
#     {'concept:name': df_activities_lst,
#      'case:concept:name': case_lst,
#      'probs': df_prob_lst
#     })
    
#     return df


def argmax_sftmx_matrice(stmx_mat):
    return np.argmax(stmx_mat, axis=0)  


# def train_test_log_split(log, n_traces, random_selection=False, sample_from_each_trace_family=True, n=1, random_seed=42):
#     print(f'Within the train_test_log_split func. number of traces={n_traces}')
#     cases_list = list(log['case:concept:name'].unique())
#     if n_traces is not None:
#         assert len(cases_list) >= n_traces, "Houston we've got a problem - more traces were demanded than there were available"
    
#     if sample_from_each_trace_family:
#         trace_family_dict = group_similar_cases(log)
#         final_cases = sample_from_dict(trace_family_dict, n=n, seed=random_seed)

#     elif random_selection:
#         random.seed(random_seed)
#         final_cases = random.sample(cases_list, n_traces)
    
#     else:
#         final_cases = cases_list[:n_traces]
        
#     train_df = log[log['case:concept:name'].isin(final_cases)]  
#     test_df = log[~log['case:concept:name'].isin(final_cases)]  
#     print(f"the train cases:{train_df['case:concept:name'].unique()}, the test cases:{test_df['case:concept:name'].unique()}")
#     return train_df, test_df


def select_stmx_mats_for_test(sfmx_mats, indices_lst):
    if isinstance(indices_lst, pd.core.frame.DataFrame):
        indices = [int(num) for num in indices_lst['case:concept:name'].unique().tolist()]
        
    elif isinstance(indices_lst, pd.core.series.Series):
        indices = [int(num) for num in indices_lst.unique().tolist()]
    
    else:
        indices = [int(num) for num in indices_lst]
    
    return [sfmx_mats[i] for i in indices]


def logarithmic(p):
    return -np.log(p) / 4.7


def exponential(p):
    return 1 - np.exp(1 - 1/p)


def linear(p):
    return 1 - p


def compare_stochastic_vs_argmax_no_loops(df, softmax_lst, n_train_traces=10, cost_function=None, round_precision=20, 
                                          random_trace_selection=True, random_seed=42, non_sync_penalty=1,
                                          conditioned_prob_compute=False, max_history_len=5, precision=3,
                                          sample_from_each_activity=False, n_activity_frames=3, lamda=0.5,
                                          sample_from_each_trace_family=True, n_samples_from_each_family=1):
    
    if cost_function is None or cost_function == 'logarithmic':
        cost_function = lambda p: -np.log(p) / 4.7
    
    if sample_from_each_activity:
        df_no_loops, stmx_lst_no_loops = sample_n_random_frames_from_each_activity_in_log_and_sftm_matrices_lst(df, softmax_lst, n_activity_frames=n_activity_frames,       random_seed=random_seed)
    else:
        df_no_loops, stmx_lst_no_loops = remove_loops_in_log_and_sftm_matrices_lst(df, softmax_lst)
    

    df_train, df_test = train_test_log_split(df_no_loops,n_traces=n_train_traces,random_selection=random_trace_selection,random_seed=random_seed,
                                             sample_from_each_trace_family=sample_from_each_trace_family, n=n_samples_from_each_family)
    
    # print('Within the compare_stochastic_vs_argmax_no_loops')
    # print(f'The length if the train set: {len(df_train)}, and the test set: {len(df_test)}')
    
    stmx_matrices_test = select_stmx_mats_for_test(stmx_lst_no_loops, df_test)
    df_train.loc[:, 'order'] = df_train.groupby('case:concept:name').cumcount()
    # reference_date = pd.Timestamp('2000-01-01')  # replace with your actual reference date
    # df_train.loc[:, 'time:timestamp'] = reference_date + pd.to_timedelta(df_train['order'], unit='D')
    # display(df_train.head())
    df_train.loc[:, 'time:timestamp'] = pd.to_datetime(df_train['order'])
    # df_train.loc[:, 'time:timestamp'] = pd.to_datetime(df_train['order'], format='%Y-%m-%d') # YYYY-MM-DD

    ## df_train['time:timestamp'] = pd.to_datetime(df_train['time:timestamp'], format='%Y%m%dT%H%M')

    ## df_train = pm4py.format_dataframe(df_train, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
    # display(df_train.head())
    net, init_marking, final_marking = pm4py.discover_petri_net_inductive(df_train)
    model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty, cost_function=cost_function, conditioned_prob_compute=conditioned_prob_compute)
        
#     gviz = pn_visualizer.apply(net, init_marking, final_marking)
#     pn_visualizer.view(gviz)
    
    if conditioned_prob_compute:
        prob_dict = build_conditioned_prob_dict(df_train, max_history_len, precision)
    else:
        prob_dict=None
    
    test_traces_cases = list(df_test['case:concept:name'].unique())
    stochastic_acc_lst = []
    argmax_acc_lst = []
    for idx, trace_case in enumerate(test_traces_cases):
        true_trace_df =  df_test[df_test['case:concept:name'] == trace_case]
        
        #----- print trace -----
#         display(true_trace_df)
#         print()
        #-----------------------
        stochastic_trace_df = sfmx_mat_to_sk_trace(stmx_matrices_test[idx], trace_case, round_precision=round_precision)

        #----- print trace -----
        display(stochastic_trace_df.head())
#         print()
        #-----------------------
        
        stochastic_acc, argmax_acc = compare_argmax_and_stochastic_alignments(stochastic_trace_df, true_trace_df,
                                                                              model, non_sync_penalty=non_sync_penalty,
                                                                              prob_dict=prob_dict, lamda=lamda)
        
        stochastic_acc_lst.append(stochastic_acc)
        argmax_acc_lst.append(argmax_acc)
        
    return stochastic_acc_lst, argmax_acc_lst


# def compare_stochastic_vs_argmax_random_indices(df, softmax_lst, cost_function, n_train_traces=10, n_indices=100, round_precision=2,
#                                                 random_trace_selection=True, random_seed=42, non_sync_penalty=1, 
#                                                 prob_dict=None, lamda=0.5):
   
#     if cost_function is None:
#         cost_function=logarithmic

#     df_random_indices, stmx_lst_random_indices = select_random_indices_in_log_and_sftm_matrices_lst(df, softmax_lst, n_indices) 
#     df_train, df_test = train_test_log_split(df_random_indices, n_traces=n_train_traces, random_selection=random_trace_selection, random_seed=random_seed)
#     stmx_matrices_test = select_stmx_mats_for_test(stmx_lst_random_indices, df_test)

#     net, init_marking, final_marking = inductive_miner.apply(df_train)
#     model = from_discovered_model_to_PetriNet(net, non_sync_move_penalty=non_sync_penalty, cost_function=cost_function)

#     test_traces_cases = list(df_test['case:concept:name'].unique())
#     stochastic_acc_lst = []
#     argmax_acc_lst = []
#     for idx, trace_case in enumerate(test_traces_cases):
#         true_trace_df =  df_test[df_test['case:concept:name'] == trace_case].reset_index(drop=True)
#         stochastic_trace_df = sfmx_mat_to_sk_trace(stmx_matrices_test[idx], trace_case, round_precision=round_precision)
#         stochastic_acc, argmax_acc = compare_argmax_and_stochastic_alignments(stochastic_trace_df, true_trace_df,
#                                                                               model, non_sync_penalty=non_sync_penalty,
#                                                                               prob_dict=prob_dict, lamda=lamda)
#         stochastic_acc_lst.append(stochastic_acc)
#         argmax_acc_lst.append(argmax_acc)
      
#     return stochastic_acc_lst, argmax_acc_lst
        
        
# def select_random_indices_in_log_and_sftm_matrices_lst(log_df, sftm_mat_lst, n_indices = 100):
#     trace_cases = list(log_df['case:concept:name'].unique())
#     filtered_trace_lst = []
#     filtered_sftm_mat_lst = []
    
#     for i, case in enumerate(trace_cases):
#         trace = log_df[log_df['case:concept:name'] == case]
#         np_sftm_mat = sftm_mat_lst[i].squeeze(0).cpu().numpy()
#         selected_indices = sample(list(range(np_sftm_mat.shape[1])), n_indices)
#         selected_indices_bool = np.zeros(np_sftm_mat.shape[1], dtype=bool)
#         np.add.at(selected_indices_bool, selected_indices, 1) 
#         sftm_filtered = np_sftm_mat[:,selected_indices_bool]
#         trace_df_filtered = trace[selected_indices_bool]
#         filtered_trace_lst.append(trace_df_filtered)
#         filtered_sftm_mat_lst.append(sftm_filtered)
#     return pd.concat(filtered_trace_lst), filtered_sftm_mat_lst


# def sample_n_random_frames_from_each_activity_in_log_and_sftm_matrices_lst(log_df, sftm_mat_lst, n_activity_frames=3, random_seed=42):
#     trace_cases = list(log_df['case:concept:name'].unique())
#     filtered_trace_lst = []
#     filtered_sftm_mat_lst = []
    
#     for i, case in enumerate(trace_cases):
#         trace = log_df[log_df['case:concept:name'] == case].reset_index()      
#         indices_to_keep = sample_n_random_frames_from_each_activity(trace, n_activity_frames, random_seed=random_seed)
#         trace = trace[trace.index.isin(indices_to_keep)]
#         np_sftm_mat = sftm_mat_lst[i].squeeze(0).cpu().numpy()
#         np_sftm_mat = np_sftm_mat[:,indices_to_keep]
#         filtered_trace_lst.append(trace)
#         filtered_sftm_mat_lst.append(np_sftm_mat)
    
#     return pd.concat(filtered_trace_lst), filtered_sftm_mat_lst


def sample_n_random_frames_from_each_activity(trace_df, n_activity_frames=3, remove_less_than_sample_sequences=True, random_seed=101):
    activity_start_indices_lst = trace_df[(trace_df['concept:name'] != trace_df.shift(1)['concept:name']) | (trace_df['case:concept:name'] != trace_df.shift(1)['case:concept:name'])].index.tolist()
    activity_start_indices_lst.append(len(trace_df))
        
    chosen_frames_lst = []
    for i in range(len(activity_start_indices_lst)-1):
        if remove_less_than_sample_sequences:
            if activity_start_indices_lst[i+1] - activity_start_indices_lst[i] < n_activity_frames:
                continue    
        chosen_frames_lst += sorted(random.Random(random_seed).sample(range(activity_start_indices_lst[i], activity_start_indices_lst[i+1]), n_activity_frames))
        
    return chosen_frames_lst


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)
    
    
def get_trace_activity_freuencies(trace, frequency_dict=None):
    
    if frequency_dict is None:
        frequency_dict = defaultdict(int)
        return_dict = True
    else:
        return_dict = False
        
    for i in range(len(trace)-1):
        frequency_dict[(trace.iloc[i,0], trace.iloc[i+1,0])] += 1
    
    if return_dict:
        return frequency_dict
    
    
def get_train_set_frequencies(df_train, frequency_dict=None):
    
    if frequency_dict is None:
        frequency_dict = defaultdict(int)
        return_dict = True
    else:
        return_dict = False
        
    trace_cases_lst = df_train['case:concept:name'].unique().tolist()
    for case in trace_cases_lst:
        trace = df_train[df_train['case:concept:name'] == case]
        get_trace_activity_freuencies(trace, frequency_dict)
    
    if return_dict:
        return frequency_dict
    
    
def get_dict_relative_frequencies(frequency_dict):
    
    relative_freq_dict = {}
    for key in frequency_dict.keys():
        relative_freq_dict[key] = round(frequency_dict[key] / sum([frequency_dict[sub_key] for sub_key in frequency_dict.keys() if sub_key[0] == key[0]]), 2)
    
    return relative_freq_dict


def remove_low_freq_activities_in_trace(trace, rel_freq_dict, thresh=0.2, start_activity='17', end_activity='18'):
    
    filtered_trace = trace.copy()
    filtered_trace['remove'] = [0] * len(filtered_trace)
    iter_flag = 1

    while len(filtered_trace) > 1 and iter_flag and filtered_trace.iloc[0,0] == start_activity and filtered_trace.iloc[-1,0] == end_activity:
        iter_flag = 0
        for i in range(len(filtered_trace)-1):
            if rel_freq_dict.get((filtered_trace.iloc[i,0], filtered_trace.iloc[i+1,0]), 0) < thresh:
                filtered_trace['remove'].iloc[i+1] = 1
                iter_flag = 1

        filtered_trace = filtered_trace[filtered_trace['remove'] == 0]          
    
    if filtered_trace.iloc[0,0] == start_activity and filtered_trace.iloc[-1,0] == end_activity:
        return filtered_trace
    
    return None


def filter_low_freq_activities_in_df(df_train, rel_freq_dict, thresh=0.2, start_activity='17', end_activity='18'):
    
    filtered_traces_lst = []
    trace_cases_lst =  df_train['case:concept:name'].unique().tolist()
    
    for case in trace_cases_lst:
        trace = df_train[df_train['case:concept:name'] == case]
        filtered_trace = remove_low_freq_activities_in_trace(trace, rel_freq_dict=rel_freq_dict, thresh=thresh, start_activity=start_activity, end_activity=end_activity)  
        if filtered_trace is not None:
            filtered_traces_lst.append(filtered_trace)
    
    if filtered_traces_lst == []:
        print('No traces are left after filtration')
        return None
    
    return pd.concat(filtered_traces_lst)


def generate_self_loops_in_df(df_train_no_loops, mul_factor=2):
    if df_train_no_loops is None:
        return df_train_no_loops
    
    traces_with_self_loops_lst = []
    trace_cases_lst =  df_train_no_loops['case:concept:name'].unique().tolist()
    
    for case in trace_cases_lst:
        trace = df_train_no_loops[df_train_no_loops['case:concept:name'] == case]
        trace_with_self_loops = trace.loc[trace.index.repeat(mul_factor)] 
        traces_with_self_loops_lst.append(trace_with_self_loops)
     
    return pd.concat(traces_with_self_loops_lst)


def additional_train_set_preprocess(df_train, thresh, mul_factor=2):
    df_train = remove_self_loops_in_dataset(df_train)
    d_frequency = get_train_set_frequencies(df_train)
    d_relative = get_dict_relative_frequencies(d_frequency)
    df_filtered = filter_low_freq_activities_in_df(df_train, d_relative, thresh)
    df_filtered_self_loops = generate_self_loops_in_df(df_filtered, mul_factor)
    
    return df_filtered_self_loops


# def get_histories_up_to_length_k(full_history, k=None, hist_min_length=1):
    
#     if k is None:
#         k = len(string)+1

#     return [full_history[i:j] for i in range(len(full_history)) for j in range(i+hist_min_length,min(i+k+1, len(full_history)+1))]


# def get_relative_freq_dict(history, precision=2):
    
#     frequencies_dict = dict(Counter(history))
#     rel_freq_dict = {}
    
#     for key in frequencies_dict:
#         if len(key) > 1:
#             total_prefix_freq = sum([frequencies_dict[sub_key] for sub_key in frequencies_dict if len(sub_key) == len(key) if key[:-1] == sub_key[:-1]])
#             rel_freq_dict[key] = round(frequencies_dict[key] / total_prefix_freq, precision)
    
#     return rel_freq_dict 


# def build_conditioned_prob_dict(df_train, max_hist_len=5, precision=2):
    
#     counter = Counter()
#     trace_cases_lst = df_train['case:concept:name'].unique().tolist()

#     for case in trace_cases_lst:
#         trace = df_train[df_train['case:concept:name'] == case]   
#         activities_seq_str = ''.join(trace['concept:name'].tolist())
#         counter += Counter(get_histories_up_to_length_k(activities_seq_str, k=max_hist_len))
    
#     rel_freq_dict = get_relative_freq_dict(counter, precision=precision)
# #     print(rel_freq_dict)
#     return rel_freq_dict


def plot_argmax_vs_stochastic(argmax_for_each_timestamp, stochastic_for_each_timestamp):
    idx = [round(val,1) for val in np.linspace(0,1,11)]
    plt.figure(figsize=(8,6))
    plt.scatter(idx, argmax_for_each_timestamp, label='argmax', color='blue', s=90)
    plt.scatter(idx, stochastic_for_each_timestamp, label='stochastic', color='red', s=50)
    plt.title('Stoachstic vs Argmax Recovery Accuracy')
    plt.xlabel('Lamda')
    plt.ylabel('Recovery Accuracy')
    plt.legend();  
    

def group_similar_cases(log_df):
    '''Returns dictionary where keys are the activity sequence (no loops) and values are a list of similar cases'''
    
    df_no_loops = remove_self_loops_in_dataset(log_df)
    trace_family_dict = defaultdict(list)
    
    for case in df_no_loops['case:concept:name'].unique():
        trace_df = df_no_loops[df_no_loops['case:concept:name'] == case]
        trace_tuple = tuple(trace_df['concept:name'].values.astype(np.int_))
        trace_family_dict[trace_tuple].append(case)

    return trace_family_dict   


def sample_from_dict(trace_family_dict,n=1, seed=42):
    """draws n elements from each list in dict returning the result as a single list"""
    random.seed(seed)
    sample = []
    for key in trace_family_dict:
        sample.extend(random.sample(trace_family_dict[key],n))
    return sample  


def evaluate_lamdas_values(df, softmax_lst, max_history_len=3, n_activity_frames=2, cost_function=None, random_seed=2, lamda_lst=[]):
    stochastic_acc_lst = []
    argmax_acc_lst = []
    
    if not cost_function:
        # logarithmic default
        cost_function = lambda p: -np.log(p) / 4.7
    
    if not lamda_lst:
        lamda_lst = [round(val,1) for val in np.linspace(0,1,11)]
        
    for lamda in lamda_lst:
        print(f'Computing for lamda={lamda}')
        stochastic_acc_conditioned, argmax_acc_conditioned = compare_stochastic_vs_argmax_no_loops(df, softmax_lst, random_seed=random_seed, cost_function=cost_function,
                                                                                                   conditioned_prob_compute=True, max_history_len=max_history_len,
                                                                                                   sample_from_each_activity=True, n_activity_frames=n_activity_frames,
                                                                                                   lamda=lamda, sample_from_each_trace_family=True, n_samples_from_each_family=1)        
        stochastic_acc_lst.append(stochastic_acc_conditioned)
        argmax_acc_lst.append(argmax_acc_conditioned)
        
    return stochastic_acc_lst, argmax_acc_lst


def generate_stats_softmax_matrices(df, softmax_lst, random_seed_lst=None, n_train_traces=None, cost_function=None, conditioned_prob_compute=True,
                                    max_history_len=3, sample_from_each_activity=True, n_activity_frames=1, lamda=0.5, sample_from_each_trace_family=True,                                                     
                                    n_samples_from_each_family=1, n_iterations=10, trace_resolution=False):                                                       
                                                                                           
    print('entering the generate_stats function')
    if random_seed_lst is None:
        random_seed_lst = [i for i in range(n_iterations)]
    
    stochastic_acc_lst = []
    argmax_acc_lst = []
    trace_resolution_stochastic_lst = []
    trace_resolution_argmax_lst = []
    
    for i in range(n_iterations):
        display(df.head(5))
        print(softmax_lst[0])
        stochastic_acc, argmax_acc = compare_stochastic_vs_argmax_no_loops(df, softmax_lst, random_seed=random_seed_lst[i], n_train_traces=n_train_traces,
                                                                                                   cost_function=cost_function, conditioned_prob_compute=conditioned_prob_compute,
                                                                                                   max_history_len=max_history_len, sample_from_each_activity=sample_from_each_activity,
                                                                                                   n_activity_frames=n_activity_frames, lamda=lamda,
                                                                                                   sample_from_each_trace_family=sample_from_each_trace_family,
                                                                                                   n_samples_from_each_family=n_samples_from_each_family)
        
        if trace_resolution:
            print(f'stochastic acc: {stochastic_acc}, argmax acc: {argmax_acc}')
            trace_resolution_stochastic_lst += stochastic_acc
            trace_resolution_argmax_lst += argmax_acc
            
        stochastic_acc_lst.append(mean(stochastic_acc))
        argmax_acc_lst.append(mean(argmax_acc))
   
    
    if trace_resolution:
        print(f'trace resolution is true and the results are: {trace_resolution_stochastic_lst, trace_resolution_argmax_lst}')
        return trace_resolution_stochastic_lst, trace_resolution_argmax_lst
    
    print(stochastic_acc_lst)
    print(argmax_acc_lst)
    return stochastic_acc_lst, argmax_acc_lst


def generate_stats_stochastic_vs_argmax(df, softmax_lst, random_seed_lst=None, n_train_traces=None,
                                        cost_function=None, conditioned_prob_compute=True,
                                        max_history_len=3, sample_from_each_activity=True, n_activity_frames=2,
                                        lamda=None, sample_from_each_trace_family=True, n_samples_from_each_family=1,
                                        dataset=None, n_repetitions=10):
    
    stochastic_stats_lst = []
    argmax_stats_lst = []
    
    if lamda is None:
        if dataset in ['50salads', 'gtea']:
            lamda = 0.2
        else:
            lamda = 0.5
        
    if random_seed_lst is None:
        random_seed_lst = [i for i in range(n_repetitions)]
        
    for idx, random_seed in enumerate(random_seed_lst):
        print(f'idx = {idx+1}')
        stocastic_acc_lst, argmax_acc_lst = compare_stochastic_vs_argmax_no_loops(df=df, softmax_lst=softmax_lst, random_seed=random_seed, n_train_traces=n_train_traces,
                                                                                  cost_function=cost_function, conditioned_prob_compute=conditioned_prob_compute,
                                                                                  max_history_len=max_history_len, sample_from_each_activity=sample_from_each_activity,
                                                                                  n_activity_frames=n_activity_frames, lamda=lamda,
                                                                                  sample_from_each_trace_family=sample_from_each_trace_family,
                                                                                  n_samples_from_each_family=n_samples_from_each_family)
        stochastic_stats_lst.append(stocastic_acc_lst)
        argmax_stats_lst.append(argmax_acc_lst)
        
        
    return stochastic_stats_lst, argmax_stats_lst  


def remove_loops_in_log_only(log_df):
    trace_cases = list(log_df['case:concept:name'].unique())
    no_loops_trace_lst = []
    
    for i, case in enumerate(trace_cases):
        trace = log_df[log_df['case:concept:name'] == case]
        loop_indices = trace_self_loops_indices(trace)
        no_loops_trace = remove_self_loops_in_trace(trace)
        no_loops_trace_lst.append(no_loops_trace)
    
    return pd.concat(no_loops_trace_lst)


def compute_softmax_lst_accuracy(softmax_lst, target_lst):
    acc_lst = []

    for i, mat in enumerate(softmax_lst):
        acc = (torch.argmax(mat, dim=1) == target_lst[i]).numpy().mean()
        acc_lst.append(acc)

    return  mean(acc_lst)



# def map_to_string_numbers(df, map_dict=None, return_map_dict=True):
#     if map_dict:
#         df['concept:name'] = df['concept:name'].map(map_dict)
#     elif not df['concept:name'].str.isnumeric().all():
#         print('The activities are not numeric strings thus defining arbitrary map_dict and mapping the column')
#         map_dict = {value: str(i) for i, value in enumerate(df['concept:name'].unique())}
#         df['concept:name'] = df['concept:name'].map(map_dict)

#     return (df, map_dict) if return_map_dict else df


    
# def prepare_df(dataset_name, return_softmax_matrices_lst=True, return_map_dict=False):
    
#     if dataset_name == '50salads':
#         with open('50salads_softmax_lst.pickle', 'rb') as handle:
#             softmax_lst = CPU_Unpickler(handle).load()

#         with open('50salads_target_lst.pickle', 'rb') as handle:
#             target_lst = CPU_Unpickler(handle).load()
    
#     elif dataset_name == 'gtea':
#         with open('gtea_softmax_lst.pickle', 'rb') as handle:
#             softmax_lst = CPU_Unpickler(handle).load()

#         with open('gtea_target_lst.pickle', 'rb') as handle:
#             target_lst = CPU_Unpickler(handle).load()

#     elif dataset_name == 'breakfast':
#         with open('breakfast_softmax_lst.pickle', 'rb') as handle:
#             softmax_lst = CPU_Unpickler(handle).load()

#         with open('breakfast_target_lst.pickle', 'rb') as handle:
#             target_lst = CPU_Unpickler(handle).load()  
    
#     else:
#         raise Exception("Dataset name must be 50salads, gtea, or breakfast.")
 

#     concant_tensor_lst = []
#     concat_idx_lst = []
    
#     for i, tensor in enumerate(target_lst):
#         tensor_lst = tensor.tolist()
#         tensor_lst = [str(elem) for elem in tensor_lst]
#         idx_lst = [str(i)] * len(tensor_lst)
#         concant_tensor_lst += tensor_lst
#         concat_idx_lst += idx_lst

#     df = pd.DataFrame(
#         {'concept:name': concant_tensor_lst,
#          'case:concept:name': concat_idx_lst
#         })
    
    
#     if return_map_dict and return_softmax_matrices_lst:
#         return df, softmax_lst, map_dict
#     elif return_map_dict:
#         return df, map_dict
#     elif return_softmax_matrices_lst:
#         return df, softmax_lst
#     else:
#         return df
    
#     return df


def run_stochastic_vs_argmax_experiment(dataset_names, **kwargs):
    
    if not isinstance(dataset_names, list):
        dataset_names_lst = [dataset_names]
    else:
        dataset_names_lst = dataset_names
    
    datasets = [prepare_df(dataset_names_lst[i]) for i in range(len(dataset_names_lst))] 
    hyperparameters_dict = kwargs.get('hyperparameters_dict', None) # [(4,0.99), (4,0.1), (2,0.99)]
    n_traces_from_each_dataset = kwargs.get('n_traces_from_each_dataset', None)
    
    res_lst = []
    for i, data in enumerate(datasets):
        hist_len, lamda = hyperparameters_dict[dataset_names_lst[i]]
        df, softmax_lst = datasets[i]
        df_filtered = select_rows(df, n_traces_from_each_dataset)
        softmax_lst_filtered = softmax_lst[:n_traces_from_each_dataset]

        stochastic_acc_lst, argmax_acc_lst = generate_stats_softmax_matrices(df_filtered, softmax_lst_filtered, n_train_traces=3, cost_function='logarithmic', conditioned_prob_compute=True,
                                        max_history_len=hist_len, sample_from_each_activity=True, n_activity_frames=1,
                                       lamda=lamda, sample_from_each_trace_family=True, n_samples_from_each_family=1, n_iterations=1, trace_resolution=True)

        res_lst.append((stochastic_acc_lst, argmax_acc_lst))
    
    return res_lst


def select_rows(df, n_traces):
    
    if n_traces is None:
        return df
    
    # Get the first n_traces unique values in the 'case:concept:name' column
    unique_values = df['case:concept:name'].unique()[:n_traces]

    # Select the rows that correspond to these unique values
    selected_rows = df[df['case:concept:name'].isin(unique_values)]

    return selected_rows



def generate_stats_bpi(df, non_sync_penalty=1, n_traces_for_model_building=10, add_heuristic=False, generate_probs_for_argmax=True, by_trace=True, true_trans_prob=[0.8],
                       uncertainty_levels=[1], expansion_min=2, expansion_max=2, custom_traces_addition=False, cost_functions_lst=None, max_history_len=0, lamda=0.5,
                       n_traces_overall=100, n_iterations=10, top_k=False, k=None, utilize_trace_family=True, max_len=10, min_trace_frequency=10, 
                       top_activities_for_model_discovery=True, frequent_traces_first=True, max_expansion=True):
    
    stats_lst = []
    stochastic_acc_lst = []
    argmax_acc_lst = []
    stochastic_acc_lst_trace_resolution = []
    argmax_acc_lst_trace_resolution = []
    
    df_accepted_traces = filter_log(df, n_traces=None, max_len=max_len, min_frequency=min_trace_frequency)
    mapping_dict = {string:i for i, string in enumerate(df["concept:name"].unique())}
    df_accepted_traces['concept:name'] = df_accepted_traces['concept:name'].apply(lambda s: str(mapping_dict[s]))  
    
    if cost_functions_lst is None:
        cost_functions_lst = [lambda p: -np.log(p) / 5]
    
    if not isinstance(cost_functions_lst,  list):
        cost_functions_lst = [cost_functions_lst]
    
    if top_k:
        all_activities_unique = find_top_k_activities_in_df(df_accepted_traces, k=k)
    else:
        all_activities_unique = set(df_accepted_traces['concept:name'].unique().tolist())
        
    for i in range(n_iterations):
        with np.errstate(divide='ignore'):
#             stats_dict = generate_trace_freq_dict(df_accepted_traces)
#             freq_lst = [item['frequency'] for item in stats_dict.values()]
#             print(len(freq_lst))
#             print(freq_lst)
            df_sample = filter_log(df_accepted_traces, n_traces=n_traces_overall, max_len=None, random_selection=True, random_seed=i)

            stochastic_acc, argmax_acc = calculate_statistics_for_different_uncertainty_levels(df_sample, non_sync_penalty=non_sync_penalty,
                                                                     n_traces_for_model_building=n_traces_for_model_building, true_trans_prob=true_trans_prob,
                                                                     add_heuristic=add_heuristic, generate_probs_for_argmax=generate_probs_for_argmax, by_trace=by_trace,
                                                                     uncertainty_levels=uncertainty_levels, expansion_min=expansion_min, expansion_max=expansion_max,
                                                                     cost_functions_lst=cost_functions_lst, max_history_len=max_history_len, lamda=lamda, 
                                                                     all_activities_unique=all_activities_unique, utilize_trace_family=utilize_trace_family,
                                                                     top_activities_for_model_discovery=top_activities_for_model_discovery,
                                                                     frequent_traces_first=frequent_traces_first, max_expansion=max_expansion)


#             stats_dict = calculate_statistics_for_different_uncertainty_levels(df_sample, non_sync_penalty=non_sync_penalty,
#                                                              n_traces_for_model_building=n_traces_for_model_building, true_trans_prob=true_trans_prob,
#                                                              add_heuristic=add_heuristic, generate_probs_for_argmax=generate_probs_for_argmax, by_trace=by_trace,
#                                                              uncertainty_levels=uncertainty_levels, expansion_min=expansion_min, expansion_max=expansion_max,
#                                                              cost_functions_lst=cost_functions_lst, max_history_len=max_history_len, lamda=lamda, 
#                                                              all_activities_unique=all_activities_unique, utilize_trace_family=utilize_trace_family,
#                                                              top_activities_for_model_discovery=top_activities_for_model_discovery,
#                                                              frequent_traces_first=frequent_traces_first)
            

#             stats_lst.append(stats_dict)
        
        stochastic_acc_lst.append([mean(acc) for acc in stochastic_acc])  # [[cost_func_1, const_func_2], [cost_func_1, const_func_2]]
        argmax_acc_lst.append(mean(argmax_acc))
   
    return stochastic_acc_lst, argmax_acc_lst   


def find_top_k_activities_in_df(df, k=3):
    activities_lst = df['concept:name'].tolist()
    activity_frequency_counter = Counter(activities_lst)
    to_k_activities_set = set(activity for activity, frequency in activity_frequency_counter.most_common(k))
    return to_k_activities_set



def model_fitness_and_precision_effect(df, n_repetitions_for_model=1, n_traces_for_discovery_range=None, utilize_trace_family=False, true_trans_prob=[0.8],
                                       n_traces_overall=1000, cost_functions_lst=None, max_history_len=0, lamda=1, traces_lenths_list=None, expansion_min=2, expansion_max=2,
                                       top_activities_for_model_discovery=True, frequent_traces_first=True, min_trace_frequency=10, max_trace_len=14, final_all_traces_eval=True,
                                       max_expansion=False):
    res_dict = defaultdict(list) 
    
    if n_traces_for_discovery_range:
        traces_lenths_list = [i for i in range(1,n_traces_for_discovery_range+1)]
    
    if final_all_traces_eval:
        traces_lenths_list.append(None)
        
    for n_traces in traces_lenths_list:
        print(f'Evaluating {n_traces} traces for model building')
        stochastic_bpi, _ = generate_stats_bpi(df=df, n_traces_for_model_building=n_traces, true_trans_prob=true_trans_prob, utilize_trace_family=utilize_trace_family,
                                               cost_functions_lst=cost_functions_lst, n_traces_overall=n_traces_overall, n_iterations=n_repetitions_for_model,
                                               top_activities_for_model_discovery=top_activities_for_model_discovery, frequent_traces_first=frequent_traces_first,
                                               min_trace_frequency=min_trace_frequency, max_len=max_trace_len, max_expansion=max_expansion)

        res_dict[n_traces].append([float(sum(col))/len(col) for col in zip(*stochastic_bpi)]) # [[cost_func_1, const_func_2], [cost_func_1, const_func_2]]
#         stats_bpi = generate_stats_bpi(df=df, n_traces_for_model_building=n_traces, true_trans_prob=true_trans_prob, utilize_trace_family=utilize_trace_family,
#                                                cost_functions_lst=cost_functions_lst, n_traces_overall=n_traces_overall, n_iterations=n_repetitions_for_model,
#                                                top_activities_for_model_discovery=top_activities_for_model_discovery, frequent_traces_first=frequent_traces_first)
#         res_dict[n_traces].append(stats_bpi)
        
    return res_dict


def sample_traces(df: pd.DataFrame, n_traces: int = None, unique: bool = True) -> pd.DataFrame:
    '''Samples traces randomly, optionally selecting unique traces, and returns a DataFrame with the selected cases'''

    if unique:
        trace_dict = defaultdict(list)
        for case in df['case:concept:name'].unique():
            trace = tuple(df[df['case:concept:name'] == case]['concept:name'].tolist())
            trace_dict[trace].append(case)
        
        unique_traces = list(trace_dict.items())  # [(trace, cases:list)]
        
        if n_traces is None:
            n_traces = len(unique_traces)
        
        if n_traces > len(unique_traces):
            warnings.warn("There are fewer unique traces than the number of traces requested. Selecting one trace from each unique trace family.")
            n_traces = len(unique_traces)
                
        sampled_cases = [random.choice(cases) for trace, cases in unique_traces[:n_traces]]
    else:
        all_cases = df['case:concept:name'].unique().tolist()
        sampled_cases = random.sample(all_cases, n_traces if n_traces is not None else len(all_cases))
    
    return df[df['case:concept:name'].isin(sampled_cases)]


def compute_model_statistics(log, net, im, fm):
    fitness = pm4py.fitness_alignments(log, net, im, fm)
    prec = pm4py.precision_alignments(log, net, im, fm)
    gen = generalization_evaluator.apply(log, net, im, fm)
    simp = simplicity_evaluator.apply(net)
    
    return {'fitness':fitness, 'precision':prec, 'generalization':gen, 'simplicity':simp}
    
      
def exp_5(p):
    return 1 - np.exp(5*(1 - 1/p))    
    
def exp_10(p):
    return 1 - np.exp(10*(1 - 1/p))   

def linear(p):
    return 1 - p

def logarithmic_5(p):
    return -np.log(p) / 5

def logarithmic_10(p):
    return -np.log(p) / 10

def logarithmic_20(p):
    return -np.log(p) / 20



def parse_file(file_path):
    places_dict = {}
    transitions_dict = {}
    arcs = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';')  # Remove trailing semicolons
            if line.startswith('place'):
                parts = line.split()
                name = parts[1].strip('"')
                properties = {}
                if 'init' in line:
                    properties['init'] = int(parts[-1].rstrip(';'))
                place = Place(name, properties=properties)
                places_dict[name] = place
            elif line.startswith('trans'):
                parts = line.split('~')
                trans_parts = parts[0].split()
                name = trans_parts[1].strip('"')
                label = parts[1].split()[0].strip('"')
                
                in_out_part = ' '.join(parts[1].split()[1:]).replace(';', '')
                in_places_names, out_places_names = in_out_part.split(' out ')
                in_places_names = in_places_names.replace('in', '').strip().split('" "')
                out_places_names = out_places_names.strip().split('" "')
                
                transition = Transition(name, label)
                transitions_dict[name] = transition

                # Process arcs
                for place_name in in_places_names:
                    place_name = place_name.replace('"', '').strip()
                    if place_name in places_dict:
                        arc = Arc(places_dict[place_name], transition)
                        arcs.append(arc)
                        places_dict[place_name].out_arcs.add(arc)
                        transition.in_arcs.add(arc)

                for place_name in out_places_names:
                    place_name = place_name.replace('"', '').strip()
                    if place_name in places_dict:
                        arc = Arc(transition, places_dict[place_name])
                        arcs.append(arc)
                        transition.out_arcs.add(arc)
                        places_dict[place_name].in_arcs.add(arc)

    # Convert dictionaries to lists as requested
    places_list = list(places_dict.values())
    transitions_list = list(transitions_dict.values())

    return places_list, transitions_list, arcs

# original correct - 13/08/2024
# def from_lists_to_PetriNet(places, transitions, arcs, non_sync_move_penalty=1, name='discovered_net', cost_function=None, conditioned_prob_compute=False, quiet_moves_weight=0.0001, return_markings=True):
#     # Deep copy the inputs to avoid accidental modifications
#     places = copy.deepcopy(places)
#     transitions = copy.deepcopy(transitions)
#     arcs = copy.deepcopy(arcs)
    
#     # Identify the initial and final places
#     initial_place = next((place for place in places if 'init' in place.properties), None)
#     final_place = next((place for place in places if place.name == 'end'), None)

#     # Adjusted sorting logic to avoid type error
#     def place_sort_key(place):
#         if place is initial_place:
#             return (0,)  # Initial place gets the first position
#         elif place is final_place:
#             return (2,)  # Final place gets the last position
#         else:
#             return (1, place.name)  # Other places are sorted alphabetically

#     sorted_places = sorted(places, key=place_sort_key)

#     transitions = [Transition(transition.name, transition.label, transition.in_arcs, transition.out_arcs, 'model', weight=non_sync_move_penalty) \
#                                                                                 for transition in transitions]

#     assert len([tran.name for tran in transitions]) == len(set([tran.name for tran in transitions]))
#     assert len([place.name for place in sorted_places]) == len(set([place.name for place in sorted_places]))
    
#     for trans in transitions:
#         if trans.label is None:
#             trans.weight = quiet_moves_weight
#         else:
#             trans.label = trans.label.split('+')[0]
#             trans.weight = non_sync_move_penalty
#             trans.name = trans.label  # Use label as the name if defined
 
#     tran2idx = {tran.name: i for i, tran in enumerate(transitions)}
#     place2idx = {place.name: i for i, place in enumerate(sorted_places)}
#     petri_new_arcs = []
    
#     for i in range(len(transitions)):
#         new_in_arcs = set()
#         for arc in transitions[i].in_arcs:
#             new_in_arc = Arc(sorted_places[place2idx[arc.source.name]], transitions[i])
#             new_in_arcs.add(new_in_arc)
#             petri_new_arcs.append(new_in_arc)
            
#         new_out_arcs = set()   
#         for arc in transitions[i].out_arcs:
#             new_out_arc = Arc(transitions[i], sorted_places[place2idx[arc.target.name]])
#             new_out_arcs.add(new_out_arc)
#             petri_new_arcs.append(new_out_arc)
        
#         transitions[i].in_arcs = new_in_arcs
#         transitions[i].out_arcs = new_out_arcs
       
    
#     # Initialize markings based on the sorted places
#     init_mark = tuple([1] + [0] * (len(sorted_places) - 1))
#     final_mark = tuple([0] * (len(sorted_places) - 1) + [1])

#     # Create and setup the PetriNet object
#     new_PetriNet = PetriNet(name)
#     new_PetriNet.add_places(sorted_places)
#     new_PetriNet.add_transitions(transitions)
#     new_PetriNet.init_mark = Marking(init_mark)
#     new_PetriNet.final_mark = Marking(final_mark)
#     new_PetriNet.arcs = petri_new_arcs
#     new_PetriNet.cost_function = cost_function
#     new_PetriNet.conditioned_prob_compute = conditioned_prob_compute

#     if return_markings:
#         return new_PetriNet, new_PetriNet.init_mark, new_PetriNet.final_mark
    
#     return new_PetriNet


def from_lists_to_PetriNet(
    places, 
    transitions, 
    arcs, 
    non_sync_move_penalty=1, 
    name='discovered_net', 
    cost_function=None, 
    conditioned_prob_compute=False, 
    quiet_moves_weight=0.0001, 
    return_markings=True,
    sync_moves_weight=1e-6
):
    # Deep copy the inputs to avoid accidental modifications
    places = copy.deepcopy(places)
    transitions = copy.deepcopy(transitions)
    arcs = copy.deepcopy(arcs)
    
    # Identify the initial and final places
    initial_place = next((place for place in places if 'init' in place.properties), None)
    final_place = next((place for place in places if place.name == 'end'), None)

    # Adjusted sorting logic to avoid type error
    def place_sort_key(place):
        if place is initial_place:
            return (0,)  # Initial place gets the first position
        elif place is final_place:
            return (2,)  # Final place gets the last position
        else:
            return (1, place.name)  # Other places are sorted alphabetically

    sorted_places = sorted(places, key=place_sort_key)

    transitions = [Transition(transition.name, transition.label, transition.in_arcs,
                  transition.out_arcs, 'model', weight=non_sync_move_penalty) \
                  for transition in transitions]

    assert len([tran.name for tran in transitions]) == len(set([tran.name for tran in transitions]))
    assert len([place.name for place in sorted_places]) == len(set([place.name for place in sorted_places]))
    
    for trans in transitions:
        if trans.label is None:
            trans.weight = quiet_moves_weight
        else:
            trans.label = trans.label.split('+')[0]
            trans.weight = non_sync_move_penalty
            trans.name = trans.label  # Use label as the name if defined
 
    tran2idx = {tran.name: i for i, tran in enumerate(transitions)}
    place2idx = {place.name: i for i, place in enumerate(sorted_places)}
    petri_new_arcs = []
    
    for i in range(len(transitions)):
        new_in_arcs = set()
        for arc in transitions[i].in_arcs:
            new_in_arc = Arc(sorted_places[place2idx[arc.source.name]], transitions[i])
            new_in_arcs.add(new_in_arc)
            petri_new_arcs.append(new_in_arc)
            
        new_out_arcs = set()   
        for arc in transitions[i].out_arcs:
            new_out_arc = Arc(transitions[i], sorted_places[place2idx[arc.target.name]])
            new_out_arcs.add(new_out_arc)
            petri_new_arcs.append(new_out_arc)
        
        transitions[i].in_arcs = new_in_arcs
        transitions[i].out_arcs = new_out_arcs
    
    # Initialize markings based on the sorted places
    init_mark = tuple([1] + [0] * (len(sorted_places) - 1))
    final_mark = tuple([0] * (len(sorted_places) - 1) + [1])

    # Create and setup the PetriNet object
    new_PetriNet = PetriNet(name)
    new_PetriNet.add_places(sorted_places)
    new_PetriNet.add_transitions(transitions)
    new_PetriNet.init_mark = Marking(init_mark)
    new_PetriNet.final_mark = Marking(final_mark)
    new_PetriNet.arcs = petri_new_arcs
    new_PetriNet.cost_function = cost_function
    new_PetriNet.conditioned_prob_compute = conditioned_prob_compute
    new_PetriNet.epsilon = sync_moves_weight
    
    # pm4py-related fields
    pm4py_net = pm4py_PetriNet(name)
    
    pm4py_places = [pm4py.objects.petri_net.obj.PetriNet.Place(p.name) for p in sorted_places]
    for place in pm4py_places:
        pm4py_net.places.add(place)
        
    pm4py_transitions = [pm4py.objects.petri_net.obj.PetriNet.Transition(t.name, t.label) for t in transitions]
    for t in pm4py_transitions:
        pm4py_net.transitions.add(t)
    
    # Create arcs in pm4py PetriNet
    for arc in petri_new_arcs:
        if isinstance(arc.source, Place):
            pm4py_petri_utils.add_arc_from_to(pm4py_places[place2idx[arc.source.name]], pm4py_transitions[tran2idx[arc.target.name]], pm4py_net)
        else:
            pm4py_petri_utils.add_arc_from_to(pm4py_transitions[tran2idx[arc.source.name]], pm4py_places[place2idx[arc.target.name]], pm4py_net)

    pm4py_init_marking = pm4py.objects.petri_net.obj.Marking()
    pm4py_init_marking[pm4py_places[0]] = 1
    
    pm4py_final_marking = pm4py.objects.petri_net.obj.Marking()
    pm4py_final_marking[pm4py_places[-1]] = 1
    
    new_PetriNet.pm4py_net = pm4py_net
    new_PetriNet.pm4py_initial_marking = pm4py_init_marking
    new_PetriNet.pm4py_final_marking = pm4py_final_marking

    # Create place mapping to map between your PetriNet places and their indices
    place_mapping = {place: idx for idx, place in enumerate(pm4py_places)}
    reverse_place_mapping = {idx: place for idx, place in enumerate(pm4py_places)}
    new_PetriNet.place_mapping = place_mapping
    new_PetriNet.reverse_place_mapping = reverse_place_mapping
    
    if return_markings:
        return new_PetriNet, new_PetriNet.init_mark, new_PetriNet.final_mark
    
    return new_PetriNet

# original
# def generate_model_from_file(file_path, activity_mapping_dict=None, return_markings=True):
#     places, transitions, arcs = parse_file(file_path)
    
#     if activity_mapping_dict is not None:
#         for t in transitions:
#             # Split the label at '+', take the first part, and update if it exists in the dictionary
#             label_base = t.label.split('+')[0]
#             if label_base in activity_mapping_dict:
#                 t.label = activity_mapping_dict[label_base]
            
#     if return_markings:
#         net, init_marking, final_marking = from_lists_to_PetriNet(places, transitions, arcs, return_markings=return_markings)
#         return net, init_marking, final_marking
#     else:
#         net = from_lists_to_PetriNet(places, transitions, arcs, return_markings=return_markings)
#         return net


def generate_model_from_file(file_path, activity_mapping_dict=None, return_markings=True):
    """
    Generates a Petri net model from a file and optionally applies an activity mapping
    to transition labels. Raises a KeyError with details if a required mapping is missing.
    
    Parameters:
        file_path (str): The path to the file to parse.
        activity_mapping_dict (dict, optional): A dictionary mapping base activity labels
            to new labels. If provided, each transition's label (split at '+') is replaced 
            by the corresponding value in this dictionary.
        return_markings (bool): If True, returns the net along with initial and final markings.
        
    Returns:
        If return_markings is True:
            tuple: (net, initial_marking, final_marking)
        Otherwise:
            net

    Raises:
        KeyError: If a transition's base label is not found in activity_mapping_dict.
    """
    # Parse the file to get places, transitions, and arcs
    places, transitions, arcs = parse_file(file_path)
    
    # Apply the activity mapping if provided
    if activity_mapping_dict is not None:
        for transition in transitions:
            # Extract the base label (before any '+')
            base_label = transition.label.split('+')[0]
            if base_label not in activity_mapping_dict:
                available_keys = list(activity_mapping_dict.keys())
                raise KeyError(
                    f"Mapping for activity '{base_label}' not found in activity_mapping_dict. "
                    f"Available keys: {available_keys}"
                )
            # Update the transition's label with the mapped value
            transition.label = activity_mapping_dict[base_label]
    
    # Convert the lists into a Petri net model
    if return_markings:
        net, initial_marking, final_marking = from_lists_to_PetriNet(
            places, transitions, arcs, return_markings=True
        )
        return net, initial_marking, final_marking
    else:
        net = from_lists_to_PetriNet(
            places, transitions, arcs, return_markings=False
        )
        return net

    
def convert_xes_to_csv(file_name, save_to_csv=True, return_df=True, mxml=False):
    # Ensure the filename ends with '.xes'
    if mxml:
        if not file_name.endswith('.mxml'): 
            file_name += '.mxml'
        
    elif not file_name.endswith('.xes'):
        file_name += '.xes'
    
    if mxml:
        log = pm4py.read_mxml(file_name)
    else:
        log = pm4py.read_xes(file_name)
        
    dataframe = pm4py.convert_to_dataframe(log)
    
    if save_to_csv:
        # Replace the '.xes' extension with '.csv' in the filename for the CSV file
        csv_file_name = file_name[:-4] + '.csv'
        dataframe.to_csv(csv_file_name, index=False)
    
    if return_df: 
        return dataframe
    
    
def unique_trace_frequencies(df, sort_by='frequency'):
    # Validate sort_by parameter
    if sort_by not in ['frequency', 'trace_length']:
        raise ValueError("sort_by must be either 'frequency' or 'trace_length'")
    
    # Step 1: Aggregate activities into a tuple for each case
    traces = df.groupby('case:concept:name')['concept:name'].apply(tuple).reset_index(name='trace_activities')
    
    # Step 2: Identify unique traces and count their frequencies
    trace_counts = traces['trace_activities'].value_counts().reset_index()
    trace_counts.columns = ['trace_activities', 'frequency']
    
    # Step 3: Add the length of each trace
    trace_counts['trace_length'] = trace_counts['trace_activities'].apply(len)
    
    # Step 4: Sort by specified criteria (frequency or trace_length) from highest to lowest
    trace_counts = trace_counts.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
    
    return trace_counts


def plot_histogram(df, column, bins=10, color='skyblue', edgecolor='black'):
    """
    Plots a histogram for the specified column in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - column: The column to plot ('frequency' or 'trace_length').
    - bins: Number of bins in the histogram.
    - color: Color of the bars.
    - edgecolor: Color of the edge of the bars.
    """
    # Check if the column is valid
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    
    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=bins, color=color, edgecolor=edgecolor)
    plt.title(f'Histogram of Trace {column.capitalize()}')
    plt.xlabel(column.capitalize())
    plt.ylabel('Count')
    plt.show()
    

def array_to_tuple(arr):
    """Convert a NumPy array to a tuple recursively."""
    try:
        return tuple(array_to_tuple(a) for a in arr)
    except TypeError:
        return arr

def count_unique_arrays(arrays):
    """Count the number of unique arrays in a list of NumPy arrays."""
    unique_arrays = set()
    for arr in arrays:
        # Convert the array to a hashable tuple
        arr_tuple = array_to_tuple(arr)
        # Add the tuple representation to the set of unique arrays
        unique_arrays.add(arr_tuple)
    return len(unique_arrays)


def print_trace_lengths_list_from_df(df):
    """
    Prints the lengths of traces from a DataFrame log as a list.
    
    Parameters:
    - df: A pandas DataFrame representing the log, with 'case:concept:name' and 'concept:name' columns.
    """
    trace_lengths = df.groupby('case:concept:name')['concept:name'].count().tolist()
    print(f'trace lengths: {trace_lengths}')
    
    
# def map_to_string_numbers(df, return_map_dict=False):
#     # Initialize an empty dictionary to store the mapping
#     mapping_dict = {}
    
#     # Check if 'concept:name' column already contains string numbers
#     if not df['concept:name'].str.isnumeric().all():
#         # Generate a mapping from unique names to string numbers
#         unique_activities = df['concept:name'].unique()
#         mapping_dict = {activity: str(i) for i, activity in enumerate(unique_activities)}
        
#         # Map the activities in 'concept:name' to string numbers using the mapping dictionary
#         df['concept:name'] = df['concept:name'].map(mapping_dict)
    
#     if return_map_dict:
#         return df, mapping_dict
#     else:
#         return df

    
def read_dataframe(file_path, return_map_dict=False):
    """
    Reads a CSV file and processes it based on specific column presence. Maps activity names to string numbers.
    
    Args:
        file_path (str): The file path of the CSV file to be read.
        return_map_dict (bool): If True, returns both the DataFrame and the activity mapping dictionary.

    Returns:
        pd.DataFrame, dict (optional): The processed DataFrame, and optionally the mapping dictionary.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check for 'Case ID' and 'Activity' columns and rename them if present
    if 'Case ID' in df.columns and 'Activity' in df.columns:
        df = df[['Case ID', 'Activity']]
        df.columns = ['case:concept:name', 'concept:name']
    # If 'Case ID' and 'Activity' are not present, ensure 'case:concept:name' and 'concept:name' are selected
    elif 'case:concept:name' in df.columns and 'concept:name' in df.columns:
        df = df[['case:concept:name', 'concept:name']]
    else:
        raise ValueError("Required columns are missing in the DataFrame.")
    
    # Map the activities in 'concept:name' to string numbers and decide what to return based on return_map_dict
    if return_map_dict:
        df, activity_mapping_dict = map_to_string_numbers(df, return_map_dict=True)
        return df, activity_mapping_dict
    else:
        df = map_to_string_numbers(df, return_map_dict=False)
        return df
    
    
def find_cases_with_same_trace(log_df):
    # Group by case and concatenate activities into a single trace string for each case
    trace_df = log_df.groupby('case:concept:name')['concept:name'].apply('->'.join).reset_index(name='trace')
    
    # Find duplicates by trace
    duplicated_traces = trace_df[trace_df.duplicated('trace', keep=False)]
    
    # Group duplicated cases by trace and aggregate case names into tuples
    cases_with_same_trace = duplicated_traces.groupby('trace')['case:concept:name'].apply(tuple).to_dict()
    
    # Flip the dictionary to have tuples of case names as keys and traces as values
    result = {v: k for k, v in cases_with_same_trace.items()}
    
    return result


def trace_frequencies(df):
    # Group by case ID and aggregate the activities into a list for each case
    traces = df.groupby('case:concept:name')['concept:name'].apply(list)

    # Convert each list of activities into a tuple (to make it hashable) and then into a string representation
    # to count unique sequences. Tuples are used because lists cannot be used as keys in a dictionary due to being unhashable.
    trace_strs = traces.apply(lambda activities: '->'.join(activities)).tolist()

    # Count the frequency of each unique trace
    trace_freq = {}
    for trace in trace_strs:
        if trace in trace_freq:
            trace_freq[trace] += 1
        else:
            trace_freq[trace] = 1
    
    return trace_freq


def create_case_id_to_frequency_dict(df):
    # Step 1: Create a Trace Representation.
    df_copy = df.copy()
    df_copy['trace_representation'] = df_copy.groupby('case:concept:name')['concept:name'].transform(lambda x: '->'.join(x))
    
    # Step 2: Calculate Frequencies of Each Unique Trace Representation
    trace_representation_to_frequency = df_copy.drop_duplicates('case:concept:name')['trace_representation'].value_counts().to_dict()
    
    # Step 3: Map Trace Frequencies to Each Case ID
    df_copy['trace_frequency'] = df_copy['trace_representation'].apply(lambda x: trace_representation_to_frequency[x])
    
    # Step 4: Create a dictionary mapping from case ID to its trace frequency
    case_id_to_frequency = pd.Series(df_copy.drop_duplicates('case:concept:name').set_index('case:concept:name')['trace_frequency']).to_dict()
    
    return case_id_to_frequency


def get_marked_places(model, marking):
    """
    Extracts the names of places in a model that have a corresponding '1' in the marking.

    Args:
        model: A dictionary-like object with a 'places' field (a list of strings).
        marking: A tuple of 1s and 0s, where each index corresponds to a place in the model.

    Returns:
        A list of strings representing the names of the marked places.
    """

    marked_places = []
    for i, place_name in enumerate(model.places):
        if marking[i] == 1:
            marked_places.append(place_name)
    return marked_places


def transform_dictionary(model, marking_dict):
    """
    Transforms a dictionary with markings as keys and nested dictionaries with 'parents' field.

    Args:
        marking_dict: A dictionary where:
            - Keys: Markings (tuples of 1s and 0s)
            - Values: Dictionaries with a 'parents' field containing a set of parent markings.
        model: A dictionary-like object with a 'places' field (a list of strings).

    Returns:
        A new dictionary where:
            - Keys: Tuples of marked place names.
            - Values: Dictionaries with the 'parents' field transformed into a set of sets of marked parent place names.

    Raises:
        ValueError: If a marking tuple in the dictionary does not match the number of places in the model.
    """

    marked_places_dict = {}
    for marking, inner_dict in marking_dict.items():
        if len(marking) != len(model.places):
            raise ValueError(f"Marking {marking} does not match the number of places in the model.")

        marked_places = get_marked_places(model, marking)

        # Transform the 'parents' set
        transformed_parents = set()
        for parent_marking in inner_dict['parents']:
            if len(parent_marking) != len(model.places):
                raise ValueError(f"Parent marking {parent_marking} does not match the number of places in the model.")
            transformed_parents.add(frozenset(get_marked_places(model, parent_marking)))  # Use frozenset for hashability

        # Update the nested dictionary
        transformed_inner_dict = inner_dict.copy()  # Avoid modifying original dict
        transformed_inner_dict['parents'] = transformed_parents

        # Store in the result dictionary
        marked_places_dict[tuple(marked_places)] = transformed_inner_dict

    return marked_places_dict


def visualize_petri_net(net, marking=None, output_path="./model"):
    """
    Generates a visual representation of a Petri Net model using Graphviz.
    Transitions are displayed as rectangles, and places as circles.
    Tokens are represented as filled black circles within places if a marking is provided.

    Args:
        net: The self-defined Petri Net model defining the structure of the net (places, transitions, arcs).
        marking (tuple, optional): A tuple representing the current marking of the net.
            Each entry corresponds to a place in the order defined by the place_mapping.
            If None, the net will be visualized without tokens.
        output_path (str, optional): Path (without extension) to save the visualization. Defaults to "./model".
    """
    if not hasattr(net, 'place_mapping') or not hasattr(net, 'reverse_place_mapping'):
        raise AttributeError("The provided Petri net does not have the required place_mapping and reverse_place_mapping attributes")

    viz = Digraph(engine='dot')
    viz.attr(rankdir='TB')  # Set rank direction to top-to-bottom

    # Convert tuple marking to dictionary if provided
    marking_dict = {}
    if marking is not None:
        if len(marking) != len(net.place_mapping):
            raise ValueError(f"The length of the marking tuple ({len(marking)}) does not match the number of places in the net ({len(net.place_mapping)})")
        for idx, tokens in enumerate(marking):
            if tokens > 0:
                place = net.reverse_place_mapping[idx]
                marking_dict[place.name] = tokens  # Store by place name

    # Add Places (Circles)
    for place in net.places:
        label = place.name
        if marking is not None and place.name in marking_dict:  # Check by place name
            tokens = marking_dict[place.name]
            if tokens > 0:
                # Add a large token to the label with larger font size
                label += f"\n<FONT POINT-SIZE='30'>●</FONT>"
        
        # Ensure that the label is treated as an HTML-like label
        viz.node(str(place), label=f"<{label}>", shape='circle', style='filled', fillcolor='white', fixedsize='true', width='0.75', height='0.75')

    # Add Transitions (Rectangles)
    for transition in net.transitions:
        label = transition.label if transition.label else str(transition)
        viz.node(str(transition), label=label, shape='box')

    # Add Arcs
    for arc in net.arcs:
        viz.edge(str(arc.source), str(arc.target))

    # Explicitly set the rank of the source and sink nodes
    with viz.subgraph() as s:
        s.attr(rank='source')
        s.node(str(net.places[0]))  # Assuming the first place is the source

    with viz.subgraph() as s:
        s.attr(rank='sink')
        s.node(str(net.places[-1]))  # Assuming the last place is the sink

    # Redirect stderr to null to suppress warnings
    with open(os.devnull, 'w') as f, redirect_stderr(f):
        # Save Visualization 
        viz.render(output_path, format='pdf', cleanup=True)

    print(f"Visualization saved to: {output_path}")
 

def array_list_to_event_log(array_list, case_id_prefix="Case"):
    """Converts a list of numpy arrays into an event log DataFrame.

    Args:
        array_list: A list of numpy arrays, each representing a unique case.
        case_id_prefix: An optional prefix for case IDs (e.g., "Case", "Patient").

    Returns:
        A pandas DataFrame in event log format.
    """

    event_log_data = []

    for i, array in enumerate(array_list):
        case_id = f"{case_id_prefix}_{i+1}"  # Generate unique case ID
        for activity in array:
            event_log_data.append({
                'case:concept:name': case_id,
                'concept:name': str(activity)
            })

    return pd.DataFrame(event_log_data)