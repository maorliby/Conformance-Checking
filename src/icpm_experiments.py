import pandas as pd  
import numpy as np  
import random 
import time  
from typing import List, Dict, Tuple, DefaultDict, Union, Optional, Any
from RunningHorizon import *
import time
from threading import Thread
import pickle
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from matplotlib.figure import Figure
import warnings

def filter_log_efficiently(log: pd.DataFrame, 
               min_len: int = None, 
               max_len: int = None, 
               n_traces: int = None,
               random_seed: int = None) -> pd.DataFrame:
    """
    Filters the log based on trace length (min_len and max_len) and the number of traces.

    Parameters:
    - log (pd.DataFrame): The event log as a pandas DataFrame.
    - min_len (int, optional): Minimum allowed length of traces. No lower limit if None.
    - max_len (int, optional): Maximum allowed length of traces. No upper limit if None.
    - n_traces (int, optional): Number of traces to include. Includes all if None.
    - random_seed (int, optional): Seed for reproducible random trace selection. No randomness if None.

    Returns:
    - pd.DataFrame: The filtered log as a pandas DataFrame.
    """

    # Filter based on min_len and max_len
    if min_len is not None or max_len is not None:
        trace_lengths = log.groupby('case:concept:name').size().reset_index(name='length')
        
        if min_len is not None:
            trace_lengths = trace_lengths[trace_lengths['length'] >= min_len]
        
        if max_len is not None:
            trace_lengths = trace_lengths[trace_lengths['length'] <= max_len]
        
        accepted_cases = trace_lengths['case:concept:name'].tolist()
    else:
        accepted_cases = log['case:concept:name'].unique()
    
    # If n_traces is None, return the filtered log
    if n_traces is None:
        return log[log['case:concept:name'].isin(accepted_cases)]
    
    # If n_traces is specified, limit the number of cases with reproducibility
    if random_seed is not None:
        selected_cases = pd.Series(accepted_cases).sample(n=min(len(accepted_cases), n_traces), random_state=random_seed).tolist()
    else:
        selected_cases = accepted_cases[:n_traces]
    
    return log[log['case:concept:name'].isin(selected_cases)]


def compute_case_variants(dk_df: pd.DataFrame) -> Dict[str, Tuple[str, ...]]:
    """
    Computes the variant (sequence of activities) for each case.

    Parameters:
    dk_df (pd.DataFrame): DataFrame containing the log data with 'case:concept:name' and 'concept:name' columns.

    Returns:
    Dict[str, Tuple[str, ...]]: Mapping from case ID to variant.
    """
    variant_series = dk_df.groupby('case:concept:name')['concept:name'].apply(tuple)
    return variant_series.to_dict()


def select_unique_cases(
    cases_list: List[str],
    case_to_variant: Dict[str, Tuple[str, ...]],
    n_cases: int,
    random_selection: bool = True,
    random_seed: int = 42
) -> List[str]:
    """
    Selects n_cases from cases_list ensuring that each selected case has a unique variant.
    
    Parameters:
    cases_list (List[str]): List of case IDs.
    case_to_variant (Dict[str, Tuple[str, ...]]): Mapping from case ID to its variant.
    n_cases (int): Number of cases to select.
    random_selection (bool): Whether to select randomly.
    random_seed (int): Seed for random selection.
    
    Returns:
    List[str]: Selected case IDs with unique variants.
    
    Raises:
    ValueError: If requested number of cases exceeds available unique variants.
    """
    # First, count unique variants available
    unique_variants = set(case_to_variant[case] for case in cases_list)
    n_unique_variants = len(unique_variants)
    
    if n_cases > n_unique_variants:
        raise ValueError(
            f"Requested {n_cases} unique variants but only {n_unique_variants} "
            "unique variants are available in the provided cases"
        )
    
    # Group cases by variant
    variant_to_cases: Dict[Tuple[str, ...], List[str]] = {}
    for case in cases_list:
        variant = case_to_variant[case]
        if variant not in variant_to_cases:
            variant_to_cases[variant] = []
        variant_to_cases[variant].append(case)
    
    # Prepare variants for selection
    variants = list(variant_to_cases.keys())
    if random_selection:
        random.seed(random_seed)
        random.shuffle(variants)
    
    # Select exactly n_cases variants
    selected_variants = variants[:n_cases]
    
    # Select one case for each variant
    selected_cases = []
    for variant in selected_variants:
        cases = variant_to_cases[variant]
        if random_selection:
            selected_case = random.choice(cases)
        else:
            selected_case = cases[0]
        selected_cases.append(selected_case)
    
    return selected_cases


def get_cases(log: pd.DataFrame, unique_variants: bool) -> List[str]:
    """
    Get list of case IDs, optionally filtering to keep only unique variants.
    
    Args:
        log (pd.DataFrame): Log DataFrame with case:concept:name and concept:name columns
        unique_variants (bool): If True, return only one case per unique sequence variant
        
    Returns:
        List[str]: List of case IDs
    """
    if not unique_variants:
        return log['case:concept:name'].unique().tolist()
    
    # Create a DataFrame with case ID and its activity sequence
    sequences = log.groupby('case:concept:name')['concept:name'].agg(tuple).reset_index()
    sequences.columns = ['case:concept:name', 'sequence']
    
    # Keep first case for each unique sequence
    unique_variant_cases = sequences.drop_duplicates(subset='sequence')['case:concept:name'].tolist()
    
    return unique_variant_cases
    

def train_test_log_split(
    dk_df: pd.DataFrame,
    n_train_traces: int = None,
    n_test_traces: int = None,
    train_traces: List[str] = None,
    test_traces: List[str] = None,
    random_selection: bool = True,
    random_seed: int = 42,
    unique_variants: bool = False,
    allow_intersection: bool = False,
    sk_df: pd.DataFrame = None,
    sftmax_lst: List[pd.DataFrame] = None,
    unique_train_variants: bool = False,  # New parameter
    unique_test_variants: bool = False    # New parameter
) -> Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]:
    """
    Splits the log into training and testing sets based on the specified number of unique traces or specific trace lists.
    Enforces uniqueness of trace variants in training and/or testing sets when specified.

    Parameters:
    dk_df (pd.DataFrame): DataFrame containing the log data with 'case:concept:name' and 'concept:name' columns.
    n_train_traces (int, optional): Number of traces to include in the training set. Default is None.
    n_test_traces (int, optional): Number of traces to include in the testing set. Default is None.
    train_traces (List[str], optional): List of specific trace IDs to include in the training set. Default is None.
    test_traces (List[str], optional): List of specific trace IDs to include in the testing set. Default is None.
    random_selection (bool, optional): Whether to randomly select the traces. Default is True.
    random_seed (int, optional): Seed for random selection. Default is 42.
    unique_variants (bool, optional): Whether to enforce unique variants. Default is False.
    allow_intersection (bool, optional): Whether to allow the same trace sequences in both training and testing sets. Default is False.
    sk_df (pd.DataFrame, optional): Second DataFrame to use as the source for the test set. Default is None.
    sftmax_lst (List[pd.DataFrame], optional): List of softmax matrices corresponding to the traces in dk_df. Default is None.
    unique_train_variants (bool, optional): Whether to enforce unique trace variants in the training set. Default is False.
    unique_test_variants (bool, optional): Whether to enforce unique trace variants in the testing set. Default is False.

    Returns:
    Dict[str, Union[pd.DataFrame, List[pd.DataFrame]]]: A dictionary with keys 'train_df', 'test_df', 'test_ground_truth',
    and 'test_sftmax_lst' (if applicable).

    Raises:
    ValueError: 
        - If the number of unique variants is insufficient to fulfill the requested n_train_traces or n_test_traces.
        - If there is an overlap in variants between training and testing sets when allow_intersection is False.

    Example:
    --------
    ```python
    split_result = train_test_log_split(
        dk_df=your_dataframe,
        n_train_traces=100,
        n_test_traces=50,
        random_selection=True,
        random_seed=123,
        unique_train_variants=True,  # Enforce unique variants in training set
        unique_test_variants=True,   # Enforce unique variants in testing set
        allow_intersection=False     # Ensure no variant overlap between train and test
    )

    train_df = split_result['train_df']
    test_df = split_result['test_df']
    ```
    """
    
    result = {}
    
    # Compute variants for each case
    case_to_variant = compute_case_variants(dk_df)
    
    # Get list of all cases
    cases_list = get_cases(dk_df, unique_variants=unique_variants)
    
    # Selecting Train and Test Cases
    if train_traces is not None:
        train_cases = train_traces.copy()
        if unique_train_variants:
            train_cases = select_unique_cases(train_cases, case_to_variant, len(train_cases), 
                                             random_selection=False, random_seed=random_seed)
        
        if test_traces is not None:
            test_cases = test_traces.copy()
            if unique_test_variants:
                test_cases = select_unique_cases(test_cases, case_to_variant, len(test_cases), 
                                                random_selection=False, random_seed=random_seed)
        else:
            remaining_cases = cases_list if allow_intersection else [case for case in cases_list if case not in train_cases]
            if unique_test_variants:
                if n_test_traces is None:
                    n_test_traces = len(remaining_cases)
                test_cases = select_unique_cases(remaining_cases, case_to_variant, n_test_traces, 
                                                random_selection, random_seed)
            else:
                test_cases = select_cases(remaining_cases, n_test_traces, random_selection, random_seed)
    elif test_traces is not None:
        test_cases = test_traces.copy()
        if unique_test_variants:
            test_cases = select_unique_cases(test_cases, case_to_variant, len(test_cases), 
                                            random_selection=False, random_seed=random_seed)
        
        remaining_cases = cases_list if allow_intersection else [case for case in cases_list if case not in test_cases]
        if unique_train_variants:
            if n_train_traces is None:
                n_train_traces = len(remaining_cases)
            train_cases = select_unique_cases(remaining_cases, case_to_variant, n_train_traces, 
                                             random_selection, random_seed)
        else:
            train_cases = select_cases(remaining_cases, n_train_traces, random_selection, random_seed)
    else:
        # Neither train_traces nor test_traces are provided
        if unique_train_variants and unique_test_variants:
            # Both sets require unique variants
            # First select unique train cases
            if n_train_traces is None:
                n_train_traces = len(cases_list) // 2
            train_cases = select_unique_cases(cases_list, case_to_variant, n_train_traces, 
                                             random_selection, random_seed)
            # Exclude train variants if intersection is not allowed
            if not allow_intersection:
                train_variants = {case_to_variant[case] for case in train_cases}
                remaining_cases = [case for case in cases_list if case_to_variant[case] not in train_variants]
            else:
                remaining_cases = cases_list.copy()
            # Then select unique test cases
            if n_test_traces is None:
                n_test_traces = len(remaining_cases) // 2
            test_cases = select_unique_cases(remaining_cases, case_to_variant, n_test_traces, 
                                            random_selection, random_seed)
        else:
            # Handle other combinations
            train_cases, test_cases = split_cases(cases_list, n_train_traces, n_test_traces, 
                                                 allow_intersection, random_selection, random_seed)
            if unique_train_variants:
                train_cases = select_unique_cases(train_cases, case_to_variant, len(train_cases), 
                                                 random_selection=False, random_seed=random_seed)
            if unique_test_variants:
                test_cases = select_unique_cases(test_cases, case_to_variant, len(test_cases), 
                                                random_selection=False, random_seed=random_seed)
    
    # If uniqueness within train or test sets is enforced, verify no overlaps if intersection is not allowed
    if not allow_intersection:
        if unique_train_variants and unique_test_variants:
            # Ensure no variant is shared between train and test
            train_variants = {case_to_variant[case] for case in train_cases}
            test_variants = {case_to_variant[case] for case in test_cases}
            if train_variants.intersection(test_variants):
                raise ValueError("Variants overlap between train and test sets while allow_intersection is False.")
    
    # Prepare the result DataFrames
    if sk_df is not None:
        result['train_df'] = filter_dataframe(dk_df, train_cases)
        result['test_df'] = filter_dataframe(sk_df, test_cases)
        result['test_ground_truth'] = filter_dataframe(dk_df, test_cases)
        return result

    if sftmax_lst is not None:
        if len(dk_df['case:concept:name'].unique()) != len(sftmax_lst):
            raise ValueError("The length of dk_df and sftmax_lst must be the same")
        
        result['train_df'] = filter_dataframe(dk_df, train_cases)
        result['test_df'] = filter_dataframe(dk_df, test_cases)
        
        # Map the test cases to their original indices
        original_case_indices = {case: i for i, case in enumerate(dk_df['case:concept:name'].unique())}
        test_indices = [original_case_indices[case] for case in test_cases]
        result['test_sftmax_lst'] = [sftmax_lst[i] for i in test_indices]
        
        return result

    result['train_df'] = filter_dataframe(dk_df, train_cases)
    result['test_df'] = filter_dataframe(dk_df, test_cases)
    
    return result


def train_test_log_split_simplified(
    log_df: pd.DataFrame,
    n_train_traces: int,  # Required parameter
    n_test_traces: Optional[int] = None,  # Optional parameter
    random_seed: int = 42,
    unique_train_variants: bool = False,
    unique_test_variants: bool = False,
    allow_variant_intersection: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Splits a process log into training and testing sets with detailed variant control.

    Parameters:
    -----------
    log_df : pd.DataFrame
        DataFrame containing the log data with 'case:concept:name' and 'concept:name' columns
    n_train_traces : int
        Number of traces (cases) to include in the training set
    n_test_traces : int, optional
        Number of traces to include in the test set. If None, includes all remaining traces
    random_seed : int
        Random seed for reproducibility
    unique_train_variants : bool
        If True, ensures each variant appears only once in training set
    unique_test_variants : bool
        If True, ensures each variant appears only once in testing set
    allow_variant_intersection : bool
        If False, ensures no variant appears in both train and test sets

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - 'train_df': Training set DataFrame
        - 'test_df': Testing set DataFrame

    Raises:
    -------
    ValueError
        - If n_train_traces or n_test_traces exceed available cases
        - If the DataFrame doesn't contain required columns
        - If variant constraints cannot be satisfied (except for empty train/test sets,
          which now produce warnings)
    """
    # Input validation
    total_cases = len(log_df['case:concept:name'].unique())
    
    if not 0 <= n_train_traces < total_cases:
        raise ValueError(f"n_train_traces must be between 0 and {total_cases} (total number of cases)")
    
    if n_test_traces is not None:
        if not 0 <= n_test_traces <= total_cases - n_train_traces:
            raise ValueError(
                f"n_test_traces must be between 0 and {total_cases - n_train_traces} "
                "(remaining cases after train set selection)"
            )
            
    required_cols = {'case:concept:name', 'concept:name'}
    if not required_cols.issubset(log_df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Set random seed
    random.seed(random_seed)
    
    # Get all cases and their variants
    all_cases = log_df['case:concept:name'].unique().tolist()
    case_variants = get_case_variants(log_df)
    random.shuffle(all_cases)
    
    if unique_train_variants or unique_test_variants or not allow_variant_intersection:
        # Handle variant constraints
        if not allow_variant_intersection:
            # Select training cases with unique variants
            train_cases = select_unique_variant_cases(all_cases, case_variants, n_train_traces, random_seed)
            
            # Select test cases from remaining variants
            train_variants = {case_variants[case] for case in train_cases}
            remaining_cases = [
                case for case in all_cases 
                if case_variants[case] not in train_variants
            ]
            
            test_cases = (
                select_unique_variant_cases(remaining_cases, case_variants, n_test_traces, random_seed)
                if unique_test_variants
                else (random.sample(remaining_cases, n_test_traces) if n_test_traces is not None else remaining_cases)
            )
        else:
            # Randomly select cases for training and testing
            # For train set: first get unique variants if needed
            if unique_train_variants:
                train_cases = select_unique_variant_cases(all_cases, case_variants, n_train_traces, random_seed)
                if len(train_cases) < n_train_traces:
                    raise ValueError(
                        f"Could not find {n_train_traces} unique variants for training set. "
                        f"Only {len(train_cases)} unique variants available."
                    )
            else:
                train_cases = random.sample(all_cases, n_train_traces)

            # Get remaining cases for test set
            remaining_cases = [case for case in all_cases if case not in train_cases]
            
            # For test set: handle unique variants if needed
            if unique_test_variants:
                if n_test_traces is not None:
                    test_cases = select_unique_variant_cases(remaining_cases, case_variants, n_test_traces, random_seed)
                    if len(test_cases) < n_test_traces:
                        raise ValueError(
                            f"Could not find {n_test_traces} unique variants for test set. "
                            f"Only {len(test_cases)} unique variants available in remaining cases."
                        )
                else:
                    test_cases = select_unique_variant_cases(remaining_cases, case_variants, random_seed=random_seed)
            else:
                if n_test_traces is not None:
                    test_cases = random.sample(remaining_cases, n_test_traces)
                else:
                    test_cases = remaining_cases
    else:
        # Simple split without variant constraints
        train_cases = random.sample(all_cases, n_train_traces)
        remaining_cases = [case for case in all_cases if case not in train_cases]
        
        if n_test_traces is not None:
            test_cases = random.sample(remaining_cases, n_test_traces)
        else:
            test_cases = remaining_cases
    
    # Create final DataFrames
    train_df = log_df[log_df['case:concept:name'].isin(train_cases)].copy()
    test_df = log_df[log_df['case:concept:name'].isin(test_cases)].copy()
    
    # Validate results: Warn instead of raising an error if one of the sets is empty.
    if len(train_df) == 0 or len(test_df) == 0:
        warnings.warn("Resulted in an empty train or test set due to variant constraints. "
                      "Returning the DataFrames as-is.", UserWarning)
    
    return {
        'train_df': train_df,
        'test_df': test_df
    }


def get_case_variants(log_df: pd.DataFrame) -> Dict[str, Tuple[str, ...]]:
    """
    Creates a mapping of case IDs to their activity sequences.
    
    Parameters:
    -----------
    log_df : pd.DataFrame
        DataFrame with 'case:concept:name' and 'concept:name' columns
    
    Returns:
    --------
    Dict[str, Tuple[str, ...]]
        Mapping of case IDs to their activity sequences
    """
    
    return log_df.groupby('case:concept:name')['concept:name'].agg(tuple).to_dict()

    
def select_unique_variant_cases(
    cases: List[str],
    case_variants: Dict[str, Tuple[str, ...]],
    n_cases: Union[int, None] = None,
    random_seed: int = 42
) -> List[str]:
    """
    Selects cases ensuring each has a unique variant sequence, with random sampling.
    
    Parameters:
    -----------
    cases : List[str]
        List of case IDs to select from
    case_variants : Dict[str, Tuple[str, ...]]
        Mapping of case IDs to their variants
    n_cases : Union[int, None]
        Number of cases to select, if None selects all unique variants
    random_seed : int
        Seed for random selection, by default 42
    
    Returns:
    --------
    List[str]
        Selected cases with unique variants
    
    Raises:
    -------
    ValueError
        If n_cases is greater than the number of available unique variants
    """
    # Group cases by their variants
    variant_to_cases: Dict[Tuple[str, ...], List[str]] = {}
    for case in cases:
        variant = case_variants[case]
        if variant not in variant_to_cases:
            variant_to_cases[variant] = []
        variant_to_cases[variant].append(case)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # For each variant, randomly select one case
    unique_cases = [random.choice(cases_list) for cases_list in variant_to_cases.values()]
    
    if n_cases is not None:
        if n_cases > len(unique_cases):
            raise ValueError(f"Requested {n_cases} cases but only {len(unique_cases)} unique variants available")
        # Randomly sample n_cases from the unique cases
        unique_cases = random.sample(unique_cases, n_cases)
    
    return unique_cases
    

def compare_search_algorithms(
    df: pd.DataFrame,
    df_name: str,
    cost_function: Optional[Any] = None, 
    n_train_traces: int = 10,
    n_test_traces: int = 10,
    train_cases: Optional[List[str]] = None,
    test_cases: Optional[List[str]] = None,
    random_seed: int = 304,
    non_sync_penalty: int = 1,
    only_return_model: bool = False,
    allow_intersection: bool = False,
    read_model_from_file: bool = False,
    model_path: Optional[str] = None,
    activity_mapping_dict: Optional[Dict[str, str]] = None,
    algorithms: Optional[List[str]] = None,
    time_limit: Optional[int] = None,
    return_results: bool = False,
    unique_train_variants: bool = True,
    unique_test_variants: bool = True
) -> Optional[Any]:  
    """
    Function to compare specified search algorithms (A*, A* Extended, Reach) 
    by either training a model on the training data or reading it from a file, 
    and then evaluating it on the test data. Optionally returns the results 
    instead of saving them.
    """
    if algorithms is None:
        algorithms = ['astar', 'astar_extended', 'reach']

    cost_function = select_cost_function(cost_function)
    np.random.seed(random_seed)
    
    result = train_test_log_split(
        df, 
        n_train_traces=n_train_traces, 
        n_test_traces=n_test_traces,
        train_traces=train_cases,
        test_traces=test_cases,
        random_selection=(train_cases is None and test_cases is None), 
        random_seed=random_seed,
        allow_intersection=allow_intersection,
        unique_train_variants=unique_train_variants,
        unique_test_variants=unique_test_variants
    )
    
    train_df, test_df = result['train_df'], result['test_df']
    
    if read_model_from_file:
        if not model_path or not activity_mapping_dict:
            raise ValueError("Both 'model_path' and 'activity_mapping_dict' must be provided when reading a model from a file.")
        
        model = generate_model_from_file(
            model_path,
            activity_mapping_dict=activity_mapping_dict,
            return_markings=False
        )
    else:
        model = prepare_model(train_df, non_sync_penalty)
    
    if only_return_model:
        return model

    model = add_transition_mappings_to_model(model)
 
    # Group similar traces
    trace_dict, lookup_dict = group_similar_traces(test_df)
    total_traces = len(trace_dict)
    
    # Initialize results dictionary
    results = {alg: {'cost': [], 'time': [], 'nodes_popped': [], 'case_id': []} for alg in algorithms}
    
    computed_results = {}
    
    print(f"Overall computing statistics for {len(trace_dict)} traces")
    for idx, (trace, cases) in enumerate(trace_dict.items(), 1):
        print(f'\rComputing trace {idx}/{total_traces}', end='')
        representative_case = cases[0]
        true_trace_df = test_df[test_df['case:concept:name'] == representative_case].reset_index(drop=True)
        true_trace_df['probs'] = [[1.0] for _ in range(len(true_trace_df))]
        stochastic_trace = construct_stochastic_trace_model(true_trace_df, non_sync_penalty)
        sync_prod = SyncProduct(model, stochastic_trace, cost_function=cost_function)
        
        computed_results[trace] = {}
        
        for alg in algorithms:
            computed_results[trace][alg] = run_algorithm_with_timeout(
                sync_prod, alg, timeout=time_limit
            )

        # Store results for all cases with the same trace
        for case_id in cases:
            for alg in algorithms:
                if computed_results[trace][alg]:
                    results[alg]['cost'].append(computed_results[trace][alg]['cost'])
                    results[alg]['time'].append(computed_results[trace][alg]['time'])  
                    results[alg]['nodes_popped'].append(computed_results[trace][alg]['nodes_popped'])
                else:
                    results[alg]['cost'].append(None)
                    results[alg]['time'].append(None)
                    results[alg]['nodes_popped'].append(None)
                results[alg]['case_id'].append(case_id)

    if return_results:
        return results

    # Save results as DataFrames with the dataframe name included in the filename
    for alg in algorithms:
        save_results_as_dataframe(results[alg], df_name, alg)

    return None  # Return None if only_return_model is False and no other return is specified
    

def group_similar_traces(df):
    # Group by case:concept:name and aggregate activities into a tuple
    grouped_traces = df.groupby('case:concept:name')['concept:name'].apply(tuple)
    
    # Create a dictionary to store the groups of similar traces
    trace_dict = defaultdict(list)
    
    # Populate the dictionary
    for case_id, trace in grouped_traces.items():
        trace_dict[trace].append(case_id)
    
    # Create a lookup dictionary to find similar traces
    lookup_dict = {}
    for trace, case_ids in trace_dict.items():
        for case_id in case_ids:
            lookup_dict[case_id] = [cid for cid in case_ids if cid != case_id]
    
    return trace_dict, lookup_dict


def select_cost_function(cost_function):
    if cost_function == 'logarithmic':
        return lambda x: -np.log(x) / 4.7
    elif cost_function == 'linear':
        return lambda x: 1 - x
    return cost_function


def prepare_model(train_df: pd.DataFrame, non_sync_penalty: int):
    """
    Prepares a PetriNet model by discovering it from the provided data and computing the necessary transition mappings.

    Parameters:
    - train_df (pd.DataFrame): The training data as a pandas DataFrame.
    - non_sync_penalty (int): Penalty value for non-synchronous moves.

    Returns:
    - model: The prepared PetriNet model.
    """
    train_df = prepare_df_cols_for_discovery(train_df)
    net, init_marking, final_marking = pm4py.discover_petri_net_inductive(train_df)
    model = from_discovered_model_to_PetriNet(
        net, 
        non_sync_move_penalty=non_sync_penalty,
        pm4py_init_marking=init_marking, 
        pm4py_final_marking=final_marking
    )

    return model


def add_transition_mappings_to_model(model):
    """
    Computes the mandatory and alive transitions for the given PetriNet model.

    Parameters:
    - model: The PetriNet model object.

    Returns:
    - model: The updated PetriNet model with mandatory and alive transitions computed.
    """
    # Measure time for computing mandatory transitions
    if model.mandatory_transitions_map is None:
        print('Starting to compute mandatory transitions...')
        start_time = time.time()
        
        model.mandatory_transitions_map = compute_mandatory_transitions(
            model.pm4py_net, 
            model.pm4py_initial_marking, 
            model.pm4py_final_marking,
            model.place_mapping
        )
        
        end_time = time.time()
        print(f"Mandatory transitions computed in {end_time - start_time:.4f} seconds")

    # Measure time for computing alive transitions
    if model.alive_transitions_map is None:
        print('Starting to compute alive transitions...')
        start_time = time.time()
        
        model.alive_transitions_map = map_markings_to_reachable_transitions(model)
        
        end_time = time.time()
        print(f"Alive transitions computed in {end_time - start_time:.4f} seconds")
    
    return model


def run_search_algorithm(sync_prod, algorithm: str, results: Dict[str, Dict[str, List]], trace_case: str, store_result: bool = True):
    start_time = time.time()
    
    if algorithm == 'astar':
        alignment, cost, nodes_popped = sync_prod.astar_search()
    elif algorithm == 'astar_extended':
        alignment, cost, nodes_popped = sync_prod.astar_incremental()
    elif algorithm == 'reach':
        alignment, cost, nodes_popped = sync_prod.reach_search()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    elapsed_time = time.time() - start_time
    
    if store_result:
        results[algorithm]['cost'].append(cost)
        results[algorithm]['time'].append(elapsed_time)
        results[algorithm]['nodes_popped'].append(nodes_popped)
        results[algorithm]['case_id'].append(trace_case)
    
    # Return the results so they can be reused if needed
    return {'cost': cost, 'time': elapsed_time, 'nodes_popped': nodes_popped}

def save_results_as_dataframe(results: Dict[str, List], df_name: str, algorithm_name: str):
    """
    Save the results as a DataFrame to a CSV file with the dataframe name included in the filename.

    Parameters:
    - results (Dict[str, List]): The results dictionary to be saved.
    - df_name (str): The name of the dataframe, used in the filename.
    - algorithm_name (str): The name of the algorithm, used in the filename.
    """
    # Convert the results dictionary into a DataFrame
    df = pd.DataFrame(results)
    
    # Ensure 'case_id' is the first column
    df = df[['case_id'] + [col for col in df.columns if col != 'case_id']]
    
    # Round 'cost' and 'time' columns to 5 decimal places
    df['cost'] = df['cost'].round(5)
    df['time'] = df['time'].round(5)
    
    # Construct the filename with the dataframe name and algorithm name
    filename = f"{df_name}_{algorithm_name}_results.csv"
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)


def split_transition_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split transition_names column into separate log and model columns and include transition weights.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing 'case_id', 'transition_names', and 
                          'transition_weights' columns where transition_names contains 
                          strings like '(sync_0_ts=0, sync_0_ts=0)'
    
    Returns:
        pd.DataFrame: DataFrame with 'case_id', 'log', 'model', and 'weight' columns
    
    Raises:
        ValueError: If required columns are missing or if transition_names format is invalid
    """
    # Validate input DataFrame
    required_cols = ['case_id', 'transition_names', 'transition_weights']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    def extract_activities(transition: str) -> Tuple[str, str]:
        """
        Extract log and model activities from transition string.
        
        Args:
            transition (str): String containing the transition information
                            in format '(sync_x_ts=y, sync_x_ts=y)'
                            
        Returns:
            Tuple[str, str]: Pair of extracted activities (log, model)
        """
        try:
            # Remove parentheses and split by comma
            cleaned = transition.strip('()').split(',')
            if len(cleaned) != 2:
                return ('', '')  # Return empty strings for invalid format
                
            # Extract activities and clean them
            log_activity = cleaned[0].strip()
            model_activity = cleaned[1].strip()
            
            return (log_activity, model_activity)
        except (AttributeError, IndexError):
            return ('', '')  # Return empty strings for any parsing errors
    
    # Create new DataFrame with required columns
    result_df = pd.DataFrame()
    result_df['case_id'] = df['case_id']
    
    # Extract activities using vectorized operations
    activities = df['transition_names'].apply(extract_activities)
    result_df['log'] = activities.str[0]
    result_df['model'] = activities.str[1]
    
    # Add transition weights column
    result_df['weight'] = df['transition_weights']
    
    return result_df


def create_window_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary dataset comparing costs and times between windowed and optimal markings,
    with all numerical values rounded to 2 decimal places. Window size is automatically detected
    from the input DataFrame's column names.
    
    If optimal markings (window_None_markings) are not present, the function will return only
    the window markings data without comparison metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing case_id, window markings, and optionally optimal markings data
    
    Returns:
        pd.DataFrame: Summary DataFrame containing:
            - case_id
            - window_cost: Cost for detected window size (rounded to 2 decimals)
            - window_time: Time for detected window size (if available, rounded to 2 decimals)
            - optimal_cost: Cost for optimal solution (if available, rounded to 2 decimals)
            - optimal_time: Time for optimal solution (if available, rounded to 2 decimals)
            - cost_increase_percentage: Percentage increase of window cost compared to optimal cost 
              (if optimal cost is available, rounded to 2 decimals)
            - alignment_length: Number of events in the alignment (computed per case)
    """
    import re
    # Detect window size from column names
    window_cols = [col for col in df.columns if 'window_' in col and 'None' not in col]
    if not window_cols:
        raise ValueError("No window columns found in DataFrame")
    
    # Extract window size using regex
    window_size = None
    for col in window_cols:
        match = re.search(r'window_(\d+)_', col)
        if match:
            window_size = match.group(1)
            break
    
    if not window_size:
        raise ValueError("Could not detect window size from column names")
    
    # Define column names based on detected window size
    window_cost_col = f'window_{window_size}_markings_1_cost'
    window_time_col = f'window_{window_size}_markings_1_time'
    optimal_cost_col = 'window_None_markings_1_cost'
    optimal_time_col = 'window_None_markings_1_time'
    
    # Check if window cost column is available (mandatory)
    if window_cost_col not in df.columns:
        raise ValueError(f"Missing required window cost column: {window_cost_col}")
    
    # Check which optional columns are available
    has_window_time = window_time_col in df.columns
    has_optimal_cost = optimal_cost_col in df.columns
    has_optimal_time = optimal_time_col in df.columns
    
    # Create aggregation dictionary with available columns
    agg_dict = {window_cost_col: 'first'}
    
    if has_window_time:
        agg_dict[window_time_col] = 'first'
    
    if has_optimal_cost:
        agg_dict[optimal_cost_col] = 'first'
    
    if has_optimal_time:
        agg_dict[optimal_time_col] = 'first'
    
    # Create summary DataFrame: take the first occurrence per case_id
    summary = df.groupby('case_id').agg(agg_dict).reset_index()
    
    # Define column mappings based on available columns
    column_mappings = {
        window_cost_col: f'window_{window_size}_cost',
    }
    
    if has_window_time:
        column_mappings[window_time_col] = f'window_{window_size}_time'
    
    if has_optimal_cost:
        column_mappings[optimal_cost_col] = 'full_window_cost'
    
    if has_optimal_time:
        column_mappings[optimal_time_col] = 'full_window_time'
    
    # Rename columns
    summary = summary.rename(columns=column_mappings)
    
    # Calculate cost increase percentage if optimal cost is available
    if has_optimal_cost:
        summary['cost_increase_percentage'] = (
            (summary[f'window_{window_size}_cost'] - summary['full_window_cost']) / 
            summary['full_window_cost'] * 100
        )
    
    # Compute alignment length for each case
    alignment_length_df = df.groupby('case_id').size().reset_index(name='alignment_length')
    
    # Merge the alignment length into the summary DataFrame
    summary = summary.merge(alignment_length_df, on='case_id')
    
    # Round all numeric columns to 2 decimal places
    numeric_columns = [col for col in summary.columns 
                       if col not in ['case_id', 'alignment_length']]
    
    if numeric_columns:  # Only round if there are numeric columns
        summary[numeric_columns] = summary[numeric_columns].round(2)
    
    return summary

    
def map_columns_to_sequential_ids(
    df: pd.DataFrame,
    return_map: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Maps both 'case:concept:name' and 'concept:name' columns to sequential string integers.
    Each unique value in each column gets a sequential ID starting from '0'.
    If return_map is True, returns the mapping dictionary for 'concept:name' only.
    
    Args:
        df: DataFrame containing both 'case:concept:name' and 'concept:name' columns
        return_map: If True, returns both DataFrame and the concept:name mapping dictionary
    
    Returns:
        DataFrame with sequential string integers in both columns,
        optionally with the mapping dictionary for concept:name column
        
    Example:
        Input:
            case:concept:name: ['A', 'B', 'A', 'C']
            concept:name: ['register', 'review', 'register', 'approve']
        Output:
            case:concept:name: ['0', '1', '0', '2']
            concept:name: ['0', '1', '0', '2']
    """
    if not {'case:concept:name', 'concept:name'}.issubset(df.columns):
        raise KeyError("DataFrame must contain both 'case:concept:name' and 'concept:name' columns")
    
    result_df = df.copy()
    
    # Map case:concept:name column
    unique_cases = pd.unique(result_df['case:concept:name'])
    case_map = {case: str(i) for i, case in enumerate(unique_cases)}
    result_df['case:concept:name'] = result_df['case:concept:name'].map(case_map)
    
    # Map concept:name column
    unique_activities = pd.unique(result_df['concept:name'])
    activity_map = {activity: str(i) for i, activity in enumerate(unique_activities)}
    result_df['concept:name'] = result_df['concept:name'].map(activity_map)
    
    return (result_df, activity_map) if return_map else result_df

    
def load_and_preprocess_log(
    df_name: str, 
    min_len: int = None, 
    max_len: int = None, 
    n_traces: int = None, 
    random_seed: int = None,
    path: str = None,
    subfolder: str = None,
    max_samples_per_activity: Optional[int] = None,
    stats: Optional[Dict] = None  # Pass the stats dict as reference
) -> Tuple[pd.DataFrame, Dict]:
    """
    Loads, preprocesses, and filters an event log with optional activity sampling.
    
    Parameters:
    - df_name (str): The filename of the CSV file to load.
    - min_len (int, optional): Minimum allowed length of traces for filtering.
    - max_len (int, optional): Maximum allowed length of traces for filtering.
    - n_traces (int, optional): Number of traces to include after filtering.
    - random_seed (int, optional): Seed for reproducible trace selection.
    - path (str, optional): The base path to the dataset directory.
    - subfolder (str, optional): Subfolder within the base path where the dataset resides.
    - max_samples_per_activity (int, optional): Maximum sequential occurrences of each activity in traces.
    - stats (Dict, optional): Dictionary to store statistics if provided.
    
    Returns:
    - Tuple[pd.DataFrame, Dict]: The preprocessed DataFrame and mapping dictionary
    """
    # Construct the full file path
    if path:
        full_path = os.path.join(path, subfolder) if subfolder else path
        file_path = os.path.join(full_path, df_name)
    else:
        file_path = df_name
    
    # Load the DataFrame from CSV
    df = pd.read_csv(file_path)
    
    # Keep only the required columns
    columns_to_keep = ['case:concept:name', 'concept:name']
    df = df[columns_to_keep]
    
    if stats is not None:
        stats['original'] = get_dataset_statistics(df, "Original")
        case_lengths = get_case_lengths(df)
        stats['original'].update({
            'avg_trace_length': case_lengths['length'].mean(),
            'median_trace_length': case_lengths['length'].median(),
            'max_trace_length': case_lengths['length'].max(),
            'min_trace_length': case_lengths['length'].min()
        })
    
    # Map strings to string numbers
    df, map_dict = map_columns_to_sequential_ids(df, return_map=True)
    
    # Assign unique numeric identifiers to each case
    df['case:concept:name'] = df.groupby('case:concept:name').ngroup().astype(str)
    
    # Apply activity sampling if specified
    if max_samples_per_activity is not None:
        df = sample_trace_activities(df, max_samples_per_activity)
        if stats is not None:
            stats['after_sampling'] = get_dataset_statistics(df, "After Sampling")
            case_lengths = get_case_lengths(df)
            stats['after_sampling'].update({
                'avg_trace_length': case_lengths['length'].mean(),
                'median_trace_length': case_lengths['length'].median(),
                'max_trace_length': case_lengths['length'].max(),
                'min_trace_length': case_lengths['length'].min()
            })
    
    # Filter the DataFrame based on trace length and number of traces
    df_filtered = filter_log_efficiently(df, min_len=min_len, max_len=max_len, 
                                       n_traces=n_traces, random_seed=random_seed)
    
    if stats is not None:
        stats['final'] = get_dataset_statistics(df_filtered, "After Length Filtering")
        case_lengths = get_case_lengths(df_filtered)
        stats['final'].update({
            'avg_trace_length': case_lengths['length'].mean(),
            'median_trace_length': case_lengths['length'].median(),
            'max_trace_length': case_lengths['length'].max(),
            'min_trace_length': case_lengths['length'].min()
        })
    
    return df_filtered, map_dict


def get_dataset_statistics(df: pd.DataFrame, label: str = "") -> Dict[str, Union[int, float]]:
    """
    Calculate and return comprehensive dataset statistics including trace variants and length metrics.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'case:concept:name' and 'concept:name' columns
        label (str): Label for logging purposes
    
    Returns:
        Dict[str, Union[int, float]]: Dictionary containing:
            - n_traces: Number of traces
            - n_activities: Number of unique activities
            - n_variants: Number of unique trace variants
            - variant_trace_ratio: Ratio of variants to traces
            - avg_trace_length: Average length of traces
            - median_trace_length: Median length of traces
            - min_trace_length: Minimum trace length
            - max_trace_length: Maximum trace length
            - std_trace_length: Standard deviation of trace lengths
            
    Raises:
        ValueError: If required columns are missing in the DataFrame
    """
    # Input validation
    required_cols = ['case:concept:name', 'concept:name']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Check if the DataFrame is empty
    if df.empty:
        warnings.warn("The input DataFrame is empty. Returning default statistics.", UserWarning)
        # Return default statistics
        stats = {
            'n_traces': 0,
            'n_activities': 0,
            'n_variants': 0,
            'variant_trace_ratio': np.nan,  # undefined when there are no traces
            'avg_trace_length': np.nan,
            'median_trace_length': np.nan,
            'min_trace_length': np.nan,
            'max_trace_length': np.nan,
            'std_trace_length': np.nan
        }
        print(f"\n{label} Dataset Statistics:")
        print("The dataset is empty. No statistics to display.")
        return stats
    
    # Basic statistics
    n_traces = df['case:concept:name'].nunique()
    n_activities = df['concept:name'].nunique()
    
    # Calculate unique variants
    variants = df.groupby('case:concept:name')['concept:name'].agg(tuple)
    n_variants = len(set(variants))
    # Safe division: if n_traces is zero (should not happen here), set ratio to NaN
    variant_trace_ratio = n_variants / n_traces if n_traces > 0 else np.nan
    
    # Calculate trace length statistics
    trace_lengths = df.groupby('case:concept:name').size()
    avg_length = trace_lengths.mean()
    min_length = trace_lengths.min()
    max_length = trace_lengths.max()
    
    # Print detailed statistics
    print(f"{label} Dataset Statistics:")
    print(f"Number of traces: {n_traces}")
    print(f"Number of unique activities: {n_activities}")
    print(f"Number of unique trace variants: {n_variants}")

    print("\nTrace Length Statistics:")
    print(f"Average length: {avg_length:.2f}")
    print(f"Min length: {min_length}")
    print(f"Max length: {max_length}\n")
    
    return {
        'n_traces': n_traces,
        'n_activities': n_activities,
        'n_variants': n_variants,
        'variant_trace_ratio': variant_trace_ratio,
        'avg_trace_length': avg_length,
        'min_trace_length': min_length,
        'max_trace_length': max_length,
    }
    

def compare_window_based_baselines(
    df_name: Union[str, pd.DataFrame] = '',
    model_path: str = '',    
    n_train_traces: int = None,
    n_test_traces: int = None,
    train_cases: List[str] = None,
    test_cases: List[str] = None,
    window_lengths_lst: List[int] = None,
    n_final_markings_lst: List[int] = None,
    only_return_model: bool = False,
    window_overlap: int = 0,
    read_model_from_file: bool = False,
    non_sync_penalty: int = 1,
    allow_variant_intersection: bool = True,
    cost_function: Any = None,
    use_heuristics: bool = False,
    max_len: int = None,
    min_len: int = None,
    n_traces: int = None,
    random_seed: int = 304,
    map_dict: Dict = None,
    return_model: bool = False,  # no longer used in return selection
    data_path: str = None,
    subfolder: str = None,
    print_dataset_stats: bool = False,
    unique_train_variants: bool = False,  
    unique_test_variants: bool = False,
    max_samples_per_activity: Optional[int] = None,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    return_test_df: bool = False,
    portion: float = 0.2,
    nonsync_density_tolerance: float = 0.1,
    use_memo: bool = False,
    max_successive_merges: int = 5
) -> Union[
    Tuple[pd.DataFrame, Any],
    Tuple[pd.DataFrame, Any, pd.DataFrame]
]:
    """
    Compare window-based baselines for process mining conformance checking
    with enhanced dataset statistics tracking (statistics are always collected).
    Supports both automatic train-test splitting and direct DataFrame inputs.
    
    Data Splitting Priority:
      - If train_df and test_df are provided, they are used directly.
      - If train_cases or test_cases are provided, they are used for splitting.
      - Otherwise, random splitting is applied based on n_train_traces and n_test_traces.
    
    Args:
        df_name (Union[str, pd.DataFrame]): Input event log file path or DataFrame.
        model_path (str): Path to process model file when reading from file.
        n_train_traces (int, optional): Number of traces for training.
        n_test_traces (int, optional): Number of traces for testing.
        train_cases (List[str], optional): Specific case IDs for training.
        test_cases (List[str], optional): Specific case IDs for testing.
        window_lengths_lst (List[int], optional): List of window lengths to evaluate.
        n_final_markings_lst (List[int], optional): List of final markings to consider.
        only_return_model (bool): If True, only returns the model without checking.
        window_overlap (int): Number of overlapping events between windows.
        read_model_from_file (bool): Whether to read model from file instead of training.
        non_sync_penalty (int): Penalty for non-synchronous moves.
        allow_variant_intersection (bool): Allow same variants in train and test.
        cost_function (Any): Custom cost function for conformance checking.
        use_heuristics (bool): Whether to use heuristics in model checking.
        max_len (int, optional): Maximum trace length to include.
        min_len (int, optional): Minimum trace length to include.
        n_traces (int, optional): Total number of traces to use.
        random_seed (int): Seed for reproducibility.
        map_dict (Dict): Activity to model element mapping.
        explor_reward (float): Exploration reward factor.
        return_model (bool): (Deprecated) The model is always returned.
        data_path (str, optional): Base path for data files.
        subfolder (str, optional): Subfolder within data_path.
        print_dataset_stats (bool): Whether to attach dataset statistics to the result.
        unique_train_variants (bool): Ensure unique variants in training.
        unique_test_variants (bool): Ensure unique variants in testing.
        max_samples_per_activity (int, optional): Maximum samples per activity.
        train_df (pd.DataFrame, optional): Pre-split training DataFrame.
        test_df (pd.DataFrame, optional): Pre-split test DataFrame.
        return_test_df (bool): Whether to include the test_df in the returned tuple.

    Returns:
        Tuple:
          - Always returns a tuple (res_df, model), where:
              - res_df is the results DataFrame (with dataset statistics attached if print_dataset_stats=True)
              - model is the process model used.
          - If return_test_df is True, returns (res_df, model, test_df).
    
    Raises:
        ValueError: For invalid input combinations.
    """

    # --- Helper Functions ---

    def validate_inputs():
        if train_df is not None and test_df is None:
            raise ValueError("If train_df is provided, test_df must also be provided")
        if test_df is not None and train_df is None:
            raise ValueError("If test_df is provided, train_df must also be provided")
        if not df_name and train_df is None and test_df is None:
            raise ValueError("Either df_name or both train_df and test_df must be provided")

    def load_data() -> pd.DataFrame:
        nonlocal map_dict  # update the mapping dictionary if not provided
        if isinstance(df_name, pd.DataFrame):
            return df_name
        else:
            df, new_map = load_and_preprocess_log(
                df_name,
                min_len=min_len,
                max_len=max_len,
                n_traces=n_traces,
                random_seed=random_seed,
                path=data_path,
                subfolder=subfolder,
                max_samples_per_activity=max_samples_per_activity,
                stats=None
            )
            if map_dict is None:
                map_dict = new_map
            return df

    def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(random_seed)
        if train_cases is not None or test_cases is not None:
            # Determine training cases.
            if train_cases is not None:
                selected_train = train_cases
            else:
                all_cases = df['case:concept:name'].unique()
                available_cases = list(set(all_cases) - set(test_cases)) if test_cases is not None else all_cases
                if n_train_traces is not None:
                    if n_train_traces > len(available_cases):
                        warnings.warn(
                            f"Requested {n_train_traces} training traces but only {len(available_cases)} are available. "
                            "Using all available traces."
                        )
                        selected_train = available_cases
                    else:
                        selected_train = np.random.choice(available_cases, size=n_train_traces, replace=False)
                else:
                    selected_train = available_cases
            train_df_local = df[df['case:concept:name'].isin(selected_train)].copy()

            # Determine test cases.
            if test_cases is not None:
                test_df_local = df[df['case:concept:name'].isin(test_cases)].copy()
            else:
                used_train = set(train_df_local['case:concept:name'].unique())
                available_cases = list(set(df['case:concept:name'].unique()) - used_train)
                if n_test_traces is not None:
                    if n_test_traces > len(available_cases):
                        warnings.warn(
                            f"Requested {n_test_traces} test traces but only {len(available_cases)} are available. "
                            "Using all available traces."
                        )
                        selected_test = available_cases
                    else:
                        selected_test = np.random.choice(available_cases, size=n_test_traces, replace=False)
                else:
                    selected_test = available_cases
                test_df_local = df[df['case:concept:name'].isin(selected_test)].copy()
            return train_df_local, test_df_local
        else:
            result = train_test_log_split_simplified(
                df, 
                n_train_traces=n_train_traces, 
                n_test_traces=n_test_traces,
                random_seed=random_seed,
                allow_variant_intersection=allow_variant_intersection,
                unique_train_variants=unique_train_variants,
                unique_test_variants=unique_test_variants
            )
            return result['train_df'], result['test_df']

    def compute_statistics(train_df_local: pd.DataFrame, test_df_local: pd.DataFrame) -> Dict:
        stats_local = {}
        stats_local['train'] = get_dataset_statistics(train_df_local, "Training")
        train_lengths = get_case_lengths(train_df_local)
        stats_local['train'].update({
            'avg_trace_length': train_lengths['length'].mean(),
            'median_trace_length': train_lengths['length'].median(),
            'max_trace_length': train_lengths['length'].max(),
            'min_trace_length': train_lengths['length'].min()
        })
        stats_local['test'] = get_dataset_statistics(test_df_local, "Test")
        test_lengths = get_case_lengths(test_df_local)
        stats_local['test'].update({
            'avg_trace_length': test_lengths['length'].mean(),
            'median_trace_length': test_lengths['length'].median(),
            'max_trace_length': test_lengths['length'].max(),
            'min_trace_length': test_lengths['length'].min()
        })
        return stats_local

    def prepare_conformance_model(train_df_local: pd.DataFrame):
        if read_model_from_file:
            if not model_path or not map_dict:
                raise ValueError("Both 'model_path' and 'map_dict' must be provided when reading a model from a file.")
            m = generate_model_from_file(
                model_path,
                activity_mapping_dict=map_dict,
                return_markings=False
            )
        else:
            m = prepare_model(train_df_local, non_sync_penalty)
        if use_heuristics:
            m = add_transition_mappings_to_model(m)
        return m

    # --- Main Flow ---

    # 1. Validate inputs.
    validate_inputs()

    # 2. Load and split data if direct DataFrame inputs are not provided.
    if train_df is None or test_df is None:
        df = load_data()
        train_df_local, test_df_local = split_data(df)
    else:
        train_df_local, test_df_local = train_df, test_df

    # 3. Compute dataset statistics.
    stats = compute_statistics(train_df_local, test_df_local)

    # 4. Select and set up the cost function.
    selected_cost_function = select_cost_function(cost_function)
    np.random.seed(random_seed)

    # 5. Initialize the 'probs' column in test_df.
    test_df_local['probs'] = [[1.0] for _ in range(len(test_df_local))]

    # 6. Prepare the process model.
    model = prepare_conformance_model(train_df_local)

    # 7. Run window-based conformance checking (statistics are always collected).
    
    results = wcb_perform_conformance_checking(
        test_df_local,
        model,
        window_lengths_lst,
        n_final_markings_lst,
        use_heuristics,
        selected_cost_function,
        True,
        portion,
        nonsync_density_tolerance,
        use_memo,
        max_successive_merges
    )

    # 8. Convert results to a DataFrame and attach dataset statistics if requested.
    res_df = wcb_convert_to_dataframe(results)
    summary_df = create_window_summary(res_df)
    
    # 9. Always return (res_df, model) and optionally add test_df.
    if return_test_df:
        return summary_df, model, results, test_df_local
    else:
        return summary_df, model, results


def wcb_perform_conformance_checking(
    test_df: pd.DataFrame,
    net: Any,
    window_lengths_lst: List[int],
    n_final_markings_lst: List[int],
    use_heuristics: bool,
    cost_function: Any,
    collect_statistics: bool,
    portion: float,
    nonsync_density_tolerance: float,
    use_memo: bool,
    max_successive_merges: int
) -> Dict[str, DefaultDict]:
    """
    Performs window-based conformance checking over a set of traces and parameter variants.
    Results are collected per-case and per-variant for downstream analysis.

    Parameters
    ----------
    test_df : pd.DataFrame
        DataFrame containing event log traces, indexed by 'case:concept:name'.
    net : Any
        Process model (Petri net or compatible object) to check conformance against.
    window_lengths_lst : List[int]
        List of window lengths to evaluate.
    n_final_markings_lst : List[int]
        List of numbers of final markings to evaluate.
    explor_reward : float
        Exploration reward parameter for the conformance algorithm.
    use_heuristics : bool
        Whether to enable heuristic guidance in the conformance algorithm.
    cost_function : Any
        Cost function to use in conformance computation.
    collect_statistics : bool
        Whether to collect and store extended statistics (alignments, trace length, etc.).
    portion : float
        Portion of trace to include in each window for conformance checking.
    nonsync_density_tolerance : float
        Non-synchronous move density tolerance parameter.
    use_memo : bool
        Whether to use memoization for repeated subproblem solutions.

    Returns
    -------
    Dict[str, DefaultDict]
        Nested dictionary of results, indexed by parameter configuration.
    """
    results: DefaultDict = defaultdict(lambda: defaultdict(list))
    trace_dict, _ = group_similar_traces(test_df)
    total_traces = len(trace_dict)

    for window_len in window_lengths_lst:
        memo = ConformanceMemo(window_length=window_len, n_model_places=len(net.places)) if use_memo else None

        for n_markings in n_final_markings_lst:
            print(f"Evaluating variant: n_markings={n_markings}, window_len={window_len}")

            for idx, (_, cases) in enumerate(trace_dict.items(), 1):
                print(f"\rComputing trace {idx}/{total_traces}", end="")

                representative_case = cases[0]
                trace_df = test_df[test_df['case:concept:name'] == representative_case]

                start_time = time.time()
                dist, full_alignment, nodes_opened = horizon_based_conformance_with_backtrack(
                    net=net,
                    trace_df=trace_df,
                    window_len=window_len,
                    cost_function=cost_function,
                    partial_conformance=True,
                    use_heuristics=use_heuristics,
                    portion=portion,
                    nonsync_density_tolerance=nonsync_density_tolerance,
                    memo=memo,
                    max_successive_merges=max_successive_merges
                )
                computation_time = time.time() - start_time
                trace_length = len(trace_df)

                for case in cases:
                    if collect_statistics:
                        wcb_update_results(
                            case=case,
                            results=results,
                            n_markings=n_markings,
                            window_len=window_len,
                            dist=dist,
                            nodes_opened=nodes_opened,
                            computation_time=computation_time,
                            full_alignment=full_alignment,
                            trace_length=trace_length
                        )
                    else:
                        wcb_update_results_basic(
                            case=case,
                            results=results,
                            n_markings=n_markings,
                            window_len=window_len,
                            dist=dist,
                            nodes_opened=nodes_opened,
                            computation_time=computation_time
                        )

            print()  # Newline after progress for this variant

            if window_len is None:
                break  # Early termination if window length is not specified

        if use_memo and memo is not None:
            memo.print_stats()

    return results


def wcb_update_results(case, results, n_markings, window_len, dist, nodes_opened, computation_time, full_alignment, trace_length):
    key = f'window_{window_len}_markings_{n_markings}'
    
    # Initialize the window number starting from 1
    window_number = 1
    
    # Initialize a counter for sync/trace moves within a window
    sync_trace_counter = 0
    
    # Capture detailed statistics for each transition in the alignment
    steps = []
    window_numbers = []
    transition_names = []
    transition_weights = []
    
    # Initialize the step counter for each window
    trace_step_counter = 1
    
    for transition in full_alignment:
        # Always increment the step counter for each move and assign it
        steps.append(trace_step_counter)
        trace_step_counter += 1
        
        # Increment sync_trace_counter only for sync and trace moves
        if transition.move_type in ['trace', 'sync']:
            sync_trace_counter += 1
        
        # Assign the current window number to this step
        window_numbers.append(window_number)

        # Capture transition details
        transition_names.append(transition.name if hasattr(transition, 'name') else str(transition))
        transition_weights.append(round(transition.weight, 2) if hasattr(transition, 'weight') else None)

        # Check if we've reached the window length for sync/trace moves
        if sync_trace_counter == window_len:
            window_number += 1
            trace_step_counter = 1  # Reset the step counter for the new window
            sync_trace_counter = 0  # Reset the sync/trace move counter for the new window
    
    # Update results dictionary
    results[key]['case_id'].append(case)
    results[key]['cost'].append(dist)
    results[key]['time'].append(computation_time)
    results[key]['nodes_opened'].append(nodes_opened)
    results[key]['steps'].append(steps)
    results[key]['window_number'].append(window_numbers)
    results[key]['transition_names'].append(transition_names)
    results[key]['transition_weights'].append(transition_weights)


def wcb_convert_to_dataframe(data):
    """
    Converts a nested dictionary with multiple outer keys into a pandas DataFrame.
    Rounds the 'cost' and 'time' columns to 5 decimal places and prefixes these columns with the outer key.
    Retains additional statistics like steps, window numbers, transition names, and transition weights.
    
    Args:
        data (dict): A dictionary where the keys are outer keys and the values are dictionaries containing lists.
    
    Returns:
        pd.DataFrame: A DataFrame with prefixed columns for each outer key.
    """
    all_dfs = []
    for key, inner_dict in data.items():
        # Retain the relevant columns
        filtered_dict = {
            'case_id': inner_dict.get('case_id'),
            'cost': inner_dict.get('cost'),
            'time': inner_dict.get('time'),
            'nodes_opened': inner_dict.get('nodes_opened'),
            'steps': inner_dict.get('steps'),
            'window_number': inner_dict.get('window_number'),
            'transition_names': inner_dict.get('transition_names'),
            'transition_weights': inner_dict.get('transition_weights')
        }
        
        # Convert the filtered dictionary to a DataFrame
        df = pd.DataFrame(filtered_dict)
        
        # Expand lists into separate rows (one row per transition)
        df = df.explode(['steps', 'window_number', 'transition_names', 'transition_weights']).reset_index(drop=True)
        
        # Round the 'cost' and 'time' columns to 5 decimal places, if they exist
        if 'cost' in df.columns:
            df['cost'] = df['cost'].round(5)
        if 'time' in df.columns:
            df['time'] = df['time'].round(5)
        
        # Prefix columns with the outer key, except for 'case_id' and detailed statistics
        df = df.rename(columns={
            'cost': f'{key}_cost',
            'time': f'{key}_time',
            'nodes_opened': f'{key}_nodes_opened'
        })
        
        # Append the DataFrame to the list
        all_dfs.append(df)
    
    # Merge all DataFrames on 'case_id'
    final_df = pd.concat(all_dfs, axis=1)
    
    # Remove duplicate 'case_id' columns that might appear due to concatenation
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    
    return final_df


def wcb_update_results_basic(case, results, n_markings, window_len, dist, nodes_opened, computation_time):
    key = f'window_{window_len}_markings_{n_markings}'
    results[key]['case_id'].append(case)
    results[key]['cost'].append(dist)
    results[key]['time'].append(computation_time)
    results[key]['nodes_opened'].append(nodes_opened)
    # This function only captures basic statistics, not detailed per-transition stats


def run_algorithm_with_timeout(sync_prod, algorithm: str, timeout: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Runs a specified search algorithm on the provided sync product with a timeout.
    Handles exceptions and returns None if the algorithm fails or times out.
    Includes the computation time in the results.
    """
    result = {}
    
    def target():
        try:
            start_time = time.time()  # Start the timer
            if algorithm == 'astar':
                result['alignment'], result['cost'], result['nodes_popped'] = sync_prod.astar_search()
            elif algorithm == 'astar_extended':
                result['alignment'], result['cost'], result['nodes_popped'] = sync_prod.astar_incremental()
            elif algorithm == 'reach':
                result['alignment'], result['cost'], result['nodes_popped'] = sync_prod.reach_search()
            end_time = time.time()  # End the timer
            result['time'] = end_time - start_time  # Calculate the elapsed time
        except Exception as e:
            result['error'] = e
    
    thread = Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        print(f"Timeout occurred for {algorithm}.")
        return None
    
    if 'error' in result:
        print(f"Error occurred for {algorithm}: {result['error']}")
        return None
    
    return {
        'cost': result.get('cost'),
        'nodes_popped': result.get('nodes_popped'),
        'time': result.get('time')
    }


def process_pickle_file(pickle_file_path):
    """
    Process the data from a pickle file to create a DataFrame and print trace information.

    Args:
        pickle_file_path (str): Path to the pickle file.

    Returns:
        pd.DataFrame: DataFrame containing the processed data.
    """
    # Open the file in binary read mode
    with open(pickle_file_path, 'rb') as file:
        # Load the content of the file into a Python object
        data = pickle.load(file)

    # Print the number of unique traces
    print(f'number of unique traces={count_unique_arrays(data)}')

    rows_list = []

    # Iterate through the array of arrays
    for case_index, case_activities in enumerate(data):
        # Iterate through each activity in the case
        for activity in case_activities:
            # Append a tuple (case_id, activity) to the list
            rows_list.append((str(case_index), str(activity)))

    # Create a DataFrame from the list of tuples
    df = pd.DataFrame(rows_list, columns=['case:concept:name', 'concept:name'])

    # Print trace lengths list from DataFrame
    print_trace_lengths_list_from_df(df)

    return df


def save_results(results, filepath):
    """
    Save the results to a CSV file.

    Parameters:
    results (Any): The results data that you want to save. It should be convertible to a pandas DataFrame.
    filepath (str): The file path where the results should be saved, including the filename and .csv extension.
    """
    # Assuming `results` is in a format that can be directly converted to a DataFrame
    df = pd.DataFrame(results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)
    print(f"Results successfully saved to {filepath}")


def calculate_model_metrics(
    df_accepted,
    net,
    init_marking,
    final_marking,
) -> Tuple[float, float]:
    """
    Calculate alignment-based fitness and precision metrics for a Petri net model.
    
    Args:
        df_accepted (EventLog): The event log to evaluate
        net (PetriNet): The Petri net model
        init_marking (Marking): Initial marking of the Petri net
        final_marking (Marking): Final marking of the Petri net
        logger (Logger): Logger instance for output
        
    Returns:
        Tuple[float, float]: A tuple containing (fitness_value, precision_value)
    """
    # Calculate alignment-based fitness
    fitness = fitness_evaluator.apply(
        df_accepted, 
        net, 
        init_marking, 
        final_marking,
        variant=fitness_evaluator.Variants.ALIGNMENT_BASED
    )
    fitness_value = fitness['average_trace_fitness'] if isinstance(fitness, dict) else fitness
    
    # Calculate precision using Align-ETConformance
    precision_val = precision_evaluator.apply(
        df_accepted, 
        net, 
        init_marking, 
        final_marking,
        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE
    )
    precision_value = precision_val['precision'] if isinstance(precision_val, dict) else precision_val
    
    print(f"Fitness = {fitness_value:.4f}")
    print(f"Precision = {precision_value:.4f}")


def sample_trace_activities(
    df: pd.DataFrame, 
    max_samples_per_activity: int
) -> pd.DataFrame:
    """
    Samples up to `max_samples_per_activity` consecutive activities for each unique activity
    within every trace, preserving the sequential structure.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing columns 'case:concept:name' and 'concept:name'.
    max_samples_per_activity : int
        Maximum number of consecutive samples per activity within a trace.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sampled activities, maintaining the input columns.

    Raises
    ------
    ValueError
        If `max_samples_per_activity` < 1, or if required columns are missing.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'case:concept:name': [0, 0, 0, 0, 0],
    ...     'concept:name': ['a', 'a', 'a', 'b', 'b']
    ... })
    >>> sample_trace_activities(df, max_samples_per_activity=2)
       case:concept:name concept:name
    0                 0            a
    1                 0            a
    2                 0            b
    3                 0            b
    """
    if max_samples_per_activity < 1:
        raise ValueError("max_samples_per_activity must be at least 1.")

    required_cols = ['case:concept:name', 'concept:name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns: {required_cols} (missing: {missing_cols})")

    sampled_case_ids: List = []
    sampled_activities: List = []

    for case_id, group in df.groupby('case:concept:name', sort=False):
        prev_activity = None
        count = 0

        for activity in group['concept:name']:
            if activity != prev_activity:
                count = 0  # Reset for a new activity group
            if count < max_samples_per_activity:
                sampled_case_ids.append(case_id)
                sampled_activities.append(activity)
                count += 1
            prev_activity = activity

    return pd.DataFrame({
        'case:concept:name': sampled_case_ids,
        'concept:name': sampled_activities
    })


def get_case_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the length (number of activities) for each case in the log.
    
    Args:
        df (pd.DataFrame): The input DataFrame with columns 'case:concept:name' and 'concept:name'.
        
    Returns:
        pd.DataFrame: A DataFrame with two columns:
                     - 'case:concept:name': the case ID
                     - 'length': the number of activities in that case
                     
    Raises:
        ValueError: If required columns are missing.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'case:concept:name': [0, 0, 0, 1, 1],
        ...     'concept:name': ['a', 'b', 'c', 'd', 'e']
        ... })
        >>> get_case_lengths(df)
           case:concept:name  length
        0                 0       3
        1                 1       2
    """
    # Input validation
    if 'case:concept:name' not in df.columns:
        raise ValueError("DataFrame must contain column: 'case:concept:name'")
        
    # Group by case ID and count activities
    case_lengths = df.groupby('case:concept:name').size().reset_index()
    case_lengths.columns = ['case:concept:name', 'length']
    
    return case_lengths


def compute_window_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes average cost, time, and nodes opened for each window_length (including 'window_None'),
    ensuring a single value per (case_id, window_length).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns for case_id, window_length, and window-specific metrics
        (e.g., window_{length}_markings_1_cost, _time, _nodes_opened).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: window_length, average_cost, average_time, average_nodes_opened.
    """
    # Extract all unique window lengths available in the DataFrame
    window_lengths: List[str] = sorted({
        col.split('_')[1]
        for col in df.columns
        if col.startswith('window_') and '_markings_1_cost' in col
    })

    results: List[dict] = []

    for length in window_lengths:
        cost_col = f'window_{length}_markings_1_cost'
        time_col = f'window_{length}_markings_1_time'
        nodes_col = f'window_{length}_markings_1_nodes_opened'

        # Only process if all relevant columns exist
        if not all(col in df.columns for col in (cost_col, time_col, nodes_col)):
            continue

        # Select relevant columns and drop missing values for this window
        subset = df[['case_id', cost_col, time_col, nodes_col]].dropna(subset=[cost_col, time_col, nodes_col])

        # Group by case_id to ensure one value per case per window_length
        case_group = subset.groupby('case_id')[[cost_col, time_col, nodes_col]].mean()

        # Compute overall averages, if data is available
        if not case_group.empty:
            averages = case_group.mean()
            results.append({
                'window_length': length if length != 'None' else 'None',
                'average_cost': round(averages[cost_col], 4),
                'average_time': round(averages[time_col], 4),
                'average_nodes_opened': round(averages[nodes_col], 4),
            })

    return pd.DataFrame(results)


def create_performance_plots(
    df_no_heuristic: pd.DataFrame,
    df_heuristic: pd.DataFrame,
    dataset_name: str,
    trace_length: str,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 15)
) -> List[Figure]:
    """
    Generate comparative performance plots for conformance algorithms with and without heuristic.
    
    Parameters
    ----------
    df_no_heuristic : pd.DataFrame
        DataFrame with results for runs without heuristic.
    df_heuristic : pd.DataFrame
        DataFrame with results for runs with heuristic.
    dataset_name : str
        Dataset identifier for plot titles.
    trace_length : str
        Trace length description to annotate plots.
    output_path : Optional[str], default=None
        If provided, directory to save each plot as PNG.
    figsize : Tuple[int, int], default=(10, 15)
        Figure size in inches (width, height).
    
    Returns
    -------
    List[Figure]
        List containing matplotlib Figure objects for each performance metric.
    """
    # Defensive copy to avoid mutation
    df_no_heuristic = df_no_heuristic.copy()
    df_heuristic = df_heuristic.copy()

    # Format window length column
    def format_window_length(x: object) -> str:
        if pd.isna(x):
            return 'Full Window'
        try:
            return str(int(x))
        except (ValueError, TypeError):
            return str(x)
    
    for df in (df_no_heuristic, df_heuristic):
        if 'window_length' in df.columns:
            df['window_length'] = df['window_length'].apply(format_window_length)
    
    # Define metrics and labels for plotting
    metrics = [
        ('average_cost',     'Average Cost Comparison',        'Average Cost'),
        ('average_time',     'Average Time Comparison',        'Average Time (seconds)'),
        ('average_nodes_opened', 'Average Nodes Opened Comparison', 'Average Nodes Opened'),
    ]
    
    # Plot style (set temporarily inside function)
    plt.rcParams.update({
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    figures: List[Figure] = []

    for col, title, y_label in metrics:
        if col not in df_no_heuristic or col not in df_heuristic:
            continue  # Skip missing metrics gracefully
        
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[0] // 2))
        
        # Plot both heuristics' data
        ax.plot(
            df_no_heuristic['window_length'], df_no_heuristic[col],
            marker='o', linewidth=2, label='No Heuristic', color='#8884d8', markersize=8
        )
        ax.plot(
            df_heuristic['window_length'], df_heuristic[col],
            marker='o', linewidth=2, label='With Heuristic', color='#82ca9d', markersize=8
        )
        
        ax.set_title(f"{title}\n{dataset_name}, trace_len {trace_length}", pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel('Window Length', labelpad=10, fontsize=10)
        ax.set_ylabel(y_label, labelpad=10, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        ax.legend(
            bbox_to_anchor=(1.02, 0.5), loc='center left',
            frameon=True, fancybox=True, shadow=True
        )
        ax.tick_params(axis='both', labelsize=9)
        plt.subplots_adjust(right=0.9)
        plt.tight_layout()

        figures.append(fig)

        if output_path:
            fig.savefig(
                f"{output_path}/{col}_comparison.png",
                bbox_inches='tight', dpi=300
            )

    return figures


def dict_to_dataframe(
    results_dict: Dict[str, Any],
    window_key: str,
    case_id_filter: Optional[Union[Any, List[Any]]] = None,
) -> pd.DataFrame:
    """
    Extracts and processes a sub-dictionary from `results_dict` associated with the given `window_key`,
    returning a DataFrame where list-like columns are exploded and all numeric values are rounded to two decimals.

    Parameters
    ----------
    results_dict : Dict[str, Any]
        The results dictionary containing (typically nested) experiment results.
    window_key : str
        A string key (e.g., 'window_15') used to identify the relevant sub-dictionary.
    case_id_filter : Optional[Union[Any, List[Any]]], default=None
        Optional filter for 'case_id'. Can be a single value or a list of values. If provided,
        the returned DataFrame will only include rows where 'case_id' matches one of these values.

    Returns
    -------
    pd.DataFrame
        A processed DataFrame with exploded list columns and all numeric columns rounded to 2 decimals.

    Raises
    ------
    ValueError
        If no matching key is found in `results_dict` containing `window_key`.
    """
    # Find the first matching top-level key containing the window_key
    try:
        inner_key = next(k for k in results_dict if window_key in k)
    except StopIteration:
        raise ValueError(f"No entry found for window key '{window_key}'.")

    inner_dict = results_dict[inner_key]
    df = pd.DataFrame(inner_dict)

    # Specify which columns are list-like and need to be exploded
    list_cols = ['steps', 'window_number', 'transition_names', 'transition_weights']
    present_list_cols = [col for col in list_cols if col in df.columns]

    if present_list_cols:
        df = df.explode(present_list_cols, ignore_index=True)

    # Round all numeric columns to two decimals (vectorized and robust to missing values)
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # Optionally filter by case_id
    if case_id_filter is not None:
        if not isinstance(case_id_filter, list):
            case_id_filter = [case_id_filter]
        df = df[df['case_id'].isin(case_id_filter)]

    return df