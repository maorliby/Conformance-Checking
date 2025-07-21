from RunningHorizon import *
from Classes import *
from utils import *
import pm4py
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.objects.conversion.wf_net.converter import apply as pt_converter
from pm4py.objects.process_tree.obj import ProcessTree, Operator
from pm4py.objects.process_tree.utils import generic as pt_util
from pm4py.objects.petri_net.utils.reachability_graph import construct_reachability_graph as crg
import numpy as np
import copy

def default_info():
    return {'parents': set(), 'reachable_transitions': set()}
    
# Get the leaves of a process tree
def get_leaves(node):
    l = []
    if not node.children:
        if node.label:
            l.append(node.label)
    else:
        for child in node.children:
            l.extend(get_leaves(child))
    return l


# Filter the process tree based on the performed transitions
def filter_tree(node, performed_transitions):
    if not node.children:
        if node.label:
            return node
        else:
            return None

    if node.operator == Operator.LOOP:
        filtered_children = []
        for i in range(len(node.children) - 1):
            res = filter_tree(node.children[i], performed_transitions)
            if res is not None:
                filtered_children.append(res)
        node.children = filtered_children
        if filtered_children:
            return node
        else:
            return None

    elif node.operator == Operator.XOR:
        # Initialize variable to keep track of the child to retain
        favorite_son = None
        
        for child in node.children:
            # Get the leaves of the current child, not the entire node
            leaves = get_leaves(child)
            
            if leaves:
                # Check if any of the performed transitions are in the leaves of this child
                for item in performed_transitions:
                    if item in leaves:
                        favorite_son = child
                        break  # Stop once we find the correct child
            
            if favorite_son:
                break  # No need to check other children if we've found the correct one
    
        if favorite_son:
            # Recursively filter the subtree rooted at the favorite child
            filter_tree(favorite_son, performed_transitions)
            # Set the XOR node's children to only include the favorite child
            node.children = [favorite_son]
            return node
        else:
            # If no child is found, remove this XOR node from the tree
            return None

    else:
        filtered_children = []
        for child in node.children:
            res = filter_tree(child, performed_transitions)
            if res is not None:
                filtered_children.append(res)
        if filtered_children:
            node.children = filtered_children
            return node
        else:
            return None


# The main function of the second heuristic
def get_mandatory_transitions_for_marking(process_tree, performed_transitions):
    process_tree_copy = copy.deepcopy(process_tree)
    
    # Filter the process tree
    filtered_process_tree = filter_tree(process_tree_copy, performed_transitions)

    # Extract mandatory transitions
    if filtered_process_tree:
        mandatory_transitions = set(get_leaves(filtered_process_tree))
        
        # Remove the TAU transitions from the list of mandatory transitions
        mandatory_transitions = {t for t in mandatory_transitions if not t.startswith("!TAU_") and t not in performed_transitions}
        return mandatory_transitions
        
    else:
        return set()


def map_markings_to_mandatory_transitions(net, initial_marking, final_marking):
    # Add a label to tau transitions with a unique prefix
    for transition in net.transitions:
        if transition.label is None:
            transition.label = "!TAU_" + transition.name

    # Construct the process tree
    process_tree = pt_converter(net, initial_marking, final_marking)
    
    markings_dict = {}
    visited = []
    queue = [(initial_marking, [])]

    markings_dict[final_marking] = set()
    
    while queue:
        current_marking = queue.pop(0)
        visited.append(current_marking[0])
        
        if current_marking[0] in markings_dict:
            continue
        
        mandatory_transitions = get_mandatory_transitions_for_marking(process_tree, current_marking[1])
   
        markings_dict[current_marking[0]] = mandatory_transitions

        available_transitions = enabled_transitions(net, current_marking[0])
        for transition in available_transitions:
            new_marking = execute(transition, net, current_marking[0])
            if new_marking not in visited and new_marking not in [q[0] for q in queue]:
                queue.append((new_marking, current_marking[1] + [transition.label]))

    return markings_dict


def compute_mandatory_transitions(net, init_marking, final_marking, places_mapping=None):
    original_result_dict = map_markings_to_mandatory_transitions(net, init_marking, final_marking)

    if places_mapping is None:
        _, places_mapping = from_discovered_model_to_PetriNet(net, return_mapping=True)

    new_result_dict = {}
    for key, value in original_result_dict.items():
        new_marking = convert_marking(key, places_mapping)
        new_result_dict[new_marking.places] = value
    
    return new_result_dict


# iterative version
def map_markings_to_reachable_transitions(model):
    """
    Maps marking places to their reachable transitions in a Petri net model.

    This function performs a breadth-first traversal of the Petri net's reachability graph,
    starting from the initial marking. It tracks information about each marking place,
    including its parent marking places and reachable transitions.

    Args:
        model: The Petri net model to analyze.

    Returns:
        A dictionary mapping markings to their associated details (parents, reachable transitions).
    """

    visited = set()
    initial_marking_node = PetriNetNode(marking_places=model.init_mark.places)
    queue = deque([initial_marking_node])
    marking_details = defaultdict(default_info)

    while queue:
        current_node = queue.popleft()
        marking_places = current_node.marking_places

        if marking_places in visited:
            continue

        visited.add(marking_places)

        available_transitions = model._find_available_transitions(marking_places)
        available_transitions_labels = {t.label for t in available_transitions if t.label}
        update_reachable_transitions(marking_places, marking_details, available_transitions_labels)

        for transition in available_transitions:
            successor_marking = model._fire_transition(marking_places, transition)
            successor_places = successor_marking.places
            marking_details[successor_places]['parents'].add(marking_places)

            if successor_places not in visited:
                new_node = PetriNetNode(parent=current_node, transition_to_parent=transition, marking_places=successor_places)
                queue.append(new_node)
            
            else:
                update_reachable_transitions(marking_places, marking_details,
                                             marking_details[successor_places]['reachable_transitions'])

    return marking_details

    
def update_reachable_transitions(marking_places, marking_details, available_transitions_labels):
    """
    Iteratively updates reachable transitions for a set of marking places and their ancestors.

    Args:
        marking_places: A tuple representing a marking within the process model whose reachable transitions are to be updated.
        marking_details: A dictionary containing marking details for each marking.
        available_transitions_labels: The set of available transition labels.
    """
    
    stack = [marking_places]  # Stack of current markings
    visited = set()

    while stack:
        current_marking = stack.pop()
        
        if current_marking in visited:
            continue

        visited.add(current_marking)

        try:
            marking_details[current_marking]['reachable_transitions'].update(available_transitions_labels)
        except KeyError:
            print(f"Warning: Marking place '{current_marking}' not found in marking details.")
            continue

        for parent_marking in marking_details[current_marking]['parents']:
            if parent_marking not in visited:
                stack.append(parent_marking) 