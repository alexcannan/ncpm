import random


def node_distance(node1: int, node2: int, n_nodes: int) -> int:
    """ given 2 indices of a cycle, returns the distance between them """
    n1, n2 = sorted([node1, node2])
    return min(n2 - n1, n1 + n_nodes - n2)


def generate_matching(n_nodes: int) -> list[tuple[int]]:
    """ given a number of nodes, generates a non-crossing perfect matching """
    assert n_nodes % 2 == 0, "number of nodes must be even"
    nodes = list(range(n_nodes))
    matches = []
    selected_nodes = set()
    while (remaining_nodes := list(set(nodes) - selected_nodes)):
        if len(remaining_nodes) == 2:
            # XXX: this is a hack, sometimes this gets stuck on last 2 but unsure why
            matches.append(tuple(remaining_nodes))
            break
        pair = sorted(random.sample(remaining_nodes, 2))
        # if any of the nodes in between are already selected, try again
        if any(node in selected_nodes for node in range(*pair)):
            continue
        # if the number of unselected nodes on either side of the pair, before any other selected node, is odd, try again
        left_iter = (pair[0] - 1) % n_nodes
        left_count = 0
        while nodes[left_iter] not in selected_nodes and nodes[left_iter] != pair[1]:
            left_count += 1
            left_iter = (left_iter - 1) % n_nodes
        right_iter = (pair[1] + 1) % n_nodes
        right_count = 0
        while nodes[right_iter] not in selected_nodes and nodes[right_iter] != pair[0]:
            right_count += 1
            right_iter = (right_iter + 1) % n_nodes
        if left_count % 2 == 1 or right_count % 2 == 1:
            continue
        # if the distance between the indices is odd, that means an even number between them, add to matches
        dist = node_distance(*pair, n_nodes=n_nodes)
        if dist == 1:
            # if the distance is 1, try and choose another a certain % of the time
            if random.random() < 0.9:
                continue
        if dist % 2 == 1:
            matches.append(pair)
            selected_nodes.update(pair)
        else:
            continue
    return matches


def square_pair_type(node1: int, node2: int, x_nodes: int, y_nodes: int) -> str:
    """ given 2 indices of a square, returns the type of edge between them

    :returns: "same", "adjacent", "opposite"
    """
    def index_to_edge(index: int, x_nodes: int, y_nodes: int) -> str:
        if index < x_nodes:
            return 0
        elif index < x_nodes + y_nodes:
            return 1
        elif index < 2 * x_nodes + y_nodes:
            return 2
        else:
            return 3
    e1 = index_to_edge(node1, x_nodes, y_nodes)
    e2 = index_to_edge(node2, x_nodes, y_nodes)
    if e1 == e2:
        return "same"
    elif (e1 + 1) % 4 == e2 or (e1 - 1) % 4 == e2:
        return "adjacent"
    else:
        return "opposite"
