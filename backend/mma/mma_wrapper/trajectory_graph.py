import numpy as np
import copy
import itertools
import random
import networkx as nx
import matplotlib.pyplot as plt
import re

from typing import Any, Dict, List, Tuple, Union

from mma_wrapper.utils import cardinality, draw_networkx_edge_labels

observation = Any
action = Any
label = str
trajectory = List[Tuple[observation, action]]
trajectory_pattern = str


class trajectory_graph:
    """A class to represent a set of histories intended to be related to a single subset of organizational specifications for a single agent.
    """

    graph: Dict[label, Dict[label, Dict[int, cardinality]]]

    ordinal_counter: int

    non_optional_start_end: Tuple[label, label]

    def __init__(self, graph: Any = {}, non_optional_start_end=(None, None)):
        self.graph = graph
        self.ordinal_counter = 0
        self.non_optional_start_end = non_optional_start_end

    def to_dict(self) -> Dict:
        return {"graph": self.graph, "non_optional_start_end": self.non_optional_start_end}

    @staticmethod
    def from_dict(obj: Dict) -> 'trajectory_graph':
        return trajectory_graph(obj["graph"], obj["non_optional_start_end"])

    def __str__(self) -> str:
        return str(self.graph)

    def __repr__(self) -> str:
        return str(self.graph)

    def __eq__(self, other: 'trajectory_graph') -> bool:
        return self.graph == other.graph

    def add_observations_to_actions(self, observation_labels: List[label], action_labels: List[label]):
        """Restrict trajectory subset to those where any of the given observations is followed by any of the given actions.

        Parameters
        ----------
        observation_labels : List[label]
            The given observation labels

        action_labels : List[label]
            The given action labels

        Returns
        -------
        None

        Examples
        --------
        >>> hs = trajectory_graph()
        >>> hs.add_observations_to_actions(["obs1","obs2"], ["act1","act2"])

        See Also
        --------
        None
        """
        for label1 in observation_labels:
            for label2 in action_labels:
                self.graph.setdefault(
                    label1, {})
                self.graph[label1].setdefault(
                    label2, {})
                self.graph[label1][label2][self.ordinal_counter] = None
        self.ordinal_counter += 1

    def add_actions_to_observations(self, action_labels: List[label], observation_labels: List[label]):
        """Restrict trajectory subset to those where any of the given actions is followed by any of the given observations.

        Parameters
        ----------
        action_labels : List[label]
            The given action labels

        observation_labels : List[label]
            The given observation labels

        Returns
        -------
        None

        Examples
        --------
        >>> hs = trajectory_graph()
        >>> hs.add_actions_to_observations(["act1","act2"], ["obs1","obs2"])

        See Also
        --------
        None
        """
        for label1 in observation_labels:
            for label2 in action_labels:
                self.graph.setdefault(
                    label1, {})
                self.graph[label1].setdefault(
                    label2, {})
                self.graph[label1][label2][self.ordinal_counter] = None
        self.ordinal_counter += 1

    def add_labels_to_labels(self, src_labels: List[Union[label, label]],
                             dst_labels: List[Union[label, label]],
                             src_to_dst_cardinality: cardinality = None):
        """Restrict trajectory subset to those where any of the given actions is followed by any of the given observations.

        Parameters
        ----------
        src_labels : List[Union[label,label]]
            The given source labels

        dst_labels : List[Union[label,label]]
            The given destination labels

        Returns
        -------
        None

        Examples
        --------
        >>> hs = trajectory_graph()
        >>> hs.add_labels_to_labels(["act1","act2"], ["obs1","obs2"])

        See Also
        --------
        None
        """
        for src_label in src_labels:
            for dst_label in dst_labels:
                self.graph.setdefault(
                    src_label, {})
                self.graph[src_label].setdefault(
                    dst_label, {})
                self.graph[src_label][dst_label][self.ordinal_counter] = src_to_dst_cardinality
        self.ordinal_counter += 1

    def get_fathers(self, label: Union[label, label]) -> List[Union[label, label]]:
        """Get the fathers of given label
        """
        father_labels = []
        for label1 in self.graph:
            if label in list(self.graph.get(label1).keys()):
                father_labels += [label1]
        return father_labels

    def get_sons(self, label: Union[label, label]) -> List[Union[label, label]]:
        """Get the sons of given label
        """
        return list(self.graph.get(label).keys())

    def add_trajectory(self, trajectory: trajectory):
        """Add the given trajectory in the trajectory subset.

        Parameters
        ----------
        trajectory : trajectory
            The given trajectory

        Returns
        -------
        None

        Examples
        --------
        >>> hs = trajectory_graph()
        >>> hs.add_trajectory([("obs1","act1"),("obs2","act2")])

        See Also
        --------
        None
        """
        last_act = None
        for obs, act in trajectory:
            if last_act is not None:
                self.add_actions_to_observations([last_act], [obs])
            self.add_observations_to_actions([obs], [act])
            last_act = act

    @staticmethod
    def parse_into_tuple(trajectory_string: trajectory_pattern) -> Tuple[List, cardinality]:

        def uniformize(string_pattern: str) -> str:

            # lonely labels in the begining of sequences
            regex = r'\[([\",0-9A-Za-z]{2,}?),\('
            matches = re.findall(regex, string_pattern)

            for group in matches:
                if group != "":
                    string_pattern = string_pattern.replace(
                        '[' + group + ',(', '[' + f"([{group}],('1','1'))" + ',(')

            # lonely labels in the middle of sequences
            regex = r'\)\),([\",0-9A-Za-z]{2,}?),\('
            matches = re.findall(regex, string_pattern)

            for group in matches:
                if group != "":
                    string_pattern = string_pattern.replace(
                        ')),' + group + ',(', ')),' + f"([{group}],('1','1'))" + ',(')

            # lonely labels in the end of sequences
            regex = r'\),([\",0-9A-Za-z]{2,}?)\],\('
            matches = re.findall(regex, string_pattern)

            for group in matches:
                if group != "":
                    string_pattern = string_pattern.replace(
                        '),' + group + '],(', '),' + f"([{group}],('1','1'))" + '],(')

            return string_pattern

        def convert_to_tuple(string_pattern: str) -> Tuple[List, cardinality]:
            stack = []
            i = 0
            while i < len(string_pattern):
                character = string_pattern[i]
                if character == "[":
                    stack += ["["]
                elif character == "]":
                    stack[-1] += "]"
                    if string_pattern[i+1] == "(":
                        card = ""
                        i += 1
                        while i < len(string_pattern):
                            char_card = string_pattern[i]
                            card += char_card
                            if char_card == ")":
                                break
                            i += 1
                        sequence = stack.pop()
                        sequence = f'({sequence},{card})'
                        if (i == len(string_pattern) - 1):
                            seq = sequence.replace("[", "[\"").replace("]", "\"]") \
                                .replace("(", "(\"").replace(")", "\")").replace(",", "\",\"") \
                                .replace("\"(", "(").replace(")\"", ")") \
                                .replace("\"[", "[").replace("]\"", "]")
                            seq = uniformize(seq)
                            return eval(seq)
                        stack[-1] += sequence
                else:
                    stack[-1] += character
                i += 1
        return convert_to_tuple(trajectory_string)

    @staticmethod
    def is_only_labels(tuple_pattern: Tuple[List, cardinality]) -> bool:
        label_or_tuple_list, card = tuple_pattern
        for label_or_tuple in label_or_tuple_list:
            if type(label_or_tuple) == tuple:
                return False
        return True

    def add_pattern(self, trajectory_pattern: trajectory_pattern):
        """Restrict trajectory subset to those matching the given sequence pattern.

        Parameters
        ----------
        trajectory : str
            The given sequence pattern

        Returns
        -------
        None

        Examples
        --------
        >>> hs = trajectory_graph()
        >>> hs.add_pattern("[obs1,act2,[obs4,act4](0,1)](1,*),obs2,[act2|act4],obs3,act3](1,*)")

        See Also
        --------
        None
        """

        tuple_pattern = trajectory_graph.parse_into_tuple(trajectory_pattern)

        self.last_label = None
        self.seq_start_label = None
        self.seq_end_label = None

        def parse_into_graph(tuple_pattern: Tuple[List, cardinality], optional_way: bool = False) -> Tuple[str, str]:

            if trajectory_graph.is_only_labels(tuple_pattern):

                labels_sequence, labels_card = tuple_pattern

                for i, label in enumerate(labels_sequence):
                    if self.last_label is None:
                        self.last_label = label
                    else:
                        if optional_way and i == 0:
                            self.add_labels_to_labels(
                                [self.last_label], [label], src_to_dst_cardinality=cardinality(0, 1))
                        else:
                            self.add_labels_to_labels(
                                [self.last_label], [label])
                        self.last_label = label
                if not ((labels_card[0] == "1" and labels_card[1] == "1") or (labels_card[0] == "0" and labels_card[1] == "0") or (labels_card[0] == "0" and labels_card[1] == "1")):
                    self.add_labels_to_labels([self.last_label], [labels_sequence[0]],
                                              src_to_dst_cardinality=cardinality(labels_card[0], labels_card[1]))
                return labels_sequence[0], labels_sequence[-1]

            else:
                tuples_sequence, sequences_card = tuple_pattern
                start_label = copy.copy(self.last_label)

                optional_way_start = []

                seq_start_label = None
                seq_end_label = None

                for i, sub_tuple in enumerate(tuples_sequence):

                    sub_tuple_card = sub_tuple[1]
                    old_length = len(optional_way_start)
                    if sub_tuple_card[0] == "0" and i > 0:
                        if len(optional_way_start) == 0:
                            optional_way_start += [(copy.copy(self.last_label), i)]
                        elif len(optional_way_start) > 0 and optional_way_start[-1][1] > i - 1:
                            optional_way_start += [(copy.copy(self.last_label), i)]
                    sl, el = parse_into_graph(sub_tuple, optional_way=(
                        len(optional_way_start) - old_length > 0) or optional_way)

                    if len(optional_way_start) > 0 and optional_way_start[-1][0] is None:
                        optional_way_start[-1] = copy.copy(
                            (sl, optional_way_start[-1][1]))

                    if i == 0:
                        start_label = sl

                    if not (sub_tuple_card[0] == "0") and seq_start_label is None:
                        seq_start_label = copy.copy(sl)

                    if not (sub_tuple_card[0] == "0"):
                        seq_end_label = copy.copy(el)

                    if len(optional_way_start) - old_length == 0:
                        optional_way = False

                    # and sub_tuple_card[0] != "0":
                    if len(optional_way_start) > 0 and optional_way_start[0][1] != i:
                        if sub_tuple_card[0] != "0":
                            self.add_labels_to_labels(
                                [optional_way_start.pop(0)[0]], [sl])

                if not ((sequences_card[0] == "1" and sequences_card[1] == "1")) and not (sequences_card[0] == "0"):
                    self.add_labels_to_labels([self.last_label], [start_label],
                                              src_to_dst_cardinality=cardinality(sequences_card[0], sequences_card[1]))

                return seq_start_label, seq_end_label

        self.non_optional_start_end = parse_into_graph(tuple_pattern)

    def sample(self, seed: int = 42) -> trajectory:
        # TODO: Finalize the sample code
        hist: trajectory = []
        order_index = 0
        MAX_CARDINALITY = 10

        passed_transitions: Dict[label,
                                 Dict[label, Dict[int, cardinality]]] = {}

        current_transition = [[(label1, label2) for label2 in self.graph[label1].keys(
        ) if order_index in self.graph[label1][label2].keys()] for label1 in self.graph.keys()][0][0]
        random.seed(seed)
        if random.random() > 0.5:
            current_transition = self.non_optional_start_end
            order_indexes = list(
                self.graph[current_transition[0]][current_transition[1]].keys())
            order_index = min(order_indexes)

        if self.graph[current_transition[0]][current_transition[1]][order_index] == None:

            print(current_transition[0], current_transition[1], order_index)
            hist += current_transition

        order_index += 1

        def next_transition(label: label, order_index: int) -> Tuple[label, label]:

            label1 = label
            label2 = None

            order_sorted_transition = [[(label2, ord_index) for ord_index in order_card.keys(
            )] for label2, order_card in self.graph[label1].items()]
            order_sorted_transition = list(
                itertools.chain.from_iterable(order_sorted_transition))
            order_sorted_transition.sort(key=lambda x: x[1])

            for _label2, ord2 in order_sorted_transition:

                if ord2 == order_index:

                    if self.graph[label1][_label2][order_index] == None:
                        label2 = _label2
                        break

                    elif passed_transitions.get(label1, {}).get(_label2, {}).get(order_index, None) == None:
                        passed_transitions.setdefault(label1, {})
                        passed_transitions[label1].setdefault(_label2, {})
                        passed_transitions[label1][_label2][order_index] = self.graph[label1][_label2][order_index]

                    if not (int(passed_transitions[label1][_label2][order_index].lower_bound) == "0" and
                            int(passed_transitions[label1][_label2][order_index].upper_bound) == "0"):

                        if passed_transitions[label1][_label2][order_index].upper_bound == "*":
                            passed_transitions[label1][_label2][order_index].upper_bound = random.randint(
                                passed_transitions[label1][_label2][order_index].lower_bound, passed_transitions[label1][_label2][order_index].lower_bound + MAX_CARDINALITY)

                        if passed_transitions[label1][_label2][order_index].lower_bound == "0" and random.random() > 0.5:
                            order_index += 1
                            continue
                        else:
                            # decrement the special transition cardinaltiy
                            trans_card = passed_transitions[label1][_label2][order_index]
                            lower_bound = "*" if trans_card.lower_bound == "*" else str(
                                int(trans_card.lower_bound) - 1 if int(trans_card.lower_bound) - 1 >= 0 else 0)
                            upper_bound = "*" if trans_card.upper_bound == "*" else str(
                                int(trans_card.upper_bound) - 1 if int(trans_card.upper_bound) - 1 >= 0 else 0)
                            if lower_bound == "1":
                                lower_bound = "0"
                            if upper_bound == "1":
                                upper_bound = "0"
                            passed_transitions[label1][_label2][order_index] = cardinality(
                                lower_bound, upper_bound)

                            # if it reaches a new sub-cycle, reset its cardinality
                            father_labels = [l for l in passed_transitions.keys(
                            ) if passed_transitions[l].get(_label2, None) != None]
                            for father_label in father_labels:
                                if father_label != label1:
                                    for card_i in passed_transitions[father_label][_label2].keys():
                                        if passed_transitions[father_label][_label2][card_i].lower_bound == "0" and \
                                                passed_transitions[father_label][_label2][card_i].upper_bound == "0" and card_i < order_index:
                                            passed_transitions[father_label][_label2][card_i] = copy.deepcopy(
                                                self.graph[father_label][_label2][card_i])

                            label1 = _label2
                            order_sorted_transition = [[(label2, ord_index) for ord_index in order_card.keys(
                            )] for label2, order_card in self.graph[label1].items()]
                            order_sorted_transition = list(
                                itertools.chain.from_iterable(order_sorted_transition))
                            order_sorted_transition.sort(
                                key=lambda x: abs(x[1]-order_index))
                            label2 = order_sorted_transition[0][0]
                            order_index = order_sorted_transition[0][1]
                            break

            return label1, label2, order_index

        label1 = None
        label2 = None
        for i in range(0, 10):
            label1, label2, order_index = next_transition(
                current_transition[1], order_index)

            print(label1, label2, order_index)

            current_transition = (label1, label2)
            hist += [current_transition]
            order_index += 1

    def next_actions(self, trajectory: trajectory, observation: observation) -> Union[List[action], None]:
        # TODO: Take into account the trajectory when sample() is finished
        if self.graph.get(observation, None) == None:
            return None
        return list(self.graph[observation].keys())

    def plot_graph(self, show: bool = False, render_rgba: bool = False, save: bool = False):

        # Create a directed graph object
        G = nx.DiGraph()

        oriented_edges = {}

        for src_label in self.graph:
            for dst_label in self.graph[src_label]:
                G.add_edge(src_label, dst_label)
                if (src_label, dst_label) in oriented_edges.keys():
                    return
                oriented_edges[(src_label, dst_label)] = str(
                    self.graph[src_label][dst_label])

        # Draw the graph
        pos = nx.spring_layout(G)
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)

        curved_edges = [
            edge for edge in G.edges() if reversed(edge) in G.edges()]
        straight_edges = list(set(G.edges()) - set(curved_edges))
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
        arc_rad = 0.25
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')

        curved_edge_labels = {edge: oriented_edges[(
            f'{edge[0]}', f'{edge[1]}')] for edge in curved_edges}
        straight_edge_labels = {edge: oriented_edges[(
            f'{edge[0]}', f'{edge[1]}')] for edge in straight_edges}
        draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=curved_edge_labels, rotate=False, rad=arc_rad)
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False)

        # Show the plot
        if (save):
            fig.savefig(f"{random.randint(1, 100)}.png",
                        bbox_inches='tight', pad_inches=0)
        if (show):
            plt.show()

        if (render_rgba):
            # fig.canvas.draw()
            # return Image.frombytes('RGB',
            # fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

            fig.canvas.draw()  # Draw the canvas, cache the renderer
            image_flat = np.frombuffer(
                fig.canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
            # NOTE: reversed converts (W, H) from get_width_height to (H, W)
            image = image_flat.reshape(
                *reversed(fig.canvas.get_width_height()), 3)  # (H, W, 3)
            return image


if __name__ == '__main__':

    hs = trajectory_graph()

    # # hs.add_pattern("[[0,1](0,1),2,3,[4,5](1,3),6](1,1)")
    # # hs.add_pattern("[[0,1](2,2),2](3,3)")
    # # hs.add_pattern("[0,[1,2](0,4),3](1,2)")
    # hs.add_pattern("[[0,1](0,2),2,3,[0,1](3,3),2,3](1,2)")
    # hs.add_observations_to_actions(["obs1", "obs2"], ["act1", "act2", "act3"])
    # # hs.plot_graph(show=True)
    # # print(hs.next_actions(trajectory=None, observation="obs1"))

    # print(hs)
    # # print(hs.to_dict())
    # print(hs.from_dict(hs.to_dict()))

    # # hs.sample(seed=89)
