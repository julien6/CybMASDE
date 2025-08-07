import numpy as np
import binascii
import importlib
import inspect
import os
import re
import numpy as np
import textwrap

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Union


MATCH_REWARD = 100
COVERAGE_REWARD = 10
OBLIGATION_REWARD_FACTOR = 10

# we assume "observation" and "action" are already one-hot encoded
observation = Union[int, np.ndarray]
action = Union[int, np.ndarray]
label = str
pattern_trajectory = str
trajectory = List[Tuple[observation, action]]
labeled_trajectory = Union[List[Tuple[label, label]], List[label], str]
trajectory_pattern_str = str
trajectory_str = str


@dataclass
class cardinality:
    lower_bound: Union[int, str]
    upper_bound: Union[int, str]

    def __str__(self) -> str:
        return f'({self.lower_bound},{self.upper_bound})'

    def __repr__(self) -> str:
        return f'({self.lower_bound},{self.upper_bound})'

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'cardinality':
        return cardinality(
            lower_bound=d['lower_bound'],
            upper_bound=d['upper_bound']
        )


def replace_all_character(string, old_character, new_character):
    while True:
        _string = string.replace(old_character, new_character)
        if string == _string:
            break
        string = _string
    return string


def generate_random_hash(num_bytes=16):
    # Génère des octets aléatoires
    random_bytes = os.urandom(num_bytes)
    # Convertit ces octets en chaîne hexadécimale
    random_hash = binascii.hexlify(random_bytes).decode('utf-8')
    return random_hash


def handle_lambda(source: str) -> str:
    source = textwrap.dedent(source)
    # check the function's source is a lambda function
    if 'lambda' in source and not "def" in source:
        _source = replace_all_character(source, "\n", " ")
        _source = replace_all_character(source, "  ", " ")
        _source += "\n"
        match = re.search(
            r"^.*(lambda [\w\s,]+:\s*\[\(.+?\s*\)\])[\n,]|^.*(lambda [\w\s,]+:.+?)[\n,]", _source, re.MULTILINE)
        if match:
            function_name = None
            source = [m for m in match.groups() if m is not None][0]
            # check if it already has a name
            match = re.search(
                r"^\s*(.*=.*lambda [\w\s,]+:\s*\[\(.+?\s*\)\])[\n,]|^\s*(.*=.*lambda [\w\s,]+:.+?)[\n,]", _source, re.MULTILINE)
            if match:
                source = [m for m in match.groups() if m is not None][0]
                function_name, source = source.split("=")
                if ":" in function_name:
                    function_name = function_name.split(":")[0]
                function_name = replace_all_character(
                    function_name, " ", "")
            function_args, function_body = source.split(":")
            function_args = replace_all_character(
                function_args.replace("lambda", ""), " ", "")
            if function_name is None:
                function_name = f"lamba_{generate_random_hash()}"
            source = f'def {function_name}({function_args}):\n    return {function_body}'
    return source


def load_function(function_data: Any) -> Tuple[Callable, bool]:
    if not 'module_name' in function_data:
        raise Exception("Module should be given")
    module_name = function_data['module_name']
    module = importlib.import_module(module_name)
    if 'source' in function_data:
        source = handle_lambda(function_data['source'])
        match = re.search(
            r"^\s*def\s+([a-zA-Z_]\w*)\s*\(", source, re.MULTILINE)
        if match:
            function_name = match.group(1)
        lcs = {}
        exec(source, module.__dict__, lcs)
        _function = lcs.get(function_name)
        return _function, True, function_name, source
    elif 'function_name' in function_data:
        function_name = function_data['function_name']
        _function = getattr(module, function_name)
        return _function, False, None, None


def dump_function(function: Callable, save_source: bool = False, function_name=None, function_source=None):
    if save_source:
        try:
            source = textwrap.dedent(inspect.getsource(function))
        except Exception as e:
            source = function_source
        # check the function's source is a lambda function
        source = handle_lambda(source)

        return {'function_name': function.__name__, 'module_name': function.__module__, 'source': source}
    else:
        return {'function_name': function.__name__, 'module_name': function.__module__}


def draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0, 1), (-1, 0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(
                1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items
