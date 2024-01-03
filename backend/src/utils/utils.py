from collections import OrderedDict
from copy import copy
from typing import Dict, List, Any, Union
from random import shuffle

class Observation:
    id: str
    value: Union[int, str]