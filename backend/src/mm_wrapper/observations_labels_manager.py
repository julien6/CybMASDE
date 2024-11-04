from typing import Dict, List, Tuple

from mm_wrapper.history_graph import observation
from mm_wrapper.llm_manager import llm_manager


class observations_labels_manager:

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm
        self.text_id_to_observations: Dict[Tuple[str,
                                                 str], List[observation]] = {}
        self.llm_manager = None
        if self.use_llm:
            self.llm_manager = llm_manager()

    def map_text_description_to_observations(self, text_description: str, observations: List[observation], use_llm: bool = True, observations_identifier: str = None) -> 'observations_labels_manager':
        """One-to-one map a textual description to a set of observations using a regular dict and by default a LLM is prompt-engineered to learn the mapping as well.

        Parameters
        ----------
        text_description : str
            The text describing a set of observations.

        observations: List[str]
            The list of observations to be mapped.

        use_llm: bool (default=True)
            Whether using a LLM to learn how to match observations to a text description in addition to the ones mapped
            with a regular dictionary

        observations_identifier: str (default=None)
            An short-way to identify a unique set of observations

        Returns
        -------
        observations_labels_manager
            The current instance

        Examples
        --------
        >>> 

        See Also
        --------
        None
        """
        self.use_llm = use_llm
        if self.use_llm:
            pass
        else:
            self.text_id_to_observations[(
                observations_identifier, text_description)] = observations
        return self

    def add_observations_from_text_description(self, text_description: str, use_llm: bool = True) -> 'observations_labels_manager':
        """Add a set of observations that match given text description.

        Parameters
        ----------
        text_description : str
            The text describing a set of observations.

        use_llm: bool (default=True)
            Whether using a LLM to find corresponding observations in addition to the ones mapped
            with a regular dictionary

        Returns
        -------
        observations_labels_manager
            The current instance

        Examples
        --------
        >>> 

        See Also
        --------
        None
        """
        self.use_llm = use_llm
        return self

    def set_use_llm(self, use_llm) -> 'observations_labels_manager':
        self.use_llm = use_llm
        return self


def assisted_observations_labels_manager_create(use_llm: bool = True) -> observations_labels_manager:
    """Create an hand-crafted observations_labels_manager by running an interactive assisted process
    consists in running a series of episodes during learning. In each episode, users have to
    describe a visual representation of agents' action for some/all frames.

    Parameters
    ----------
    use_llm: bool (default=True)
        Whether using a LLM to learn how to match observations to a text description in
        addition to the ones mapped with a regular dictionary

    Returns
    -------
    observations_labels_manager
        The current instance

    Examples
    --------
    >>> 

    See Also
    --------
    None
    """
    return observations_labels_manager()


simple_spread_obs_mngr = observations_labels_manager()

simple_world_comm_obs_mngr = observations_labels_manager()
