# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
from typing import List, Optional

import numpy as np


class LaneSegment:
    def __init__(
        self,
        id: int,
        l_neighbor_id: Optional[int],
        r_neighbor_id: Optional[int],
        centerline: np.ndarray,
    ) -> None:
        """
        Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way"
            l_neighbor_id: Unique ID for left neighbor
            r_neighbor_id: Unique ID for right neighbor
            centerline: The coordinates of the lane segment's center line.
        """
        self.id = id
        self.l_neighbor_id = l_neighbor_id
        self.r_neighbor_id = r_neighbor_id
        self.centerline = centerline

class Road:
    def __init__(
        self,
        id: int,
        l_bound: np.ndarray,
        r_bound: np.ndarray,
    ) -> None:
        """Initialize the lane segment.

        Args:
            id: Unique lane ID that serves as identifier for this "Way".
            l_bound: The coordinates of the lane segment's left bound.
            r_bound: The coordinates of the lane segment's right bound.
        """
        self.id = id
        self.l_bound = l_bound
        self.r_bound = r_bound

