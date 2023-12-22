#!/usr/bin/env python3
# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""
Utility to load the Argoverse vector map from disk, where it is stored in an XML format.

We release our Argoverse vector map in a modified OpenStreetMap (OSM) form. We also provide
the map data loader. OpenStreetMap (OSM) provides XML data and relies upon "Nodes" and "Ways" as
its fundamental element.

A "Node" is a point of interest, or a constituent point of a line feature such as a road.
In OpenStreetMap, a `Node` has tags, which might be
        -natural: If it's a natural feature, indicates the type (hill summit, etc)
        -man_made: If it's a man made feature, indicates the type (water tower, mast etc)
        -amenity: If it's an amenity (e.g. a pub, restaurant, recycling
            centre etc) indicates the type

In OSM, a "Way" is most often a road centerline, composed of an ordered list of "Nodes".
An OSM way often represents a line or polygon feature, e.g. a road, a stream, a wood, a lake.
Ways consist of two or more nodes. Tags for a Way might be:
        -highway: the class of road (motorway, primary,secondary etc)
        -maxspeed: maximum speed in km/h
        -ref: the road reference number
        -oneway: is it a one way road? (boolean)

However, in Argoverse, a "Way" corresponds to a LANE segment centerline. An Argoverse Way has the
following 9 attributes:
        -   id: integer, unique lane ID that serves as identifier for this "Way"
        -   has_traffic_control: boolean
        -   turn_direction: string, 'RIGHT', 'LEFT', or 'NONE'
        -   is_intersection: boolean
        -   l_neighbor_id: integer, unique ID for left neighbor
        -   r_neighbor_id: integer, unique ID for right neighbor
        -   predecessors: list of integers or None
        -   successors: list of integers or None
        -   centerline_node_ids: list

In Argoverse, a `LaneSegment` object is derived from a combination of a single `Way` and two or more
`Node` objects.
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union, cast

import numpy as np
import matplotlib.pyplot as plt

from dataloader.utils.lane_segment import LaneSegment, Road

logger = logging.getLogger(__name__)


_PathLike = Union[str, "os.PathLike[str]"]


class Node:
    """
    e.g. a point of interest, or a constituent point of a
    line feature such as a road
    """

    def __init__(self, id: int, x: float, y: float, height: Optional[float] = None):
        """
        Args:
            id: representing unique node ID
            x: x-coordinate in city reference system
            y: y-coordinate in city reference system

        Returns:
            None
        """
        self.id = id
        self.x = x
        self.y = y
        self.height = height


def str_to_bool(s: str) -> bool:
    """
    Args:
       s: string representation of boolean, either 'True' or 'False'

    Returns:
       boolean
    """
    if s == "True":
        return True
    assert s == "False"
    return False


def convert_dictionary_to_lane_segment_obj(lane_id: int, lane_dictionary: Mapping[str, Any]) -> LaneSegment:
    """
    Not all lanes have predecessors and successors.

    Args:
       lane_id: representing unique lane ID
       lane_dictionary: dictionary with LaneSegment attributes, not yet in object instance form

    Returns:
       ls: LaneSegment object
    """

    l_neighbor_id = None 
    r_neighbor_id = None 
    ls = LaneSegment(
        lane_id,
        l_neighbor_id,
        r_neighbor_id,
        lane_dictionary["centerline"],
    )
    return ls


def append_additional_key_value_pair(lane_obj: MutableMapping[str, Any], way_field: List[Tuple[str, str]]) -> None:
    """
    Key name was either 'predecessor' or 'successor', for which we can have multiple.
    Thus we append them to a list. They should be integers, as lane IDs.

    Args:
       lane_obj: lane object
       way_field: key and value pair to append

    Returns:
       None
    """
    assert len(way_field) == 2
    k = way_field[0][1]
    v = int(way_field[1][1])
    lane_obj.setdefault(k, []).append(v)


def append_unique_key_value_pair(lane_obj: MutableMapping[str, Any], way_field: List[Tuple[str, str]]) -> None:
    """
    For the following types of Way "tags", the key, value pair is defined only once within
    the object:
        - has_traffic_control, turn_direction, is_intersection, l_neighbor_id, r_neighbor_id

    Args:
       lane_obj: lane object
       way_field: key and value pair to append

    Returns:
       None
    """
    assert len(way_field) == 2
    k = way_field[0][1]
    v = way_field[1][1]
    lane_obj[k] = v


def extract_node_waypt(way_field: List[Tuple[str, str]]) -> int:
    """
    Given a list with a reference node such as [('ref', '0')], extract out the lane ID.

    Args:
       way_field: key and node id pair to extract

    Returns:
       node_id: unique ID for a node waypoint
    """
    key = way_field[0][0]
    node_id = way_field[0][1]
    assert key == "ref"
    return int(node_id)


def get_lane_identifier(child: ET.Element) -> int:
    """
    Fetch lane ID from XML ET.Element.

    Args:
       child: ET.Element with information about Way

    Returns:
       unique lane ID
    """
    return int(child.attrib["id"])


def convert_node_id_list_to_xy(node_id_list: List[int], all_graph_nodes: Mapping[int, Node]) -> np.ndarray:
    """
    convert node id list to centerline xy coordinate

    Args:
       node_id_list: list of node_id's
       all_graph_nodes: dictionary mapping node_ids to Node

    Returns:
       centerline
    """
    num_nodes = len(node_id_list)

    if all_graph_nodes[node_id_list[0]].height is not None:
        centerline = np.zeros((num_nodes, 3))
    else:
        centerline = np.zeros((num_nodes, 2))
    for i, node_id in enumerate(node_id_list):
        if all_graph_nodes[node_id].height is not None:
            centerline[i] = np.array(
                [
                    all_graph_nodes[node_id].x,
                    all_graph_nodes[node_id].y,
                    all_graph_nodes[node_id].height,
                ]
            )
        else:
            centerline[i] = np.array([all_graph_nodes[node_id].x, all_graph_nodes[node_id].y])

    return centerline


def extract_node_from_ET_element(child: ET.Element) -> Node:
    """
    Given a line of XML, build a node object. The "node_fields" dictionary will hold "id", "x", "y".
    The XML will resemble:

        <node id="0" x="3168.066310258233" y="1674.663991981186" />

    Args:
        child: xml.etree.ElementTree element

    Returns:
        Node object
    """
    node_fields = child.attrib
    node_id = int(node_fields["id"])
    for element in child:
        way_field = cast(List[Tuple[str, str]], list(element.items()))
        key = way_field[0][1]
        if key == "local_x":
            x = float(way_field[1][1])
        elif key == "local_y":
            y = float(way_field[1][1])

    return Node(id=node_id, x=x, y=y)


def extract_lane_segment_from_ET_element(
    child: ET.Element, all_graph_nodes: Mapping[int, Node]
) -> Tuple[LaneSegment, int]:
    """
    We build a lane segment from an XML element. A lane segment is equivalent
    to a "Way" in our XML file. Each Lane Segment has a polyline representing its centerline.
    The relevant XML data might resemble::

        <way lane_id="9604854">
            <tag k="has_traffic_control" v="False" />
            <tag k="turn_direction" v="NONE" />
            <tag k="is_intersection" v="False" />
            <tag k="l_neighbor_id" v="None" />
            <tag k="r_neighbor_id" v="None" />
            <nd ref="0" />
            ...
            <nd ref="9" />
            <tag k="predecessor" v="9608794" />
            ...
            <tag k="predecessor" v="9609147" />
        </way>

    Args:
        child: xml.etree.ElementTree element
        all_graph_nodes

    Returns:
        lane_segment: LaneSegment object
        lane_id
    """
    lane_obj: Dict[str, Any] = {}
    lane_id = get_lane_identifier(child)
    node_id_list: List[int] = []
    for element in child:
        # The cast on the next line is the result of a typeshed bug.  This really is a List and not a ItemsView.
        way_field = cast(List[Tuple[str, str]], list(element.items()))
        field_name = way_field[0][0]
        if field_name == "k":
            key = way_field[0][1]
            if key in {"predecessor", "successor"}:
                append_additional_key_value_pair(lane_obj, way_field)
            else:
                append_unique_key_value_pair(lane_obj, way_field)
        else:
            node_id_list.append(extract_node_waypt(way_field))

    lane_obj["centerline"] = convert_node_id_list_to_xy(node_id_list, all_graph_nodes)
    lane_segment = convert_dictionary_to_lane_segment_obj(lane_id, lane_obj)
    return lane_segment, lane_id

def construct_road_from_ET_element(
    child: ET.Element, lane_objs: Mapping[int, LaneSegment]
):
    road_id = int(child.attrib["id"])
    for element in child:
        if element.tag == "member":
            relation_field = cast(List[Tuple[str, str]], list(element.items()))
            if relation_field[2][1] == "right":
                r_bound_idx = int(relation_field[1][1])
            elif relation_field[2][1] == "left":
                l_bound_idx = int(relation_field[1][1])
    l_bound = lane_objs[l_bound_idx].centerline
    r_bound = lane_objs[r_bound_idx].centerline
    road = Road(
        road_id,
        l_bound,
        r_bound
    )
    return road, road_id


def load_lane_segments_from_xml(map_fpath: _PathLike) -> Mapping[int, LaneSegment]:
    """
    Load lane segment object from xml file

    Args:
       map_fpath: path to xml file

    Returns:
       lane_objs: List of LaneSegment objects
    """
    tree = ET.parse(os.fspath(map_fpath))
    root = tree.getroot()

    logger.info(f"Loaded root: {root.tag}")

    all_graph_nodes = {}
    lane_objs = {}
    roads = {}
    # all children are either Nodes or Ways or relations
    for child in root:
        if child.tag == "node":
            node_obj = extract_node_from_ET_element(child)
            all_graph_nodes[node_obj.id] = node_obj
        elif child.tag == "way":
            lane_obj, lane_id = extract_lane_segment_from_ET_element(child, all_graph_nodes)
            lane_objs[lane_id] = lane_obj
        elif child.tag == "relation":
            road, road_id = construct_road_from_ET_element(child, lane_objs)
            roads[road_id] = road
        else:
            logger.error("Unknown XML item encountered.")
            raise ValueError("Unknown XML item encountered.")
    return roads

def build_polygon_bboxes(roads):
    """
    roads: dict, key: road id; value field: l_bound, r_bound
    polygon_bboxes: An array of shape (K,), each array element is a NumPy array of shape (4,) representing
                        the bounding box for a polygon or point cloud.
    each road_id corresponds to a polygon_bbox
    lane_start: An array of shape (,4), indicating (x_l, y_l, x_r, y_r)
    lane_end: An array of shape (,4), indicating (x_l, y_l, x_r, y_r)
    """
    polygon_bboxes = []
    lane_starts = []
    lane_ends = []
    for road_id in roads.keys():
        x = np.concatenate((roads[road_id].l_bound[:,0], roads[road_id].r_bound[:,0]))
        xmin = np.min(x)
        xmax = np.max(x)
        y = np.concatenate((roads[road_id].l_bound[:,1], roads[road_id].r_bound[:,1]))
        ymin = np.min(y)
        ymax = np.max(y)
        polygon_bbox = np.array([xmin, ymin, xmax, ymax])
        polygon_bboxes.append(polygon_bbox)
        
        lane_start = np.array([roads[road_id].l_bound[0,0], roads[road_id].l_bound[0,1],
                               roads[road_id].r_bound[0,0], roads[road_id].r_bound[0,1]])
        lane_end = np.array([roads[road_id].l_bound[-1,0], roads[road_id].l_bound[-1,1],
                               roads[road_id].r_bound[-1,0], roads[road_id].r_bound[-1,1]])
        lane_starts.append(lane_start)
        lane_ends.append(lane_end)

    return np.array(polygon_bboxes), np.array(lane_starts), np.array(lane_ends)

def find_all_polygon_bboxes_overlapping_query_bbox(polygon_bboxes: np.ndarray, 
                                                   query_bbox: np.ndarray, 
                                                   lane_starts: np.ndarray, 
                                                   lane_ends: np.ndarray) -> np.ndarray:
    """Find all the overlapping polygon bounding boxes.
    Each bounding box has the following structure:
        bbox = np.array([x_min,y_min,x_max,y_max])
    In 3D space, if the coordinates are equal (polygon bboxes touch), then these are considered overlapping.
    We have a guarantee that the cropped image will have any sort of overlap with the zero'th object bounding box
    inside of the image e.g. along the x-dimension, either the left or right side of the bounding box lies between the
    edges of the query bounding box, or the bounding box completely engulfs the query bounding box.
    Args:
        polygon_bboxes: An array of shape (K, 4), each array element is a NumPy array of shape (4,) representing
                        the bounding box for a polygon or point cloud.
        query_bbox: An array of shape (4,) representing a 2d axis-aligned bounding box, with order
                    [min_x,min_y,max_x,max_y].
        lane_starts: An array of shape (, 4), representing the start point of lane left bound and right bound
        lane_ends: An array of shape (, 4), representing the end point of lane left bound and right bound
    Returns:
        An integer array of shape (K,) representing indices where overlap occurs.
    """
    query_min_x = query_bbox[0]
    query_min_y = query_bbox[1]

    query_max_x = query_bbox[2]
    query_max_y = query_bbox[3]

    bboxes_x1 = polygon_bboxes[:, 0]
    bboxes_x2 = polygon_bboxes[:, 2]

    bboxes_y1 = polygon_bboxes[:, 1]
    bboxes_y2 = polygon_bboxes[:, 3]

    # check if falls within range
    overlaps_left = (query_min_x <= bboxes_x2) & (bboxes_x2 <= query_max_x)
    overlaps_right = (query_min_x <= bboxes_x1) & (bboxes_x1 <= query_max_x)

    x_check1 = bboxes_x1 <= query_min_x
    x_check2 = query_min_x <= query_max_x
    x_check3 = query_max_x <= bboxes_x2
    x_subsumed = x_check1 & x_check2 & x_check3

    x_in_range = overlaps_left | overlaps_right | x_subsumed

    overlaps_below = (query_min_y <= bboxes_y2) & (bboxes_y2 <= query_max_y)
    overlaps_above = (query_min_y <= bboxes_y1) & (bboxes_y1 <= query_max_y)

    y_check1 = bboxes_y1 <= query_min_y
    y_check2 = query_min_y <= query_max_y
    y_check3 = query_max_y <= bboxes_y2
    y_subsumed = y_check1 & y_check2 & y_check3
    y_in_range = overlaps_below | overlaps_above | y_subsumed

    # at least one lane endpoint in range
    # xy_check1 = (query_min_x <= lane_starts[:,0]) & (lane_starts[:,0] <= query_max_x) & \
    #             (query_min_y <= lane_starts[:,1]) & (lane_starts[:,1] <= query_max_y)
    # xy_check2 = (query_min_x <= lane_starts[:,2]) & (lane_starts[:,2] <= query_max_x) & \
    #             (query_min_y <= lane_starts[:,3]) & (lane_starts[:,3] <= query_max_y)
    # xy_check3 = (query_min_x <= lane_ends[:,0]) & (lane_ends[:,0] <= query_max_x) & \
    #             (query_min_y <= lane_ends[:,1]) & (lane_ends[:,1] <= query_max_y)
    # xy_check4 = (query_min_x <= lane_ends[:,2]) & (lane_ends[:,2] <= query_max_x) & \
    #             (query_min_y <= lane_ends[:,3]) & (lane_ends[:,3] <= query_max_y)
    # xy_in_range = xy_check1 | xy_check2 | xy_check3 | xy_check4

    # overlap_indxs = np.where(x_in_range & y_in_range & xy_in_range)[0]

    overlap_indxs = np.where(x_in_range & y_in_range)[0]
    return overlap_indxs

def get_road_ids_in_xy_bbox(
    polygon_bboxes,
    lane_starts,
    lane_ends,
    roads,
    query_x: float,
    query_y: float,
    query_search_range_manhattan: float = 50.0,
):
    """
    Prune away all lane segments based on Manhattan distance. We vectorize this instead
    of using a for-loop. Get all lane IDs within a bounding box in the xy plane.
    This is a approximation of a bubble search for point-to-polygon distance.
    The bounding boxes of small point clouds (lane centerline waypoints) are precomputed in the map.
    We then can perform an efficient search based on manhattan distance search radius from a
    given 2D query point.
    We pre-assign lane segment IDs to indices inside a big lookup array, with precomputed
    hallucinated lane polygon extents.
    Args:
        query_x: representing x coordinate of xy query location
        query_y: representing y coordinate of xy query location
        city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        query_search_range_manhattan: search radius along axes
    Returns:
        lane_ids: lane segment IDs that live within a bubble
    """
    query_min_x = query_x - query_search_range_manhattan
    query_max_x = query_x + query_search_range_manhattan
    query_min_y = query_y - query_search_range_manhattan
    query_max_y = query_y + query_search_range_manhattan

    overlap_indxs = find_all_polygon_bboxes_overlapping_query_bbox(
        polygon_bboxes,
        np.array([query_min_x, query_min_y, query_max_x, query_max_y],),
        lane_starts,
        lane_ends
    )

    if len(overlap_indxs) == 0:
        return []

    neighborhood_road_ids = []
    for overlap_idx in overlap_indxs:
        lane_segment_id = list(roads.keys())[overlap_idx]
        neighborhood_road_ids.append(lane_segment_id)

    return neighborhood_road_ids

if __name__ == "__main__":
    roads = load_lane_segments_from_xml("Town03.osm")
    polygon_bboxes = build_polygon_bboxes(roads)
    query_x = 5.772
    query_y = 119.542
    cv_range = 50
    neighborhood_road_ids = get_road_ids_in_xy_bbox(polygon_bboxes, query_x, query_y, cv_range)


# # %%
# plt.figure(dpi=200)
# fig, (ax1,ax2) = plt.subplots(1,2)
# fig.set_figheight(2)
# fig.set_figwidth(4)
# for i in roads.keys():
   
#     road_id = i
#     ax1.plot(roads[road_id].l_bound[:,0], roads[road_id].l_bound[:,1], color='k')#, marker='o', markerfacecolor='blue', markersize=5)
#     ax1.plot(roads[road_id].r_bound[:,0], roads[road_id].r_bound[:,1], color='k')#, marker='o', markerfacecolor='red', markersize=5)
#     ax1.plot((roads[road_id].l_bound[:,0]+roads[road_id].r_bound[:,0])/2, (roads[road_id].l_bound[:,1]+roads[road_id].r_bound[:,1])/2, color="0.7",linestyle='dashed')
#     ax2.plot(roads[road_id].l_bound[:,0], roads[road_id].l_bound[:,1], color='k')#, marker='o', markerfacecolor='blue', markersize=5)
#     ax2.plot(roads[road_id].r_bound[:,0], roads[road_id].r_bound[:,1], color='k')#, marker='o', markerfacecolor='red', markersize=5)
#     ax2.plot((roads[road_id].l_bound[:,0]+roads[road_id].r_bound[:,0])/2, (roads[road_id].l_bound[:,1]+roads[road_id].r_bound[:,1])/2, color="0.7",linestyle='dashed')

# ax1.set_xlim([-60,60])
# ax1.set_ylim([-60,60])
# ax2.set_xlim([60,120])
# ax2.set_ylim([80,180])
# ax1.axis("off")
# ax2.axis("off")
# # plt.show()
# plt.savefig("town03_lane_segment.jpg")
# # %%
# # plot one lane segment
# for i in roads.keys():
#     road_id = i
#     if min(roads[road_id].l_bound[:,0])>60 and max(roads[road_id].l_bound[:,1])>-20 and max(roads[road_id].r_bound[:,0])<120 and max(roads[road_id].r_bound[:,1])<70:
        
#         plt.plot(roads[road_id].l_bound[:,0], roads[road_id].l_bound[:,1], color='0.7')#, marker='o', markerfacecolor='blue', markersize=5)
#         plt.plot(roads[road_id].r_bound[:,0], roads[road_id].r_bound[:,1], color='0.7')#, marker='o', markerfacecolor='red', markersize=5)
# # plt.
# # plt.xlim((60,120))
# # plt.ylim((80,180))
# # plt.axis("off")
# plt.show()

# # %%
# for i in roads.keys():
#     road_id = i
#     plt.plot(roads[road_id].l_bound[:,0], roads[road_id].l_bound[:,1], color='0.7')#, marker='o', markerfacecolor='blue', markersize=5)
#     plt.plot(roads[road_id].r_bound[:,0], roads[road_id].r_bound[:,1], color='0.7')#, marker='o', markerfacecolor='red', markersize=5)
# plt.show()
# # %%
