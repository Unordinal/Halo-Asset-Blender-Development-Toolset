# ##### BEGIN MIT LICENSE BLOCK #####
#
# MIT License
#
# Copyright (c) 2023 Steven Garcia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ##### END MIT LICENSE BLOCK #####

import math

from .format import JMAAsset
from ..global_functions import global_functions

def read_header_16394(JMA):
    JMA.node_checksum = int(JMA.next())
    JMA.frame_count = int(JMA.next())
    framerate_string = JMA.next()
    actor_count = int(JMA.next())
    JMA.actor_name = JMA.next()
    JMA.node_count = int(JMA.next())
    try:
        JMA.frame_rate = int(framerate_string)

    except ValueError:
        JMA.frame_rate = math.floor(float(framerate_string))

    if actor_count != 1:
        raise global_functions.ParseError("File must have an actor count of 1!")

def read_header_16390(JMA):
    JMA.frame_count = int(JMA.next())
    framerate_string = JMA.next()
    actor_count = int(JMA.next())
    JMA.actor_name = JMA.next()
    JMA.node_count = int(JMA.next())
    JMA.node_checksum = int(JMA.next())
    try:
        JMA.frame_rate = int(framerate_string)

    except ValueError:
        JMA.frame_rate = math.floor(float(framerate_string))

    if actor_count != 1:
        raise global_functions.ParseError("File must have an actor count of 1!")

def read_nodes_16394(JMA):
    for node_idx in range(JMA.node_count):
        name = JMA.next()
        parent = int(JMA.next())
        JMA.nodes.append(JMAAsset.Node(name, parent=parent))

def read_nodes_16392(JMA):
    for node_idx in range(JMA.node_count):
        name = JMA.next()
        child = int(JMA.next())
        sibling = int(JMA.next())
        JMA.nodes.append(JMAAsset.Node(name, child=child, sibling=sibling))

def read_nodes_16391(JMA):
    for node_idx in range(JMA.node_count):
        JMA.nodes.append(JMAAsset.Node(JMA.next()))

def read_node_transforms_16390(JMA):
    for transform_idx in range(JMA.frame_count):
        transforms_for_frame = []
        for node_idx in range(JMA.node_count):
            transforms_for_frame.append(JMA.next_transform())

        JMA.transforms.append(transforms_for_frame)

def read_root_transforms_16395(JMA, extension):
    biped_controller_enabled = bool(int(JMA.next()))
    if biped_controller_enabled:
        # different animation file types use the data differently
        if extension == 'jma':
            JMA.biped_controller_frame_type = JMAAsset.BipedControllerFrameType.JMA

        elif extension == 'jmt':
            JMA.biped_controller_frame_type = JMAAsset.BipedControllerFrameType.JMT

        elif extension == 'jmrx':
            JMA.biped_controller_frame_type = JMAAsset.BipedControllerFrameType.JMRX

        for transform_idx in range(JMA.frame_count):
            JMA.biped_controller_transforms.append(JMA.next_transform())

def validate_node_graph_16394(JMA, report):
    # update node graph
    # loop over nodes and
    for node_idx in range(JMA.node_count):
        node = JMA.nodes[node_idx]
        if node.parent == -1:
            continue # this is a root node, nothing to update

        if node.parent >= len(JMA.nodes) or node.parent == node_idx:
            report({'WARNING'}, "Malformed node graph (bad parent index)")
            JMA.broken_skeleton = True
            break

        parent_node = JMA.nodes[node.parent]
        if parent_node.child:
            node.sibling = parent_node.child

        else:
            node.sibling = -1

        if node.sibling >= len(JMA.nodes):
            report({'WARNING'}, "Malformed node graph (sibling index out of range)")
            JMA.broken_skeleton = True
            break

        parent_node.child = node_idx

def validate_node_graph_16392(JMA, report):
    # update node graph
    # loop over nodes and
    for node_idx in range(JMA.node_count):
        node = JMA.nodes[node_idx]
        if node.child == -1:
            continue # no child nodes, nothing to update

        if node.child >= len(JMA.nodes) or node.child == node_idx:
            report({'WARNING'}, "Malformed node graph (bad child index)")
            JMA.broken_skeleton = True
            break

        child_node = JMA.nodes[node.child]
        while child_node != None:
            child_node.parent = node_idx
            if child_node.visited:
                report({'WARNING'}, "Malformed node graph (circular reference)")
                JMA.broken_skeleton = True
                break

            child_node.visited = True
            if child_node.sibling >= len(JMA.nodes):
                report({'WARNING'}, "Malformed node graph (sibling index out of range)")
                JMA.broken_skeleton = True
                break

            if child_node.sibling != -1:
                child_node = JMA.nodes[child_node.sibling]

            else:
                child_node = None

def process_file_retail(JMA, extension, game_title, retail_version_list, report):
    JMA.version = int(JMA.next())
    if not JMA.version in retail_version_list:
        raise global_functions.ParseError("Importer does not support this " + extension + " version")

    JMA.game_title = game_title
    if game_title == 'auto':
        JMA.game_title = global_functions.get_game_title(JMA.version, 'JMA')

    if JMA.version >= 16395:
        read_header_16394(JMA)
        read_nodes_16394(JMA)
        read_node_transforms_16390(JMA)
        read_root_transforms_16395(JMA, extension)
        validate_node_graph_16394(JMA, report)

    elif JMA.version == 16394:
        read_header_16394(JMA)
        read_nodes_16394(JMA)
        read_node_transforms_16390(JMA)
        validate_node_graph_16394(JMA, report)

    elif JMA.version >= 16392:
        read_header_16390(JMA)
        read_nodes_16392(JMA)
        read_node_transforms_16390(JMA)
        validate_node_graph_16392(JMA, report)

    elif JMA.version == 16391:
        read_header_16390(JMA)
        read_nodes_16391(JMA)
        read_node_transforms_16390(JMA)

    elif JMA.version == 16390:
        read_header_16390(JMA)
        read_node_transforms_16390(JMA)

    if JMA.left() != 0: # is something wrong with the parser?
        report({'WARNING'}, "%s elements left after parse end" % JMA.left())

    return JMA
