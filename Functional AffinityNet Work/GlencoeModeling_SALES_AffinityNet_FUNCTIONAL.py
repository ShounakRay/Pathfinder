# @Author: Shounak Ray <Ray>
# @Date:   31-Jul-2020 16:07:99:996  GMT-0600
# @Email:  rijshouray@gmail.com
# @Filename: GlencoeModeling_SALES_AffinityNet_FUNCTIONAL.py
# @Last modified by:   Ray
# @Last modified time: 23-Feb-2021 16:02:80:805  GMT-0700
# @License: [Private IP]


import collections
import itertools
import random
from datetime import datetime, timedelta
from itertools import chain

# Import packages
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pydot

# dataframe as input
# make Dataframe
# create a card

_ = """
#######################################################################################################################
########################################################### MASTER CONTROL VARIABLES ##################################
####################################################################################################################"""

df_feature_list = ['member_name', 'status', 'date', 'item_group', 'item_name', 'service_provider']
df_contigencies = []
top_CONST = 100                  # how many top nodes reported (isolation)
edge_class = 'service_provider'        # node/edge-connection attributes
strat_class = 'item_group'      # class for color stratification
buttons = False                 # clustering options for the graph
groups_attribute = True         # specify group for each node for auto-coloring
groups_option = True            # Default coloring already takes place, extra clustering functions are absent/inactive
physics_option = True           # specify physics for graph motion
node_labels = True              # specify labels for each node
manipulation_option = True      # specify manipulation values
interaction_option = True       # specify interaction values
layout_option = True            # specify layout appearance
shape_name = 'dot'              # shape of each node
shapes = True                   # specify if shape should be set
node_values = False                  # specify node size
VIS_parameters = [buttons, groups_attribute, groups_option, physics_option, node_labels,
                  manipulation_option, interaction_option, layout_option, shapes, node_values]

_ = """
#######################################################################################################################
################################################## DATA PROCESSING ####################################################
####################################################################################################################"""

df_SALES = pd.read_csv("/Users/Ray/Documents/DeepSea data/Glencoe/Original/SALES.csv", low_memory=False)

# REMOVE MOST NON-FINAL COLUMNS
df_SALES = df_SALES[df_feature_list + df_contigencies]
# DROP ALL NA ROWS
df_SALES.dropna(inplace=True)
# UPPER-CASE ALL VALUES
df_SALES['member_name'] = df_SALES['member_name'].str.upper()
df_SALES['status'] = df_SALES['status'].str.upper()
df_SALES['item_group'] = df_SALES['item_group'].str.upper()
df_SALES['item_name'] = df_SALES['item_name'].str.upper()
df_SALES['service_provider'] = df_SALES['service_provider'].str.upper()
# Converts date col to datetime for sorting and future comparison purposes
df_SALES['date'] = pd.to_datetime(df_SALES['date'])
df_SALES.sort_values(by='date', inplace=True)
df_SALES = df_SALES.reset_index(inplace=False).drop('index', axis=1, inplace=False)

_ = """
#######################################################################################################################
################################################## AFFINITY NET PREPARATION ###########################################
####################################################################################################################"""

# Drop useless columns
SALES_Ed = df_SALES.groupby(by=['member_name'], axis=0)[edge_class]

# Store all completegraphs in list, store all nodes in list, store all edges in list
test = [nx.complete_graph(list(list(w)[1].values)) for w in SALES_Ed]
# complete_EDGE_LIST = list(chain.from_iterable([w.edges for w in test]))
# Equate (re-order) A->B + B->A edges since Graph is Undirected
complete_EDGE_LIST = [tuple(sorted(tup)) for tup in list(chain.from_iterable([w.edges for w in test]))]

_ = """
############# ############# ############# ############# ############# ########
############# NEED TO UPDATE THIS SECTION FOR JSON # TO BE UPDATED ###########
############# ############# ############# ############# ############# #####"""
# DETERMINING EDGE WEIGHTS
edge_weights = collections.Counter(complete_EDGE_LIST)
first_edge_value, second_edge_value = [i[0] for i in list(edge_weights)], [i[1] for i in list(edge_weights)]
edge_weights_3t = list(zip(first_edge_value, second_edge_value, list(edge_weights.values()))
                       )  # based on edge weights (SORT, THEN TRUNCATE)
edge_weights_3t = sorted(edge_weights_3t, key=lambda tup: tup[2])[-top_CONST:]    # get the top edges
# Edge-weight order and iterations are independent from node lists and node-strat_class dictionary
nodes_from_edges_1 = [tup[0] for tup in edge_weights_3t]
nodes_from_edges_2 = [tup[1] for tup in edge_weights_3t]
nodes_from_edges = list(set(nodes_from_edges_1 + nodes_from_edges_2))

_ = """
#######################################################################################################################
################################################## HTML/JS/JSON WRITING TO FILE #######################################
####################################################################################################################"""
# CONDENSE COLUMNS (keep edge class and strat class)
# CONDENSE ROWS (keep ones in finalized node list)
# Sort dictionary as same order as finalized node list
# > Create the dictionary that defines the order for sorting
# > Generate a rank column that will be used to sort the dataframe numerically
# Remove escape characters in Dataframe columns
# Convert from Dataframe to dictionary
# Convert dictionary values from single-item list to string
df__dict_stratClass_edgeClass = (df_SALES[[edge_class] + [strat_class]]
                                 ).drop_duplicates(edge_class).reset_index().drop('index', 1)
df__dict_stratClass_edgeClass = df__dict_stratClass_edgeClass[df__dict_stratClass_edgeClass[edge_class].isin(
    nodes_from_edges)].reset_index().drop('index', 1)
sorterIndex = dict(zip(nodes_from_edges, range(len(nodes_from_edges))))
df__dict_stratClass_edgeClass['edge_class temp rank'] = df__dict_stratClass_edgeClass[edge_class].map(sorterIndex)
df__dict_stratClass_edgeClass.sort_values('edge_class temp rank', inplace=True)
df__dict_stratClass_edgeClass.drop('edge_class temp rank', 1, inplace=True)
df__dict_stratClass_edgeClass[edge_class] = df__dict_stratClass_edgeClass[edge_class].str.replace(
    '"', '\\"').str.replace("'", "\\'")
df__dict_stratClass_edgeClass[strat_class] = df__dict_stratClass_edgeClass[strat_class].str.replace(
    '"', '\\"').str.replace("'", "\\'")
dict_stratClass_edgeClass = df__dict_stratClass_edgeClass.set_index(edge_class).T.to_dict('list')
dict_stratClass_edgeClass = {str(k): str(v[0]) for k, v in dict_stratClass_edgeClass.items()}

edge_weights_3t = [(edge[0].replace('"', '\\"').replace("'", "\\'"), edge[1].replace(
    '"', '\\"').replace("'", "\\'"), edge[2]) for edge in edge_weights_3t]
d_keys_ind = list(dict_stratClass_edgeClass.keys())
d_values_ind = list(dict_stratClass_edgeClass.values())

# List of colors for each individual group based on strat class
for dummy in list(set(d_values_ind)):
    color_map = ["#" + color for color in list(set(["%06x" % random.randint(0, 0xFFFFFF)
                                                    for i in list(set(d_values_ind))]))]
color_strat_dict = dict(zip(d_values_ind, color_map))
# RENDERING TIME IS EXTREMELY SLOW: POSSIBLY NORMALIZE THE WEIGHTS (LN 457), LOOKS VERY MESSY
# (ASSESS REPULSION AND PHYSICS, STABILIZATION METRICS (DECREASE -> FASTER?))
# GENERATE SPECIALIZED JSON FILE
HTML_FILE = """"""
HTML_FILE += """
<html>
    <head>
        <script type="text/javascript" src="/Users/Ray/Documents/Python/Glencoe/vis/dist/vis.js"></script>
        <link href="/Users/Ray/Documents/Python/Glencoe/vis/dist/vis.css" rel="stylesheet" type="text/css" />
        <style type="text/css">
            #mynetwork
            {
                width: 100%;
                height: 100%;
                border: 1px solid lightgray;
            }
        </style>
    </head>"""
HTML_FILE += """
    <body>"""
if(buttons):
    HTML_FILE += """
        <p>These are different clustering methods (functions)</p>
        <input type = "button" onclick = "clusterByConnection()" value = "Cluster 'None' by connections">
        <br>
        <input type = "button" onclick = "clusterByHubsize()" value = "Cluster by hubsize">
        <br>
        <input type = "button" onclick = "clusterByColor()" value = "Cluster by color">"""
HTML_FILE += """
        <div id = "mynetwork"></div>"""
HTML_FILE += """
        <script type="text/javascript">"""
HTML_FILE += """
            var nodes = new vis.DataSet([\n"""
for node_num in range(len(dict_stratClass_edgeClass)):
    HTML_FILE += """                {"id": """ + str(node_num)
    if(node_labels):
        HTML_FILE += """, "label": """ + "\"" + d_keys_ind[node_num] + "\""
    if(groups_attribute):
        HTML_FILE += """, "group": """ + "\"" + d_values_ind[node_num] + "\""
    if(shapes):
        HTML_FILE += """, "shape": """ + "\"" + shape_name + "\""
    # if(node_values):
        # HTML_FILE += """, "value": """ + "\"" + someListUndefined + "\""
    if(node_num <= len(dict_stratClass_edgeClass) - 2):
        HTML_FILE += """},\n"""
    else:
        HTML_FILE += """}\n"""
HTML_FILE += """            ]);"""
HTML_FILE += """
            var edges = new vis.DataSet([\n"""
c = 0
for edge in edge_weights_3t:
    HTML_FILE += """                {"from": """ + str(d_keys_ind.index(edge[0]))
    HTML_FILE += """, "to": """ + str(d_keys_ind.index(edge[1]))
    HTML_FILE += """, "value": """ + str(edge[2])
    HTML_FILE += """, "color": {'inherit': 'both'}"""
    # if(edge_labels):
    # do something but there's no code here since what in the world would be the label (maybe the weight?)
    if(c <= len(edge_weights_3t) - 2):
        HTML_FILE += """},\n"""
    else:
        HTML_FILE += """}\n"""
    c += 1
HTML_FILE += """            ]);"""
HTML_FILE += """
            var container = document.getElementById('mynetwork');
            var data = {
                nodes: nodes,
                edges: edges
            };"""
HTML_FILE += """
            var options = {"""
if(groups_option):
    HTML_FILE += """
                groups: {\n"""
    for strat_num in range(len(list(color_strat_dict.keys()))):
        HTML_FILE += """\t\t\t\t\t\t\t\t\t\'""" + str(list(color_strat_dict.keys())[
                                                      strat_num]) + "\': {color: {background: """ + "\"" + list(color_strat_dict.values())[strat_num] + """\"}, borderWidth: 1"""
        if(strat_num <= len(list(color_strat_dict.keys())) - 2):
            HTML_FILE += """},\n"""
        else:
            HTML_FILE += """}\n"""
    HTML_FILE += """                }"""
if(physics_option):
    if(groups_option):
        HTML_FILE += ","
    HTML_FILE += """
                "physics": {
                    "enabled": true,
                    "barnesHut": {
                        "theta": 0.5,
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.3,
                        "springLength": 95,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0
                    },
                    "forceAtlas2Based": {
                        "theta": 0.5,
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springConstant": 0.01,
                        "springLength": 100,
                        "damping": 0.4,
                        "avoidOverlap": 0.6
                    },
                    "repulsion": {
                        "centralGravity": 0.2,
                        "springLength": 200,
                        "springConstant": 0.05,
                        "nodeDistance": 100,
                        "damping": 0.09
                    },
                    "maxVelocity": 50,
                    "minVelocity": 0.1,
                    "solver": 'forceAtlas2Based',
                    "stabilization": {
                      "enabled": true,
                      "iterations": 1000,
                      "updateInterval": 100,
                      "onlyDynamicEdges": false,
                      "fit": true
                    },
                    "adaptiveTimestep": true
                }"""
if(manipulation_option):
    if(physics_option):
        HTML_FILE += ","
    HTML_FILE += """
                "manipulation": {
                    "enabled": true,
                    "initiallyActive":false,
                    "addNode":true,
                    "addEdge":true,
                    "editNode":undefined,
                    "editEdge":true
                }"""
if(interaction_option):
    if(manipulation_option):
        HTML_FILE += ","
    HTML_FILE += """
                "interaction": {
                    "dragNodes":true,
                    "dragView": true,
                    "hideEdgesOnDrag": false,
                    "hideEdgesOnZoom": true,
                    "hideNodesOnDrag": false,
                    "hover": true,
                    "hoverConnectedEdges": true,
                    "multiselect": true,
                    "navigationButtons": false,
                    "selectable": true,
                    "selectConnectedEdges": true,
                    "tooltipDelay": 0,
                    "zoomView": true
                }"""
if(layout_option):
    if(interaction_option):
        HTML_FILE += ","
    HTML_FILE += """
                "layout": {
                    "improvedLayout": true,
                    "clusterThreshold":250
                }"""
HTML_FILE += """
            };"""
HTML_FILE += """
            var network = new vis.Network(container, data, options)

            network.on("selectNode", function(params)
            {
                if (params.nodes.length == 1)
                {
                    if (network.isCluster(params.nodes[0]) == true)
                    {
                        network.openCluster(params.nodes[0]);
                    }
                }
            });"""
if(buttons):
    HTML_FILE += """
            function clusterByConnection()
            {
                network.setData(data);
                network.clusterByConnection(1)
            }
            function clusterByHubsize()
            {
                network.setData(data);
                var clusterOptionsByData =
                {
                    processProperties: function(clusterOptions, childNodes)
                    {
                        clusterOptions.label = "<" + childNodes.length + ">";
                        return clusterOptions;
                    },
                    clusterNodeProperties: {borderWidth:4, shape:'database', font:{size:29}}
                };
                network.clusterByHubsize(undefined, clusterOptionsByData);
            }
            function clusterByColor()
            {
                network.setData(data);
                var colors = ['blue','pink','green'];
                var clusterOptionsByData;
                for (var i = 0; i < colors.length; i++)
                {
                    var color = colors[i];
                    clusterOptionsByData =
                    {
                        joinCondition: function (childOptions)
                        {
                        return childOptions.color.background == color;
                        },
                        processProperties: function (clusterOptions, childNodes, childEdges)
                        {
                            var totalMass = 0;
                            for (var i = 0; i < childNodes.length; i++)
                            {
                                totalMass += childNodes[i].mass;
                            }
                            clusterOptions.mass = totalMass;
                            return clusterOptions;
                        },
                        clusterNodeProperties: {id: 'cluster:' + color,
                                                borderWidth: 3,
                                                shape: 'database',
                                                color: color,
                                                label: 'color:' + color}
                    };
                    network.cluster(clusterOptionsByData);
                }
            }"""
HTML_FILE += """
        </script>
    </body>
</html>"""

with open("OG_FILE_Glencoe_AUTO%MAIN-" + str(edge_class) + "%STRAT-" + str(strat_class) + "%RANK-" +
          str(top_CONST) + ".html", "w") as file:
    file.write(HTML_FILE)

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
