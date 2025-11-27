from dataclasses import dataclass
import logging
import sys
from logging import Logger

import osmnx
from scipy.spatial import cKDTree
import pandas as pd
import geopandas as gpd
import numpy as np


def init_logger(logger: Logger) -> Logger:
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_format = logging.Formatter('[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    return logger


@dataclass
class Region:
    name: str
    geojson_filepath: str


''' OSM node ids we blacklist explicitly (either located on a highway, or unreachable) '''
MUNICH_BLACKLIST_NODES = [1788376015, 300247533, 1582278230, 28099475, 1038270087, 5837932500, 5837932501,
                          1591359267, 4921250906, 260493620, 1887115026, 639010494, 4921250905]


# https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
def ckdnearest(nodes_to_match: gpd.GeoDataFrame, graph, nodes_to_match_against: set[int]):
    """
    Mapmatching of coordinates to nodes on the graph, see
    https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas

    :param nodes_to_match:
    :param graph:
    :param nodes_to_match_against:
    :return:
    """
    graph_nodes = osmnx.graph_to_gdfs(graph, nodes=True, edges=False)
    graph_nodes = graph_nodes[~graph_nodes.index.isin(MUNICH_BLACKLIST_NODES)]
    graph_nodes = graph_nodes[graph_nodes.index.isin(nodes_to_match_against)]
    nA = np.array(list(nodes_to_match.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(graph_nodes.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = graph_nodes.iloc[idx]
    gdf = pd.concat(
        [
            nodes_to_match.reset_index(drop=True).drop(columns='geometry'),
            pd.Series(graph_nodes.iloc[idx].index, name='osmid'),
            pd.Series(gdB_nearest.reset_index(drop=True)['geometry'], name='geometry'),
        ],
        axis=1)

    return gdf
