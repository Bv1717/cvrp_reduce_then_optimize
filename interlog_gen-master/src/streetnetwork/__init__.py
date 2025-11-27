import logging
import os.path
from enum import Enum
import networkx as nx

import osmnx
import shapely

from utils import init_logger, Region

LOG = init_logger(logging.getLogger(__name__))


def get_region_polygon(geojson_path) -> shapely.Polygon:
    with open(geojson_path) as f:
        return shapely.from_geojson(f.read())


class NetworkType(Enum):
    Drive = 'drive'
    Bike = 'bike'


def load_osmnx_network_from_polygon(region: Region, polygon: shapely.Polygon, network_type: NetworkType,
                                    no_cache: bool = False):
    cache_folder = f'./resources/cache'
    path = f"{cache_folder}/{region.name}.{network_type.name}.graphml"

    if not no_cache and os.path.exists(path):
        LOG.info(f"loading network from cache (type: {network_type.name}, location: {path})")
        network = osmnx.load_graphml(path)
    else:
        LOG.info(f"loading network from osm (type: {network_type.name})")
        network = osmnx.graph_from_polygon(polygon, network_type=network_type.value, simplify=False)
        network = network.subgraph(max(nx.strongly_connected_components(network), key=len)).copy()
        if not no_cache:
            osmnx.save_graphml(network, path)

    return network
