import logging
import math
from multiprocessing import Pool

import osmnx
from shapely import LineString
import networkx as nx
from typing import Generator

from tqdm import tqdm

from utils import init_logger

LOG = init_logger(logging.getLogger(__name__))

DistanceMatrixRowT = tuple[int, int, float] | tuple[int, int, float, LineString]
DistanceMatrixRowsT = list[DistanceMatrixRowT]


def generate_all_pair_shortest_paths(mapped_nodes, osm_network_graph, with_linestring: bool = True) -> Generator[
    DistanceMatrixRowT, None, None]:
    nodes, edges = osmnx.graph_to_gdfs(osm_network_graph)
    iterrows = mapped_nodes.iterrows()

    for i, i_row in iterrows:
        origin_osmid = i_row['osmid']
        dist, paths = nx.single_source_dijkstra(osm_network_graph, origin_osmid, weight='length')

        for j, j_row in tqdm(mapped_nodes.iterrows(), desc=f"origin {i} ", total=len(mapped_nodes), unit=""):
            if j == i:
                if with_linestring:
                    yield (j, j, 0, "LINESTRING ()")
                else:
                    yield (j, j, 0)
                continue
            target_osmid = j_row['osmid']
            try:
                dist[target_osmid]
            except:
                print(j, j_row['osmid'])
                continue

            if with_linestring:
                geoms = [edges.loc[(u, v, 0), 'geometry'] for u, v in
                         zip(paths[target_osmid][:-1], paths[target_osmid][1:])]
                if len(geoms) == 0:
                    linestring_xy = "LINESTRING ()"
                else:
                    linestring_xy = LineString(
                        [[geoms[0].xy[0][0], geoms[0].xy[1][0]]] + [[geoms[k].xy[0][l], geoms[k].xy[1][l]]
                                                                    for k in range(0, len(geoms))
                                                                    for l in range(1, len(geoms[k].xy[0]))])
                yield (i, j, math.ceil(dist[target_osmid]), linestring_xy)
            else:
                yield (i, j, math.ceil(dist[target_osmid]))


def gen_one_to_all_sp(args) -> DistanceMatrixRowsT:
    origin, origin_osmid, mapped_nodes, osm_network_graph = args
    i = origin
    rows = []
    dist, _ = nx.single_source_dijkstra(osm_network_graph, origin_osmid, weight='length')
    for j, j_row in mapped_nodes.iterrows():
        if j == i:
            rows.append((j, j, 0))
        else:
            target_osmid = j_row['osmid']
            rows.append((i, j, math.ceil(dist[target_osmid])))

    LOG.info(f"partial distance matrix for rows with origin {i} (osm: {origin_osmid}) finished.")
    return rows


def generate_all_pair_shortest_paths_multiprocessing(mapped_nodes, osm_network_graph, num_processes) -> list[
    DistanceMatrixRowsT]:
    with Pool(num_processes) as pool:
        return pool.map(gen_one_to_all_sp,
                        [(i, i_row['osmid'], mapped_nodes, osm_network_graph) for i, i_row in mapped_nodes.iterrows()])
