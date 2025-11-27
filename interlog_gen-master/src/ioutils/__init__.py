import pandas
from geopandas import GeoDataFrame
from networkx import MultiDiGraph

from distancematrix import (DistanceMatrixRowsT, generate_all_pair_shortest_paths,
                            generate_all_pair_shortest_paths_multiprocessing)


def export_distancematrix_to_csv(path: str, dm: DistanceMatrixRowsT):
    pandas.DataFrame(dm, columns=["Origin","Destination","Distance"]).to_csv(path, header=False)


def export_distancematrix_multiprocessing(path: str, mapped_nodes: GeoDataFrame, network_osm: MultiDiGraph, num_processes):
    with open(path, "w") as f:
        for rows in generate_all_pair_shortest_paths_multiprocessing(mapped_nodes, network_osm, num_processes=num_processes):
            for row in rows:
                f.write(",".join(map(lambda x: str(x), row)))
                f.write("\n")


def export_distancematrix(path: str, mapped_nodes: GeoDataFrame, network_osm: MultiDiGraph):
    with open(path, "w") as f:
        for row in generate_all_pair_shortest_paths(mapped_nodes, network_osm, with_linestring=False):
            f.write(",".join(map(lambda x: str(x), row)))
            f.write("\n")


def export_distancematrix_with_linestrings(path: str, mapped_nodes: GeoDataFrame, network_osm: MultiDiGraph):
    with open(path, "w") as f:
        for row in generate_all_pair_shortest_paths(mapped_nodes, network_osm, with_linestring=True):
            f.write(",".join(map(lambda x: str(x), row)))
            f.write("\n")




