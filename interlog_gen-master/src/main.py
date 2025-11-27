import csv
import logging
import os
import pathlib
import random
import time
from os import mkdir
from os.path import exists
from typing import Optional

import click
import pandas
import shapely
from tqdm import tqdm

import demand
from ioutils import export_distancematrix, export_distancematrix_multiprocessing
from streetnetwork import load_osmnx_network_from_polygon, NetworkType
from utils import init_logger, Region

import pandas as pd
import geopandas as gpd
from utils import ckdnearest

LOG = init_logger(logging.getLogger(__name__))


def generate_distance_matrix_files(region: Region, instance_folder: str, network_types: set[NetworkType],
                                   num_processes: int = 1, no_cache: bool = False):
    assert num_processes > 0, "num_processes must be greater than 0"

    t = time.time()
    LOG.info(f"generate distance matrix file(s) for {region.name} -"
             f" {[nt.name for nt in network_types]} in {instance_folder}")

    LOG.info(f"load region polygon from {region.geojson_filepath}")
    with open(region.geojson_filepath) as f:
        shape = shapely.from_geojson(f.read())

    LOG.info(f"load instance from {instance_folder}")
    depots = pd.read_csv(f"{instance_folder}/Depot.csv")
    customers = pd.read_csv(f"{instance_folder}/Customer.csv")
    if os.path.exists(f"{instance_folder}/Satellite.csv"):
        satellites = pd.read_csv(f"{instance_folder}/Satellite.csv")
        instance = pd.concat([depots, satellites, customers]).set_index("Index")
    else:
        instance = pd.concat([depots, customers]).set_index("Index")
    

    instance_gdf = gpd.GeoDataFrame(instance, geometry=gpd.points_from_xy(instance.Y, instance.X,
                                                                          crs="EPSG:4326"))

    instance_gdf.to_csv(f"{instance_folder}/all_w_geom.csv")

    # load networks
    osm_networks = {nt: load_osmnx_network_from_polygon(region, shape, nt, no_cache=no_cache) for nt in network_types}
    # identify nodes available in all network types to be used in the mapmatching
    common_nodes = None
    for nt in network_types:
        network_osm = osm_networks[nt]
        if common_nodes is None:
            common_nodes = set(network_osm.nodes.keys())
        else:
            common_nodes.intersection_update(set(network_osm.nodes.keys()))

    for nt in network_types:
        dm_path = f"{instance_folder}/dm_{nt.value}.csv"
        if exists(dm_path):
            LOG.info(f"{nt.value} routes file exists already - skipped")
        else:
            LOG.info(f"calculate distance matrix for {nt.value} routes")
            network_osm = osm_networks[nt]
            mapped_nodes = ckdnearest(instance_gdf, network_osm, common_nodes)
            if num_processes == 1:
                export_distancematrix(dm_path, mapped_nodes, network_osm)
            else:
                export_distancematrix_multiprocessing(dm_path, mapped_nodes, network_osm, num_processes)

    LOG.info(f"finished after {round(time.time() - t, ndigits=3)}s")


def find_linestring_and_fill_map(linestring_filepath: str, sorted_legs, leg_map: dict[tuple[int, int], str]):
    iterable = iter(sorted_legs)
    next_leg = next(iterable)
    with open(linestring_filepath, "r") as f:
        for line in tqdm(f, unit="", desc="reading line"):
            # 0,1,5658,LINESTRING (11.491183 48.2063312, 11.491306 48.2061854
            row = line.rstrip().split(sep=",", maxsplit=3)
            if (int(row[0]), int(row[1])) == next_leg:
                leg_map[next_leg] = row[3]
                next_leg = next(iterable, None)
                if next_leg is None:
                    break

    return


def extract_route_linestrings(path: str):
    # read itineraries from file
    df = pd.read_csv(path, header=0)
    itineraries: list[tuple[str, list[int]]] = list()
    legs: dict[str, list[tuple[int, int]]] = {"T": list(), "B": list()}
    for _, row in df.iterrows():
        # Index, Type, Route
        if not str(row["Type"]) in legs:
            raise RuntimeError(f"Type {row['Type']} not supported")
        itinerary = list(map(lambda x: int(x), str(row["Route"]).split(sep="-")))
        itineraries.append((str(row["Type"]), itinerary))
        for i, j in zip(itinerary[:-1], itinerary[1:]):
            legs[str(row["Type"])].append((i, j))

    leg_to_linestring_map: dict[str, dict[tuple[int, int], str]] = {"T": dict(), "B": dict()}
    for kind, dm_path in [("T", "./resources/instances/500_1_2-20/dm_drive_w_geom.csv"),
                          ("B", "./resources/instances/500_1_2-20/dm_bike_w_geom.csv")]:
        # create mapping and sort pairs
        sorted_legs_of_type = sorted(legs[kind])

        # read lines from linestring file
        find_linestring_and_fill_map(dm_path, sorted_legs_of_type, leg_to_linestring_map[kind])

    route_linestrings = list()
    for kind, itinerary in tqdm(itineraries, unit="", desc="processing itineraries"):
        route_linestring = None
        for idx, j in zip(itinerary[0:-1], itinerary[1:]):
            linestring = leg_to_linestring_map[kind][(idx, j)]
            if route_linestring is None:
                route_linestring = linestring[:-1]
            else:
                route_linestring += "," + linestring[len("LINESTRING ("):-1]
        route_linestring += ")"
        route_linestrings.append(route_linestring)

    df["Geometry"] = pd.Series(route_linestrings)
    path = pathlib.Path(path)
    df.to_csv(path.parent.joinpath(f"{path.stem}_w_geom{path.suffix}"), sep=";", quoting=csv.QUOTE_NONE)


def sample(instance_folder: str, num_lockers: int, num_customers: int, seed: int,
           gis_folder="./interlog_gen-master/resources/gis/",
           preprocessed_folder="./interlog_gen-master/resources/preprocessed/munich/"):
    random.seed(seed)

    def coordinates_from_point_string(point: str):
        it = point.split("(")[1].split(")")[0].split(" ")
        return float(it[0]), float(it[1])

    # Demand/Capacity, EarliestTime, LatestTime, ServiceTime, Cost, VehicleCompatibility, SatelliteCompatibility
    default_earliest = "08:00"
    default_latest = "17:00"
    default_service_time = 0
    default_satellite_cost = 36
    default_vehicle_comp = 1
    default_satellite_comp = 1
    default_capacity = 865
    default_demand = 11

    lockers_sample = demand.sample_lockers(num_lockers, gis_folder, preprocessed_folder)
    if not lockers_sample.empty:
        satellite_frame = pandas.DataFrame(lockers_sample["id"])
        satellite_frame["Type"] = "satellite"
        satellite_frame["Name"] = lockers_sample["id"]
        satellite_frame["X"] = lockers_sample.apply(lambda row: coordinates_from_point_string(row["Coordinate"])[1], axis=1)
        satellite_frame["Y"] = lockers_sample.apply(lambda row: coordinates_from_point_string(row["Coordinate"])[0], axis=1)
        satellite_frame = satellite_frame.reset_index().drop(["id"], axis=1)
        satellite_frame["index"] = satellite_frame.index + 1
        satellite_frame.rename(columns={"index": "Index"}, inplace=True)

        # Demand/Capacity, EarliestTime, LatestTime, ServiceTime, Cost, VehicleCompatibility, SatelliteCompatibility
        satellite_frame["Demand/Capacity"] = default_capacity
        satellite_frame["EarliestTime"] = default_earliest
        satellite_frame["LatestTime"] = default_latest
        satellite_frame["ServiceTime"] = default_service_time
        satellite_frame["Cost"] = default_satellite_cost
        satellite_frame["VehicleCompatibility"] = default_vehicle_comp
        satellite_frame["SatelliteCompatibility"] = default_satellite_comp

        satellite_frame.to_csv(f"{instance_folder}/Satellite.csv", sep=",", index=False)

    # Index, Type, Name, X, Y, Demand.Capacity, EarliestTime, LatestTime, ServiceTime, Cost, initialAssignment, VehicleCompatibility, SatelliteCompatibility
    customers_sample = demand.sample_customers(num_customers, gis_folder, preprocessed_folder)
    customer_frame = pandas.DataFrame(customers_sample["id"])
    customer_frame["Type"] = "customer"
    customer_frame["Name"] = customers_sample["id"]
    customer_frame["X"] = customers_sample.apply(lambda row: coordinates_from_point_string(row["Coordinate"])[1],
                                                 axis=1)
    customer_frame["Y"] = customers_sample.apply(lambda row: coordinates_from_point_string(row["Coordinate"])[0],
                                                 axis=1)
    customer_frame = customer_frame.reset_index().drop(["id"], axis=1)
    customer_frame["index"] = customer_frame.index + num_lockers + 1
    customer_frame.rename(columns={"index": "Index"}, inplace=True)

    # Demand/Capacity, EarliestTime, LatestTime, ServiceTime, Cost, VehicleCompatibility, SatelliteCompatibility
    customer_frame["Demand/Capacity"] = default_demand
    customer_frame["EarliestTime"] = default_earliest
    customer_frame["LatestTime"] = default_latest
    customer_frame["ServiceTime"] = default_service_time
    customer_frame["Cost"] = 0
    customer_frame["VehicleCompatibility"] = default_vehicle_comp
    customer_frame["SatelliteCompatibility"] = default_satellite_comp

    customer_frame.to_csv(f"{instance_folder}/Customer.csv", sep=",", index=False)

    # copy depot location from gis data
    depot_frame = pd.read_csv(f"{gis_folder}/Depot.csv", sep=",")
    depot_frame.to_csv(f"{instance_folder}/Depot.csv", sep=",", index=False)


def process(instance_folder: str, network_types: Optional[set[NetworkType]] = None, num_processes: int = 1,
            no_cache: bool = False):
    if network_types is None:
        network_types = {NetworkType.Drive, NetworkType.Bike}
    region = Region(name="Munich", geojson_filepath="./interlog_gen-master/resources/gis/region.geojsonl.json")
    generate_distance_matrix_files(region, instance_folder, network_types, num_processes, no_cache)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--gis-folder", type=str, default="./interlog_gen-master/resources/gis/", help="path to the GIS data folder")
@click.option("--preprocess-folder", type=str, default="./interlog_gen-master/resources/preprocessed/munich",
              help="path to the targeted preprocessed data folder")
def preprocess(gis_folder: str, preprocess_folder: str):
    LOG.info(f"check whether files are available in GIS folder ({gis_folder})")

    region_bounds_file = f"{gis_folder}/region.geojsonl.json"
    addresses = f"{gis_folder}/Munich_addresses.json.zip"
    pop_dens_w_poly = f"{gis_folder}/Pop_dens_w_poly.csv"
    locker_locations = f"{gis_folder}/locker_locations.csv"

    required_files = [addresses, pop_dens_w_poly, locker_locations]
    missing_files = [file for file in required_files if not os.path.exists(file)]
    if len(missing_files) > 0:
        LOG.error("Cannot find files: %s", missing_files)
        exit(-1)

    if not os.path.exists(preprocess_folder):
        mkdir(preprocess_folder)

    t = time.time()
    LOG.info("copy region bounds (geojsonl)")
    with open(region_bounds_file, "r") as infile:
        with open(f"{preprocess_folder}/region.geojsonl.json", "w") as outfile:
            outfile.writelines(infile.readlines())
    LOG.info("prepare addresses and districts")
    demand.merge_addresses_and_districts(addresses, pop_dens_w_poly, preprocess_folder)
    LOG.info("prepare locker locations")
    demand.prepare_locker_locations(f"{gis_folder}/locker_locations.csv", pop_dens_w_poly,
                                    preprocess_folder)

    LOG.info(f"preprocessing finished after {time.time() - t}s")
    LOG.info(f"`- preprocessed files located in {preprocess_folder}")


@cli.command()
@click.option("--customers", required=True, type=int, help="Number of customers to sample")
@click.option("--satellites", required=True, type=int, help="Number of satellites to sample")
@click.option("--seed", type=int, help="Random number generator seed")
@click.option("--instance-parent-folder", type=str, default="./interlog_gen-master/resources/instances/",
              help="path to the preprocessed data folder")
@click.option("--network-types", required=True, type=click.Choice(['drive', 'bike']), default=["drive", "bike"],
              multiple=True, help="network types to generate")
@click.option("--num-processes", type=int, default=1, help="Number of processes used for DM calculations")
@click.option("--no-cache", is_flag=True, help="Disable use of cache for osm network handling")
def generate(customers: int, satellites: int, seed: int, instance_parent_folder: str, network_types: tuple[str],
             num_processes: int, no_cache: bool):
    num_customers = customers
    num_lockers = satellites
    instance_folder = f"{instance_parent_folder}/{num_customers}_{num_lockers}_{seed}/"
    if not os.path.exists(instance_folder):
        mkdir(instance_folder)

    sample(instance_folder, num_lockers, num_customers, seed)
    process(instance_folder, set([NetworkType(nt) for nt in network_types]), num_processes, no_cache)


@cli.command()
@click.option("--instance-folder", required=True, type=str, help="instance folder to process")
@click.option("--network-types", required=True, type=click.Choice(['drive', 'bike']), default=["drive", "bike"],
              multiple=True, help="network types to generate")
@click.option("--num-processes", type=int, default=1, help="Number of processes used for DM calculations")
@click.option("--no-cache", is_flag=True, help="Disable use of cache for osm network handling")
def process_only(instance_folder: str, network_types: tuple[str], num_processes: int, no_cache: bool):
    process(instance_folder, set([NetworkType(nt) for nt in network_types]), num_processes, no_cache)


if __name__ == '__main__':
    cli()
