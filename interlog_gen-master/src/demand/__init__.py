import logging
import math
import random
from dataclasses import dataclass
from zipfile import ZipFile
from pathlib import Path

import shapely
import pandas
import geopandas
import json

from geopandas import GeoDataFrame
from shapely import Point

from streetnetwork import get_region_polygon
from utils import init_logger

LOG = init_logger(logging.getLogger(__name__))


@dataclass
class OSMAddress:
    id: int
    point: tuple[float, float]
    tags: str


def load_addresses(filepath, target_folder) -> GeoDataFrame:
    """
    Load addresses from OSM extracted with overpass and stored as json
    (default: Munich_address.json), and retain only ones which are within
    the problem region.

    :param filepath:
    :param target_folder:
    :return:
    """
    region = get_region_polygon(f"{target_folder}/region.geojsonl.json")

    if filepath.endswith(".zip"):
        with ZipFile(filepath) as myzip:
            path = Path(filepath)
            myzip.extract(path.stem, path.parent)

    with open(filepath.rsplit(".zip")[0]) as f:
        addresses = [OSMAddress(node["id"], Point(node["lon"], node["lat"]), str(node["tags"]).replace(";", ",")) for
                     node in json.load(f)["elements"] if node["type"] == "node"]
        gdf = GeoDataFrame(data=addresses, geometry=[Point(it.point) for it in addresses])
        gdf = gdf.loc[gdf.within(region)]
        return gdf


def merge_addresses_and_districts(addresses_file, pop_dens_w_poly_file, target_folder):
    """
    Combine addresses from OSM extracted with overpass and stored as json with the district
    information from the population density with polygon file.

    :param addresses_file:
    :param pop_dens_w_poly_file:
    :param target_folder:
    :return:
    """
    geo_addresses = load_addresses(addresses_file, target_folder)
    pop_dense_poly = geopandas.read_file(pop_dens_w_poly_file)

    merged = geo_addresses.sjoin(pop_dense_poly[["fid", "geometry"]], how="left", predicate="within")[
        ["id", "point", "fid"]]
    merged.rename(columns={"point": "Coordinate"}, inplace=True)
    merged[~merged["fid"].isnull()].to_csv(f"{target_folder}/addresses.csv", index=False, sep=";")


def load_preprocessed_addresses(region_folder) -> pandas.DataFrame:
    return pandas.read_csv(f"{region_folder}/addresses.csv", sep=";")


@dataclass
class District:
    fid: int
    name: str
    rel_population: float


def load_districts(gis_folder="./resources/gis/") -> dict[int, District]:
    res = pandas.read_csv(f"{gis_folder}/Pop_dens.csv")
    res["PopulationShare"] = res["Population"] / res["Population"].sum()
    res = {row["fid"]: District(fid=row["fid"], name=row["NAME"], rel_population=row["PopulationShare"]) for _, row in
           res.iterrows()}
    return res


def sample_customers(num_customers: int, gis_folder, preprocessed_folder="./resources/preprocessed/munich/"):
    districts = load_districts(gis_folder)
    addresses = load_preprocessed_addresses(preprocessed_folder)

    # fixed number selected per district
    # ?alternative: assign weight per entry and use DataFrame.sample
    num_per_district = {fid: int(math.floor(d.rel_population * num_customers)) for fid, d in districts.items()}
    remaining = num_customers - sum(num_per_district.values())

    # distribute remaining
    if remaining > 0:
        for fid in random.sample([fid for fid in num_per_district.keys()], k=remaining):
            num_per_district[fid] += 1

    # now sample
    selected = {}
    for fid in districts.keys():
        selected[fid] = addresses[addresses["fid"] == fid].sample(num_per_district[fid])

    selected_frame = pandas.concat(list(selected.values()))

    return selected_frame


def prepare_locker_locations(filepath, pop_dens_w_poly, target_folder):
    """
    Create a reduced locker_locations.csv in the target folder. To do so, we read the
    original locker file and only retain those, which are within the problem region.
    Afterward, we add the district id (fid) by cross-referencing the location with the
    population density polygon data.

    :param filepath:
    :param pop_dens_w_poly:
    :param target_folder:
    """
    region = get_region_polygon(f"{target_folder}/region.geojsonl.json")
    lockers = pandas.read_csv(filepath)
    lockers['Coordinate'] = lockers.apply(lambda row: shapely.Point(row['Longitude'], row['Latitude']), axis=1)
    gdf = GeoDataFrame(data=lockers, geometry="Coordinate")
    gdf = gdf.loc[gdf.within(region)]
    gdf.drop(["Latitude", "Longitude"], axis=1, inplace=True)
    gdf['id'] = gdf.index

    pop_dense_poly = geopandas.read_file(pop_dens_w_poly)
    merged = gdf.sjoin(pop_dense_poly[["fid", "geometry"]], how="left", predicate="within")[
        ["id", "Coordinate", "fid"]]
    merged[~merged["fid"].isnull()].to_csv(f"{target_folder}/locker_locations.csv", sep=";", index=False)


def load_preprocessed_lockers(preprocessed_folder) -> pandas.DataFrame:
    return pandas.read_csv(f"{preprocessed_folder}/locker_locations.csv", sep=";")


def sample_lockers(num_lockers: int, gis_folder, preprocessed_folder="./resources/preprocessed/munich/"):
    districts = load_districts(gis_folder)
    addresses = load_preprocessed_lockers(preprocessed_folder)

    # fixed number selected per district
    # ?alternative: assign weight per entry and use DataFrame.sample
    num_per_district = {fid: int(math.floor(d.rel_population * num_lockers)) for fid, d in districts.items()}
    remaining = num_lockers - sum(num_per_district.values())

    # distribute remaining
    if remaining > 0:
        for fid in random.sample([fid for fid in num_per_district.keys()], k=remaining):
            num_per_district[fid] += 1

    # now sample
    selected = {}
    for fid in districts.keys():
        selected[fid] = addresses[addresses["fid"] == fid].sample(num_per_district[fid])

    selected_frame = pandas.concat(list(selected.values()))

    return selected_frame
