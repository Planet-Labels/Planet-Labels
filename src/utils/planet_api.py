__author__ = 'vincent'

from planet import api
import requests
import urllib
import os
import pytz
import datetime
import geojson
import const
import optparse
import csv
import random

def download_image(image_directory, url, key):
    print url
    r = requests.get(url, stream=True, auth=(key, ''))
    if 'content-disposition' in r.headers:
        local_filename = r.headers['content-disposition'] \
            .split("filename=")[-1].strip("'\"")
        print local_filename
    else:
        local_filename = '.'.join(url.split('/')[-2:])

    print os.path.join(image_directory, local_filename)

    with open(os.path.join(image_directory, local_filename), 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return local_filename

def fetch_lat_lon(image_dir, lat_lon_list, count = 1, scene_type = 'rapideye', product = 'analytic'):
    """
    Download a single rapideye image of each coordinate in a list of coordinates.

    :param image_directory: The directory where the images should be downloaded to.
    :param lat_lon_list: A list of coordinate tuples in the form of (lat, lon)
    :param scene_type: The type of satellite that the images should be from ('ortho' or 'rapideye')
    :param product: The image type. 'analytic' or 'visual'.
    :return:
    """
    points = [geojson.Point([lon, lat]) for lat, lon in lat_lon_list]
    intersects = [geojson.dumps(point) for point in points]
    sceneIDs = []

    url = "https://api.planet.com/v0/scenes/rapideye/"

    for intersect in intersects:

        params = {
            "cloud_cover.estimated.lte": 0,
            "intersects":intersect,
        }

        data = requests.get(url, params=params, auth=(const.API_KEY, ''))
        scenes_data = data.json()["features"]

        if not scenes_data:
            print "No scenes available for these coordinates: ", intersect
            continue

        counter = 0
        for scene in scenes_data:
            if counter == count: break
            sceneIDs.append(scene['id'])
            print scene['id']
            counter += 1

    print "Downloading scene list!"
    planet_client = api.Client(api_key = const.API_KEY)

    try:
        callback = api.write_to_file(image_dir)
        bodies = planet_client.fetch_scene_geotiffs(scene_ids = sceneIDs, product = product, scene_type=scene_type, callback=callback)

        for b in bodies:
            b.await()
    except Exception, e:
        print "Download failed: %s"%e

def fetch_images(image_directory, scene_type = 'rapideye', product = 'analytic', filters = {}):
    """

    :param image_directory: The directory where the original images should be saved to.
    :param scene_type: The satellite type that images should be fetched from.
    :param product: The type of imagery that will be downloaded. "analytic" is not pre-processed, "visual" is true-color.
    :param filters: Any filters for the search.
    :return:
    """

    if not os.path.isdir(image_directory):
        raise OSError("The specified path " + image_directory + " does not point to a directory!")

    if not filters:
        start = datetime.datetime(year=2011, month=1, day=1, tzinfo=pytz.utc).isoformat()
        end = datetime.datetime(year=2015, month=12, day=1, tzinfo=pytz.utc).isoformat()

        filters = {
            # Your filters here, for example:
            # Get images with estimated 0% cloud cover
            "cloud_cover.estimated.lte": 0,
            "acquired.gte": start,
            "acquired.lte": end
        }

    next_url = "https://api.planet.com/v0/scenes/" + scene_type + "/?" + urllib.urlencode(filters)

    scene_data_pages = []
    scene_IDs = []
    scene_count = 0

    print "Searching for %s images that comply with the given filters." % scene_type
    print "Scene List:"
    while next_url:
        # Note: you don't have to pass the filters in again here,
        # here, they will always be included in data.links.next
        r = requests.get(next_url, auth=(const.API_KEY, ''))
        r.raise_for_status()
        data = r.json()
        scenes_data = data["features"]
        scene_data_pages.append(scenes_data)

        for scene in scenes_data:
            scene_IDs.append(str(scene['id']))
            print str(scene['id'])

        scene_count += len(scenes_data)

        next_url = data["links"].get("next", None)

    print '\n%s total results' % scene_count
    print "Downloading scene list!"

    planet_client = api.Client(api_key = const.API_KEY)

    callback = api.write_to_file(image_directory)
    bodies = planet_client.fetch_scene_geotiffs(scene_IDs, scene_type = scene_type, product = product,  callback=callback)
    # await the completion of the asynchronous downloads, this is where
    # any exception handling should be performed
    for b in bodies:
        b.await()


def download_US_cities(no_cities, image_dir):
    fetch_lat_lon(image_dir = image_dir, lat_lon_list=const.CITY_COORDINATES)


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-p', '--path', dest='path', default=const.ORIGINALS_DIR)
    parser.add_option('-t', '--latitude', dest='lat', type=float, default=-1)
    parser.add_option('-n', '--longitude', dest='lon', type=float, default=-1)
    parser.add_option('-s', '--single', dest = 'single', default = False, action='store_true')

    (options, _) = parser.parse_args()

    fetch_lat_lon(image_dir = options.path, lat_lon_list=const.VISUALIZATION_COORDINATES, count = 2)


