# -*- coding: utf-8 -*-
import luigi
import subprocess, shlex
import utils
import os
import pdb
from dotenv import load_dotenv, find_dotenv

#python modis_luigi.py GetRasters --local-scheduler --workers 7

# NOTE: sudo apt-get install libhdf4-dev // install HDF4 driver so GDAL recognizes the MODIS format files
load_dotenv(find_dotenv())

user = os.environ.get("MODIS_USER")
password = os.environ.get("MODIS_PASSWORD")

class DownloadTile(luigi.Task):
    """
    Download Files for
    """
    date_tile_file = luigi.Parameter()
    date_tile_url = luigi.Parameter()
    modis_prod_url = luigi.Parameter()
    path_d = luigi.Parameter()

    def output(self):
        """
        This class expects to find an HDF file
        """
        return luigi.LocalTarget(self.path_d + self.date_tile_file)

    def run(self):
        """
        Downloads a MODIS-HDF file
        """
        cmd = ' '.join(['wget', '--user=' + user, '--password=' + password, '-O', self.path_d +  self.date_tile_file, self.date_tile_url])
        subprocess.call(cmd, shell=True)

        # Check if downloaded file is empty
        utils.check_empty_file(self.path_d + self.date_tile_file)

class Tile2GeoTiff(luigi.Task):
    """
    Converting horrible HDF files into GeoTiffs
    """
    date_tile_file = luigi.Parameter()
    date_tile_url = luigi.Parameter()
    modis_prod_url = luigi.Parameter()
    path_d = luigi.Parameter()

    def requires(self):
        # Check if file is empty
        utils.check_empty_file(self.path_d + self.date_tile_file)

        return DownloadTile(date_tile_file=self.date_tile_file, date_tile_url=self.date_tile_url, modis_prod_url=self.modis_prod_url, path_d = self.path_d)

    def run(self):
        preffix = 'HDF4_EOS:EOS_GRID:"'
        suffix = utils.get_suffix(self.modis_prod_url)


        cmd_list = ['gdal_translate', preffix + self.path_d + self.date_tile_file + suffix, self.path_d + self.date_tile_file + '.tiff']
        cmd = ' '.join(cmd_list)

        # We use shlex and Popen to deal with spaces in HDF layer names and with quotes marks in the command line
        cm_sp = shlex.split(cmd, posix=False)
        subprocess.Popen(cmd, shell=True)

    def output(self):
        return luigi.LocalTarget(self.path_d + self.date_tile_file + '.tiff')



class MergeDayTiles(luigi.Task):
    """
    Class for merging  all the Tiles associated to a specific day
    """
    date = luigi.Parameter()
    modis_prod_url = luigi.Parameter()
    tiles_path = luigi.Parameter()
    mosaic_path = luigi.Parameter()

    def requires(self):
        tiles = utils.create_tiles()
        tile_urls = utils.get_tile_urls(self.modis_prod_url, self.date, tiles)
        path_d = self.tiles_path + '/' + self.date + '/'
        utils.valid_path(path_d)
        tile_files = [tile + '.hdf' for tile in tiles]
        for tile_url, tile_file in zip(tile_urls, tile_files):
            yield Tile2GeoTiff(date_tile_file=tile_file, date_tile_url=tile_url, modis_prod_url=self.modis_prod_url, path_d = path_d)

    def run(self):
        path_d = self.tiles_path + '/' + self.date + '/'
        utils.valid_path(path_d)
        cmd_list = ['gdal_merge.py', '-o', self.mosaic_path, '-a_nodata "-2000"','$(ls ' +path_d + '/*.tiff)']
        subprocess.call(' '.join(cmd_list), shell=True)

    def output(self):
        return luigi.LocalTarget(self.mosaic_path)

class ReprojectMosaic(luigi.Task):
    """
    Reproject a GEOTIFF file (in this case, a mosaic) into INEGI-friendly format
    """

    date = luigi.Parameter()
    modis_prod_url = luigi.Parameter(default="https://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.006/")
    tiles_path = luigi.Parameter()
    prep_path = luigi.Parameter()
    reprojec_path = luigi.Parameter()

    def requires(self):
        mosaic_path = self.prep_path + '/'+ self.date + '_mosaic.tiff'
        return MergeDayTiles(date = self.date, modis_prod_url = self.modis_prod_url, tiles_path = self.tiles_path, mosaic_path = mosaic_path)

    def run(self):
        mosaic_path = self.prep_path + '/' + self.date + '_mosaic.tiff'
        s_srs = '"+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"'
        t_srs = '"+proj=lcc +lat_1=17.5 +lat_2=29.5 +lat_0=12 +lon_0=-102 +       x_0=2500000 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"'
        cmd_list = ['gdalwarp', '-s_srs', s_srs, '-t_srs', t_srs, mosaic_path, self.reprojec_path]
        subprocess.call(' '.join(cmd_list), shell=True)

    def output(self):
        return luigi.LocalTarget(self.reprojec_path)


class CropRaster(luigi.Task):
    """
    Using a shapefile (of Mexico), crop GEOTIFFs and get only info from within the shapefile. The GEOTIFF and shapefile must be in the same projection.
    """

    date = luigi.Parameter()
    modis_prod_url = luigi.Parameter(default="https://e4ftl01.cr.usgs.gov/MOLT/MOD13A2.006/")
    tiles_path = luigi.Parameter(default='tmp')
    shp_path = luigi.Parameter(default='../shp/simplified.shp')
    prep_path = luigi.Parameter(default='prep')

    def requires(self):
        utils.valid_path(self.prep_path)
        reprojec_path = self.prep_path + '/' + self.date + '_reproj.tiff'
        return ReprojectMosaic(date = self.date, modis_prod_url = self.modis_prod_url, tiles_path = self.tiles_path, reprojec_path = reprojec_path, prep_path = self.prep_path)

    def run(self):
        crop_path = self.prep_path + '/' + self.date + '_crop.tiff'
        reprojec_path = self.prep_path + '/' + self.date + '_reproj.tiff'
        cmd_list = ['gdalwarp', '-cutline', self.shp_path, '-crop_to_cutline', '-dstnodata -2000', reprojec_path, crop_path]

        subprocess.call(' '.join(cmd_list), shell=True)

    def output(self):
        crop_path = self.prep_path + '/' + self.date + '_crop.tiff'
        return luigi.LocalTarget(crop_path)

class GetRasters(luigi.Task):
    """
    Class get raster images of Mexico, by means of, for each date, (a) downloading tiles in HDF format, (b) converting it to GeoTiff files, (c) mosaicing the tiles to create a single file (d) reprojecting the raster to an INEGI-friendly format, (e) croping to a shapefile.
    """
    start =  luigi.Parameter('DEFAULT')
    end = luigi.Parameter('DEFAULT')
    product = luigi.Parameter('DEFAULT')

    def requires(self):
        modis_prod_url = 'https://e4ftl01.cr.usgs.gov/MOLT/' + self.product + '/'
        dates = utils.get_valid_dates(self.start, self.end, modis_prod_url)
        tiles_path = 'tmp_' + self.product
        prep_path = 'prep_' + self.product
        yield [CropRaster(date=date, modis_prod_url=modis_prod_url, tiles_path = tiles_path, prep_path = prep_path) for date in dates]


class SavitzkyGolayFilter(luigi.WrapperTask):
    period_start = luigi.configuration.get_config().get('SavitzkyGolayFilter' ,'period_start')
    period_end = luigi.configuration.get_config().get('SavitzkyGolayFilter' ,    'period_end')
    sg_window = luigi.configuration.get_config().get('SavitzkyGolayFilter' ,    'sg_window')
    sg_degree = luigi.configuration.get_config().get('SavitzkyGolayFilter' ,    'sg_degree')
    product = luigi.configuration.get_config().get('DEFAULT', 'product')
    def requires(self):
        modis_prod_url = 'https://e4ftl01.cr.usgs.gov/MOLT/' + self.product + '/'
        yield [GetRasters(self.period_start, self.period_end, self.product)]



if __name__ == '__main__':
        luigi.run()
