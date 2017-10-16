import utils
import preprocess_utils as pu
import luigi
import time
import pickle
import numpy as no


from luigi import configuration


class StackRasters(luigi.Task):
    """
    Stacking of all date rasters to consider for clusterizing.
    Note: ideally these rasters have been mosaiced, reprojected and croped in the shape of Mexico.
    """
    dates = luigi.Parameter() # pass
    stack_path = luigi.Parameter() # pass
    target_raster_path = luigi.Parameter(default='prep_MOD13A2.006/') # Add default
    suffix = luigi.Parameter('_crop.tiff') # Add default

    #def requires(self):
    #    dates_paths = [self.target_raster_path + date + self.crop_suffix for date in self.dates]
    #    return [luigi.LocalTarget(date_p) for date_p in dates_paths]

    def run(self):
        rasters = utils.rasters_stack(self.dates, self.target_raster_path, self.suffix)
        pickle.dump(rasters, open(self.stack_path, 'wb'))

    def output(self):
        return luigi.LocalTarget(self.stack_path)

class DistanceRasters(luigi.Task):
    """
Create horizontal and vertical distance rasters for NDVI and EVI data in Mexico.
    """
    dates = luigi.Parameter() #pass
    raster_name = luigi.Parameter() # pass
    stacks_path = luigi.Parameter(default='tmp/') #Add default
    dimension = luigi.Parameter() #pass
    reference_raster = luigi.Parameter(default='prep_MOD13A2.006/2004.01.01_crop.tiff') #Add default

    def requires(self):
        stack_path =  self.stacks_path + self.dates[0] + '-' + self.dates[-1] + '.p'
        yield StackRasters(dates = self.dates,
                stack_path = stack_path)

    def run(self):
        stack_path =  self.stacks_path + self.dates[0] + '-' + self.dates[-1] +'.p'
        stack = pickle.load(open(stack_path, "rb"))
        if self.dimension == 'h':
            dist_mat = pu.pixel_distance_rows(stack)
        elif self.dimension == 'v':
            dist_mat = pu.pixel_distance_cols(stack)
        #raster_name = self.dist_raster_path + self.dates[0] + '-' + self.dates[-1] + '_' + self.dimension + '.tiff'
        utils.write_raster(dist_mat, self.raster_name, self.reference_raster)

    def output(self):
        #raster_name = self.dist_raster_path + self.dates[0] + '-' + self.dates[-1] + '_' + self.dimension + '.tiff'
        return luigi.LocalTarget(self.raster_name)

class MunShape(luigi.Task):
"""
Creates a municipality shapefile
"""
    municipalities_shape_path = luigi.Parameter(default='../shp/muns.shp') #Add default
    cve_muni = luigi.Parameter() #Pass
    mun_shp_path = luigi.Parameter() #Pass

    #def requires(self):
    #    return luigi.LocalTarget(self.municipalities_shape_path)

    def run(self):
        utils.create_mun_shp(self.cve_muni, self.mun_shp_path, self.municipalities_shape_path)
        time.sleep(2) # Sometimes shp files take a second to create

    def output(self):
        return luigi.LocalTarget(self.mun_shp_path)

class CropDistanceRasters(luigi.Task):
"""
    Function for croping a distance raster in the shape of a municipio. Requires a municipio shapefile (that can be created via MunShape, and a raster file with temporal distances between pixels, created by DistanceRasters.

"""
    dates = luigi.Parameter() # Pass
    cve_muni = luigi.Parameter() # Pass
    dimension = luigi.Parameter() # Pass
    dist_raster_path = luigi.Parameter(default='tmp/') # Add default
    cropmun_raster_path = luigi.Parameter() # Pass
    muns_shps_path = luigi.Parameter(default='tmp/') # Add default

    # Create stack_path
    def requires(self):
        mun_shp_path = self.muns_shps_path + self.cve_muni + '.shp'
        raster_name = utils.date_rasternames(self.dist_raster_path, self.dates,    dimension = self.dimension)
        yield MunShape(cve_muni = self.cve_muni,
                mun_shp_path = mun_shp_path)
        yield DistanceRasters(dates = self.dates,
                dimension = self.dimension,
                raster_name = raster_name)

    def run(self):
        mun_shp_path = self.muns_shps_path + self.cve_muni + '.shp'
        raster_name = utils.date_rasternames(self.dist_raster_path, self.dates, dimension = self.dimension)
        utils.crop_raster(mun_shp_path, raster_name, self.cropmun_raster_path)

    def output(self):
        return luigi.LocalTarget(self.cropmun_raster_path)


class Clusters1D(luigi.Task):

    dates = luigi.Parameter() #Pass
    cve_muni = luigi.Parameter() #Pass
    dimension = luigi.Parameter() #Pass
    cutoff = luigi.Parameter() #Pass
    cluster_1d_path = luigi.Parameter() #Pass
    crop_raster_path = luigi.Parameter(default='tmp/') #Add default
    reference_raster = luigi.Parameter(default='prep_MOD13A2.006/2004.01.01_crop.tiff') #Add default

    def requires(self):
        cropmun_raster_path = utils.date_rasternames(self.crop_raster_path, self.dates, self.dimension, self.cve_muni)
        yield CropDistanceRasters(dates = self.dates,
                cve_muni = self.cve_muni,
                dimension = self.dimension,
                cropmun_raster_path = cropmun_raster_path)

    def run(self):
        cropmun_raster_path = utils.date_rasternames(self.crop_raster_path, self.dates, self.dimension, self.cve_muni)
        dist_array = utils.raster_to_numpy(cropmun_raster_path)
        if self.dimension == 'h':
            clust = pu.neighbor_clusters_rows(dist_array, float(self.cutoff))
        elif self.dimension == 'v':
            clust = pu.neighbor_clusters_cols(dist_array, float(self.cutoff))
        utils.write_raster(clust, self.cluster_1d_path, cropmun_raster_path)

    def output(self):
        return luigi.LocalTarget(self.cluster_1d_path)


class CreateCluster(luigi.Task):

    dates = luigi.Parameter()# Pass
    cve_muni = luigi.Parameter()# Pass
    cutoff = luigi.Parameter() # Pass
    clusters_1d_path = luigi.Parameter(default='tmp/') # Add default
    clusters_path = luigi.Parameter(default='clusters/') # Add default
    reference_raster = luigi.Parameter(default='prep_MOD13A2.006/2004.01.01_crop.tiff') # Add default
    def requires(self):
        dims = ['h', 'v']
        for dim in dims:
            cluster_1d_path = utils.date_rasternames(self.clusters_path, self.dates, dimension=dim, cve_muni = self.cve_muni, cutoff = self.cutoff)
            yield Clusters1D(dates = self.dates,
                    cve_muni = self.cve_muni,
                    dimension = dim,
                    cutoff = self.cutoff,
                    cluster_1d_path = cluster_1d_path)
    def run(self):
        h_path = utils.date_rasternames(self.clusters_path, self.dates, dimension='h', cve_muni = self.cve_muni, cutoff = self.cutoff)
        v_path = utils.date_rasternames(self.clusters_path, self.dates, dimension='v', cve_muni = self.cve_muni, cutoff = self.cutoff)
        h_clust = utils.raster_to_numpy(h_path)
        v_clust = utils.raster_to_numpy(v_path)
        cluster = pu.cluster_join(h_clust, v_clust)
        cluster_path = utils.date_rasternames(self.clusters_path, self.dates, cve_muni = self.cve_muni, cutoff = self.cutoff)
        cluster_u = pu.unify_clusters(cluster)
        utils.write_raster(cluster_u, cluster_path, h_path)

    def output(self):
        cluster_path = utils.date_rasternames(self.clusters_path, self.dates, cve_muni = self.cve_muni, cutoff = self.cutoff)
        return luigi.LocalTarget(cluster_path)

class GetClusters(luigi.WrapperTask):
    start = luigi.Parameter(default='2004.01.01')
    end = luigi.Parameter(default='2004.10.01')
    cutoffs = luigi.Parameter(default=['3', '5', '7'])
    cve_muni = luigi.Parameter(default='25017')

    def requires(self):
        dates = utils.get_valid_dates(self.start, self.end) # Add Mods Product
        for cutoff in self.cutoffs:
            yield CreateCluster(dates=dates,
                    cutoff=cutoff,
                    cve_muni=self.cve_muni)
if __name__ == '__main__':
    luigi.run()
