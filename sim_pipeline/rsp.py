import numpy as np
from matplotlib.transforms import Affine2D
import lsst.geom as geom
# Source injection
from lsst.pipe.tasks.insertFakes import _add_fake_sources
from sim_pipeline.image_simulation import galsimobj_true_flux
import galsim
from astropy.table import Table, vstack
from sim_pipeline.image_simulation import sharp_image


def DC2_cutout(ra, dec, num_pix, butler, band):
    """
    Draws a cutout from the DC2 data based on the given ra, dec pair. For this one needs to provide
    a butler to this function. To initiate Butler, you need to specify data configuration and 
    collection of the data.
    
    :param ra: ra for the cutout
    :param dec: dec for the cutout
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param: band: image band
    :returns: cutout image
    """
    skymap = butler.get("skyMap")
    point = geom.SpherePoint(ra, dec, geom.degrees)
    cutoutSize = geom.ExtentI(num_pix, num_pix)
    #print(cutoutSize)


    #Read this from the table we have at hand... 
    tractInfo = skymap.findTract(point)
    patchInfo = tractInfo.findPatch(point)
    my_tract = tractInfo.tract_id
    my_patch = patchInfo.getSequentialIndex()
    xy = geom.PointI(tractInfo.getWcs().skyToPixel(point))
    bbox = geom.BoxI(xy - cutoutSize//2, cutoutSize)
    coaddId_r = {
        'tract':my_tract, 
        'patch':my_patch,
        'band': band
    }
    coadd_cut_r = butler.get("deepCoadd", parameters={'bbox':bbox}, dataId=coaddId_r)
    return coadd_cut_r
 
    
def lens_inejection(lens_pop, ra, dec, num_pix, delta_pix, butler, lens_cut=None, flux=None):
    """
    Chooses a random lens from the lens population and injects it to a DC2 cutout image. For this 
    one needs to provide a butler to this function. To initiate Butler, you need to specify data 
    configuration and collection of the data.
    
    :param lens_pop: lens population from sim-pipeline
    :param ra: ra for the cutout
    :param dec: dec for the cutout
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param lens_cut: list of criteria for lens selection
    :param flux: flux need to be asigned to the lens image. It sould be None
    :param: path: path to save the output
    :returns: An astropy table containing Injected lens in r-band, DC2 cutout image in r-band, 
    cutout image with injected lens in r, g , and i band 
    """   
    #lens = sim_lens
    if lens_cut == None:
        kwargs_lens_cut = {}
    else:
        kwargs_lens_cut = lens_cut
    
    rgb_band_list=['r', 'g', 'i']
    lens_class = lens_pop.select_lens_at_random(**kwargs_lens_cut)
    theta_E=lens_class.einstein_radius
    skymap = butler.get("skyMap")
    point = geom.SpherePoint(ra, dec, geom.degrees)
    cutoutSize = geom.ExtentI(num_pix, num_pix)
    #Read this from the table we have at hand 
    tractInfo = skymap.findTract(point)
    patchInfo = tractInfo.findPatch(point)
    my_tract = tractInfo.tract_id
    my_patch = patchInfo.getSequentialIndex()
    xy = geom.PointI(tractInfo.getWcs().skyToPixel(point))
    bbox = geom.BoxI(xy - cutoutSize//2, cutoutSize)
    injected_final_image = []
    #band_report = []
    box_center = []
    cutout_image = []
    lens_image=[]
    for band in rgb_band_list:
        coaddId_r = {
            'tract':my_tract, 
            'patch':my_patch,
            'band': band
        }
        
        #coadd cutout image
        coadd_cut_r = butler.get("deepCoadd", parameters={'bbox':bbox}, dataId=coaddId_r)
        lens=sharp_image(lens_class=lens_class, band=band, mag_zero_point=27, delta_pix=delta_pix, 
                         num_pix=num_pix)
        if flux == None:
            gsobj = galsimobj_true_flux(lens, pix_scale=delta_pix)
        else:
            gsobj = galsim.InterpolatedImage(galsim.Image(lens), scale = delta_pix, flux = flux)

        wcs_r= coadd_cut_r.getWcs()
        bbox_r= coadd_cut_r.getBBox()
        x_min_r = bbox_r.getMinX()
        y_min_r = bbox_r.getMinY()
        x_max_r = bbox_r.getMaxX()
        y_max_r = bbox_r.getMaxY()

        # Calculate the center coordinates
        x_center_r = (x_min_r + x_max_r) / 2
        y_center_r = (y_min_r + y_max_r) / 2

        center_r = geom.Point2D(x_center_r, y_center_r)
        #geom.Point2D(26679, 15614)
        point_r=wcs_r.pixelToSky(center_r)
        ra_degrees = point_r.getRa().asDegrees()
        dec_degrees = point_r.getDec().asDegrees()
        center =(ra_degrees, dec_degrees)

        #image_r = butler.get("deepCoadd", parameters={'bbox':bbox_r}, dataId=coaddId_r)
        mat = np.eye(3)
        mat[:2,:2] = wcs_r.getCdMatrix()

        transform = Affine2D(mat)
        arr_r = np.copy(coadd_cut_r.image.array)

        _add_fake_sources(coadd_cut_r, [(point_r, gsobj)])
        inj_arr_r = coadd_cut_r.image.array
        injected_final_image.append(inj_arr_r)
        #band_report.append(band)
        box_center.append(center)
        cutout_image.append(arr_r)
        lens_image.append((inj_arr_r-arr_r))

    t = Table([[lens_image[0]], [cutout_image[0]],[injected_final_image[0]], 
               [injected_final_image[1]], [injected_final_image[2]], [box_center[0]]], 
               names=('lens','cutout_image','injected_lens_r', 'injected_lens_g', 'injected_lens_i'
                      , 'cutout_center'))
    return t


def multiple_lens_injection(lens_pop, ra, dec, num_pix, delta_pix, butler,lens_cut=None, flux=None):
    """
    Draws multiple DC2 cutout images and injects random lenses from the lens population. For this 
    one needs to provide a butler to this function. To initiate Butler, you need to specify data 
    configuration and collection of the data.
    
    :param lens_pop: lens population from sim-pipeline
    :param ra: ra for a cutout
    :param dec: dec for a cutout
    :param num_pix: number of pixel for the cutout
    :param delta_pix: pixel scale for the lens image
    :param butler: butler object
    :param flux: flux need to be asigned to the lens image. It sould be None
    :param: path: path to save the output
    :returns: An astropy table containing Injected lenses in r-band, DC2 cutout images in r-band, 
    cutout images with injected lens in r, g , and i band for a given set of ra and dec
    """
    injected_images=[]
    for i in range(len(ra)):
        injected_images.append(lens_inejection(lens_pop, ra[i], dec[i], num_pix, delta_pix, butler, 
                                               lens_cut=None, flux=None))
    injected_image_catalog=vstack(injected_images)
    return injected_image_catalog

    