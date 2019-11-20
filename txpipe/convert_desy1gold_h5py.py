from astropy.io import fits
import h5py
import os

def add_col(array, name):
    total_length = len(array)
    f.create_dataset('photometry/'+name,maxshape=(total_length,), 
                     shape=(total_length,), chunks=True) 
    f['photometry/'+name][:total_length] = array


# List all files for gold catalog (public realease des y1), one file per tile
path = '/global/projecta/projectdirs/lsst/groups/WL/projects/wl-txpipe-hack/DESY1/gold/'
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
print files[0]
print len(files)


# Compile all tiles into a single array
ra, dec = [], []
objid = []

for f in files[0:5]:
    hdu = fits.open(path+f)
    ra.extend(hdu[1].data['RA'])
    dec.extend(hdu[1].data['DEC'])
    objid.extend(hdu[1].data['COADD_OBJECTS_ID'])
    

# Write output into a h5py file
outfile = 'photometry_catalog.h5py'
f = h5py.File(outfile, 'w')

add_col(ra, 'ra')
add_col(dec, 'dec')
add_col(objid, 'objectId')

f.close()

