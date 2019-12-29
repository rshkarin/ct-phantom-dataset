import collections
import sys
import warnings
import os
import numpy as np
from pymrt import geometry
from scipy import ndimage as ndi
from skimage import io


Dataset = collections.namedtuple('Dataset', ['data', 'label'])


def report_progress(action, name, idx, length, width=50):
    """
    Provides a fancy progress reporting.
    """
    if idx % 5 == 0:
        ratio = idx / length
        filled = '=' * int(ratio * width)
        rest = '-' * (width - int(ratio * width))
        sys.stdout.write('\r' + action + ': ' + name + ' [' + filled + rest + '] ' + str(int(ratio*100.)) + '%')
        sys.stdout.flush()


def save_data(data, output_dir, outfile_fmt='slice_{0:05d}.tif'):
    """
    Save data as a set of tiff files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, dslc in enumerate(data):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            io.imsave(os.path.join(output_dir, outfile_fmt.format(i)), dslc)
            report_progress('Saving', output_dir, i, data.shape[0], 25)


def save_dataset(dataset, output_dir, data_fld='data', label_fld='label', \
                 outfile_fmt='slice_{0:05d}.tif'):
    """
    Save a dataset as a set of tiff files.
    """    
    save_data(dataset.data, os.path.join(output_dir, data_fld), outfile_fmt=outfile_fmt)
    if dataset.label is not None:
        save_data(dataset.label, os.path.join(output_dir, label_fld),
                  outfile_fmt=outfile_fmt)


def randomly_rotate(data, label=None, theta_range=(-15, 15), phi_range=(-15, 15)):
    """
    Randomly rotates a 3D dataset around its center within the specified angular ranges.
    """
    theta = np.random.uniform(low=theta_range[0], high=theta_range[-1])
    phi = np.random.uniform(low=phi_range[0], high=phi_range[-1])
    
    theta, phi = 5, 5
    
    drot = ndi.rotate(data, phi, axes=(0,2), order=3, reshape=False)
    drot = ndi.rotate(drot, theta, axes=(1,2), order=3, reshape=False)
    lrot = None
    if label is not None:
        lrot = ndi.rotate(label, phi, axes=(0,2), order=0, reshape=False)
        lrot = ndi.rotate(lrot, theta, axes=(1,2), order=0, reshape=False)
        
    return drot, lrot


def extend_bounding_box(bbox, offset_perc, shape_limit=None):
    """
    Extend a bounding box along each axis by a specified percentage
    by maintaining the shape limit.
    """
    lengths = [(bb.stop-bb.start) for bb in bbox]
    offsets = [int(np.round(l*offset_perc)) for l in lengths]
    if shape_limit is not None:
        bbox = tuple(slice(max(slc.start-offset, 0), min(slc.stop+offset, lim)) \
                             for slc, offset, lim in zip(bbox, offsets, shape_limit))
    else:
        bbox = tuple(slice(max(slc.start-offset, 0), slc.stop+offset) \
                             for slc, offset in zip(bbox, offsets))
    return bbox


def add_texture(shape, radius=3, step=7):
    """
    Generates a 3D volumes filled with spheres of specified radius and step.
    """
    output_texture = np.zeros(shape, dtype=np.bool)

    for z in range(0, shape[0], step):
        for y in range(0, shape[1], step):
            for x in range(0, shape[2], step):
                point = [z, y, x]
                idxs = geometry.sphere(shape, radius, position=np.array(point) / np.array(shape))
                output_texture[idxs] = True
        
    return output_texture


def sample_random_array(input_arr, std):
    return np.array(input_arr) + ((-2*std) * np.random.random_sample((3,)) + std)
    

def generate_phantom(components, size=512, original_size=1024, random_seed=None, 
                     gaussian_sigma=None, random_rotation_angles=None, textured_structures=None,
                     texture_radius=2, texture_step=6):
    """
    Generates a 3D phantom dataset of the specified size from a set of specified components.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    scale_factor = size / float(original_size)
    data, label = np.zeros((size, size, size), dtype=np.float32), \
                  np.zeros((size, size, size), dtype=np.uint8)
  
    for key, structure_params in components.items():
        if structure_params['type'] == 'ellipsoid':
            size_rnd = sample_random_array(structure_params['size'], structure_params['size_std']) * scale_factor
            pos_rnd = sample_random_array(structure_params['pos'], structure_params['pos_std']) * scale_factor / size
            idxs = geometry.ellipsoid(size, size_rnd, pos_rnd)
            
            texture_structure = (textured_structures is not None) and (key in textured_structures)
            if texture_structure:
                bbox = ndi.measurements.find_objects(idxs)[0]
                bbox = extend_bounding_box(bbox, 0.1)
                idxs_loc = idxs[bbox]
                texture_idxs = add_texture(idxs_loc.shape, radius=texture_radius, step=texture_step)
                texture_idxs = np.logical_and(texture_idxs, idxs_loc)
            
            data[idxs] = structure_params['value']
            if texture_structure:
                data[bbox][texture_idxs] = structure_params['value'] + 20
            
            if structure_params['label_id'] is not None:
                label[idxs] = structure_params['label_id']
            
    if gaussian_sigma is not None:
        data = ndi.gaussian_filter(data, gaussian_sigma)
            
    if random_rotation_angles is not None:
        data, label = randomly_rotate(data, label=label, 
                                      theta_range=random_rotation_angles[0], 
                                      phi_range=random_rotation_angles[-1])
            
    return data, label


def generate_phantoms(num_datasets, components, output_dir, size=512, original_size=1024, 
                      random_seed=None, gaussian_sigma=None, random_rotation_angles=None,
                      textured_structures=None, texture_radius=2, texture_step=6):
    """
    Runs a process of generating specified number of datasets from a specified set of components.
    """
    for nd in range(num_datasets):
        ph_data, ph_label = generate_phantom(components, size=size, 
                                             original_size=original_size, 
                                             gaussian_sigma=gaussian_sigma, 
                                             random_rotation_angles=random_rotation_angles,
                                             textured_structures=textured_structures,
                                             texture_radius=texture_radius,
                                             texture_step=texture_step)
        output_path = os.path.join(output_dir, 'phantom_{0:02d}'.format(nd), 'original')
        save_dataset(Dataset(data=ph_data, label=ph_label), output_path)


def main():
    components = collections.OrderedDict()
    components['container'] = dict(type='ellipsoid', size=(455., 256., 256.), pos=(512., 512., 512.), value=30, pos_std=0, size_std=10, label_id=None)
    components['sphere1'] = dict(type='ellipsoid', size=(40., 40., 40.), pos=(796., 412., 471.), value=130, pos_std=10, size_std=10, label_id=1)
    components['sphere2'] = dict(type='ellipsoid', size=(50., 50., 50.), pos=(662., 515., 385.), value=160, pos_std=10, size_std=15, label_id=2)
    components['sphere3'] = dict(type='ellipsoid', size=(125.0, 55.0, 55.0), pos=(299., 610., 512.), value=190, pos_std=10, size_std=15, label_id=3)
    components['sphere4'] = dict(type='ellipsoid', size=(75.0, 75.0, 75.0), pos=(456., 472., 650.), value=230, pos_std=10, size_std=15, label_id=4)

    generate_phantoms(1, components, 'simulation_datasets', 
                      gaussian_sigma=0.75, random_rotation_angles=((-2.5,2.5), (-2.5,2.5)),
                      textured_structures=['sphere1', 'sphere2', 'sphere3', 'sphere4'])

if __name__ == "__main__":
    main()
