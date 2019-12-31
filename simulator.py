#!/usr/bin/env python

"""Simulate crystal X-ray diffraction pattern for given parameters.

Usage: simulator.py <SF-FILE> [options]

Options:
    -h --help               Show this screen.
    -n --num NUM            Specify number of patterns to simulate [default: 3].
    -o --output FILE        Specify output filename [default: peaks.npy].
    --lattice LATTICE       Specify 6 lattice parameters in angstrom and degree [default: 100,100,100,90,90,90].
    --det-dist DIST         Specify detector distance in meter [default: 0.1].
    --photon-energy PE      Specify single or multiple photon energy in eV [default: 9000,7000].
    --show                  Whether to display simulated patterns.
"""


import h5py
import numpy as np
from docopt import docopt
from numpy.linalg import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from util import calc_transform_matrix, calc_wavelength, euler_angles_to_rotation_matrix


BLOB_SIZE = 0.0002  # in per angstrom
FS = np.array([1., 0., 0.])  # detector fast scan vector
SS = np.array([0., 1., 0.])  # detector slow scan vector
PIXEL_SIZE = 100E-6  # pixel size in meter
CENTER_FS = 720  # beam center alone fast scan direction
CENTER_SS = 720  # beam center alone slow scan direction
SIZE_FS = 1440  # detector size alone fast scan direction
SIZE_SS = 1440  # detector size alone slow scan direction
EPSILON = 0.001


if __name__ == "__main__":
    argv = docopt(__doc__)
    det_dist = float(argv['--det-dist'])
    hkl_file = argv['<SF-FILE>']
    sf_data = np.loadtxt(hkl_file, skiprows=1)[:, 0:4]
    sf_data = np.concatenate([sf_data, -sf_data])  # expand sf list
    sf_data[:, 3] = np.abs(sf_data[:, 3])
    sf_data[:, 3] = sf_data[:, 3]/100  # scaling down intensity

    lattice_constants = list(map(float, argv['--lattice'].split(',')))
    A0 = calc_transform_matrix(lattice_constants)

    # photon energy
    PEs = sorted(list(map(float, argv['--photon-energy'].split(','))))
    N = int(argv['--num'])

    peak_data = []
    for i in range(N):
        reflections = None
        # random euler angles in degrees [0, 90]
        euler_angles = np.random.rand(3) * 90
        R = euler_angles_to_rotation_matrix(euler_angles)
        for PE in PEs:
            wavelength = calc_wavelength(PE) * 1E10
            # q vectors in standard system (centered on 000 miller indice)
            Q0 = A0.dot(sf_data[:, 0:3].T).T
            Q = R.dot(Q0.T).T  # q vectors after rotation
            ewald_radius = 1. / wavelength  # radius of Ewald sphere in angstrom
            Q[:, 2] += ewald_radius  # q vectors with center on scattering point
            # Q residue of a reflection to Ewald sphere in per angstrom
            Q_res = np.abs(norm(Q, axis=1) - ewald_radius)
            fired_idx = np.where(Q_res < BLOB_SIZE)[0]

            Q_fired = Q[fired_idx]
            Q_fired /= norm(Q_fired, axis=1)[:, np.newaxis]
            I_fired = sf_data[fired_idx, 3]
            # length of fired reflections in real space
            rho = det_dist / Q_fired[:, 2]
            # fired reflections in real space, centered on scattering point
            peaks = Q_fired * rho[:, np.newaxis]
            peaks_fs = (peaks - np.array([0., 0., det_dist])
                        ).dot(FS) / norm(FS) / PIXEL_SIZE + CENTER_FS
            peaks_ss = (peaks - np.array([0., 0., det_dist])
                        ).dot(SS) / norm(SS) / PIXEL_SIZE + CENTER_SS
            valid_idx = np.where((peaks_fs >= 0) & (peaks_fs < SIZE_FS) & (
                peaks_ss >= 0) & (peaks_ss < SIZE_SS))[0]
            peaks_fs = peaks_fs[valid_idx]
            peaks_ss = peaks_ss[valid_idx]

            _reflections = np.zeros((valid_idx.size, 12))
            _reflections[:, 0:4] = sf_data[fired_idx][valid_idx][:, 0:4]
            _reflections[:, 4] = Q_res[fired_idx][valid_idx]
            _reflections[:, 5] = peaks_fs
            _reflections[:, 6] = peaks_ss
            _reflections[:, 7] = np.sqrt(
                (peaks_ss - CENTER_SS) ** 2 + (peaks_fs - CENTER_FS) ** 2)
            _reflections[:, 8] = PE
            _reflections[:, 9] = (peaks_ss - CENTER_SS) * PIXEL_SIZE
            _reflections[:, 10] = (peaks_fs - CENTER_FS) * PIXEL_SIZE
            _reflections[:, 11] = I_fired  # peak intensity
            if reflections is None:
                reflections = _reflections
            else:
                reflections = np.concatenate([reflections, _reflections])

        peak_data.append({
            'clen': det_dist,
            'euler_angles': euler_angles,
            'rotation_matrix': R,
            'A': R.dot(A0),
            'peaks': reflections[:, 5:7],
            'photon_energy': reflections[:, 8],
            'coords': reflections[:, 9:11],
            'intensity': reflections[:, 11]
        })

        if argv['--show']:
            im = np.zeros((SIZE_SS, SIZE_FS))
            fig, ax = plt.subplots()
            ax.imshow(im)
            for i in range(len(PEs)):
                this_color_idx = np.where(
                    np.abs(reflections[:, 8] - PEs[i]) < EPSILON)[0]
                ax.scatter(
                    reflections[this_color_idx][:, 5],
                    reflections[this_color_idx][:, 6],
                    color='C%d' % i, s=3
                )
                print('peaks of color %d: %d' % (i, this_color_idx.size))
            ax.scatter(720, 720, color='red')
            ax.set_title(euler_angles)
            plt.show(block=True)
        
    # save to cxi file
    cxi = h5py.File('sim.cxi', 'w')
    cxi_data = [det_dist for item in peak_data]
    cxi_data = np.array(cxi_data)
    cxi.create_dataset('clen', data=cxi_data)
    cxi_data = [PEs[0] for item in peak_data]
    cxi_data = np.array(cxi_data)
    cxi.create_dataset('photon_energy', data=cxi_data)
    cxi_data = [item['A'] for item in peak_data]
    cxi_data = np.stack(cxi_data)
    cxi.create_dataset('A', data=cxi_data)
    cxi_data = []
    for item in peak_data:
        img = (np.random.rand(SIZE_SS, SIZE_FS) * 10).astype(np.int)
        for i, peak in enumerate(item['peaks']):
            peak = np.round(peak).astype(np.int)
            img[
                peak[0]-1:peak[0]+2, 
                peak[1]-1:peak[1]+2
                ] = item['intensity'][i] / 9
            img[img > 32767] = 32767  # saturated peaks
        cxi_data.append(img)
    cxi_data = np.stack(cxi_data)
    cxi.create_dataset(
        'data', data=cxi_data, 
        compression='gzip',
        compression_opts=3,
    )
    cxi.close()

    # np.save(argv['--output'], peak_data)
    # print('Peak file saved to %s' % argv['--output'])
