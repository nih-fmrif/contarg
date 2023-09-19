from pkg_resources import resource_filename
from pathlib import Path
import configparser
import nibabel as nb
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans
from simnibs import sim_struct, run_simnibs
import argparse


""" 
The code here must run in the SimNIBS 4.0 environment defined here:
 https://github.com/simnibs/simnibs/releases/download/v4.0.0/environment_linux.yml
"""

# from https://stackoverflow.com/a/13849249

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            90
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            180
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180/np.pi

def sort_circle_coords(coords, W):
    """
    Sorts the coordinates of a circle in a counterclockwise order and calculates the mean of every W adjacent points.

    Parameters
    ----------
    coords : ndarray
        The coordinates of the points on the circle. An array of shape (n, d), where n is the number of points and d is
         the number of dimensions.
    W : int
        The number of adjacent points to calculate the mean of.

    Returns
    -------
    output : ndarray
        The mean of every W adjacent points, sorted in a counterclockwise order. An array of shape (n, d), where n is
        the number of points and d is the number of dimensions.

    Notes
    -----
    This function uses the `cdist` function from `scipy.spatial.distance` to calculate the Euclidean distance between
    points. If the input array `coords` has repeated points, the output array may contain NaN values.

    """
    W = int(W)
    sorted_coords = np.zeros_like(coords)

    used = []
    idx = 1

    for i in range(len(coords)):
        D = cdist(coords[[idx], :], coords).squeeze()
        D[used] = np.nan
        idx = np.where((D == np.nanmin(D[D!= 0])))[0][0]
        sorted_coords[i, :] = coords[idx, :]
        used.append(idx)

    output = np.zeros_like(coords)

    for i in range(len(sorted_coords)):
        idx = np.roll(np.array(range(len(sorted_coords))), W + i)
        output[i, :] = sorted_coords[idx[:(W *2)]].mean(axis=0)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description='Run simnibs simulations in batches')
    parser.add_argument('--searchgrid_path', type=str, required=True, help='Path to the search grid mat file')
    parser.add_argument('--HeadModel_dir', type=str, required=True, help='Path to the head model directory')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory path')
    parser.add_argument('--coord_start_id', type=int, required=True, help='Start index of coordinates to simulate')
    parser.add_argument('--coord_stop_id', type=int, required=True, help='End index of coordinates to simulate')
    parser.add_argument('--batch_id', type=int, required=True, help='Batch ID')
    parser.add_argument('--coil', type=str, default='MagVenture_MCF-B65.ccd', help='Coil name')
    parser.add_argument('--distancetoscalp', type=float, default=2, help='Distance to scalp in cm')
    parser.add_argument('--angleresolution', type=float, default=30, help='Angle resolution in degrees')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads to use')
    parser.add_argument('--clean', type=bool, default=True, help='Whether to clean up the output directory after simulation or not')
    return parser.parse_args()


def sim_batch(
    searchgrid_path,
    HeadModel_dir,
    outdir,
    coord_start_id,
    coord_stop_id,
    batch_id,
    coil='MagVenture_MCF-B65.ccd',
    distancetoscalp=2,
    angleresolution=30,
    nthreads=1,
    clean=True
):
    outdir = Path(outdir)
    (outdir / 'Simulation').mkdir(exist_ok=True, parents=True)
    batch_outdir = outdir / f'Simulation/batch-{batch_id:04d}'

    HeadModel_dir = Path(HeadModel_dir)
    try:
        m2m_dir = sorted(HeadModel_dir.glob('m2m*'))[0]
    except IndexError:
        raise FileNotFoundError(f"No m2m directory found in {HeadModel_dir}")
    skinsurf_path = m2m_dir / 'Skin.surf.gii'
    subject = '_'.join(m2m_dir.parts[-1].split('_')[1:])
    headmesh_path = m2m_dir / f'{subject}.msh'

    searchgrid = sio.loadmat(searchgrid_path)

    # get the skin surface vertices
    skinsurf = nb.load(skinsurf_path)
    skinsurf_verts = skinsurf.get_arrays_from_intent('pointset')[0].data

    if not Path(coil).exists():
        simnibs_coil_dir = Path(resource_filename('simnibs', 'resources/coil_models'))
        coil_path = simnibs_coil_dir / f'Drakaki_BrainStim_2022/{coil}'
        if not coil_path.exists():
            coil_path = simnibs_coil_dir / f'legacy_and_other/{coil}'
            if not coil_path.exists():
                raise FileNotFoundError(f"Could not find coil ({coil}) in {simnibs_coil_dir}.")
    else:
        coil_path = Path(coil)

    sim_records = []
    s = sim_struct.SESSION()
    s.fnamehead = headmesh_path
    s.pathfem = batch_outdir
    tms_list = s.add_tmslist()
    tms_list.fnamecoil = coil_path
    if coord_stop_id > len(searchgrid['SubSampledSearchGrid']):
        coord_stop_id = None
    for coord in searchgrid['SubSampledSearchGrid'][coord_start_id:coord_stop_id]:
        d = cdist(skinsurf_verts, coord.reshape(1, -1))
        circle = skinsurf_verts[((d > 19) & (d < 20)).squeeze()]
        # get coordinates relative to a point that's about in the center of the circle
        # this point may be different from the coord due to the curvature of the cortex
        coplanar_center = circle.mean(0)
        rel_coords = circle - coplanar_center

        k = int(np.round(180 / angleresolution))
        # drop the negative y half of the circle
        circle = circle[rel_coords[:, 1] >= 0]
        rel_coords = rel_coords[rel_coords[:, 1] >= 0]
        # Calculate angles relative to line between center and point on circle with highest z coord
        zero_ind = np.where(rel_coords[:, 2] == rel_coords[:, 2].max())[0]
        zc = rel_coords[zero_ind]
        angles = np.array([angle_between(rc, zc.squeeze()) for rc in rel_coords])
        slop = 3
        # we'll take our 0 coord for 0 and don't need 180, so only go from 1:-1
        angle_targets = np.linspace(0, 180, k + 1)[1:-1]

        refdirs = []
        refdirs.append(circle[zero_ind].squeeze())
        for at in angle_targets:
            if at > angles.max():
                at = angles.max() - slop
            angle_ind = (angles > (at - slop)) & (angles < (at + slop))
            ave_coords = circle[angle_ind].mean(0)
            if not (ave_coords == np.nan).any():
                refdirs.append(ave_coords)

        refdirs = np.array(refdirs)

        for refdir in refdirs:
            position = tms_list.add_position()
            position.centre = coord
            position.pos_ydir = list(refdir)
            position.didt = 1 * 1e6
            position.distance = distancetoscalp

    s.map_to_vol = True
    s.fields = 'e'

    run_simnibs(s, cpus=nthreads)

    if clean:
        print("Removing col_po.geo, .msh, and .msh.opt files")
        to_clean = []
        to_clean.extend(sorted(batch_outdir.glob('*_scalar.msh')))
        to_clean.extend(sorted(batch_outdir.glob('*_scalar.msh.opt')))
        to_clean.extend(sorted(batch_outdir.glob('*_coil_pos.geo')))

        for tc in to_clean:
            tc.unlink()
    else:
        print("NOT removing col_po.geo, .msh, and .msh.opt files. These take up ~200MB per simulation.")

if __name__ == "__main__":
    args = parse_args()
    sim_batch(
        searchgrid_path=args.searchgrid_path,
        HeadModel_dir=args.HeadModel_dir,
        outdir=args.outdir,
        coord_start_id=args.coord_start_id,
        coord_stop_id=args.coord_stop_id,
        batch_id=args.batch_id,
        coil=args.coil,
        distancetoscalp=args.distancetoscalp,
        angleresolution=args.angleresolution,
        nthreads=args.nthreads,
        clean=args.clean
    )