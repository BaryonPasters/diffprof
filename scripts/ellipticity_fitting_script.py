"""Script to fit Bolshoi, MDPL2, or TNG MAHs with the diffmah model."""
import numpy as np
import os
from mpi4py import MPI
import argparse
from time import time
from diffprof.load_bpl_histories import load_histories, TASSO, BEBOP
from diffprof.fit_ellipticity_helpers import get_outline, get_outline_bad_fit
from diffprof.fit_ellipticity_helpers import fit_ellipticity, get_header
import subprocess
import h5py

TMP_OUTPAT = "_tmp_ellipticity_fits_rank_{0}.dat"


def _write_collated_data(outname, data):
    nrows, ncols = np.shape(data)
    colnames = get_header()[1:].strip().split()
    assert len(colnames) == ncols, "data mismatched with header"
    with h5py.File(outname, "w") as hdf:
        for i, name in enumerate(colnames):
            if name == "halo_id":
                hdf[name] = data[:, i].astype("i8")
            else:
                hdf[name] = data[:, i]


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()

    parser = argparse.ArgumentParser()

    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("outbase", help="Basename of the output hdf5 file")
    parser.add_argument("-indir", help="Input directory", default="BEBOP")
    parser.add_argument("-test", help="Short test run?", type=bool, default=False)

    args = parser.parse_args()
    rank_basepat = args.outbase + TMP_OUTPAT
    rank_outname = os.path.join(args.outdir, rank_basepat).format(rank)

    if args.indir == "TASSO":
        indir = TASSO
    elif args.indir == "BEBOP":
        indir = BEBOP
    else:
        indir = args.indir
    halo_ids, e_sims, log_mahs, t, LGM_MIN = load_histories(indir, "ellipticity")

    # Get data for rank
    if args.test:
        nhalos_tot = nranks * 25
    else:
        nhalos_tot = len(halo_ids)
    _a = np.arange(0, nhalos_tot).astype("i8")
    indx = np.array_split(_a, nranks)[rank]

    halo_ids_for_rank = halo_ids[indx]
    log_mahs_for_rank = log_mahs[indx]
    es_for_rank = e_sims[indx]
    nhalos_for_rank = len(halo_ids_for_rank)

    header = get_header()
    start = time()
    with open(rank_outname, "w") as fout:
        fout.write(header)

        for i in range(nhalos_for_rank):
            halo_id = halo_ids_for_rank[i]
            lgmah = log_mahs_for_rank[i, :]
            eh = es_for_rank[i, :]

            p_best, loss, method, loss_data = fit_ellipticity(t, eh, lgmah, LGM_MIN)

            if np.isfinite(loss):
                outline = get_outline(halo_id, p_best, loss, method)
            else:
                outline = get_outline_bad_fit(halo_id, p_best, loss, method)

            fout.write(outline)

    comm.Barrier()
    end = time()

    msg = (
        "\n\nWallclock runtime to fit {0} galaxies with {1} ranks = {2:.1f} seconds\n\n"
    )
    if rank == 0:
        runtime = end - start
        print(msg.format(nhalos_tot, nranks, runtime))

        #  collate data from ranks and rewrite to disk
        pat = os.path.join(args.outdir, rank_basepat)
        fit_data_fnames = [pat.format(i) for i in range(nranks)]
        data_collection = [np.loadtxt(fn) for fn in fit_data_fnames]
        all_fit_data = np.concatenate(data_collection)
        outname = os.path.join(args.outdir, args.outbase)
        _write_collated_data(outname, all_fit_data)

        #  clean up temporary files
        _remove_basename = pat.replace("{0}", "*")
        command = "rm -rf " + _remove_basename
        raw_result = subprocess.check_output(command, shell=True)
