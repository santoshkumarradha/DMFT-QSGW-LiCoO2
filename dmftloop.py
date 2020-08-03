from __future__ import print_function
import numpy as np
import subprocess
import os
import shutil
import glob
import matplotlib.pyplot as plt
import similaritymeasures


def excute(command, args=None, directory="./", fname="log"):
    f = open(fname, "a")
    pass_cmd = [command]
    if args != None:
        pass_cmd.extend(args)
    p = subprocess.Popen(pass_cmd, cwd=directory, stdout=f, shell=True)
    p.wait()


# excute(command, args=args, directory=directory, fname=fname)


def check_iter(fname):
    try:
        os.makedirs(fname)
        return True
    except OSError:
        print("{} exists".format(fname))
        return False


#convergence test


def fprint(text, fname="Convergence.text"):
    with open(fname, 'w+') as f:
        print(text, file=f)


def converge(fname):

    fname = ["it" + str(i) for i in iters]
    G = []
    Gl = np.loadtxt(fname[0] + "/gl.inp").T
    fig, ax = plt.subplots(Gl.shape[0] - 1, 1, figsize=(10, Gl.shape[0] * 2))
    for i in fname:
        S = np.loadtxt(i + "/sig.inp").T
        Sig = S[1::2] + 1j * S[2::2]
        w = S[0]
        for i1 in range(Sig.shape[0]):
            ax1 = ax[i1]
            ax1.plot(w, Sig[i1].imag, label=i + "-" + str(i1 + 1))
            ax1.legend(loc='upper right')
        G.append(Sig.imag)
    plt.tight_layout()
    plt.savefig("convergence_sig.png", dpi=200)
    plt.close()

    from scipy.spatial.distance import directed_hausdorff as dist
    from tabulate import tabulate

    def converge_txt(s):
        rms = []
        val = []
        headings = ["Iter"]
        headings.extend(["Orb-" + str(i + 1) for i in range(10)])
        headings.extend(["Total"])
        for cnt in [[i, i + 1] for i in range(0, s.shape[0] - 1)]:
            tmp = [
                str(np.array(cnt) + 1).replace("[", "").replace("]",
                                                                "").replace(
                                                                    " ", "->")
            ]
            tmp.extend([
                dist([s[cnt[0]][norb]], [s[cnt[1]][norb]])[0]
                for norb in range(10)
            ])
            tmp.extend([np.sum(tmp[1:])])
            rms.append(tmp)
            val.append(tmp[1:])
        print(tabulate(rms, headers=headings))
        fprint(tabulate(rms, headers=headings))
        return (np.array(val))

    fname = ["it" + str(i) for i in iters]
    s = []
    for i in fname:
        S = np.loadtxt(i + "/sig.inp").T
        Sig = S[1::2] + 1j * S[2::2]
        w = S[0]
        s.append(Sig.imag)
    s = np.array(s)

    vals = converge_txt(s)
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    for j, i in enumerate(vals.T[:-1]):
        ax[1].plot(np.arange(len(i)) + 1, i, label=str(j + 1), marker="o")
    ax[0].plot(np.arange(len(i)) + 1,
               vals.T[-1],
               label="Total",
               c="k",
               marker="o")
    for i in ax:
        i.set_ylabel("RMS diff")
        i.set_xlabel("Iter")
        i.legend(ncol=3)
    plt.tight_layout()
    plt.savefig("Convergence_iter.png")
    plt.show()
    plt.close()


# plotting sig and gl
def sig_plot(fname, save=False):
    S = np.loadtxt(fname + "/sig.inp").T
    fig, ax1 = plt.subplots(2, 1, figsize=(10, 5))
    Sig = S[1::2] + 1j * S[2::2]
    w = S[0]
    ax = ax1[0]
    for i in range(Sig.shape[0]):
        ax.plot(w, Sig[i].imag, label=i + 1)
    ax.set_xlabel(r'$\Omega_n$', fontsize=12)
    ax.set_ylabel(r'$\Sigma$', fontsize=12)
    ax.set_xlim(0, 200)
    ax.legend(ncol=3)

    ax = ax1[1]
    Gl = np.loadtxt(fname + "/gl.inp").T
    wl = Gl[0]
    Gl = Gl[1:]
    for i in range(Gl.shape[0]):
        ax.plot(wl, Gl[i], label=i + 1)
    ax.set_xlabel(r'$n_l$', fontsize=12)
    ax.set_ylabel(r'$G_l$', fontsize=12)
    ax.legend(ncol=3)
    plt.tight_layout()
    if save:
        plt.savefig(fname + "/sig_g.png", dpi=300)
    plt.close('all')


def run_iter(iter, ctrl="temp", np_lmfdmft=1, np_ctqmc=1):
    print("\n Starting iter {} \n-----------------------------------\n".format(
        iter))
    iter_folder = "it{}".format(int(iter))
    prev_iter_folder = "it{}".format(int(iter - 1))
    if check_iter(iter_folder):
        source_dir = prev_iter_folder
        dest_dir = iter_folder
        if iter == 1:
            source_dir = "./"
        for filename in glob.glob(os.path.join(source_dir, '*.*')):
            shutil.copy(filename, dest_dir)

        print("Loading modules for lmf")
        # load confliciting modules ! and set python path and set openmpi processors
        # cmd = "module load openmpi"
        # excute(cmd,
        #        args=None,
        #        directory="./{}".format(iter_folder),
        #        fname="log")
        cmd = "export OMP_NUM_THREADS=1"
        excute(cmd,
               args=None,
               directory="./{}".format(iter_folder),
               fname="log")
        cmd = "echo $OMP_NUM_THREADS"
        excute(cmd,
               args=None,
               directory="./{}".format(iter_folder),
               fname="log")

        # Run lmfdmft For iter==1, create empty impurity self energy first and then proceed
        if iter == 1:
            print("creating empty impurity self energy")
            cmd = "mpirun -n {np_lmfdmft}  lmfdmft --ldadc~fn=dc -job=1 ctrl.{ctrl}".format(
                np_lmfdmft=np_lmfdmft, ctrl=ctrl)
            excute(cmd,
                   args=None,
                   directory="./{}".format(iter_folder),
                   fname="log")
        print("creating impurity self energy")
        cmd = "mpirun -n {np_lmfdmft}  lmfdmft --ldadc~fn=dc -job=1 ctrl.{ctrl}".format(
            np_lmfdmft=np_lmfdmft, ctrl=ctrl)
        excute(cmd,
               args=None,
               directory="./{}".format(iter_folder),
               fname="log")

        # Remove eigenvvectors and projections of previous runs
        print("Remove eigenvvectors and projections of previous runs..")
        cmd = "rm -f evec* proj*"
        excute(cmd,
               args=None,
               directory="./{}".format(iter_folder),
               fname="log")

        # Unload confliciting modules !
        print("Unload confliciting modules !")
        # cmd = "module unload hdf5"
        # excute(cmd,
        #        args=None,
        #        directory="./{}".format(iter_folder),
        #        fname="log")
        cmd = 'export PYTHONPATH="${PYTHONPATH}:/home/srr70/lm_ver6/lm/dmft/interface_solver/"'
        excute(cmd,
               args=None,
               directory="./{}".format(iter_folder),
               fname="log")

        # Run TRIQS Continious time monte carlo solver
        print("Run TRIQS Continious time monte carlo solver")
        cmd = "/home/srr70/miniconda3/envs/triqs/bin/mpirun -np {np_ctqmc} lmtriqs {ctrl}".format(
            np_ctqmc=np_ctqmc, ctrl=ctrl)
        excute(cmd,
               args=None,
               directory="./{}".format(iter_folder),
               fname="log")

        print("Plotting the values")
        fname = iter_folder
        sig_plot(fname, save=True)

        print("done \n-------------------------------\n")


for iter in np.arange(1, 10):
    run_iter(iter, ctrl="temp", np_lmfdmft=1, np_ctqmc=32)
    if iter > 1:
        iters = np.arange(1, iter + 1)
        converge(iters)