#!/usr/bin/env python3

import sys
import time
import random
from datetime import datetime
import numpy as np
from copy import deepcopy
import qml
from qml.math import cho_solve
from qml.math import svd_solve
from qml.representations import *
from qml.kernels import get_local_symmetric_kernel
from qml.kernels import get_local_kernel
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel

from qml.kernels import get_local_symmetric_kernel_mbdf, get_local_kernel_mbdf

import itertools
from time import time

from tqdm import tqdm
import pandas as pd

from colored import fg
from yaspin import yaspin

import MBDF

RED = fg('red')
WHITE = fg('white')
GREEN = fg('green')

def read_costs(f):
    return pd.read_csv(f)

def do_ML(X, X_test, Yprime, sigma, Q, Q_test):
  # train ML model and return predictions (cross validated w/ # nModels)


  K      = get_local_symmetric_kernel_mbdf(X,  Q, sigma)
  K_test = get_local_kernel_mbdf(X_test, X, Q_test, Q, sigma)

  Y = Yprime
  C = deepcopy(K)
  alpha = svd_solve(C, Y)

  Yss = np.dot((K_test).T, alpha)

  return Yss

def opt_sigma(N, X, Q, Y):
    sigmas = [0.01, 0.1, 1.0, 10., 100., 1000.]

    total = list(range(X.shape[0]))
    random.seed(42)
    maes = np.array([])
    for sigma in sigmas:
      MAE = np.array([])
      for n in range(5):
        random.shuffle(total)
        training_index = total[:N]
        test_index     = total[-200:]

        X_train = X[training_index]
        X_test  = X[test_index]
        Y_train = Y[training_index]
        Y_test  = Y[test_index]
        Q_train = Q[training_index]
        Q_test  = Q[test_index]

        K      = get_local_symmetric_kernel_mbdf(X_train,  Q_train, sigma)
        K_test = get_local_kernel_mbdf(X_test, X_train, Q_test, Q_train, sigma)

        C = deepcopy(K)
        alpha = svd_solve(C, Y_train)

        pred = np.dot((K_test).T, alpha)

        MAE = np.append(MAE, np.mean(np.abs(pred-Y_test)))

      maes = np.append(maes, np.mean(MAE))
    #print(sigmas[np.argmin(maes)])

    return sigmas[np.argmin(maes)]


#def do_LC(X, X_test, total, nModels, mols_train, mols_test, HF_times, MP_times, CC_times, Q, Q_test
def do_LC(X_train, Q_train, X_test, Q_test, Y_PBE, Y_PBE_PBE0, Y_PBE0_PBE86, Y_PBE86_wPBE, Y_wPBE):

  # fancy guess
  N_PBE   = [ 128, 256, 512, 1024 ]
  N_PBE0  = [  32,  64, 128,  256 ]
  N_PBE86 = [   8,  16,  32,   64 ]
  N_wPBE  = [   2,   4,   8,   16 ]

  random.seed(42)
  total = list(range(X_train.shape[0]))

  e_multi    = np.array([])
  #split = list(range(total))
  nModels = 10
  for i in range(nModels):
      MAE_multi   = np.array([])

      with yaspin(text="Generating Model {}".format(i+1), color="cyan") as sp:
        for train in range(len(N_PBE)):
          random.shuffle(total)
          # Direct Learning
          ti = total[:N_PBE[train]]
          sigma_PBE = opt_sigma(N_PBE[train], X_train, Q_train, Y_PBE)
  #        sigma_PBE = 1.
          Yp_PBE    = do_ML(X_train[ti], X_test, Y_PBE[ti], sigma_PBE, Q_train[ti], Q_test)

          # Delta learning (PBE - PBE0)
          sigma_PBE0  = opt_sigma(N_PBE0[train], X_train, Q_train, Y_PBE_PBE0)
          ti = total[:N_PBE0[train]]
  #        sigma_PBE0 = 1.
          Yp_PBE_PBE0 = do_ML(X_train[ti], X_test, Y_PBE_PBE0[ti], sigma_PBE0, Q_train[ti], Q_test)

          # Delta learning (PBE0 - PBE86)
          ti = total[:N_PBE86[train]]
          sigma_PBE86   = opt_sigma(N_PBE86[train], X_train, Q_train, Y_PBE0_PBE86)
  #        sigma_PBE86 = 1.
          Yp_PBE0_PBE86 = do_ML(X_train[ti], X_test, Y_PBE0_PBE86[ti], sigma_PBE86, Q_train[ti], Q_test)

          # Delta learning (PBE0 - PBE86)
          ti = total[:N_wPBE[train]]
          sigma_wPBE    = opt_sigma(N_wPBE[train], X_train, Q_train, Y_PBE86_wPBE)
  #        sigma_wPBE = 1.
          Yp_PBE86_wPBE = do_ML(X_train[ti], X_test, Y_PBE86_wPBE[ti], sigma_wPBE, Q_train[ti], Q_test)

          # get energy prediction 2- and 3- levels
          Y_multi  = Yp_PBE + Yp_PBE_PBE0 + Yp_PBE0_PBE86 + Yp_PBE86_wPBE

          mae_multi   = np.mean(np.abs(Y_multi-Y_wPBE))
          MAE_multi   = np.append(MAE_multi, mae_multi)
          sp.write("> N = {:>2}{:>5}{:>5}{:>5}, MAE: {:>4,.8f} [✔]".format(N_wPBE[train], N_PBE86[train], N_PBE0[train], N_PBE[train], mae_multi))
#        print("N: {:.2f},\tMAE: {:.2f}".format(N_wPBE[train], mae_multi))
        sp.ok("✔")

        e_multi   = np.append(e_multi, MAE_multi)

  e_multi   = e_multi.reshape(nModels,len(N_PBE)).mean(axis=0)

  print("\nN_wPBE,N_PBE86,N_PBE0,N_PBE,e_direct")
  for i in range(len(e_multi)):
      print("{},{},{},{},{:.4f}".format(N_wPBE[i], N_PBE86[i], N_PBE0[i], N_PBE[i], e_multi[i]))

  return True

def get_properties(filename):
    df = pd.read_csv(filename)

    properties = dict()

    names   = df['xyzfile'].to_numpy()
    Y_PBE   = df['e_PBE'].to_numpy(dtype=float)
    Y_PBE0  = df['e_PBE0'].to_numpy(dtype=float)
    Y_PBE86 = df['e_PBE86'].to_numpy(dtype=float)
    Y_wPBE  = df['e_wPBE'].to_numpy(dtype=float)

    for i, name in enumerate(names):
        properties[name] = [ Y_PBE[i], Y_PBE0[i], Y_PBE86[i], Y_wPBE[i] ]

    return properties

def get_mols(data):
    mols = []

    for name in sorted(data.keys()):
        mol = qml.Compound()
        mol.read_xyz("../xyz/" + name + ".xyz")
        mol.properties = data[name]
        mols.append(mol)

    return mols


def get_representation(mols):
    coords = np.asarray([mol.coordinates for mol in mols])
    coords *= 1.88973
    Q = np.asarray([mol.nuclear_charges for mol in mols])


    spinner = yaspin(text="Calculate Representation", color="yellow")
    spinner.start()
    start = time()
    X = MBDF.generate_mbdf(Q,coords,pad=29,cutoff_r=10)
    np.savez("qm9_PBE_XQ.npz", X=X, Q=Q)
    #readins = np.load("qm7b.npz", allow_pickle=True)
    #X = readins['X']
    end = time()
    spinner.stop()
    print("\n [ {}OK{} ] Calculate in Representation ({:.2f} min)\n".format(GREEN, WHITE, (end-start)/60.))

    return X, Q



def main():

  print("")
  filename = 'data.txt'
  data = get_properties(filename)
  mols = get_mols(data)

  Xall, Qall   = get_representation(mols)
  Y_PBE        = np.array([ mol.properties[0] for mol in mols])
  Y_PBE_PBE0   = np.array([ mol.properties[1]-mol.properties[0] for mol in mols])
  Y_PBE0_PBE86 = np.array([ mol.properties[2]-mol.properties[1] for mol in mols])
  Y_PBE86_wPBE = np.array([ mol.properties[3]-mol.properties[2] for mol in mols])
  Y_wPBE       = np.array([ mol.properties[3] for mol in mols])

  total = list(range(Xall.shape[0]))

  random.seed(42)
  random.shuffle(total)

  train = total[:1800]
  test  = total[1800:]

  X_train = Xall[train]
  X_test  = Xall[test]
  Q_train = Qall[train]
  Q_test  = Qall[test]

  Y_PBE        = Y_PBE[train]
  Y_PBE_PBE0   = Y_PBE_PBE0[train]
  Y_PBE0_PBE86 = Y_PBE0_PBE86[train]
  Y_PBE86_wPBE = Y_PBE86_wPBE[train]
  Y_wPBE       = Y_wPBE[test]


  start = time()
  isDone = do_LC(X_train, Q_train, X_test, Q_test, Y_PBE, Y_PBE_PBE0, Y_PBE0_PBE86, Y_PBE86_wPBE, Y_wPBE)
  end = time()

  print("\n [ {}OK{} ] Generate Learning Curves ({:.2f} min)\n".format(fg('green'), fg('white'), (end-start)/60.))


if __name__ == "__main__":
  main()

