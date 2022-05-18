import scipy.sparse, ipt, slepc, pyscf_davidson
import numpy as np
import time
import primme

def eigs(mat, algorithm, tol = 1e-10, maxiter = 10000, aa_memory = 0):
    n = np.argmin(mat.diagonal())
    v0 = np.zeros(mat.shape[0]); v0[n] = 1

    if scipy.sparse.issparse(mat):
        symmetric = scipy.sparse.linalg.norm(mat - mat.T, scipy.Inf) < tol
    else:
        symmetric = np.linalg.norm(mat - mat.T, np.inf) < tol


    if algorithm == 'ipt':
        return(ipt.eigs_ipt(mat, i = n, mem=aa_memory, tol=tol, v0=v0, maxiter = maxiter))

    elif algorithm != 'ipt' and symmetric:

        if algorithm == 'slepc-ks':
            return(slepc.eigs_slepc(mat, isherm = True, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'krylovschur', maxiter = maxiter))

        elif algorithm == 'scipy-lanczos':
            tic = time.time()
            e, v = scipy.sparse.linalg.eigsh(mat, k = 1, v0 = v0, which = 'SM', tol = tol, return_eigenvectors = True)
            toc = time.time()
            return({'eigenvalue' : e, 'eigenvector' : v, 'time' : toc - tic, 'residual' : np.linalg.norm(mat@v - e*v)/np.linalg.norm(v)})

        elif algorithm == 'slepc-lobpcg-jacobi':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'preonly')
            st_opts.setdefault('PCType', 'jacobi')

            return(slepc.eigs_slepc(mat, isherm = True, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'lobpcg', maxiter = maxiter, st_opts = st_opts))

        elif algorithm == 'slepc-lobpcg-lu':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'preonly')
            st_opts.setdefault('PCType', 'lu')

            return(slepc.eigs_slepc(mat, isherm = True, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'lobpcg', maxiter = maxiter, st_opts = st_opts))

        elif algorithm == 'slepc-lobpcg':

            return(slepc.eigs_slepc(mat, isherm = True, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'lobpcg', maxiter = maxiter))

        elif algorithm == 'slepc-gd':


            return(slepc.eigs_slepc(mat, isherm = True, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'gd', maxiter = maxiter))

        elif algorithm == 'slepc-gd-jacobi':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'preonly')
            st_opts.setdefault('PCType', 'jacobi')

            return(slepc.eigs_slepc(mat, isherm = True, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'gd', maxiter = maxiter, st_opts=st_opts))

        elif algorithm == 'slepc-gd-none':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'preonly')
            st_opts.setdefault('PCType', 'none')


            return(slepc.eigs_slepc(mat,isherm = True,k = 1,tol = tol,v0 = v0,which = 'SR',EPSType = 'gd', maxiter = maxiter, st_opts=st_opts))

        elif algorithm == 'slepc-gd-lu':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'preonly')
            st_opts.setdefault('PCType', 'lu')

            return(slepc.eigs_slepc(mat,isherm = False,k = 1,tol = tol,v0 = v0,which = 'SR',EPSType = 'gd',maxiter = maxiter, st_opts=st_opts))

        elif algorithm == 'slepc-jd-jacobi':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'bcgs')
            st_opts.setdefault('PCType', 'jacobi')

            return(slepc.eigs_slepc(mat, isherm = True, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'jd', maxiter = maxiter, st_opts=st_opts))

        elif algorithm == 'slepc-jd-none':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'bcgs')
            st_opts.setdefault('PCType', 'none')

            return(slepc.eigs_slepc(mat,isherm = True,k = 1,tol = tol,v0 = v0,which = 'SR',EPSType = 'jd',maxiter = maxiter, st_opts=st_opts))

        elif algorithm == 'pyscf-d':
            return(pyscf_davidson.davidson(lambda x: mat.dot(x), x0 = v0, precond = mat.diagonal(), tol = tol, max_cycle = maxiter))

        elif algorithm == 'primme':
            pm.eigsh(S, 1, which = "SA", OPinv = prec, return_stats = True)

    elif algorithm != 'ipt' and not symmetric:

        if algorithm == 'slepc-ks':
            return(slepc.eigs_slepc(mat, isherm = False, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'krylovschur', maxiter = maxiter))

        elif algorithm == 'scipy-arnoldi':
            tic = time.time()
            e, v = scipy.sparse.linalg.eigs(mat, k = 1, v0 = v0, which = 'SM', tol = tol, return_eigenvectors = True)
            toc = time.time()
            return({'eigenvalue' : e, 'eigenvector' : v, 'time' : toc - tic, 'residual' : np.linalg.norm(mat@v - e*v)/np.linalg.norm(v)})

        elif algorithm == 'slepc-gd-jacobi':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'preonly')
            st_opts.setdefault('PCType', 'jacobi')

            return(slepc.eigs_slepc(mat, isherm = False, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'gd', maxiter = maxiter, st_opts=st_opts))

        elif algorithm == 'slepc-gd-none':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'preonly')
            st_opts.setdefault('PCType', 'none')

            return(slepc.eigs_slepc(mat,isherm = False,k = 1,tol = tol,v0 = v0,which = 'SR',EPSType = 'gd',maxiter = maxiter, st_opts=st_opts))



        elif algorithm == 'slepc-jd-jacobi':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'bcgs')
            st_opts.setdefault('PCType', 'jacobi')

            return(slepc.eigs_slepc(mat, isherm = False, k = 1, tol = tol, v0 = v0, which = 'SR', EPSType = 'jd', maxiter = maxiter, st_opts=st_opts))

        elif algorithm == 'slepc-jd-none':

            st_opts = {}
            st_opts.setdefault('STType', 'precond')
            st_opts.setdefault('KSPType', 'bcgs')
            st_opts.setdefault('PCType', 'none')

            return(slepc.eigs_slepc(mat,isherm = False,k = 1,tol = tol,v0 = v0,which = 'SR',EPSType = 'jd',maxiter = maxiter, st_opts=st_opts))


        elif algorithm == 'pyscf-d':
            return(pyscf_davidson.davidson_nosym(lambda x: mat.dot(x), x0 = v0, precond = mat.diagonal(), tol = tol))
