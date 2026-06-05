import numpy as np
import math as mt
import os
import xarray

# PyWake / EllipSys imports
from py_wake_ellipsys.wind_farm_models.ellipsys import EllipSys
from py_wake_ellipsys.wind_farm_models.ellipsys_lib import (
    FlatBoxGrid,
    WFPostFlow,
    set_cluster_vars,
    AD,
    WFRun,
    Cluster,
    Forest,
    E3D
)

from py_wake.examples.data.hornsrev1 import Hornsrev1Site
from py_wake_ellipsys_examples.data.turbines.dummy_wt import Dummy
from py_wake_ellipsys.utils.canopy import CAN2CANunf, read_CAN

class StationaryForest(Forest):
    """Forest that doesn't rotate with wind direction"""
    def write_input(self, wd, grid_type, grid_wd=270.0, mode='w'):
        f = open('input.dat', mode)
        f.write('# forest drag force\n')
        f.write('forceallocation' + '\n')
        f.write('forest\n')
        f.write('relaxforest %g\n' % self.relax)
        if grid_type in ['flatbox', 'flatboxper']:
            # For flat terrain we rotate the canopy where the rotation center is always at x,y=0.0,0.0
            f.write('rotate_canopy %g %g %g\n' % (270.0 - grid_wd, 0.0, 0.0))
        # SKIP rotate_canopy - forest stays stationary
        f.write('forest_out_res\n')
        if self.ke:
            f.write('ke_forest true\n')
        f.write('lowtreefilter %g\n' % self.lowtreefilter)
        f.write('\n')
        f.close()


def get_TI(z0):
    kappa = 0.4
    z_ref = 10.0
    C_mu = 0.03
    return (kappa * np.sqrt(2/3))/(C_mu**(0.25)*np.log((z_ref + z0)/z0))

def set_coarse(wfm):
    """Sets a coarse grid and loose convergence for faster testing."""
    wfm.set_subattr('e3d.reslim', 1e-2)
    wfm.set_subattr('grid.cells1_D', 0.5)
    wfm.set_subattr('grid.z_cells1_D', 5.0)
    Cluster.maxnodes = 1
    wfm.wfrun.cluster.walltime = '0:20:00'

def keSogCnst(wfm):
    """Standard k-epsilon Sogachev constants for forest canopies."""
    cnst = wfm.cnst
    cnst.cmu = 0.03
    cnst.kappa = 0.4
    cnst.ce1 = 1.52
    cnst.ce2 = 1.833
    cnst.pred = cnst.kappa ** 2 / (np.sqrt(cnst.cmu) * (cnst.ce2 - cnst.ce1))
    cnst.prtke = cnst.pred
    wfm.cnst = cnst

def kefPCnst(wfm):
    cnst = wfm.cnst
    cnst.ce1 = -1
    cnst.ce2 = 1.92
    cnst.prtke = 1.00
    cnst.pred = 1.30
    wfm.cnst = cnst

def main():
    # ---------------------------------------------------------------
    # 1. Define Canopy Parameters and Process CAN File
    # ---------------------------------------------------------------

    CANfile = 'fairway60m_4dx_512m.CAN'
    if not os.path.exists(CANfile):
        print(f"Error: {CANfile} not found in the current directory.")
        return
    
    # We read the original CAN file to get dimensions (xmin, xmax, ymin, ymax)
    nx, ny, xmin, xmax, ymin, ymax, FH, PAD, nlevelslist, totalnlevels = read_CAN(CANfile, canfileformat=1)

    # Calculate delta from CAN file: (512-0)/128 = 4m
    delta = (xmax - xmin) / nx  # This should match the grid spacing

    # Since we use shift2center=True, the canopy will be centered at (0,0)
    grid_center = [0.0, 0.0]

    # Convert to binary and center the data
    CAN2CANunf(CANfile, 'grid.CANunf', canfileformat=1, shift2center=True)

    # ---------------------------------------------------------------
    # 2. Define CFD Grid Parameters (Flat Terrain)
    # ---------------------------------------------------------------
    Dref = 100 # scaling parameter
    zlen = 500 # height of domain
    zFirstCell = 0.1    # first cell height above ground

    # Dimensions of inner grid domain in units of Dref (e,w,n,s) (512 x 512)
    m1_D = 2.56
    m1_e_D = m1_D; m1_w_D = m1_D; m1_n_D = m1_D; m1_s_D = m1_D
    cells1_D = Dref / delta # number of cells per Dref in inner domain

    # Use FlatBoxGrid for a completely flat terrain case (no .grd files needed)
    grid = FlatBoxGrid(Dref, 
                            origin = grid_center,
                            cells1_D = cells1_D,
                            zFirstCell_D = zFirstCell / Dref, 
                            z_cells1_D = 25,
                            zWakeEnd_D = 3.0,
                            bsize = 32,
                            zlen_D = zlen / Dref,
                            radius_D = 50,
                            m1_w_D = m1_w_D, 
                            m1_e_D = m1_e_D, 
                            m1_n_D = m1_n_D,
                            m1_s_D = m1_s_D, 
                            dwd = 360, 
                            cluster = Cluster(gbar_mem = 3, walltime = '4:00:00'))
        
    # ---------------------------------------------------------------
    # 3. Define Simulation Parameters
    # ---------------------------------------------------------------
    wt = Dummy()
    wt_x = np.array([0.0]) 
    wt_y = np.array([0.0]) 
    type_i = np.array([0])

    hub_height = 90.0
    h_i = np.array([hub_height])

    z0 = 0.03

    #---------------------------------------------------------------
    # Settings for different wind directions (uncomment the desired case)
    #---------------------------------------------------------------

    # Fairway Laying horizontal (wind from west (tilted) = 45 degrees)
    # grid_wd = 270.0
    # wd = [225.0]

    # Fairway laying horizontal (wind from west = 0 degrees)
    # grid_wd = 270.0
    # wd = [180.0]

    # Fairway Laying horizontal (wind from south = 90 degrees)
    grid_wd = 270.0
    wd = [180.0]

    #---------------------------------------------------------------

    ws = [6.0]
    TI = get_TI(z0)
    # TI = 0.1
    zRef = 10.0

    # ---------------------------------------------------------------
    # 4. Setup Cluster and Flow Model
    # ---------------------------------------------------------------
    run_machine = 'gbar'
    set_cluster_vars(run_machine, True, 'hpc', corespernode=24, maxnodes=2)

    wfpostflow = WFPostFlow(outputformat='netCDF', single_precision_netCDF=True, cluster = Cluster(gbar_mem = 6, walltime = '6:00:00', blockdistribution = 'equal'))

    e3d = E3D(turbmodel='ke', nstepmin=2, relaxu=0.5, relaxp=0.5, relaxturb=0.5, reslim = 1e-4, start_grlvl = 1)

    # Ensure ke=True is set if we are using keSogCnst or kefPCnst
    forest = StationaryForest(canfile='grid.CANunf', cd=0.2, ke=False, lowtreefilter=0.0) 

    flowmodel = EllipSys(Hornsrev1Site(), wt, grid,TI, zRef, ad=AD(force=None, run_pre=False), e3d = e3d,
                                wfrun=WFRun(casename='Fairway60m_cd0.2_4xycd_4zcd_512m_filter0', cluster = Cluster(walltime = '72:00:00'), write_restart=True), forest = forest,
                                wfpostflow=wfpostflow, run_wd_con=False)
    

    # We can choose to set either keSogCnst or kefPCnst
    # keSogCnst(flowmodel)
    # kefPCnst(flowmodel)
    
    # ---------------------------------------------------------------
    # 5. Run Simulation
    # ---------------------------------------------------------------

    coarse = False

    if coarse: 
        set_coarse(flowmodel)

    flowmodel.run_grid = True
    flowmodel.run_cal = False # this is only relevant when simulating turbines
    flowmodel.run_wf = True
    flowmodel.run_post = True


    # 1. Explicitly create the grid with your turbine coordinates 
    flowmodel.create_windfarm_grid(wt_x, wt_y, grid_wd = grid_wd)

    # 2. Run the flow solver matching the test script pipeline
    flowmodel.run_windfarm(wt_x, wt_y, wd, ws, type_i, grid_wd = grid_wd)

    # Store 3D flow data
    iwd, iws = 0, 0

    flowmodel.post_windfarm_flow(wd[iwd], ws[iws], precursor=False, grid_wd = grid_wd)

if __name__ == '__main__':
    main()
