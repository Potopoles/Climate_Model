import numpy as np
import time
from datetime import timedelta
from grid import Grid
from fields import initialize_fields, CPU_Fields, GPU_Fields
from nc_IO import constant_fields_to_NC, output_to_NC
from IO import write_restart
from multiproc import create_subgrids
from namelist import i_time_stepping, \
                    i_load_from_restart, i_save_to_restart, \
                    i_radiation, njobs, comp_mode, \
                    i_microphysics, i_soil
from diagnostics import diagnose_secondary_fields
from bin.diagnostics_cython import diagnose_secondary_fields_c
from IO_helper_functions import print_ts_info, print_computation_time_info
if i_time_stepping == 'MATSUNO':
    from matsuno import step_matsuno as time_stepper
elif i_time_stepping == 'RK4':
    from RK4 import step_RK4 as time_stepper

from numba import cuda

GR = Grid()
#GR, subgrids = create_subgrids(GR, njobs)
subgrids = {}

# INITIALIZE FIELDS ON CPU
CF = CPU_Fields(GR, subgrids)

#COLP_OLD, COLP, COLP_NEW, dCOLPdt, PAIR, PHI, PHIVB, \
#UWIND_OLD, UWIND, VWIND_OLD, VWIND, WIND, WWIND,\
#UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,\
#BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, \
#HSURF, POTT_OLD, POTT, dPOTTdt, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
#RAD, SOIL, MIC, TURB = initialize_fields(GR, subgrids, F)
PAIR, PHI, PHIVB, \
UWIND_OLD, UWIND, VWIND_OLD, VWIND, WIND, WWIND,\
UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,\
BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, \
HSURF, POTT_OLD, POTT, dPOTTdt, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
RAD, SOIL, MIC, TURB = initialize_fields(GR, subgrids, CF)

# INITIALIZE FIELDS ON GPU AND COPY TO DEVICE
if comp_mode == 2:
    GF = GPU_Fields(GR, subgrids, CF)

constant_fields_to_NC(GR, HSURF, RAD, SOIL)


if i_load_from_restart:
    outCounter = GR.outCounter
else:
    outCounter = 0


if comp_mode == 2:
    ##############################
    t_start = time.time()
    stream = GR.stream
    #COLP_OLDd         = cuda.to_device(COLP_OLD, stream)
    #COLPd             = cuda.to_device(COLP, stream)
    #COLP_NEWd         = cuda.to_device(COLP_NEW, stream)
    #dCOLPdtd          = cuda.to_device(dCOLPdt, stream)
    UFLXd             = cuda.to_device(UFLX, stream)
    dUFLXdtd          = cuda.to_device(dUFLXdt, stream)
    VFLXd             = cuda.to_device(VFLX, stream)
    dVFLXdtd          = cuda.to_device(dVFLXdt, stream)
    FLXDIVd           = cuda.to_device(FLXDIV, stream)
    BFLXd             = cuda.to_device(BFLX, stream)
    CFLXd             = cuda.to_device(CFLX, stream)
    DFLXd             = cuda.to_device(DFLX, stream)
    EFLXd             = cuda.to_device(EFLX, stream)
    RFLXd             = cuda.to_device(RFLX, stream)
    QFLXd             = cuda.to_device(QFLX, stream)
    SFLXd             = cuda.to_device(SFLX, stream)
    TFLXd             = cuda.to_device(TFLX, stream)
    UWIND_OLDd        = cuda.to_device(UWIND_OLD, stream)
    UWINDd            = cuda.to_device(UWIND, stream)
    VWIND_OLDd        = cuda.to_device(VWIND_OLD, stream)
    VWINDd            = cuda.to_device(VWIND, stream)
    WWINDd            = cuda.to_device(WWIND, stream)
    POTT_OLDd         = cuda.to_device(POTT_OLD, stream)
    POTTd             = cuda.to_device(POTT, stream)
    POTTVBd           = cuda.to_device(POTTVB, stream)
    dPOTTdtd          = cuda.to_device(dPOTTdt, stream)
    RAD.dPOTTdt_RADd  = cuda.to_device(RAD.dPOTTdt_RAD, stream)
    MIC.dPOTTdt_MICd  = cuda.to_device(MIC.dPOTTdt_MIC, stream)
    PVTFd             = cuda.to_device(PVTF, stream)
    PVTFVBd           = cuda.to_device(PVTFVB, stream)
    PHId              = cuda.to_device(PHI, stream)
    PHIVBd            = cuda.to_device(PHIVB, stream)
    MIC.QV_OLDd       = cuda.to_device(MIC.QV_OLD, stream)
    MIC.QVd           = cuda.to_device(MIC.QV, stream)
    MIC.dQVdtd        = cuda.to_device(MIC.dQVdt, stream)
    MIC.dQVdt_MICd    = cuda.to_device(MIC.dQVdt_MIC, stream)
    MIC.QC_OLDd       = cuda.to_device(MIC.QC_OLD, stream)
    MIC.QCd           = cuda.to_device(MIC.QC, stream)
    MIC.dQCdtd        = cuda.to_device(MIC.dQCdt, stream)
    MIC.dQCdt_MICd    = cuda.to_device(MIC.dQCdt_MIC, stream)
    # diagnostics
    PAIRd             = cuda.to_device(PAIR, stream)
    TAIRd             = cuda.to_device(TAIR, stream)
    TAIRVBd           = cuda.to_device(TAIRVB, stream)
    RHOd              = cuda.to_device(RHO, stream)
    WINDd             = cuda.to_device(WIND, stream)
    HSURFd            = cuda.to_device(HSURF, stream)

    t_end = time.time()
    GR.copy_time += t_end - t_start
    ##############################



while GR.ts < GR.nts:
    real_time_ts_start = time.time()
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt
    GR.GMT += timedelta(seconds=GR.dt)

    print_ts_info(GR, WIND, UWIND, VWIND, CF.COLP, POTT)

    ########## DIAGNOSTICS (not related to dynamics)
    #if i_radiation or i_microphysics or i_soil:
    #    t_start = time.time()
    #    PAIR, TAIR, TAIRVB, RHO, WIND = \
    #            diagnose_secondary_fields(GR, CF.COLP, PAIR, PHI, POTT, POTTVB,
    #                                    TAIR, TAIRVB, RHO,\
    #                                    PVTF, PVTFVB, UWIND, VWIND, WIND)
    #    #PAIR, TAIR, TAIRVB, RHO, WIND = \
    #    #        diagnose_secondary_fields_c(GR, CF.COLP, PAIR, PHI, POTT, POTTVB,
    #    #                                TAIR, TAIRVB, RHO,\
    #    #                                PVTF, PVTFVB, UWIND, VWIND, WIND)
    #    #PAIR = np.asarray(PAIR)
    #    #TAIR = np.asarray(TAIR)
    #    #TAIRVB = np.asarray(TAIRVB)
    #    #RHO = np.asarray(RHO)
    #    #WIND = np.asarray(WIND)
    #    t_end = time.time()
    #    GR.diag_comp_time += t_end - t_start
    ##########

    ########## RADIATION
    #if i_radiation:
    #    t_start = time.time()
    #    RAD.calc_radiation(GR, TAIR, TAIRVB, RHO, PHIVB, SOIL, MIC)
    #    t_end = time.time()
    #    GR.rad_comp_time += t_end - t_start
    ##########

    ########## MICROPHYSICS
    #if i_microphysics:
    #    t_start = time.time()
    #    MIC.calc_microphysics(GR, WIND, SOIL, TAIR, PAIR, RHO, PHIVB)
    #    t_end = time.time()
    #    GR.mic_comp_time += t_end - t_start
    ##########

    ########## SOIL
    #if i_soil:
    #    t_start = time.time()
    #    SOIL.advance_timestep(GR, RAD, MIC)
    #    t_end = time.time()
    #    GR.soil_comp_time += t_end - t_start
    ##########

    ######## DYNAMICS
    t_dyn_start = time.time()
    if comp_mode in [0,1]:
        #F.COLP, PHI, PHIVB, POTT, POTTVB, \
        #UWIND, VWIND, WWIND,\
        #UFLX, VFLX, MIC.QV, MIC.QC \
        #            = time_stepper(GR, subgrids,
        #                    F.COLP_OLD, F.COLP, F.COLP_NEW, dCOLPdt, PHI, PHIVB,
        #                    POTT_OLD, POTT, dPOTTdt, POTTVB,
        #                    UWIND_OLD, UWIND, VWIND_OLD, VWIND, WWIND,
        #                    UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
        #                    BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
        #                    HSURF, PVTF, PVTFVB, 
        #                    RAD.dPOTTdt_RAD, MIC.dPOTTdt_MIC,
        #                    MIC.QV_OLD, MIC.QV, MIC.dQVdt,
        #                    MIC.QC_OLD, MIC.QC, MIC.dQCdt,
        #                    MIC.dQVdt_MIC, MIC.dQCdt_MIC)
        PHI, PHIVB, POTT, POTTVB, \
        UWIND, VWIND, WWIND,\
        UFLX, VFLX, MIC.QV, MIC.QC \
                    = time_stepper(GR, subgrids, CF,
                            PHI, PHIVB,
                            POTT_OLD, POTT, dPOTTdt, POTTVB,
                            UWIND_OLD, UWIND, VWIND_OLD, VWIND, WWIND,
                            UFLX, dUFLXdt, VFLX, dVFLXdt, FLXDIV,
                            BFLX, CFLX, DFLX, EFLX, RFLX, QFLX, SFLX, TFLX, 
                            HSURF, PVTF, PVTFVB, 
                            RAD.dPOTTdt_RAD, MIC.dPOTTdt_MIC,
                            MIC.QV_OLD, MIC.QV, MIC.dQVdt,
                            MIC.QC_OLD, MIC.QC, MIC.dQCdt,
                            MIC.dQVdt_MIC, MIC.dQCdt_MIC)

    elif comp_mode == 2:
        PHId, PHIVBd, POTTd, POTTVBd, \
        UWINDd, VWINDd, WWINDd,\
        UFLXd, VFLXd, MIC.QVd, MIC.QCd\
                    = time_stepper(GR, subgrids, GF,
                            PHId, PHIVBd,
                            POTT_OLDd, POTTd, dPOTTdtd, POTTVBd,
                            UWIND_OLDd, UWINDd, VWIND_OLDd, VWINDd, WWINDd,
                            UFLXd, dUFLXdtd, VFLXd, dVFLXdtd, FLXDIVd,
                            BFLXd, CFLXd, DFLXd, EFLXd, RFLXd, QFLXd, SFLXd, TFLXd, 
                            HSURFd, PVTFd, PVTFVBd, 
                            RAD.dPOTTdt_RADd, MIC.dPOTTdt_MICd,
                            MIC.QV_OLDd, MIC.QVd, MIC.dQVdtd,
                            MIC.QC_OLDd, MIC.QCd, MIC.dQCdtd,
                            MIC.dQVdt_MICd, MIC.dQCdt_MICd)

    t_dyn_end = time.time()
    GR.dyn_comp_time += t_dyn_end - t_dyn_start
    ########

    ## TEST FOR CRASH
    #for k in range(0,GR.nz):
    #    if (np.sum(np.isnan(UWIND[:,:,k][GR.iisjj])) > 0) | \
    #            (np.max(UWIND[:,:,k][GR.iisjj]) > 500):
    #        quit()


    # WRITE NC FILE
    if GR.ts % GR.i_out_nth_ts == 0:

        if comp_mode == 2:
            ##############################
            t_start = time.time()
            GF.COLP             .to_host(stream)
            PAIRd             .to_host(stream)
            PHId              .to_host(stream)
            PHIVBd            .to_host(stream)
            UWINDd            .to_host(stream)
            VWINDd            .to_host(stream)
            WINDd             .to_host(stream)
            WWINDd            .to_host(stream) 
            POTTd             .to_host(stream)
            TAIRd             .to_host(stream)
            RHOd              .to_host(stream)
            PVTFVBd           .to_host(stream)
            MIC.QV_OLDd       .to_host(stream)
            MIC.QVd           .to_host(stream)
            MIC.dQVdtd        .to_host(stream)
            MIC.QC_OLDd       .to_host(stream)
            MIC.QCd           .to_host(stream)
            MIC.dQCdtd        .to_host(stream)
            MIC.dQCdt_MICd    .to_host(stream)
            RAD.dPOTTdt_RADd  .to_host(stream)
            MIC.dPOTTdt_MICd  .to_host(stream)

            stream.synchronize()

            t_end = time.time()
            GR.copy_time += t_end - t_start
            ##############################

        t_start = time.time()
        outCounter += 1
        #WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
        #                                WIND, UWIND, VWIND, COLP, POTT)
        output_to_NC(GR, outCounter, CF.COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,
                    POTT, TAIR, RHO, PVTF, PVTFVB,
                    RAD, SOIL, MIC)
        t_end = time.time()
        GR.IO_time += t_end - t_start

    # WRITE RESTART FILE
    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        GR.outCounter = outCounter
        write_restart(GR, F.COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
                        UFLX, VFLX, \
                        HSURF, POTT, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
                        RAD, SOIL, MIC, TURB)


    GR.total_comp_time += time.time() - real_time_ts_start

print_computation_time_info(GR)


