import numpy as np

def tracer_tendency_upstream(GR, TRACER, HGHT, UWIND, VWIND, UFLX, VFLX, WIND):
    UFLX[GR.iisjj] = \
            np.maximum(UWIND[GR.iisjj],0) * TRACER[GR.iisjj_im1] * HGHT[GR.iisjj_im1] * GR.dy + \
            np.minimum(UWIND[GR.iisjj],0) * TRACER[GR.iisjj] * HGHT[GR.iisjj] * GR.dy

    VFLX[GR.iijjs] = \
            np.maximum(VWIND[GR.iijjs],0) * TRACER[GR.iijjs_jm1] * HGHT[GR.iijjs_jm1] * GR.dx[GR.iijjs_jm1] + \
            np.minimum(VWIND[GR.iijjs],0) * TRACER[GR.iijjs] * HGHT[GR.iijjs] * GR.dx[GR.iijjs]

    dTRACERdt = ( - (UFLX[GR.iijj_ip1] - UFLX[GR.iijj]) - (VFLX[GR.iijj_jp1] - VFLX[GR.iijj]) ) / \
            (GR.A[GR.iijj] * HGHT[GR.iijj]) + np.maximum(0., 0.00001*WIND[GR.iijj]*(5.0 - TRACER[GR.iijj]))

    return(dTRACERdt)

