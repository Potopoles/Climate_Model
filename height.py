import numpy as np

def height_tendency_upstream(GR, HGHT, UWIND, VWIND, UFLX, VFLX):
    UFLX[GR.iisjj] = \
            np.maximum(UWIND[GR.iisjj],0) * HGHT[GR.iisjj_im1] * GR.dy + \
            np.minimum(UWIND[GR.iisjj],0) * HGHT[GR.iisjj] * GR.dy

    VFLX[GR.iijjs] = \
            np.maximum(VWIND[GR.iijjs],0) * HGHT[GR.iijjs_jm1] * GR.dx[GR.iijjs_jm1] + \
            np.minimum(VWIND[GR.iijjs],0) * HGHT[GR.iijjs] * GR.dx[GR.iijjs]

    dHGHTdt = ( - (UFLX[GR.iijj_ip1] - UFLX[GR.iijj]) - (VFLX[GR.iijj_jp1] - VFLX[GR.iijj]) ) / GR.A[GR.iijj]

    return(dHGHTdt)
