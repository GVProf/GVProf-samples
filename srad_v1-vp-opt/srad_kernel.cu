// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, WRONG MEMORY
// ACCESS

// srad kernel
__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, 
                     fp *d_c,
                      fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // get current horizontal block index
    int tx = threadIdx.x;              // get current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // more threads than actual elements !!!
    int row;                           // column, x position
    int col;                           // row, y position

    // variables
    fp d_Jc;
    fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
    fp d_c_loc;
    fp d_G2, d_L, d_num, d_den, d_qsqr;

    // figure out row/col location in new matrix
    row = (ei + 1) % d_Nr - 1;     // (0-n) row
    col = (ei + 1) / d_Nr + 1 - 1; // (0-n) column
    if ((ei + 1) % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run
    int iN, iS, jW, jE;
        if(row == 0) {
        iN = 0;
        }else{
            iN = row -1;
        }
        if(row == d_Nr -1){
            iS = d_Nr -1;
        }else{
            iS = row+1;
        }
        if(col ==0){
            jW = 0;
        }else{
            jW =col -1;
        }
        if(col == d_Nc -1){
            jE = d_Nc - 1;
        }else{
            jE = col +1;
        }
        // directional derivatives, ICOV, diffusion coefficent
        d_Jc = d_I[ei]; // get value of the current element

        // directional derivates (every element of IMAGE)(try to copy to shared
        // memory or temp files)
        d_dN_loc =
            d_I[iN + d_Nr * col] - d_Jc; // north direction derivative
        d_dS_loc =
            d_I[iS + d_Nr * col] - d_Jc; // south direction derivative
        d_dW_loc =
            d_I[row + d_Nr * jW] - d_Jc; // west direction derivative
        d_dE_loc =
            d_I[row + d_Nr * jE] - d_Jc; // east direction derivative

        // normalized discrete gradient mag squared (equ 52,53)
        d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
                d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) /
               (d_Jc * d_Jc); // gradient (based on derivatives)

        // normalized discrete laplacian (equ 54)
        d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) /
              d_Jc; // laplacian (based on derivatives)

        // ICOV (equ 31/35)
        d_num = (0.5 * d_G2) -
                ((1.0 / 16.0) *
                 (d_L * d_L));    // num (based on gradient and laplacian)
        d_den = 1 + (0.25 * d_L); // den (based on laplacian)
        d_qsqr = d_num / (d_den * d_den); // qsqr (based on num and den)

        // diffusion coefficent (equ 33) (every element of IMAGE)
        d_den = (d_qsqr - d_q0sqr) /
                (d_q0sqr * (1 + d_q0sqr)); // den (based on qsqr and q0sqr)
        d_c_loc = 1.0 / (1.0 + d_den); // diffusion coefficient (based on den)

        // save data to global memory
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;

        // saturate diffusion coefficent to 0-1 range
        if (d_c_loc < 0.0) {        // if diffusion coefficient < 0
            d_c[ei] = 0.0;
        } else if (d_c_loc < 1.0) {
            d_c[ei] = d_c_loc;
        }
    }
}
