// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, MEMORY
// CORRUPTION

// srad kernel
__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, uint32_t *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // get current horizontal block index
    int tx = threadIdx.x;              // get current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // more threads than actual elements !!!
    int row;                           // column, x position
    int col;                           // row, y position

    // variables
    int d_cN, d_cS, d_cW, d_cE;
    fp d_D;

    // figure out row/col location in new matrix
    row = (ei + 1) % d_Nr - 1;     // (0-n) row
    col = (ei + 1) / d_Nr + 1 - 1; // (0-n) column
    if ((ei + 1) % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  uint32_t mask = 1<<laneid;

        // diffusion coefficent
        d_cN = (int)(d_c[ei/32] & mask);                     // north diffusion coefficient
        d_cS = d_c[(d_iS[row] + d_Nr * col)/32] & mask; // south diffusion coefficient
        d_cW = d_c[ei/32]&mask;                     // west diffusion coefficient
        d_cE = d_c[(row + d_Nr * d_jE[col])/32]&mask; 

        // divergence (equ 58)
        d_D = d_cN * d_dN[ei] + d_cS * d_dS[ei] + d_cW * d_dW[ei] +
              d_cE * d_dE[ei]; // divergence

        // image update (equ 61) (every element of IMAGE)
        d_I[ei] =
            d_I[ei] +
            0.25 * d_lambda *
                d_D; // updates image (based on input time step and divergence)
    }
}
