#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

extern double size;
//
//  benchmarking program
//

void compute_grid_offset(int offset[9], int num_bins, int bin_size)
{
    offset[0] = -num_bins * bin_size - bin_size;
    offset[1] = -num_bins * bin_size;
    offset[2] = -num_bins * bin_size + bin_size;
    offset[3] = -bin_size;
    offset[4] = 0;
    offset[5] = bin_size;
    offset[6] = num_bins * bin_size - bin_size;
    offset[7] = num_bins * bin_size;
    offset[8] = num_bins * bin_size + bin_size;
}

void compute_bin_offset(int offset[9], int num_bins)
{
    offset[0] = -num_bins - 1;
    offset[1] = -num_bins;
    offset[2] = -num_bins + 1;
    offset[3] = -1;
    offset[4] = 0;
    offset[5] = 1;
    offset[6] = num_bins - 1;
    offset[7] = num_bins;
    offset[8] = num_bins + 1;
}

int grid_size()
{
    return (int)ceil(size/(2*cutoff)) + 2;
}

__device__ int get_bin_index(particle_t& p, int num_bins)
{
    int i = p.x / (2 * cutoff) + 1;
    int j = p.y / (2 * cutoff) + 1;
    return i * num_bins + j;
}

__device__ int get_grid_index(particle_t& p, int num_bins, int bin_size)
{
    int i = p.x / (2 * cutoff) + 1;
    int j = p.y / (2 * cutoff) + 1;
    return i * num_bins * bin_size + j * bin_size;
}

__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
  double dx = neighbor.x - particle.x;
  double dy = neighbor.y - particle.y;
  double r2 = dx * dx + dy * dy;
  if( r2 > cutoff*cutoff )
      return;
  //r2 = fmax( r2, min_r*min_r );
  r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
  double r = sqrt( r2 );

  //
  //  very simple short-range repulsive force
  //
  double coef = ( 1 - cutoff / r ) / r2 / mass;
  particle.ax += coef * dx;
  particle.ay += coef * dy;

}

__global__ void clear_bins(unsigned int* bin_sizes, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) {
        return;
    }
    
    bin_sizes[tid] = 0;
}

__global__ void bin_particles(particle_t* grid, particle_t* particles, unsigned int* bin_sizes, int n, int num_bins, int bin_size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) {
        return;
    }

    int bindex = get_bin_index(particles[tid], num_bins);
    int gindex = get_grid_index(particles[tid], num_bins, bin_size);

    int pos = atomicAdd(&bin_sizes[bindex], (unsigned int)1);
    grid[gindex + pos] = particles[tid];
}

__global__ void compute_forces_gpu(particle_t* grid, particle_t* particles, unsigned int* bin_sizes, int* goffset, int* boffset, int n, int num_bins, int bin_size)
{
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) {
        return;
    }

    particles[tid].ax = particles[tid].ay = 0;

    int bindex = get_bin_index(particles[tid], num_bins);
    int gindex = get_grid_index(particles[tid], num_bins, bin_size);

    for(int i = 0; i < 9; i++) {
        int gdx = gindex + goffset[i];
        int bdx = bindex + boffset[i];
        for(int j = 0; j < bin_sizes[bdx]; j++) {
            apply_force_gpu(particles[tid], grid[gdx + j]);
        }
    }
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) { 
        return;
    }

    particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}

int main( int argc, char **argv )
{    
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize(); 

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char* savename = read_string( argc, argv, "-o", NULL );
    
    FILE* fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t* particles = (particle_t*) malloc( n * sizeof(particle_t) );
    particle_t* d_particles;
    particle_t* grid;
    unsigned int* bin_sizes;
    int* goffset;
    int* boffset;

    int offset[9];
    int num_bins;
    int bin_size = 10;
    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    int clear_blks;

    set_size(n);
    init_particles(n, particles);

    num_bins = grid_size();    
    clear_blks = (num_bins*num_bins + NUM_THREADS - 1) / NUM_THREADS;

    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    cudaMalloc((void **) &grid, num_bins * num_bins * bin_size * sizeof(particle_t));
    cudaMalloc((void **) &bin_sizes, num_bins * num_bins * sizeof(unsigned int));
    cudaMalloc((void **) &goffset, 9 * sizeof(int));
    cudaMalloc((void **) &boffset, 9 * sizeof(int));    
 
    compute_grid_offset(offset, num_bins, bin_size);
    cudaMemcpy(goffset, offset, 9 * sizeof(int), cudaMemcpyHostToDevice);
    compute_bin_offset(offset, num_bins);
    cudaMemcpy(boffset, offset, 9 * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for(int step = 0; step < NSTEPS; step++) {

        clear_bins <<< clear_blks, NUM_THREADS >>> (bin_sizes, num_bins * num_bins);

        bin_particles <<< blks, NUM_THREADS >>> (grid, d_particles, bin_sizes, n, num_bins, bin_size);

        compute_forces_gpu <<< blks, NUM_THREADS >>> (grid, d_particles, bin_sizes, goffset, boffset, n, num_bins, bin_size);

        move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save(fsave, n, particles);
	    }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
   
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    cudaFree(grid);
    cudaFree(bin_sizes);
    cudaFree(goffset);
    cudaFree(boffset);
    if(fsave) {
        fclose(fsave);
    }
    return 0;
}
