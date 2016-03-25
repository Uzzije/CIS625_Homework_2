#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256 //change?

extern double size;
//
//  benchmarking program
//

/// The size of each given bin.
__device__ double bin_size;
double host_bin_size;
/// integer number of bins (with some slop)
__device__ int bins_per_side;
int host_bins_per_side;

//device variable macros
//not sure how these work
#define BIN_SIZE bin_size
#define NUM_BINS_PER_SIDE bins_per_side
#define NUM_BINS (NUM_BINS_PER_SIDE*NUM_BINS_PER_SIDE)
#define HOST_BIN_SIZE host_bin_size
#define HOST_NUM_BINS_PER_SIDE host_bins_per_side
#define HOST_NUM_BINS (HOST_NUM_BINS_PER_SIDE*HOST_NUM_BINS_PER_SIDE)

cudaError_t status;

void set_device_vars(double local_bin_size, int local_bins_per_side) {
    host_bin_size = local_bin_size;
    host_bins_per_side = local_bins_per_side;
    //Store to device variables

    status = cudaMemcpyToSymbol(bins_per_side, &local_bins_per_side, sizeof(int));
    assert( cudaSuccess == status);
    status = cudaMemcpyToSymbol(bin_size, &local_bin_size, sizeof(double));
    assert( cudaSuccess == status);
}
void set_bin_size( int n )
{
    double local_bin_size;
    int local_bins_per_side;


    local_bin_size = sqrt(density*5);
    local_bins_per_side = floor(size/local_bin_size)+1;
    int num_bins = local_bins_per_side*local_bins_per_side;

    assert( local_bin_size > 2*cutoff );

    printf("%d, %d, %g\n", num_bins, local_bins_per_side, local_bin_size);
    printf("%g\n", size);
    printf("%d\n", n);

    //Store to device variables
    set_device_vars(local_bin_size, local_bins_per_side);
}

//This is a complete new design - Thrust is essentially useless since
// you can't navigate the device_vector object on the device.  (WTF?)
namespace {
   std::vector< std::vector<int> > bin_members;
   // These two are HOST ptrs to device memory
   int* hptr_bin_list; //index of all particles, grouped by bin
   int* hptr_bin_index; //index of first particle in a given bin
   //These are DEVICE ptrs to device memory
   __device__ int* bin_list; //index of all particles, grouped by bin
   __device__ int* bin_index; //index of first particle in a given bin
   //Yes, we really do need two sets, and yes, we have to update them
   // when calling the appropriate kernel.  Stupid, yes, it is.
}
/// This will only hold AFTER particles are binned
/// Can only be called on GPU since memory is allocated there
__device__
void check_bin_invariants_impl(int* bin_list_p, int* bin_index_p, int n) {
  for(int i = 0; i < n; i++) {
    assert(bin_list[i] >= 0 && bin_list[i] < n);
    //printf("%d - %d\n", i, bin_list[i] );
  }

  for(int i = 0; i < NUM_BINS; i++) {
    //empty bins on either end
    assert( bin_index[i] >= 0 && bin_index[i] <= n );
    //printf("%d - %d\n", i, bin_index[i] );
    if( i != 0 ) {
      //printf("%d <= %d\n", bin_index[i-1], bin_index[i] );
      assert( bin_index[i-1] <= bin_index[i] );
    } else {
      assert( bin_index[0] == 0 );
    }
  }
}
__global__
void check_bin_invariants(int* bin_list_p, int* bin_index_p, int n) {
  check_bin_invariants_impl(bin_list_p, bin_index_p, n);
}
int which_bin(particle_t &particle) {
  int col = floor(particle.x/HOST_BIN_SIZE);
  int row = floor(particle.y/HOST_BIN_SIZE);
  assert ( col < HOST_NUM_BINS_PER_SIDE);
  assert ( row < HOST_NUM_BINS_PER_SIDE);

  int index = row*HOST_NUM_BINS_PER_SIDE + col;
  assert( index >= 0 && index < HOST_NUM_BINS );
  return index;
}

__global__ void update_device_pointers(int* bin_list_p, int* bin_index_p, int n)
{
  bin_list = bin_list_p;
  bin_index = bin_index_p;

  //sanity checking values
  for(int i = 0; i < n; i++) bin_list[i] = -1;
  for(int i = 0; i < NUM_BINS; i++) bin_index[i] = -1;
}

void bin_init(int n) {
  bin_members.resize(HOST_NUM_BINS);
  hptr_bin_list = NULL;
  status = cudaMalloc((void**)&hptr_bin_list, n * sizeof(int));
  assert( cudaSuccess == status);
  hptr_bin_index = NULL;
  status = cudaMalloc((void**)&hptr_bin_index, HOST_NUM_BINS * sizeof(int));
  assert( cudaSuccess == status);

  update_device_pointers<<<1,1>>>(hptr_bin_list, hptr_bin_index, n);
}
//Figure out which indexes live in each bin, then copy
//all that information to the device memory in bin_list
void bin_particles(particle_t* particles, int n) {
   // First map each particle to a bin (in host memory, using std)
   for(int i = 0; i < HOST_NUM_BINS; i++) {
     bin_members[i].clear();
   }
   for(int i = 0; i < n; i++) {
     int bin = which_bin(particles[i]);
     assert( bin >= 0 && bin < HOST_NUM_BINS );
     bin_members[bin].push_back(i);
  }
  // Then copy that into device memory
  int sofar = 0;
  for(int i = 0; i < HOST_NUM_BINS; i++) {
     const int size = bin_members[i].size();
     assert( size >= 0 && size < n);
     status = cudaMemcpy(&(hptr_bin_list[sofar]), &(bin_members[i])[0], 
                size * sizeof(int), cudaMemcpyHostToDevice);
     assert( cudaSuccess == status);
     status = cudaMemcpy(&(hptr_bin_index[i]), &sofar, 
                sizeof(int), cudaMemcpyHostToDevice);
     assert( cudaSuccess == status);
     sofar += size;
  }
  assert( sofar == n);

  check_bin_invariants<<<1,1>>>(hptr_bin_list, hptr_bin_index, n);
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

///Compute all of the forces for the particles local to a specific box
/// i is the box index
__device__
void compute_forces_for_box(particle_t * particles, int n, int r, int c) {
  //Computing forces between all points in the box and all points in the neighbouring boxes:
  const int i = r * NUM_BINS_PER_SIDE + c;
  const int box_size = NUM_BINS_PER_SIDE;

  const int box_begin = bin_index[i];
  assert( box_begin >= 0 && box_begin <= n);
  const int box_end = i+1 < NUM_BINS ? bin_index[i+1] : n;
  assert( box_end >= 0 && box_end <= n);
  assert( box_begin <= box_end);

  for(int bi = box_begin; bi < box_end; bi++) {
    const int p = bin_list[bi]; //the particle index in particles
    assert( p >= 0 && p < n);
    particles[p].ax = 0;
    particles[p].ay = 0;
#define COMPARE_TO_BIN(index) \
    do { \
      const int other_box_begin = bin_index[index]; \
      const int other_box_end = (index)+1 < NUM_BINS ? bin_index[(index)+1] : n; \
      for(int bi2 = other_box_begin; bi2 < other_box_end; bi2++) { \
        const int p2 = bin_list[bi2]; \
	assert( p2 >= 0 && p2 < n); \
        apply_force_gpu( particles[p], particles[p2] ); \
      } \
    } while(false);
    
    //all particles in this bin 
    COMPARE_TO_BIN(i);

    //For the box to the left:
    if( c-1 >= 0 )
      COMPARE_TO_BIN(i-1);
    //For the box to the right:
    if( c+1 < NUM_BINS_PER_SIDE )
      COMPARE_TO_BIN(i+1);
    //For the box to the top:
    if (r-1 >= 0)
      COMPARE_TO_BIN(i-box_size);
    //For the box to the bottom:
    if (r+1 < NUM_BINS_PER_SIDE)
      COMPARE_TO_BIN(i+box_size);

      //For the box int the upper left-hand corner:
    if ((c - 1 >= 0) && (r-1 >= 0))
      COMPARE_TO_BIN(i -box_size -1 );
      //For the box int the upper right-hand corner:
    if ((c + 1 < NUM_BINS_PER_SIDE) && (r-1 >= 0))
      COMPARE_TO_BIN(i-box_size+1);
      //For the box int the lower left-hand corner:
    if ((c - 1 >= 0) && (r+1 < NUM_BINS_PER_SIDE))
      COMPARE_TO_BIN(i+box_size-1);
    //For the box int the lower right-hand corner:
    if ((c + 1 < NUM_BINS_PER_SIDE) && (r+1 < NUM_BINS_PER_SIDE))
      COMPARE_TO_BIN(i+box_size+1);
#undef COMPARE_TO_BIN
  }
}


__global__ void compute_forces_gpu(particle_t * particles, int n)
{
  check_bin_invariants_impl(bin_list, bin_index, n);

  assert( bin_size > 0 );
  assert( bins_per_side > 0 );

  // Get thread (bin) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= NUM_BINS) return;

  //TODO: should use a block group per box..

  int i = tid;
  int r = i/NUM_BINS_PER_SIDE;
  int c = i % NUM_BINS_PER_SIDE;
  compute_forces_for_box(particles, n, r, c); // O(#bins * c)
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

  // Get thread (particle) ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= n) return;

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
    status = cudaThreadSynchronize(); 
    assert( cudaSuccess == status);

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    status = cudaMalloc((void **) &d_particles, n * sizeof(particle_t));
    assert( cudaSuccess == status);

    set_size( n );
    set_bin_size( n );

    init_particles( n, particles );

    //allocate all the global storage
    bin_init(n);
    bin_particles(particles, n);


    assert( cudaSuccess == cudaThreadSynchronize() );
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    status = cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);
    assert( cudaSuccess == status);

    status = cudaThreadSynchronize();
    assert( cudaSuccess == status);
    copy_time = read_timer( ) - copy_time;
    
    //
    //  simulate a number of time steps
    //
    status = cudaThreadSynchronize();
    assert( cudaSuccess == status);
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces (block of bins)
        //
	int blks = (HOST_NUM_BINS + NUM_THREADS - 1) / NUM_THREADS;
	compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n);

	// Note: This organization is horrible for inner block performance
	// currently every single thread will diverge.  This is worst case
	// inner loop performance for a GPU.  The O(n) vs O(n^2) still makes
	// it a net win though.  To do this for real, you'd want as many
	// threads working on the same block as possible (to share memory
	// accesses in the inner loop) and would probably want many more
	// particles per block. (Guess: 1000 vs 5).  To maximimize the
	// benefit, you'd want to figure out which particle in which block
	// each thread should be working on in a given iteration.  
	// i.e. prefix sum index array, mod by num active threads, 
	// create O(N) temp buffer
	// It's not completely clear whether a thread taking multiple particles
	// from the same block or multiple threads per block would be better.
	// Suspect you'd want to autotune.

	// p.s. All this kernel launch overhead is a waste.  Does CUDA allow
	// inside kernel barriers?  If so, this entire loop could be moved
	// into a single kernel call.  (a.k.a. gpu-gc style)
        
        //
        //  move particles (block of particles)
        //
	int move_blks = (n + NUM_THREADS - 1) / NUM_THREADS;
	move_gpu <<< move_blks, NUM_THREADS >>> (d_particles, n, size);
        
	// Copy the particles back to the CPU
        status = cudaMemcpy(particles, d_particles, 
	           n * sizeof(particle_t), cudaMemcpyDeviceToHost);
    	assert( cudaSuccess == status);
	//rebin them - this writes back to the GPU directly
	bin_particles(particles, n);	
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
#if 0 //not needed due to previous read
	    // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
#endif
            save( fsave, n, particles);
	}
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );
    
    return 0;
}
