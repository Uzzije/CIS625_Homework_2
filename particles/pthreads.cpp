#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <algorithm>
#include "pthread_barrier.h"
#include "common.h"

using std::vector;
//
//  global variables
//

extern double size;

double cutoff = 0.01;
int offset[9];
int num_bins;

vector<vector<particle_t> > grid;

int n, n_threads, no_output = 0;
FILE *fsave,*fsum;
double gabsmin = 1.0, gabsavg = 0.0;

particle_t *particles;

pthread_barrier_t barrier;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

//
//  check that pthreads routine call was successful
//
#define P( condition ) {if( (condition) != 0 ) { printf( "\n FAILURE in %s, line %d\n", __FILE__, __LINE__ );exit( 1 );}}

int index(int i, int j)
{
    return i*num_bins + j;
}

void compute_offset()
{
    offset[0] = index(-1,-1);
    offset[1] = index(-1,0);
    offset[2] = index(-1,1);
    offset[3] = index(0,-1);
    offset[4] = index(0,0);
    offset[5] = index(0,1);
    offset[6] = index(1,-1);
    offset[7] = index(1,0);
    offset[8] = index(1,1);      
}

int get_bin_index(particle_t& p)
{
    int i = p.x / (2 * cutoff) + 1;
    int j = p.y / (2 * cutoff) + 1;
    return index(i,j);
}

void resize_grid()
{
    num_bins = (int)ceil(size/(2*cutoff)) + 2;
    grid.resize(num_bins*num_bins);
}

void bin_particles(int n, particle_t* particles)
{
    for(int i = 0; i < n; i++) {
        int j = get_bin_index(particles[i]);
        grid[j].push_back(particles[i]);
    }
}

vector<particle_t> get_neighbors(int idx)
{
    vector<particle_t> neighbors;
    for(int i = 0; i < 9; i++) {
        int bin = idx + offset[i];
        for(int j = 0; j < grid[bin].size(); j++) {
            neighbors.push_back(grid[bin][j]);
        }
    }
    return neighbors;
}

//
//  This is where the action happens
//
void *thread_routine( void *pthread_id )
{
    int navg, nabsavg=0;
    double dmin, absmin = 1.0, davg,absavg = 0.0;
    int thread_id = *(int*)pthread_id;

    //printf("Thread %d: Begin\n", thread_id);

    int total_bins = num_bins * num_bins;
    int bins_per_thread = (total_bins + n_threads - 1) / n_threads;
    int first = min(thread_id * bins_per_thread + 1, total_bins);
    int last = min((thread_id + 1) * bins_per_thread + 1, total_bins);

    // These values are NOT the particles that this thread simulates
    // These are only the particles that this thread is responsible for rebinning
    int particles_per_thread = (n + n_threads - 1) / n_threads;
    int first_particle = min(thread_id * particles_per_thread, n);
    int last_particle = min((thread_id + 1)* particles_per_thread, n);

    //printf("Thread %d: bins_owned (%d) = (%d, %d)\n", thread_id, bins_per_thread, first, last);
    //printf("Thread %d: parts_owned (%d) = (%d, %d)\n", thread_id, particles_per_thread, first_particle, last_particle);
    
    //
    //  simulate a number of time steps
    //
    for( int step = 0; step < NSTEPS; step++ )
    {
     //   printf("Thread %d: Begin Iteration %d\n", thread_id, step);
        dmin = 1.0;
        navg = 0;
        davg = 0.0;
        //
        //  compute forces
        //
        for(int bin = first; bin < last; bin++) {
            if(grid[bin].size() > 0) { //False for all boundary bins
                vector<particle_t> neighbors = get_neighbors(bin);
                for(int k = 0; k < grid[bin].size(); k++) {
                    grid[bin][k].ax = grid[bin][k].ay = 0;
                    for(int l = 0; l < neighbors.size(); l++) {
                        apply_force(grid[bin][k], neighbors[l], &dmin, &davg, &navg);
                    }
                }
            }
        }
       // printf("Thread %d: Reached First Barrier, Iteration %d\n", thread_id, step);
        pthread_barrier_wait(&barrier);
        
        if( no_output == 0 )
        {
          //
          // Computing statistical data
          // 
            if (navg) {
                absavg +=  davg/navg;
                nabsavg++;
            }
            if (dmin < absmin) {
                absmin = dmin;
            }
        }
    
        // Move each particle for this thread
        // Copy particles back to original position in particles array
        for(int bin = first; bin < last; bin++) {
            for(int k = 0; k < grid[bin].size(); k++) {
                move(grid[bin][k]);
                particles[grid[bin][k].index] = grid[bin][k];
            }
            grid[bin].clear(); //Clear our bins
        }
        //printf("Thread %d: Reached Second Barrier, Iteration %d\n", thread_id, step);
        pthread_barrier_wait(&barrier);
        
        //At this point all threads have computed new positions for
        //their particles and cleared their bins
        //Now we can safely rebin all our particles
        //Sadly it seems this part has to be essentially serial
        //since the STL isn't thread safe for writes
        pthread_mutex_lock(&mutex); 
        for(int i = first_particle; i < last_particle; i++) {
            int j = get_bin_index(particles[i]);
            grid[j].push_back(particles[i]);
        }
        pthread_mutex_unlock(&mutex);
        //printf("Thread %d: Reached Thrid Barrier, Iteration %d\n", thread_id, step);
        
        //Need to wait for all threads to finish rebinning
        //One more sync point than starter code, but can't
        //seem to avoid it
        pthread_barrier_wait(&barrier);
        
        //
        //  save if necessary
        //
        if (no_output == 0) 
          if( thread_id == 0 && fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
    }
     
    if (no_output == 0 )
    {
      absavg /= nabsavg; 	
      //printf("Thread %d has absmin = %lf and absavg = %lf\n",thread_id,absmin,absavg);
      pthread_mutex_lock(&mutex);
      gabsavg += absavg;
      if (absmin < gabsmin) gabsmin = absmin;
      pthread_mutex_unlock(&mutex);    
    }

    return NULL;
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    //
    //  process command line
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-p <int> to set the number of threads\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");        
        return 0;
    }
    
    n = read_int( argc, argv, "-n", 1000 );
    n_threads = read_int( argc, argv, "-p", 2 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    fsave = savename ? fopen( savename, "w" ) : NULL;
    fsum = sumname ? fopen ( sumname, "a" ) : NULL;

    if(find_option( argc, argv, "-no" ) != -1) {
      no_output = 1;
    }

    //
    //  allocate resources
    //
    particles = (particle_t*) malloc(n * sizeof(particle_t));
    set_size(n);
    init_particles(n, particles);

    // Store the position of a particle in the list
    // with the particle so each thread knows where
    // to place the result
    for(int i = 0; i < n; i++) {
        particles[i].index = i;
    }    

    resize_grid();
    compute_offset();
    bin_particles(n, particles);

    pthread_attr_t attr;
    P(pthread_attr_init(&attr));
    P(pthread_barrier_init(&barrier, NULL, n_threads));

    int *thread_ids = (int *) malloc(n_threads * sizeof(int));
    for(int i = 0; i < n_threads; i++ ) {
        thread_ids[i] = i;
    }

    pthread_t *threads = (pthread_t *) malloc( n_threads * sizeof( pthread_t ) );
    
    //
    //  do the parallel work
    //
    double simulation_time = read_timer( );
    for(int i = 1; i < n_threads; i++) {
        P(pthread_create(&threads[i], &attr, thread_routine, &thread_ids[i]));
    }

    thread_routine(&thread_ids[0]);
    
    for(int i = 1; i < n_threads; i++) { 
        P(pthread_join(threads[i], NULL));
    }
    simulation_time = read_timer( ) - simulation_time;
   
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      gabsavg /= (n_threads*1.0);
      // 
      //  -the minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", gabsmin, gabsavg);
      if (gabsmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting ");
      if (gabsavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting ");
    }
    printf("\n");

    //
    // Printing summary data
    //
    if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_threads,simulation_time); 
    
    //
    //  release resources
    //
    P( pthread_barrier_destroy( &barrier ) );
    P( pthread_attr_destroy( &attr ) );
    free( thread_ids );
    free( threads );
    free( particles );
    if( fsave )
        fclose( fsave );
    if( fsum )
        fclose ( fsum );
    
    return 0;
}
