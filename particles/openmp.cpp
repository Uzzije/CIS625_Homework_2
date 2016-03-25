#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <stdint.h>

#define TIMERS 0
#define MAXPARTILCESPERBOX 8

//
//  benchmarking program
//
int main(int argc, char **argv)
{
    int navg, nabsavg = 0;
    double davg, dmin, absmin = 1.0, absavg = 0.0;

    if(find_option(argc, argv, "-h") >= 0) {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    const uint32_t n = read_int(argc, argv, "-n", 1000);

    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    FILE *fsave = savename ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname ? fopen(sumname, "a") : NULL;

    particle_t *particles = (particle_t*) malloc(n * sizeof(particle_t));
    set_size(n);
    init_particles(n, particles);

    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
#if TIMERS==1
    double force_time = 0.0;
    double move_time = 0.0;
    double running_time;
#endif

    double size =  sqrt(0.0005 * n);
    double interaction_length = 0.01;


    int blocksize = (int) ceil(size/interaction_length);
    double block_width = size/blocksize;

    particle_t*** blocks = (particle_t***) malloc(blocksize*blocksize * sizeof(particle_t**));
    for(int b=0; b<blocksize*blocksize; b++){
        blocks[b] = (particle_t**)malloc(MAXPARTILCESPERBOX*sizeof(particle_t*));
    }


    for(int step = 0; step < NSTEPS; step++) {
        navg = 0;
        davg = 0.0;
        dmin = 1.0;
        //
        //  compute forces
        //
#if TIMERS==1
        running_time = read_timer();
#endif

        int number_in_block[blocksize*blocksize];
#pragma omp parallel shared(number_in_block) //declares the following section to be parallel with the number_in_block shared between threads
        {
#pragma omp for //this for is parallel, I can experiment with clauses like "schedule (dynamic, chunk_size)" maybe try 50 for chunk size <this reduces the grade to 40...
        for(int b=0; b<blocksize*blocksize; b++){
            number_in_block[b] = 0; // starts with no particles in any box;
        }

#pragma omp for nowait //parallel for, I wonder what the nowait clause would do here... <not much.
        for(size_t p = 0; p < n; p++) {
            double x = particles[p].x;
            double y = particles[p].y;

            int x_index = (int)floor(x/block_width);
            int y_index = (int)floor(y/block_width);
#pragma omp atomic //this avoids a race condition in updating number_in_block(?), this is effectively a critical section that acts only on one statement
            number_in_block[x_index + y_index*blocksize]++;
//not sure why the following statement isn't also atomic/critical
            blocks[x_index + y_index*blocksize][number_in_block[x_index + y_index*blocksize]-1] = particles+p;
        }
        } // note: this is the end of the parallel section, the following loops are new branches in the program
#pragma omp parallel for  shared(blocks, number_in_block) reduction (+:navg) reduction(+:davg)
//reduction clause causes the private coppies of navg and davg to be added to the global coppies at the end of the loop
//not sure which of the following nested for loops are actually being parallelized. probably only the outermost, and thats fine as long as all the cores stay busy
//might want to see if the second for can also be parallelized to reduce the thread size and even out the ammount of work (i.e. an idle processor won't have to wait as long at the end for other processors to finish if the thread size is smaller)
        for(int j=0; j<blocksize; j++){
//#pragma omp parallel for  shared(blocks, number_in_block) reduction (+:navg) reduction(+:davg) <this caused the grade to drop to 39...
            for(int i=0; i<blocksize; i++){
                for(int p=0; p<number_in_block[i + j*blocksize]; p++ ){
                    blocks[i + j*blocksize][p]->ax = blocks[i + j*blocksize][p]->ay = 0;
                    //interact with blocks and neighbors
                    for(int xoffset=-1; xoffset<2; xoffset++){
                        int xblockindex = i+xoffset;
                        // dont go out of bounds
                        if(xblockindex<0 || xblockindex >= blocksize ){
                            continue;
                        }
                        for(int yoffset=-1; yoffset<2; yoffset++){
                            int yblockindex = j+yoffset;
                            // dont go out of bounds
                            if(yblockindex<0 || yblockindex >= blocksize ){
                                continue;
                            }
                            for(int num=0; num<number_in_block[xblockindex + yblockindex*blocksize]; num++ ){
                                // apply the force
                                apply_force(*(blocks[i + j*blocksize][p]), // this particle
                                            *(blocks[xblockindex + yblockindex*blocksize][num]), // its neighbor
                                            &dmin, &davg, &navg);
                            }
                        }
                    }
                }
            }
        }

#if TIMERS==1
        force_time += read_timer() - running_time;
        running_time = read_timer();
#endif
//
//  move particles
//
#pragma omp parallel for
        for(size_t i = 0; i < n; i++) {
            move(particles[i]);
        }
#if TIMERS==1
        move_time += read_timer() - running_time;
#endif

        if(find_option(argc, argv, "-no") == -1) {
            //
            // Computing statistical data
            //
            if(navg) {
                absavg +=  davg / navg;
                nabsavg++;
            }
            if(dmin < absmin) {
                absmin = dmin;
            }

            //
            //  save if necessary
            //
            if(fsave && (step % SAVEFREQ) == 0) {
                save(fsave, n, particles);
            }
        }
    }
    simulation_time = read_timer() - simulation_time;
//#pragma omp parallel //not sure why this section is parallel since the only statemend in it is only exicuted by the master thread
  //  {
//#pragma omp master
        printf( "n = %d,threads = %d, simulation time = %g seconds", n,omp_get_num_threads(), simulation_time);
  //  }

    if(find_option(argc, argv, "-no") == -1) {
        if(nabsavg) {
            absavg /= nabsavg;
        }
        //
        //  -the minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf(", absmin = %lf, absavg = %lf", absmin, absavg);
        if(absmin < 0.4) {
            printf("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        }
        if(absavg < 0.8) {
            printf("\nThe average distance is below 0.8 meaning that most particles are not interacting");
        }
    }
    printf("\n");
#if TIMERS==1
    printf(" force time = %g seconds", force_time);
    printf(" move time = %g seconds\n", move_time);
#endif

//
// Printing summary data
//
//#pragma omp parallel //again, why parallel? <still don't know, but removing it (and the other one) caused the grade to jump to 79! I suspect the majority of the speedup came from not doing the following if in parallel
//    {
        if(fsum) {
//#pragma omp master
            fprintf(fsum, "%d %d %g\n", n, omp_get_num_threads(), simulation_time);
        }
//    }

//
// Clearing space
//
    if(fsum) {
        fclose(fsum);
    }
    free(particles);
    if(fsave) {
        fclose(fsave);
    }

    return 0;
}
