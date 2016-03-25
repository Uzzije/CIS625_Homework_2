

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "common.h"
#define MAXPARTILCESPERBOX 8


int belongs_on(int num_procs, double ymin[], double ymax[], particle_t &p){
    for(int r=0; r<num_procs; r++){
        if(p.y <= ymax[r] && p.y >= ymin[r]){
            return r;
        }
    }
    //should never happen, should be in one of them!!
    return -1;
}

void recompute_offsets(int num_procs, int sizes[], int *offsets){
    offsets[0] = 0;
    for (int r=0; r<num_procs; r++){
        offsets[r+1] = offsets[r]+sizes[r];
    }
    return;
}

//
//  benchmarking program
//
int main(int argc, char **argv)
{
    int navg, nabsavg = 0;
    double dmin, absmin = 1.0, davg, absavg = 0.0;
    double rdavg, rdmin;
    int rnavg;

    //
    //  process command line parameters
    //
    if(find_option(argc, argv, "-h") >= 0) {
        printf("Options:\n");
        printf("-h to see this help\n");
        printf("-n <int> to set the number of particles\n");
        printf("-o <filename> to specify the output file name\n");
        printf("-s <filename> to specify a summary file name\n");
        printf("-no turns off all correctness checks and particle output\n");
        return 0;
    }

    int n = read_int(argc, argv, "-n", 1000);
    char *savename = read_string(argc, argv, "-o", NULL);
    char *sumname = read_string(argc, argv, "-s", NULL);

    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen(savename, "w") : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen(sumname, "a") : NULL;
    // FILE *fsave = savename ? fopen(savename, "a") : NULL;
    int saveoutput = 0;
    if(rank==0 && fsave){
        saveoutput = 1;
    }
    MPI_Bcast(&saveoutput, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    
    particle_t *particles = (particle_t*) malloc(n * sizeof(particle_t));
    double size =  sqrt(0.0005 * n);
    double interaction_length = 0.01;

    int blocksize = (int) ceil(size/interaction_length);

    double block_width = size/blocksize;

    particle_t*** blocks = (particle_t***) malloc(blocksize*blocksize * sizeof(particle_t**));
    for(int b=0; b<blocksize*blocksize; b++){
        blocks[b] = (particle_t**)malloc(MAXPARTILCESPERBOX*sizeof(particle_t*));
    }
    int number_in_block[blocksize*blocksize];
    for(int b=0; b<blocksize*blocksize; b++){
        number_in_block[b] = 0; // starts with no particles in any box;
    }

    particle_t **particles_to_send = (particle_t**) malloc(n_proc * sizeof(particle_t*));
    for(int r = 0; r < n_proc; r++){
        particles_to_send[r] = (particle_t*) malloc(n * sizeof(particle_t));
    }

    particle_t **particles_to_recv = (particle_t**) malloc(n_proc * sizeof(particle_t*));
    for(int r = 0; r < n_proc; r++){
        particles_to_recv[r] = (particle_t*) malloc(n * sizeof(particle_t));
    }

    particle_t **particles_to_ghost_send = (particle_t**) malloc(n_proc * sizeof(particle_t*));
    for(int r = 0; r < n_proc; r++){
        particles_to_ghost_send[r] = (particle_t*) malloc(n * sizeof(particle_t));
    }

    particle_t **particles_to_ghost_recv = (particle_t**) malloc(n_proc * sizeof(particle_t*));
    for(int r = 0; r < n_proc; r++){
        particles_to_ghost_recv[r] = (particle_t*) malloc(n * sizeof(particle_t));
    }
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous(6, MPI_DOUBLE, &PARTICLE);
    MPI_Type_commit(&PARTICLE);

    MPI_Datatype BLOCK;
    MPI_Type_contiguous(6*MAXPARTILCESPERBOX, MPI_DOUBLE, &BLOCK);
    MPI_Type_commit(&BLOCK);

    //
    //  set up the data partitioning across processors
    //
    int blocks_per_proc = (blocksize + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc((n_proc + 1) * sizeof(int));
    for(int i = 0; i < n_proc + 1; i++) {
        partition_offsets[i] = min(i * blocks_per_proc, blocksize);
    }

    int *partition_sizes = (int*) malloc(n_proc * sizeof(int));
    for(int i = 0; i < n_proc; i++) {
        partition_sizes[i] = partition_offsets[i + 1] - partition_offsets[i];
    }

    double ymax[n_proc];
    double ymin[n_proc];
    // ymin[0] = 0;
    for(int i = 0; i < n_proc; i++) {
        ymin[i] = i*(size/n_proc);
        ymax[i] = (i+1)*(size/n_proc);
    }

    
    //allocate storage for local partition
    
    // int num_local_blocks = partition_sizes[rank];

    // particle_t *local_blocks = (particle_t*) malloc(num_local_blocks*MAXPARTILCESPERBOX * sizeof(particle_t));
    
    
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //

    set_size(n);
    if(rank == 0) {
        init_particles(n, particles);


        for(int p = 0; p < n; p++) {
            double x = particles[p].x;
            double y = particles[p].y;

            int x_index = (int)floor(x/block_width);
            int y_index = (int)floor(y/block_width);
            int particle_index = number_in_block[x_index + y_index*blocksize]++;
            blocks[x_index + y_index*blocksize][particle_index] = particles+p;
        }

    }
    particle_t *contig_particles = (particle_t*) malloc(n * sizeof(particle_t));

    int particles_on_proc[n_proc];
    int particles_on_proc_offsets[n_proc+1];
    particles_on_proc_offsets[0] = 0;
    if(rank == 0) {
        int contig_index = 0;
        for (int r=0; r<n_proc; r++){
            particles_on_proc[r] = 0;
            for(int j=partition_offsets[r]; j<partition_offsets[r+1]; j++){
                for(int i=0; i<blocksize; i++){
                    for(int p=0; p<number_in_block[i + j*blocksize]; p++ ){
                        // for(int p=0; p<number_in_block[i + j*blocksize]; p++ ){
                        contig_particles[contig_index] = *(blocks[i + j*blocksize][p]);
                        contig_index++;
                        particles_on_proc[r]++;
                    }
                }
                
            }
            particles_on_proc_offsets[r+1] = particles_on_proc_offsets[r]+particles_on_proc[r];
        }
    }
    MPI_Bcast(particles_on_proc, n_proc, MPI_INT, 0, MPI_COMM_WORLD);

    int myparticle_count = particles_on_proc[rank];
    particle_t *myparticles = (particle_t*) malloc((n)* sizeof(particle_t) );
    
    MPI_Scatterv(contig_particles, particles_on_proc, particles_on_proc_offsets, PARTICLE, myparticles, myparticle_count, PARTICLE, 0, MPI_COMM_WORLD);


    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer();
    for(int step = 0; step < NSTEPS; step++) {
        // printf("step: %d\n", step);
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        /*****************************************************************************************************
         *
         * Begin particle communication
         *
         *****************************************************************************************************/ 
        int number_to_send_to[n_proc];
        int number_to_recv_from[n_proc];
        for(int r=0; r<n_proc; r++){
            number_to_send_to[r] = 0;
            number_to_recv_from[r] = 0;
        }
        

        for(int p = myparticle_count-1; p>=0  ; p--) {
            int this_belongs_on = belongs_on(n_proc, ymin, ymax, myparticles[p]);
            if(this_belongs_on != rank){
                particles_to_send[this_belongs_on][number_to_send_to[this_belongs_on]] = myparticles[p];
                number_to_send_to[this_belongs_on]++;
                // copy the end of the array to this position, should
                // work even if this one is the end;
                myparticle_count--;
                myparticles[p] = myparticles[myparticle_count];
            }
        }
        for(int r=0; r<n_proc; r++){
            if(r != rank){
                MPI_Send(number_to_send_to+r, 1, MPI_INT, r, 1, MPI_COMM_WORLD);
            }
            else{
                for(int recv=0; recv<n_proc; recv++){
                    if(recv == rank){
                        continue;
                    }
                    MPI_Recv(number_to_recv_from+recv, 1, MPI_INT, recv, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

        }
    
        for(int r=0; r<n_proc; r++){
            if(r != rank){
                if(number_to_send_to[r] > 0){
                    MPI_Send(particles_to_send[r], number_to_send_to[r], PARTICLE, r, 2, MPI_COMM_WORLD);
                }
            }
            else{
                for(int recv=0; recv<n_proc; recv++){
                    if(recv == rank){
                        continue;
                    }
                    if(number_to_recv_from[recv] > 0){
                        MPI_Recv(particles_to_recv[recv], number_to_recv_from[recv], PARTICLE, recv, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
        }


        for(int r=0; r<n_proc; r++){
            for(int i=0; i<number_to_recv_from[r]; i++){
                myparticles[myparticle_count] = particles_to_recv[r][i];
                myparticle_count++;
                
            }
        }
        /*****************************************************************************************************
         *
         * End particle communication
         *
         *****************************************************************************************************/

        /*****************************************************************************************************
         *
         * Start Ghost communication
         *
         *****************************************************************************************************/
        int number_to_send_ghost_to[n_proc];
        int number_to_recv_ghost_from[n_proc];
        for(int r=0; r<n_proc; r++){
            number_to_send_ghost_to[r] = 0;
            number_to_recv_ghost_from[r] = 0;
        }
        int prev = rank-1;
        int next = rank+1;

        
        for(int p = 0; p< myparticle_count  ; p++) {
            double y = myparticles[p].y;
            if(prev > 0 && y >= (ymin[rank] - interaction_length) ){
                particles_to_ghost_send[prev][number_to_send_ghost_to[prev]] = myparticles[p];
                number_to_send_ghost_to[prev]++;
            }
            if(next < n_proc && y <= (ymax[rank] + interaction_length) ){
                particles_to_ghost_send[next][number_to_send_ghost_to[next]] = myparticles[p];
                number_to_send_ghost_to[next]++;
            }
        }
        for(int r=0; r<n_proc; r++){
            if(r != rank){
                MPI_Send(number_to_send_ghost_to+r, 1, MPI_INT, r, 3, MPI_COMM_WORLD);
            }
            else{
                for(int recv=0; recv<n_proc; recv++){
                    if(recv == rank){
                        continue;
                    }
                    MPI_Recv(number_to_recv_ghost_from+recv, 1, MPI_INT, recv, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

        }

        for(int r=0; r<n_proc; r++){
            if(r != rank){
                if(number_to_send_ghost_to[r] > 0){
                    MPI_Send(particles_to_ghost_send[r], number_to_send_ghost_to[r], PARTICLE, r, 4, MPI_COMM_WORLD);
                }
            }
            else{
                for(int recv=0; recv<n_proc; recv++){
                    if(recv == rank){
                        continue;
                    }
                    if(number_to_recv_ghost_from[recv] > 0){
                        MPI_Recv(particles_to_ghost_recv[recv], number_to_recv_ghost_from[recv], PARTICLE, recv, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                }
            }
        }

        
        /*****************************************************************************************************
         *
         * End Ghost communication
         *
         *****************************************************************************************************/

        for(int b=0; b<blocksize*blocksize; b++){
            number_in_block[b] = 0; // starts with no particles in any box;
        }
        for(int p = 0; p < myparticle_count; p++) {
            double x = myparticles[p].x;
            double y = myparticles[p].y;
            int x_index = (int)floor(x/block_width);
            int y_index = (int)floor(y/block_width);
            int particle_index = number_in_block[x_index + y_index*blocksize]++;
            blocks[x_index + y_index*blocksize][particle_index] = myparticles+p;
        }
        for(int r=0; r<n_proc; r++){
            for(int i=0; i<number_to_recv_ghost_from[r]; i++){
                double x =  particles_to_ghost_recv[r][i].x;
                double y =  particles_to_ghost_recv[r][i].y;
                int x_index = (int)floor(x/block_width);
                int y_index = (int)floor(y/block_width);
                int particle_index = number_in_block[x_index + y_index*blocksize]++;
                blocks[x_index + y_index*blocksize][particle_index] = &(particles_to_ghost_recv[r][i]);
            }
        }
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if(find_option(argc, argv, "-no") == -1){
            // if(fsave && (step % SAVEFREQ) == 0) {
            if( saveoutput && ( step % SAVEFREQ) == 0) {
                MPI_Gather(&myparticle_count, 1, MPI_INT, 
                           particles_on_proc, 1, MPI_INT, 
                           0, MPI_COMM_WORLD);
                recompute_offsets(n_proc, particles_on_proc, particles_on_proc_offsets);
                MPI_Gatherv(myparticles, myparticle_count, PARTICLE, particles, particles_on_proc, particles_on_proc_offsets, PARTICLE, 0, MPI_COMM_WORLD);
                if(rank == 0){
                    save(fsave, n, particles);
                }
                
            }
        }
        //
        //  compute all forces
        //
        for(int j=0; j<blocksize; j++){
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

        if(find_option(argc, argv, "-no") == -1) {

            MPI_Reduce(&davg, &rdavg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&navg, &rnavg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dmin, &rdmin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);


            if(rank == 0) {
                //
                // Computing statistical data
                //
                if(rnavg) {
                    absavg +=  rdavg / rnavg;
                    nabsavg++;
                }
                if(rdmin < absmin) {
                    absmin = rdmin;
                }
            }
        }

        //
        //  move particles
        //x
        
        for(int i = 0; i < myparticle_count; i++) {
            move(myparticles[i]);
        }
        
            
        
    }
    simulation_time = read_timer() - simulation_time;

    if(rank == 0) {
        printf("n = %d, simulation time = %g seconds", n, simulation_time);

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

        //
        // Printing summary data
        //
        if(fsum) {
            fprintf(fsum, "%d %d %g\n", n, n_proc, simulation_time);
        }
    }

    //
    //  release resources
    //
    if(fsum) {
        fclose(fsum);
    }
    // free(partition_offsets);
    // free(partition_sizes);
    // free(local);
    free(particles);
    if(fsave) {
        fclose(fsave);
    }

    MPI_Finalize();

    return 0;
}
