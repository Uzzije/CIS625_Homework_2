#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "common.h"

using std::vector;
using std::cout;
using std::endl;
using std::cin;

extern double size;

double cutoff = 0.01;
int offset[9];
int num_bins;

vector<vector<particle_t> > grid;

int index(double i, double j)
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
    double j = p.x / (2 * cutoff) + 1;
    double i = p.y / (2 * cutoff) + 1;
    return index(i,j);
}

void resize_grid()
{
    num_bins = (int)ceil(size/(2*cutoff)) + 2;
    grid.resize(num_bins*num_bins);
	printf("bin_size = %d\n", num_bins);
	printf("grid_size = %zu\n", grid.size());
}

void clear_grid()
{
    for(int i = 0; i < num_bins*num_bins; i++) {
        grid[i].clear();
    }
}

void bin_particles(int n, particle_t* particles)
{
    for(int i = 0; i < n; i++) {
        int j = get_bin_index(particles[i]);
        grid[j].push_back(particles[i]);


		printf("index = %d\n", j);

    }
	int z;
	cin >> z;
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
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;    

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    // init_particles(n, particles);
	particles[0].x = 0.0;
	particles[0].y = 0.0;
	particles[1].x = 0.05;
	particles[1].y = 0.05;
	particles[2].x = 0.05;
	particles[2].y = 0.0;
	particles[3].x = 0.0;
	particles[3].y = 0.05;
	particles[4].x = 0.035;
	particles[4].y = 0.035;
	for ( int i = 0; i < n; i++ )
	{
		printf("%d.\t( %f, %f )\n", i, particles[i].x, particles[i].y);
	}
    resize_grid();
    compute_offset();
    bin_particles(n, particles);
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
	
    for( int step = 0; step < NSTEPS; step++ )
    {
	    navg = 0;
        davg = 0.0;
	    dmin = 1.0;

        for(int i = 1; i < num_bins - 1; i++) {
            for(int j = 1; j < num_bins - 1; j++) {
                int bin = index(i, j);
                vector<particle_t> neighbors = get_neighbors(bin);
                for(int k = 0; k < grid[bin].size(); k++) {
                    grid[bin][k].ax = grid[bin][k].ay = 0;
                    for(int l = 0; l < neighbors.size(); l++) {
                        apply_force(grid[bin][k], neighbors[l], &dmin, &davg, &navg);
                    }
                }
            }
        }

        int count = 0;
        for(int i = 1; i < num_bins - 1; i++) {
            for(int j = 1; j < num_bins - 1; j++) {
                int bin = index(i,j);
                for(int k = 0; k < grid[bin].size(); k++) {
                    particles[count++] = grid[bin][k];
                }
            }
        }  

        for( int i = 0; i < n; i++ ) {
            move(particles[i]);
        }
        clear_grid();
        bin_particles(n, particles);	

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
		
          //
          //  save if necessary
          //
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, particles );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  -the minimum distance absmin between 2 particles during the run of the simulation
    //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
    //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
    //
    //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");     

    //
    // Printing summary data
    //
    if( fsum) 
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );    
    free( particles );
    if( fsave )
        fclose( fsave );

    return 0;
}
