#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include "common.h"

using std::vector;
using std::cout;
using std::cin;
using std::endl;

//
//  benchmarking program
//

int	   bin_num;
double grid_length, bin_sz;

//
//  tuned constants
//
#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005

void init_bins( int n, particle_t *particles, vector<bin_type> &bins )
{
	grid_length = sqrt( density * n );
	bin_sz = 0.025;
	bin_num = ceil( grid_length/bin_sz );
	
	bins.resize( bin_num*bin_num );
	
	cout << "Grid Length: " << grid_length << endl;
    cout << "Number of Bins: " << bin_num << endl;
	cout << endl;
	
	for (int k = 0; k < n; k++ )
	{
		int i = particles[ k ].x/bin_sz;
		int j = particles[ k ].y/bin_sz;
				
		bins[ (i*bin_num + j) ].push_back( particles[k] );
		
		cout << k << ".\t( " << particles[ k ].x << ", " << particles[ k ].y << " )";
		cout << "\tIndex = " << ( i*bin_num + j ) << endl;
	}
	int z;
	//cin >> z;
}
	
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
	vector<bin_type> bins;
	
    set_size( n );
    init_particles( n, particles );
	init_bins( n, particles, bins );
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
	bool flag = true;
	
    for( int step = 0; step < NSTEPS; step++ )
    {
	    navg = 0;
        davg = 0.0;
	    dmin = 1.0;
        //
        //  compute forces
        //
        /*
        for( int i = 0; i < n; i++ )
        {
            particles[i].ax = particles[i].ay = 0;
            for (int j = 0; j < n; j++)
				apply_force(particles[i], particles[j],&dmin,&davg,&navg);
        }
         */
        // This checks if two particles a about to collide, and bounces them off each other.
        for( int i = 0; i < bin_num; i++ ) // starting at the left bottom of the bin; go through each column
        {
            for (int j = 0; j < bin_num; j++) // for each column go up each row of that column
            {
                bin_type& searching_bin = bins[i*bin_num + j]; // get bin memory address at each row
                for (int each_particle = 0; each_particle < searching_bin.size(); each_particle++) // for each particle object in that bin memory address
                {
                    searching_bin[each_particle].ax = searching_bin[each_particle].ay = 0;
                }
                    // search nearby neigbour's bins (8) plus self 9 total search.
                    // first loop is a 3 iteration by the next lines three iteration (3x3). i.e resulting in the 9 total iteration required for the neigbour search
                    for (int pos_x = -1; pos_x <= 1; pos_x++)
                    {
                        for(int pos_y = -1; pos_y <= 1; pos_y++)
                        {
                            // checks the index of the bin that is searching its nearest neighbour.
                              // If that bin is on an edge move on to next bin address.
                            if((i + pos_x >= 0) && ((i + pos_x) < bin_num) && ((j + pos_y) >= 0) && ((j + pos_y) < bin_num))
                            {
                                bin_type& neighbours_bins = bins[(i+pos_x)*bin_num + j + pos_y]; // gets bin address of neigbouring been to search
                                //
                                for(int each_bin_particle = 0; each_bin_particle < searching_bin.size(); each_bin_particle++)
                                {
                                    for (int neighbours_bin = 0; neighbours_bin < neighbours_bins.size(); neighbours_bin++)
                                    {
                                        apply_force(searching_bin[each_bin_particle], neighbours_bins[neighbours_bin], &dmin, &davg, &navg);
                                    }
                                }

                            }
                        }

                    }
            }
        }


        //
        //  move particles, this would have to change when dealing with
        //

        for( int i = 0; i < n; i++ )
            move( particles[i] );



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
