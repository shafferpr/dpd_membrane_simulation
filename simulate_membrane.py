r"""
This script can be used from the command line to perform a dpd simulation of a membrane structure

Command Line Arguments
----------------------

This script has several command line arguments which you can view by running:

.. code-block:: bash

    python simulate_membrane.py --help


The output should look like this:

.. code-block:: bash

    usage: simulate_membrane.py [-h] [--inputfile MEMBRANE_INPUT_FILE]
                                [--output OUTPUT_PREFIX] [--box_size BOX_SIZE]
                                [--composite_solute] [--solute_mesh SOLUTE_MESH]
                                [--solute_rigid_coords SOLUTE_RIGID_COORDS]
                                [--solvent_radius SOLVENT_RADIUS]
                                [--solvent_density SOLVENT_DENSITY]
                                [--solvent_force SOLVENT_FORCE]
                                [--solute_solvent_interaction SOLUTE_SOLVENT_INTERACTION]
                                [--solute_wall_interaction SOLUTE_WALL_INTERACTION]
                                [--dump_every DUMP_EVERY] [--steps STEPS]
                                [--membrane_lower_boundary MEMBRANE_LOWER_BOUNDARY]
                                [--solute_z_position SOLUTE_Z_POSITION]
                                [--sqrt_n_solutes SQRT_N_SOLUTES]

    generate a membrane structure from sdf file, and perform a DPD simulation

    optional arguments:
      -h, --help            show this help message and exit
      --inputfile MEMBRANE_INPUT_FILE
                            sdf membrane structure inputfile
      --output OUTPUT_PREFIX
                            directory for all of the output files (including
                            membrane geometry, particle trajectories, and stats
                            files and figures)
      --box_size BOX_SIZE   total boxsize for cubic box, default=50
      --composite_solute    include composite solute particles, this does not
                            require an argument, just include --composite_solute
                            to use this
      --solute_mesh SOLUTE_MESH
                            the .off mesh file containing the solute shape
      --solute_rigid_coords SOLUTE_RIGID_COORDS
                            the .txt file containing the coordinates of particles
                            in a single solute molecule
      --solvent_radius SOLVENT_RADIUS
                            size of the solvent particles, default=1.0
      --solvent_density SOLVENT_DENSITY
                            density of the solvent, default=5
      --solvent_force SOLVENT_FORCE
                            force acting on the solvent particles pushing them
                            through the membrane, default=-0.1
      --solute_solvent_interaction SOLUTE_SOLVENT_INTERACTION
                            strength of solute solvent interaction, >35 means
                            hydrophobic, default=35 (neutral)
      --solute_wall_interaction SOLUTE_WALL_INTERACTION
                            strength of solute membrane interaction, <35 means
                            attractive, default=35
      --dump_every DUMP_EVERY
                            number of steps between writing snapshots of the
                            particles, default is 1500
      --steps STEPS         number of simulation steps, default is 100000
      --membrane_lower_boundary MEMBRANE_LOWER_BOUNDARY
                            the lower boundary of the membrane, this is used to
                            compute flow rate, default=15
      --solute_z_position SOLUTE_Z_POSITION
                            the initial z position of the solutes, default=45
      --sqrt_n_solutes SQRT_N_SOLUTES
                            the square root of the number of solute particles,
                            default=4

"""

import argparse
import os, sys
import json
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../webapp'))
from porousMediaSimulation.dpdsimulation import DPDSimulation, Statistics
from application.models import *
from application import db


def save_params(args,output):
    with open("%s/sim_params.json"%output, "w") as f:
        json.dump(vars(args),f,indent=3)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='generate a membrane structure from sdf file, and perform a DPD simulation')
    parser.add_argument('--label',dest='label',default='membrane_simulation',help='the label you want to give this simulation')
    parser.add_argument('--membrane_input_directory',dest='membrane_input_directory',default='four_pore_membrane',help="sdf membrane structure inputfile, default=four_pore_membrane")
    #parser.add_argument('--output',dest='output_prefix',default='sim_output',help="directory for all of the output files (including membrane geometry, \
        #particle trajectories, and stats files and figures)")
    parser.add_argument('--box_size', type=int, dest='box_size',default=42,help="total boxsize for cubic box, default=42")
    parser.add_argument('--z_box_size', type=int, dest='z_box_size',default=80,help="z boxsize for cubic box, default=80")
    parser.add_argument('--composite_solute',dest='composite_solute',action='store_true',default=True,help="include composite solute particles, \
        this does not require an argument, just include --composite_solute to use this")
    parser.add_argument('--solute_mesh',dest='solute_mesh',default="sphere_small.off", help="the .off mesh file containing the solute shape")
    parser.add_argument('--solute_rigid_coords',dest='solute_rigid_coords',default="rigid_coords_sphere_small.txt",help="the .txt file containing the coordinates of particles in a single solute molecule")
    parser.add_argument('--solvent_radius',type=float,dest='solvent_radius',default=1.0,help="size of the solvent particles, default=1.0")
    #parser.add_argument('--solute_radius',type=float,dest='solute_radius', default=0.0, help="size of the solute particles, default=0.0, which corresponds to no solute")
    parser.add_argument('--solvent_density',type=float, dest='solvent_density', default=5, help="density of the solvent, default=5")
    #parser.add_argument('--solute_density',type=float, dest='solute_density', default=0.1, help="density of the solute, default=0.1")
    parser.add_argument('--solvent_force', type=float, dest='solvent_force', default=-0.04, help="force acting on the solvent particles pushing them through the membrane, default=-0.04")
    #parser.add_argument('--solute_force', type=float, dest='solute_force', default=0.1, help="force acting on the solute particles pushing the through the membrane, default=0.1")
    parser.add_argument('--solute_solvent_interaction',type=float,dest='solute_solvent_interaction', default=46, help="strength of solute solvent interaction, >35 means hydrophobic, default=46 (hydrophobic)")
    parser.add_argument('--solute_wall_interaction',type=float,dest='solute_wall_interaction', default=28, help="strength of solute membrane interaction, <35 means attractive, default=28 (solute attracted to wall)")
    parser.add_argument('--dump_every',type=int,dest='dump_every', default=10000, help="number of steps between writing snapshots of the particles, default is 10000")
    parser.add_argument('--steps',type=int, dest='steps', default=1000000, help="number of simulation steps, default is 1000000")
    parser.add_argument('--membrane_lower_boundary', type=float, dest='membrane_lower_boundary', default=5, help="the lower boundary of the membrane, this is used to compute flow rate, default=5")
    parser.add_argument('--solute_z_position', type=float, dest='solute_z_position', default=34, help="the initial z position of the solutes, default=45")
    parser.add_argument('--solute_layers', type=int, dest='solute_layers',default=12, help="the number of layers of solute particles, default=1")
    parser.add_argument('--sqrt_n_solutes', type=int, dest='sqrt_n_solutes', default=9, help="the square root of the number of solute particles, default=4")
    parser.add_argument('--no_db_update',dest='no_db_update',action='store_true',default=True,help="don't record simulation in  the database")
    args=parser.parse_args()
    #os.system("mkdir %s"%args.output_prefix)
    #save_params(args)

    #check that box size from input and box size from sdf file are the same

    #with open(args.membrane_input_file.split('.')[0]) as f:
        #box_size_from_file = int(f.readline().split(' ')[0])
    #if box_size_from_file != args.box_size:
        #raise RuntimeError("box size provided does not match box size in input file, input file says box size is %d"%box_size_from_file)

    #comm=MPI.COMM_WORLD
    #rank=comm.Get_rank()

    #sim_db_entry=Simulations(label=args.label,solute_mesh=args.solute_mesh,solute_rigid_coords=args.solute_rigid_coords,solvent_force=args.solvent_force, \
                             #solute_solvent_interaction=args.solute_solvent_interaction,solute_wall_interaction=args.solute_wall_interaction,n_solutes=args.sqrt_n_solutes**2, \
                             #steps=args.steps,status='started')
    #db.session.add(sim_db_entry)
    #db.session.commit()
    #sim_index=sim_db_entry.id
    
    

    
        #simulation=DPDSimulation(membrane_input_file=args.membrane_input_file,output_prefix=args.output_prefix,\
            #box_size=args.box_size,composite_solute=args.composite_solute)
    membrane_input_file="%s/qq.dat"%args.membrane_input_directory
    print(membrane_input_file)
    simulation=DPDSimulation(membrane_input_file=membrane_input_file,label=args.label,solute_mesh=args.solute_mesh, solute_rigid_coords=args.solute_rigid_coords, solvent_force=args.solvent_force, solute_solvent_interaction=args.solute_solvent_interaction, solute_wall_interaction=args.solute_wall_interaction, n_solutes=args.sqrt_n_solutes**2,\
                             box_size=args.box_size,z_box_size=args.z_box_size,nsteps=args.steps,composite_solute=args.composite_solute,db_flag=args.no_db_update)
    print("initialized")
    save_params(args,simulation.output_prefix)

    simulation.initializeSolvent(density=args.solvent_density,radius=args.solvent_radius,force=args.solvent_force)
    if args.composite_solute:
        simulation.initializeRigidSolute(a=args.solute_solvent_interaction, solute_solute_a=0.9*args.solute_solvent_interaction, z_position=args.solute_z_position, \
                                            num_particles_sqroot=args.sqrt_n_solutes, mesh_file=args.solute_mesh, coord_file=args.solute_rigid_coords,layers=args.solute_layers)
    simulation.initializeWall(solute_interaction_strength=args.solute_wall_interaction)
    simulation.createOutput(dump_every=args.dump_every)
    #initialize the simulations statistics object
    simulation.statistics()


    simulation.runSimulation(steps=args.steps,stats_every=1000)

