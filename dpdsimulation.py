r"""
Contains the DPDSimulation class and the Statistics class

"""


import mirheo as mir
from mpi4py import MPI
import ctypes
import matplotlib.pyplot as plt
import trimesh
import numpy as np
import os,sys
sys.path.insert(0, os.path.abspath('../webapp'))
from application.models import *
from application import db
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)

class Statistics(object):
    """Initializes the Statistics object

    Attributes
    ----------
    output: str
       Path to the output directory for the simulation
    flowrate_stats: Boolean
       A boolean flag that causes the flowrate to be calculated, default=True
    membrane_lower_boundary: float
       the lower boundary of the membrane, used to compute flowrate
    boxsize: float
       the size of the (cubic) simulation box, default=50
    """
    def __init__(self,output,flowrate_stats=True,membrane_lower_boundary=15,boxsize=50,\
            solvent_density_profile=True,solvent_velocity_profile=True,solute_density_profile=True,solute_velocity_profile=True):
        self.output=output
        self.flowrate_stats=flowrate_stats
        self.membrane_lower_boundary=membrane_lower_boundary
        self.boxsize=boxsize
        self.solvent_velocity_profile=solvent_velocity_profile
        self.solvent_density_profile=solvent_density_profile
        self.solute_density_profile=solute_density_profile
        self.solute_velocity_profile=solute_velocity_profile
        self.all_solute_positions=np.asarray([])
        self.all_solvent_positions=np.asarray([])
        self.all_solute_velocities=np.asarray([])
        self.all_solvent_velocities=np.asarray([])
        self.times=[]
        self.flowrates=[]
        self.flowrates_average=[]
        os.system("rm %s/stats/*"%self.output)
        os.system("mkdir %s/stats"%self.output)

    def compute_flowrate(self,particle_zpositions_old,particle_zpositions_new,simulation_step):
        flowrate=0
        i=0
        for elem in zip(particle_zpositions_old,particle_zpositions_new):
            i+=1
            #print(elem)
            if elem[0]>self.membrane_lower_boundary and elem[1]<self.membrane_lower_boundary:
                #print(i,elem[0],elem[1])
                flowrate += 1.0
            if elem[0]<self.membrane_lower_boundary and elem[1]>self.membrane_lower_boundary:
                flowrate -= 1.0
                #print(elem[0],elem[1])
        #flowrate /= float(stride)
        self.flowrates.append(flowrate)
        fr_mean=np.mean(self.flowrates[-60:])
        self.flowrates_average.append(fr_mean)
        with open("%s/stats/flowrate.dat"%self.output,"a") as f:
            f.write("%d %f %f\n"%(simulation_step,flowrate, fr_mean))

    def compute_solvent_density_profile(self, particle_zpositions_new, simulation_step):
        self.all_solvent_positions=np.concatenate((self.all_solvent_positions,particle_zpositions_new))

    def compute_solute_density_profile(self,solute_zpositions, simulation_step):
        self.all_solute_positions=np.concatenate((self.all_solute_positions,solute_zpositions))

    def plot_flowrate(self):
        plt.xlabel('simulation step')
        plt.ylabel('flowrate')
        plt.title("flowrate of solvent vs time")
        plt.plot(self.times,self.flowrates)
        plt.savefig("%s/stats/flowrate.png"%self.output)
        plt.close()
        plt.xlabel('simulation step')
        plt.ylabel('flowrate')
        plt.title("mean flowrate of solvent vs time")
        plt.plot(self.times,self.flowrates_average)
        plt.savefig("%s/stats/flowrate_mean.png"%self.output)
        plt.close()

    def plot_solute_density_profiles(self):
        plt.xlabel("z_position")
        plt.ylabel("density")
        plt.title("histogram of solute density (z-axis)")
        hist=np.histogram(self.all_solute_positions,bins=int(self.boxsize))
        plt.hist(self.all_solute_positions,bins='auto')
        plt.savefig("%s/stats/solute_density.png"%self.output)
        np.save("%s/stats/solute_density"%self.output,hist)
        plt.close()

    def plot_solvent_density_profiles(self):
        plt.xlabel("z_position")
        plt.ylabel("density")
        plt.title("histogram of solvent density (z-axis)")
        hist=np.histogram(self.all_solvent_positions,bins=int(self.boxsize))
        plt.hist(self.all_solvent_positions,bins=int(self.boxsize))
        plt.savefig("%s/stats/solvent_density.png"%self.output)
        plt.close()

    def calculate(self,particle_zpositions_old,particle_zpositions_new,particle_velocities,solute_zpositions,solute_velocities,simulation_step):
        self.times.append(simulation_step)
        if self.flowrate_stats:
            self.compute_flowrate(particle_zpositions_old,particle_zpositions_new,simulation_step)
        if self.solute_density_profile:
            self.compute_solvent_density_profile(particle_zpositions_new, simulation_step)
        if self.solvent_density_profile:
            self.compute_solute_density_profile(solute_zpositions, simulation_step)


    def plot(self):
        if self.flowrate_stats:
            self.plot_flowrate()
        if self.solute_density_profile:
            self.plot_solute_density_profiles()
        if self.solvent_density_profile:
            self.plot_solvent_density_profiles()


class DPDSimulation(object):
    def __init__(self, membrane_input_file, label, solute_mesh='', solute_rigid_coords='', solvent_force=0, solute_solvent_interaction=0, solute_wall_interaction=0, n_solutes=0, box_size=80, z_box_size=80, nsteps=0, composite_solute=True, ranks=1, db_flag=True):
        ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)
        self.membrane_input_file=membrane_input_file
        self.label=label
        self.output_prefix=label
        self.box_size=box_size
        self.z_box_size=z_box_size
        self.dt=0.01
        self.domain=(self.z_box_size,self.box_size,self.box_size)
        self.u=mir.mirheo((ranks,1,1),self.domain,self.dt,debug_level=2)
        self.composite_solute=composite_solute
        self.db_flag=db_flag
        comm=MPI.COMM_WORLD
        rank=comm.Get_rank()
        print(rank,"rank")
        if not self.db_flag:
            if rank==1:
                sim_db_entry=Simulations(label=label,solute_mesh=solute_mesh, solute_rigid_coords=solute_rigid_coords, solvent_force=solvent_force, solute_solvent_interaction=solute_solvent_interaction,solute_wall_interaction=solute_wall_interaction, n_solutes=n_solutes,steps=nsteps,status='started')
                db.session.add(sim_db_entry)
                db.session.commit()
                self.output_prefix="../data/%s"%sim_db_entry.id
                self.sim_index=sim_db_entry.id
                print(sim_db_entry.id)
                print("updating db")
            if comm.Get_size()==1:
                self.output_prefix=label
        os.system("mkdir %s"%self.output_prefix)
            
    def initializeSolvent(self, density, radius, force):
        self.solvent_density=density
        self.solvent_radius=radius
        self.solvent_force=force
        self.solvent_pv  = mir.ParticleVectors.ParticleVector('solvent_pv', mass = 1.0)
        self.ic  = mir.InitialConditions.Uniform(self.solvent_density)
        self.solvent_dpd = mir.Interactions.DPD('solvent_dpd', self.solvent_radius, a=35.0, gamma=4, kbt=0.8, power=0.5)
        self.solvent_vv = mir.Integrators.VelocityVerlet('solvent_vv')
        self.u.registerPlugins(mir.Plugins.createAddForce('solvent_force', self.solvent_pv, force=[self.solvent_force,0.0,0.0]))
        self.u.registerInteraction(self.solvent_dpd)
        self.u.registerParticleVector(self.solvent_pv, self.ic)
        self.u.registerIntegrator(self.solvent_vv)
        self.u.setInteraction(self.solvent_dpd,self.solvent_pv,self.solvent_pv)
        self.u.setIntegrator(self.solvent_vv, self.solvent_pv)

    def initializeSolute(self, density, radius, force):
        self.solute_density=density
        self.solute_radius=radius
        self.solute_force=force
        self.solute_pv  = mir.ParticleVectors.ParticleVector('solute_pv', mass = 1.0)
        self.solute_ic  = mir.InitialConditions.Uniform(self.solute_density)
        self.solute_dpd = mir.Interactions.DPD('solute_dpd', self.solute_radius, a=35.0, gamma=4, kbt=1.0, power=0.5)
        self.solute_vv = mir.Integrators.VelocityVerlet_withConstForce('solute_vv',force=[self.solute_force,0.0,0.0])
        self.u.registerInteraction(self.solute_dpd)
        self.u.registerParticleVector(self.solute_pv, self.solute_ic)
        self.u.registerIntegrator(self.solute_vv)
        self.u.setInteraction(self.solute_dpd,self.solute_pv,self.solute_pv)

    def initializeRigidSolute(self,coord_file="rigid_coords.txt", mesh_file="sphere_mesh2.off", num_particles_sqroot=4, a=20, solute_solute_a=20, z_position=45,layers=1):
        xy=np.arange(5,self.box_size,self.box_size/num_particles_sqroot)
        com_q=[]
        print(xy)
        for x in xy:
            for y in xy:
                for l in range(layers):
                    com_q.append([z_position+4*l, x, y, 1.0, 0.0, 0.0, 0.0])
        #com_q = [[40.0, 6.0, 2.0,   1.0, 0.0, 0.0, 0.0],
                 #[40.0, 17.0, 6.0,   1.0, 0.0, 0.0, 0.0],
                 #[40.0, 20.0, 6.0,   1.0, 0.0, 0.0, 0.0],
                 #[40.0, 25.0, 6.0,   1.0, 0.0, 0.0, 0.0]]
        m = trimesh.load(mesh_file)
        inertia = [row[i] for i, row in enumerate(m.moment_inertia)]
        if inertia[0]<0:
            for i in range(len(inertia)):
                inertia[i] *= -1
        self.rigid_coords = np.loadtxt(coord_file).tolist()
        self.mesh = mir.ParticleVectors.Mesh(m.vertices.tolist(), m.faces.tolist())
        self.pv_rigid = mir.ParticleVectors.RigidObjectVector('spheres', 1.0, inertia, len(self.rigid_coords), self.mesh)
        self.ic_rigid = mir.InitialConditions.Rigid(com_q, self.rigid_coords)
        self.vv_rigid = mir.Integrators.RigidVelocityVerlet("vv_rigid")
        self.u.registerParticleVector(self.pv_rigid, self.ic_rigid)
        #self.u.registerPlugins(mir.Plugins.createAddForce('solute_force', self.pv_rigid, force=[self.solvent_force,0.0,0.0]))
        self.solvent_rigid_dpd = mir.Interactions.DPD('dpd', self.solvent_radius, a=a, gamma=4, kbt=0.8, power=0.5)
        #repulsive LJ to avoid overlap between spheres
        #self.cnt = mir.Interactions.LJ('cnt', self.solvent_radius, epsilon=0.35, sigma=0.8, max_force=400.0)
        self.cnt=mir.Interactions.DPD('dpd2', self.solvent_radius, a=a, gamma=4, kbt=0.8, power=0.5)
        self.u.registerPlugins(mir.Plugins.createAddForce('solute_force', self.pv_rigid, force=[self.solvent_force,0.0,0.0]))
        self.u.registerInteraction(self.cnt)
        self.u.registerInteraction(self.solvent_rigid_dpd)
        self.u.setInteraction(self.cnt, self.pv_rigid, self.pv_rigid)
        self.u.setInteraction(self.solvent_rigid_dpd,self.solvent_pv,self.pv_rigid)
        self.u.registerIntegrator(self.vv_rigid)
        self.u.setIntegrator(self.vv_rigid, self.pv_rigid)

        #ensure that solvent particles are removed from the solute rigid bodies
        #self.belonging_checker = mir.BelongingCheckers.Mesh("mesh checker")
        #self.u.registerObjectBelongingChecker(self.belonging_checker, self.pv_rigid)
        #self.u.applyObjectBelongingChecker(self.belonging_checker, self.solvent_pv, correct_every=100000, inside="none", outside="")

        


    def setSoluteSolventInteraction(self,interaction_strength):
        if self.solute_radius > 0.0:
            self.solute_solvent_dpd=mir.Interactions.DPD('solute_solvent_dpd', (self.solvent_radius+self.solute_radius)/2, a=interaction_strength, gamma=10.0, kbt=1.0, power=0.5)
            self.u.registerInteraction(self.solute_solvent_dpd)
            self.u.setInteraction(self.solute_solvent_dpd,self.solvent_pv,self.solute_pv)

    def initializeWall(self,solute_interaction_strength=10):
        self.wall = mir.Walls.SDF("sdfwall",sdfFilename=self.membrane_input_file)
        self.u.registerWall(self.wall,check_every=0) # register the wall in the coordinator
        # we now create the frozen particles of the walls
        self.solvent_wall_dpd=mir.Interactions.DPD('solvent_wall_dpd', self.solvent_radius, a=40, gamma=4, kbt=0.8, power=0.5)
        self.u.registerInteraction(self.solvent_wall_dpd)
        self.pv_wall = self.u.makeFrozenWallParticles(pvName="wall", walls=[self.wall], interactions=[self.solvent_wall_dpd], integrator=self.solvent_vv, density=5)
        # set the wall for pv
        # this is required for non-penetrability of the solvent thanks to bounce-back
        # this will also remove the initial particles which are not inside the wall geometry
        #self.u.setWall(self.wall, self.solvent_pv)
    
        
        #set the interaction between the solute particle vector and the wall pv, and the interaction between the solvent particle vectors and the wall particle vectors
        self.u.setInteraction(self.solvent_wall_dpd, self.solvent_pv, self.pv_wall)

        if self.composite_solute:
            #self.solute_wall_lj = mir.Interactions.LJ('solute_wall_lj', self.solvent_radius, epsilon=0.35, sigma=0.8, max_force=400.0)
            self.solute_wall_dpd = mir.Interactions.DPD('solute_wall_dpd', self.solvent_radius, a=solute_interaction_strength, gamma=4, kbt=1.0, power=0.5)
            self.u.registerPlugins(mir.Plugins.createWallRepulsion('wall_repulsion', self.pv_rigid, self.wall, C=200, h=0.4, max_force=2000))
            self.u.registerPlugins(mir.Plugins.createWallRepulsion('wall_repulsion2', self.solvent_pv, self.wall, C=200, h=0.4, max_force=2000))
            self.u.registerInteraction(self.solute_wall_dpd)
            self.u.setInteraction(self.solute_wall_dpd, self.pv_rigid, self.pv_wall)

    def drawWall(self):
        self.wall = mir.Walls.SDF("sdfwall",sdfFilename=self.membrane_input_file)
        self.u.registerWall(self.wall)
        self.u.dumpWalls2XDMF([self.wall], h = (0.5, 0.5, 0.5), filename = "%s/wall"%self.output_prefix)

    def initializeIntegrators(self):
        self.u.setIntegrator(self.solvent_vv, self.solvent_pv)
        if self.solute_radius > 0.0 :
            self.u.setIntegrator(self.solute_vv, self.solute_pv)

    def createOutput(self,dump_every):
        #these are the statistics generated automatically by ymero, which are printed to the console and not generally of very much interest
        self.u.registerPlugins(mir.Plugins.createStats('stats', every=5000))
        self.u.registerPlugins(mir.Plugins.createDumpAverage('field', [self.solvent_pv],

                                                             50, 10000, (1.0, 1.0, 1.0),
                                                [("velocity", "vector_from_float4")], '%s/velocity/solvent-'%self.output_prefix))
        self.u.registerPlugins(mir.Plugins.createDumpAverage('field2', [self.pv_rigid],
                                                50, 10000, (1.0, 1.0, 1.0),
                                                [("velocity", "vector_from_float4")], '%s/velocity/solute-'%self.output_prefix))

        self.u.registerPlugins(mir.Plugins.createDumpParticles('part_dump', self.solvent_pv, 10*dump_every, [], '%s/particles/solvent_particles-'%self.output_prefix))

        if self.composite_solute:
            self.u.registerPlugins(mir.Plugins.createDumpParticles('rigid_dump', self.pv_rigid, dump_every, [], '%s/particles/solute_particles-'%self.output_prefix))
        #if self.solvent_radius > 0.0:
            #self.u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_solute', self.solute_pv, dump_every, [], '%s/particles/solute_particles-'%self.output_prefix))
        # we can also dump the frozen particles for visualization purpose
        #self.u.registerPlugins(mir.Plugins.createDumpParticles('part_dump_wall', self.pv_wall, dump_every, [], '%s/particles/wall_particles-'%self.output_prefix))
        print("plugins registered")
        # we can dump the wall sdf in xdmf + h5 format for visualization purpose
        # the dumped file is a grid with spacings h containing the SDF values
        self.u.dumpWalls2XDMF([self.wall], h = (0.5, 0.5, 0.5), filename = "%s/wall/wall"%self.output_prefix)

    def statistics(self,flowrate_stats=True,membrane_lower_boundary=10,solvent_density_profile=True,solvent_velocity_profile=True,solute_density_profile=True,solute_velocity_profile=True):
        self.Statistics=Statistics(output=self.output_prefix,membrane_lower_boundary=membrane_lower_boundary,boxsize=self.box_size)


    def runSimulation(self,steps=20000,stats_every=200):
        #equilibrate
        self.u.run(2000)
        for i in range(int(steps/stats_every)):
            if self.u.isMasterTask():
                particle_zpositions_old=[x[0] for x in self.solvent_pv.getCoordinates()]

            self.u.run(stats_every)
            if self.u.isMasterTask():
                particle_zpositions_new=[x[0] for x in self.solvent_pv.getCoordinates()]
                solute_zpositions=[x[0] for x in self.pv_rigid.getCoordinates()]

                particle_velocities=self.solvent_pv.getVelocities()
                solute_velocities=self.pv_rigid.getVelocities()
                self.Statistics.calculate(particle_zpositions_old,particle_zpositions_new,particle_velocities,solute_zpositions,solute_velocities,i*stats_every)
                #x=list(zip(particle_zpositions_old,particle_zpositions_new))

                #print(np.mean(np.asarray(particle_zpositions_old)-np.asarray(particle_zpositions_new)))
        if self.u.isMasterTask():
            self.Statistics.plot()
        if not self.db_flag:
            comm=MPI.COMM_WORLD
            rank=comm.Get_rank()
            if rank == 1:
                sim_db_entry2=Simulations.query.get(self.sim_index)
                sim_db_entry2.status='finished'
                db.session.commit()
                os.system("cp -r %s/wall/ ../data/%s/."%(self.label,self.sim_index))
