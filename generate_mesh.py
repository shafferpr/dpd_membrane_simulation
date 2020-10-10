import sys
import pygalmesh as pgm
import meshio

output=sys.argv[1]

sphere1=pgm.Ball([0.0, 0, 0], 1.3)

sphere2=pgm.Ball([0.0, -1.5, 0], 1.3)

sphere3=pgm.Ball([0.866, 0.5, 0], 1.3)

sphere4=pgm.Ball([-0.866, 0.5, 0], 1.3)

u=pgm.Union([sphere1, sphere2, sphere3, sphere4])

combo_mesh=pgm.generate_surface_mesh(u,outfile=output,angle_bound=30,distance_bound=0.1,radius_bound=0.1)

#meshio.write(output, combo_mesh)
