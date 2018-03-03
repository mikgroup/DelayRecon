# DelayRecon: 
Prerequisite:

Need to install bart: https://mrirecon.github.io/bart/

Instruction:

Data: center-out radial dataset and projection reconstruction dataset are prepared in cfl format.

	•	radial_data.cfl, radial_data.hdr, radial_traj.cfl, radial_traj.hdr,
	•	PR_data.cfl, PR_data.hdr, PR_traj.cfl, PR_traj.hdr

Code: 
	1.	Main files for projection reconstruction trajectory corrections, and radial trajectory correction (center-out spiral trajectory can use radial code directly ); they are only different in the way of how to get the central part: 
 
    •	delay_PR.m
    •	delay_radial.m 

  2.  Auxiliairy files: 
  
	•	ksp_interp.m  (kspace interpolation)
	•	partial_derivative.m  (take partial derivative)
	•	lowrank_thresh.m (low-rank threshold)
