#Temperature Cool Walking
#James Lincoff
#February 2017

#This script sets up and runs a temperature cool walking simulation.
#It takes in inputs using argparse.
#Based off of the inputs, it then sets up simulation objects, reporters, and minimizes, equilibrates, and runs a TCW simulation.

import math
import numpy as np
import random
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import os
import argparse

parser = argparse.ArgumentParser(description = 'Provide inputs for TCW simulation using argparse and run.')
parser.add_argument('--pdb', type = str, help = 'Provide pdb path.')
parser.add_argument('--temperature_file', type = str, help = 'Provide path to text file that contains list of temperatures, low to high. Should be formatted like example file.')
parser.add_argument('--forcefield', type = str, help = 'Provide the name of the openmm .xml file being used for the forcefield.')
parser.add_argument('--watermodel', type = str, help = 'Provide the name of the openmm .xml file being used for the water model.')
parser.add_argument('--timestep', type = float, default = 0.002, help = 'Set the timestep, in picoseconds. Default is 0.002.')
parser.add_argument('--step_probability', type = float, default = 0.002, help = 'Set the amount of propagation performed at each intermediate temperature during annealing. step_probability = 0.002 for example corresponds to 500 steps at each temperature on average. That is the default value.')
parser.add_argument('--p_down', type = float, default = 0.0001, help = 'Set rate of attempting cool walking runs. p_down = 0.0001 corresponds to one cool walking run for every 10,000 standard MD steps of low temperature walker propagation on average. That is the default value.')
parser.add_argument('--decorrelation_num', type = int, default = 2, help = 'Set the ratio of additional high temperature decorrelation performed during annealing. The default is 2 MD steps at the high temperature performed per step of annealing.')
parser.add_argument('--checkpointconstant', type = int, default = 500000, help = 'Set the rate at which restart files are saved. Default is every 500000 MD steps.')
parser.add_argument('--simulation_length', type = int, default = 50000000, help = 'Set the total length of the simulation. Default is 50000000 steps.')
parser.add_argument('--outputfilenames', type = str, default = 'TCW_test', help = 'Set the naming convention for output files.')
parser.add_argument('--writeoutfrequency', type = int, default = 500, help = 'Set the frequency at which pdb and state data is reported. Default is every 500 steps.')
parser.add_argument('--equilibration_length', type = int, default = 0, help = 'Set the number of equilibration steps performed before production. Default is 0 steps.')
parser.add_argument('--tominimize', type = int, default = 0, help = 'Specify whether or not to minimize before production. Default is 0.')
parser.add_argument('--load_restarts', type = int, default = 0, help = 'Specify whether to load restart files for high and low temperature walkers, if continuing a simulation. Default is 0.')
parser.add_argument('--low_restart', type = str, help = 'Specify file path for restart file for low walker, if using restarts')
parser.add_argument('--high_restart', type = str, help = 'Specify file path for restart file for high walker, if using restarts')
parser.add_argument('--platform', type = str, default = 'CUDA', help = 'Specify the platform. Default is CUDA.')

args = parser.parse_args()

pdb = PDBFile(args.pdb)
temperatures = []
for line in open(args.temperature_file):
    temp = float(line)
    temperatures.append(temp)
forcefield = ForceField(args.forcefield, args.watermodel)
step_probability = args.step_probability
p_down = args.p_down
decorrelation_num = args.decorrelation_num
checkpointconstant = args.checkpointconstant
simulation_length = args.simulation_length
outputfilenames = args.outputfilenames
writeoutfrequency = args.writeoutfrequency
equilibration_length = args.equilibration_length
tominimize = args.tominimize
load_restarts = args.load_restarts
low_restart = args.low_restart
high_restart = args.high_restart

beta = Quantity(-1.0 / (1.3806488 * 6.022 / 1000.0),1.0/kilojoules_per_mole) 
platform = Platform.getPlatformByName(args.platform)
properties = {'CudaPrecision':'mixed'}
num_temps = len(temperatures)

# Create system objects. low_temp is the low temperature walker, at the temperature of interest.
# high_temp is the high temperature walker.
# annealer will be used for all annealing cycles.
# This naming convention applies to both integrators and simulation objects as well.
low_temp = forcefield.createSystem(pdb.topology, nonbondedMethod = PME,nonbondedCutoff = 0.95*nanometers, constraints = HBonds)
low_temp.addForce(AndersenThermostat(temperatures[0]*kelvin, 50.0/picosecond))
high_temp = forcefield.createSystem(pdb.topology, nonbondedMethod = PME,nonbondedCutoff = 0.95*nanometers, constraints = HBonds)
high_temp.addForce(AndersenThermostat(temperatures[num_temps-1]*kelvin, 50.0/picosecond))
annealer = forcefield.createSystem(pdb.topology,nonbondedMethod = PME,nonbondedCutoff = 0.95*nanometers, constraints = HBonds)
annealer.addForce(AndersenThermostat(temperatures[num_temps-1]*kelvin, 500.0/picosecond))

#Create integrators.
low_integrator = VerletIntegrator(args.timestep*picoseconds)
high_integrator = VerletIntegrator(args.timestep*picoseconds)
anneal_integrator = VerletIntegrator(args.timestep*picoseconds)

#Create simulation objects.
platform = Platform.getPlatformByName(args.platform)
properties = {'CudaPrecision':'mixed'}

low_sim = Simulation(pdb.topology, low_temp, low_integrator,platform,properties)
low_sim.context.setPositions(pdb.positions)
high_sim = Simulation(pdb.topology, high_temp, high_integrator,platform,properties)
high_sim.context.setPositions(pdb.positions)
anneal_sim = Simulation(pdb.topology, annealer, anneal_integrator,platform,properties)
anneal_sim.context.setPositions(pdb.positions)
low_sim.context.setVelocitiesToTemperature(temperatures[0]*kelvin)
high_sim.context.setVelocitiesToTemperature(temperatures[num_temps - 1]*kelvin)

#Add in reporters.
low_sim.reporters.append(PDBReporter(outputfilenames + '_low.pdb', writeoutfrequency))
low_sim.reporters.append(StateDataReporter(outputfilenames + '_low.txt', writeoutfrequency, step=True, potentialEnergy=True, totalEnergy=True, temperature=True))
high_sim.reporters.append(StateDataReporter(outputfilenames + '_high.txt', writeoutfrequency, step=True, potentialEnergy=True, totalEnergy=True, temperature=True))
countsfile = open(outputfilenames + '_counts.txt','w') #countsfile tracks acceptance rates for cool walking exchanges

#Load in restart states if contuing a simulation.
if load_restarts:
    low_sim.loadState(low_restart)
    high_sim.loadState(high_restart)

#Minimize.
if tominimize:
    low_sim.minimizeEnergy()
    high_sim.minimizeEnergy()

#Equilibrate.
low_sim.step(equilibration_length)
high_sim.step(equilibration_length)

#att_down tracks the number of attempted cool walking exchanges.
#acc_down tracks the number of accepted cool walking exchanged.
#totstep is used to track how far the simulation has progressed. Production stops once totstep exceeds simulation_length.
att_down = 0 #leave these three alone
acc_down = 0
totstep = 0
checkpointcounter = int(checkpointconstant)

#Begin production run. All code below here can generally be left alone, as all variable are specified above.
while totstep <= (simulation_length):
    countsfile.flush() 
    
    #Iteratively increment num_steps and totstep according to p_down, to specify the amount of propagation to be performed by the low and high temperature walkers before the next cool walking run.
    totstep = totstep + 1
    num_steps = 1
    
    #Iteratively increment num_steps and totstep according to p_down, to specify the amount of propagation to be performed by the low and high temperature walkers before the next cool walking run.
    while random.uniform(0.0,1.0) >= p_down:
        num_steps = num_steps + 1
        totstep = totstep + 1
    
    #Propagate replicas.
    low_sim.step(num_steps)
    high_sim.step(num_steps)
    
    #Check if a restart file should be saved by comparing totstep to checkpointcounter. If totstep exceeds checkpointercounter, new restarts are saved and checkpointcounter is increased.
    if(totstep > checkpointcounter):
        checkpointcounter = checkpointcounter + checkptconst
        low_sim.saveState(outputfilenames + '_low_restart.xml') 
        high_sim.saveState(outputfilenames + '_high_restart.xml') 
    
    #Increment att_down and prepare for the cool walking run by saving the states of the two walkers and setting the annealer to the current high temperature state.
    att_down = att_down + 1
    initial_high_state = high_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
    initial_low_state = low_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
    anneal_sim.context.setPositions(initial_high_state.getPositions())
    anneal_sim.context.setVelocities(initial_high_state.getVelocities())
    anneal_sim.context.setParameter(AndersenThermostat.Temperature(),temperatures[num_temps-1]*kelvin)
    Tcws = np.zeros(num_temps) #Tcws stores probability for heating and cooling during annealing.
    #Note that formally all probabilities listed are implemented as the log(probability) to avoid using exponentials of large numbers
    
    #Iterate through intermediate temperatures, as tracked by cool_step.
    for cool_step in range(1,num_temps,1):
        
        #Store the current energy of the annealer as the "current" state, then move to the next lowest temperature and recalculate the energy.
        #Use these two energies to calculate Tcwdown.
        PE = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
        KE1 = anneal_sim.context.getState(getEnergy = True).getKineticEnergy()
        energy1 = PE + KE1
        energy2 = PE + KE1*temperatures[num_temps-cool_step-1]/temperatures[num_temps-cool_step]
        anneal_sim.context.setVelocities(anneal_sim.context.getState(getVelocities=True).getVelocities()*math.sqrt(temperatures[num_temps-cool_step-1]/temperatures[num_temps-cool_step]))
        anneal_sim.context.setParameter(AndersenThermostat.Temperature(),temperatures[num_temps-cool_step-1]*kelvin)
        Tcws[num_temps-cool_step-1] =  beta*(energy2/temperatures[num_temps-cool_step-1]-energy1/temperatures[num_temps-cool_step])
         
        anneal_num = 1
        while random.uniform(0.0,1.0) >= step_probability: 
            anneal_num = anneal_num + 1
        anneal_sim.step(anneal_num)
        high_sim.step(anneal_num*decorrelation_num)
        
        #Check whether or not to attempt an exchange at this temperature, restricted to lowest temperatures for efficiency.
        #If not, continue straight to the next temperature.
        if (cool_step < num_temps-3  or random.uniform(0.0,1.0) > 0.5):
            continue
        
        #If it is decided to perform an exchange attempt at this temperature, store the current annealer state as cooled_state.
        #This is the state/configuration that we will attempt to impose on the low temperature walker.
        cooled_state = anneal_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
        
        #Prepare for heating.
        anneal_sim.context.setPositions(initial_low_state.getPositions())
        anneal_sim.context.setVelocities(initial_low_state.getVelocities()*math.sqrt(temperatures[num_temps-cool_step-1]/temperatures[0]))
        anneal_sim.step(anneal_num)
        s = np.array_repr(Tcws)
        countsfile.write(s)
        #Iterate through each intermediate temperature above the current one, to the high temperature.
        for heat_step in range(num_temps-cool_step, num_temps, 1):
            
            #Increase the velocity and propagate, as done for the cooling.
            anneal_sim.context.setVelocities(anneal_sim.context.getState(getVelocities=True).getVelocities()*math.sqrt(temperatures[heat_step]/temperatures[heat_step-1])*kelvin)
            anneal_sim.context.setParameter(AndersenThermostat.Temperature(),temperatures[heat_step]*kelvin)
            anneal_num = 0
            while random.uniform(0.0,1.0) >= step_probability:
                anneal_num = anneal_num + 1    
            anneal_sim.step(anneal_num)
            
            #Calculate Tcwup for heating to the current temperature.
            PE = anneal_sim.context.getState(getEnergy=True).getPotentialEnergy()
            KE1 = anneal_sim.context.getState(getEnergy=True).getKineticEnergy()
            energy1 = PE + KE1
            anneal_sim.context.setVelocities(anneal_sim.context.getState(getVelocities = True).getVelocities()*math.sqrt(temperatures[heat_step-1]/temperatures[heat_step]))
            energy2 = PE + KE1*temperatures[heat_step]/temperatures[heat_step-1]
            Tcws[heat_step-1] =  Tcws[heat_step-1] - (beta*(energy2/temperatures[heat_step-1] - energy1/temperatures[heat_step]))
        
        #Begin calculating the acceptance probability. Round down individual paired cooled/heating probabilities to 0.0 if greater than 0.0.
        s = np.array_repr(Tcws)
        countsfile.write(s)
        Tcw = 0.0
        for temp in range(num_temps):
            if Tcws[temp] < 0.0:
                Tcw = Tcw + Tcws[temp]
        acc = Tcw
        
        #Calculate the total acceptance probability.
        low_PE = initial_low_state.getPotentialEnergy()
        cooled_PE = cooled_state.getPotentialEnergy()
        ener = beta*low_PE/temperatures[num_temps-cool_step-1] + beta*cooled_PE/temperatures[0]
        ener2 = beta*low_PE/temperatures[0] + beta*cooled_PE/temperatures[num_temps-cool_step-1]
        acc = acc + ener-ener2
        
        #Write out information to the countsfile to keep track of acceptances.
        s = repr(totstep) + ' ' + repr(temperatures[num_temps-cool_step-1]) + ' ' + repr(Tcw) + ' ' + repr(ener - ener2) + ' ' + repr(acc) + '\n'
        countsfile.write(s)
        
        #Check whether or not to exchange. If the exchange is rejected, then cooling will continue with the next lower temperature.
        if (acc > math.log(random.uniform(0.0,1.0))):
            
            #If accepted, increase acc_down, at set low_sim to the appropriate configuration and velocities and break out of the cooling run.
            acc_down = acc_down + 1
            low_sim.context.setPositions(cooled_state.getPositions())
            low_sim.context.setVelocities(cooled_state.getVelocities()*math.sqrt(temperatures[0]/temperatures[num_temps-cool_step-1]))
            s = repr(acc_down) + ' ' + repr(att_down) + '\n'
            last_acc = 1
            countsfile.write(s)
            break

#Close countsfile after all production is finished.
countsfile.close()
