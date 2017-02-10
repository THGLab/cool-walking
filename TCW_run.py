#Temperature Cool Walking
#James Lincoff
#February 2017

#This script sets up and runs a temperature cool walking simulation.
#It takes in inputs using argparse.
#Based off of the inputs, it then sets up simulation objects, reporters, and minimizes, equilibrates, and runs a TCW simulation.

import math
import random
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import sys, os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'Provide inputs for TCW simulation using argparse and run.')
parser.add_argument('--pdb', type = str, help = 'Provide pdb path.')
parser.add_argument('--temperature_file', type = str, help = 'Provide path to text file that contains list of temperatures, low to high. Should be formatted like example file.')
parser.add_argument('--forcefield', type = str, help = 'Provide the name of the openmm .xml file being used for the forcefield.')
parser.add_argument('--watermodel', type = str, help = 'Provide the name of the openmm .xml file being used for the water model.')
parser.add_argument('--timestep', type = float, default = 0.001, help = 'Set the timestep, in picoseconds. Default is 0.001.')
parser.add_argument('--step_probability', type = float, default = 0.025, help = 'Set the amount of propagation performed at each intermediate temperature during annealing. step_probability = 0.025 for example corresponds to 40 steps at each temperature on average. That is the default value.')
parser.add_argument('--p_down', type = float, default = 0.000125, help = 'Set rate of attempting cool walking runs. p_down = 0.000125 corresponds to one cool walking run for every 8,000 standard MD steps of low temperature walker propagation on average. That is the default value.')
parser.add_argument('--decorrelation_num', type = int, default = 4, help = 'Set the ratio of additional high temperature decorrelation performed during annealing. The default is 4 MD steps at the high temperature performed per step of annealing.')
parser.add_argument('--jump_frequency', type = float, default = 0.2, help = 'Set how frequently an exchange attempt is performed during a cool walking run. The default is 0.2, which is appropriate for 5 intermediate temperatures.')
parser.add_argument('--checkpointconstant', type = int, default = 1000000, help = 'Set the rate at which restart files are saved. Default is every 1000000 MD steps.')
parser.add_argument('--simulation_length', type = int, default = 100000000, help = 'Set the total length of the simulation. Default is 100000000 steps.')
parser.add_argument('--outputfilenames', type = str, default = 'TCW_test', help = 'Set the naming convention for output files.')
parser.add_argument('--writeoutfrequency', type = int, default = 1000, help = 'Set the frequency at which pdb and state data is reported. Default is every 1000 steps.')
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
jump_frequency = args.jump_frequency
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
low_temp = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.95 * nanometers)
high_temp = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.95 * nanometers)
annealer = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.95 * nanometers)

# Create integrators.
low_integrator = LangevinIntegrator(temperatures[0] * kelvin, 1.0 / picosecond, args.timestep * picoseconds)
high_integrator = LangevinIntegrator(temperatures[num_temps - 1] * kelvin, 1.0 / picosecond, args.timestep * picoseconds)
anneal_integrator = LangevinIntegrator(temperatures[num_temps - 1] * kelvin, 1.0 / picosecond, args.timestep * picoseconds)

#Create simulation objects.
low_sim = Simulation(pdb.topology, low_temp, low_integrator,platform,properties)
high_sim = Simulation(pdb.topology, high_temp, high_integrator,platform,properties)
anneal_sim = Simulation(pdb.topology, annealer, anneal_integrator,platform,properties)

#Add in reporters.
low_sim.reporters.append(PDBReporter(outputfilenames + '_low.pdb', writeoutfrequency))
low_sim.reporters.append(StateDataReporter(outputfilenames + '_low.txt', writeoutfrequency, step=True, potentialEnergy=True, totalEnergy=True, temperature=True))
high_sim.reporters.append(StateDataReporter(outputfilenames + '_high.txt', writeoutfrequency, step=True, potentialEnergy=True, totalEnergy=True, temperature=True))
countsfile = open(outputfilenames + '_counts.txt','w') #countsfile tracks acceptance rates for cool walking exchanges

#Set positions and velocities.
low_sim.context.setPositions(pdb.positions)
high_sim.context.setPositions(pdb.positions)
low_sim.context.setVelocitiesToTemperature(temperatures[0]*kelvin)
high_sim.context.setVelocitiesToTemperature(temperatures[num_temps - 1]*kelvin)

#Load in restart states if contuing a simulation
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

#Begin production run. All code below here can generally be left alone, as all variable are specified above EXCEPT the names of the restart files.
while totstep <= (simulation_length):
    print 'hi'
    countsfile.flush()

    #Increment totstep. num_steps will be used to track the number of steps of standard propagation for this iteration before attempting a cool walking run.
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
    if totstep > checkpointcounter:
        checkpointcounter = checkpointcounter + checkpointconstant
        low_sim.saveState(outputfilenames + '_low_restart_' + str(checkpointcounter/checkpointconstant) + '.xml')
        high_sim.saveState(outputfilenames + '_high_restart_'+ str(checkpointcounter/checkpointconstant) + '.xml')

    #Increment att_down and prepare for the cool walking run by saving the states of the two walkers and setting the annealer to the current high temperature state.
    att_down = att_down + 1
    initial_high_state = high_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
    initial_low_state = low_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)
    anneal_sim.context.setState(initial_high_state)
    Tcwdown = 0.0 #Tcwdown is the probability associated with cooling from the high temperature down to the current annealing temperature.
    #Note that formally all probabilities listed are implemented as the log(probability) to avoid using exponentials of large numbers

    #Iterate through intermediate temperatures, as tracked by cool_step.
    #i.e. 387.5 --> 367.0 --> 347.5 --> 331.0 --> 314.5 for this alanine example.
    for cool_step in range(1,num_temps,1):

        #Store the current energy of the annealer as the "current" state, then move to the next lowest temperature and recalculate the energy.
        #Use these two energies to calculate Tcwdown.
        energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()+anneal_sim.context.getState(getEnergy = True).getKineticEnergy()
        anneal_sim.context.setVelocitiesToTemperature(temperatures[num_temps-cool_step-1]*kelvin)
        anneal_integrator.setTemperature(temperatures[num_temps-cool_step-1]*kelvin)
        energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()+anneal_sim.context.getState(getEnergy = True).getKineticEnergy()
        Tcwdown = Tcwdown + beta*(energy2/temperatures[num_temps-cool_step-1]-energy1/temperatures[num_temps-cool_step])

        #Prepare for propagation at this temperature.
        anneal_num = 1

        #Iterate to obtain the number of steps of propagation to be performed at this temperature, according to step_probability.
        while random.uniform(0.0,1.0) >= step_probability:    
            anneal_num = anneal_num + 1

        #Propagate the annealer, and perform additional propagation of the high temperature walker to further decorrelate between cool walking runs.
        anneal_sim.step(anneal_num)
        high_sim.step(anneal_num*decorrelation_num)

        #Check whether or not to attempt an exchange at this temperature according to jump_frequency.
        #If not, continue straight to the next temperature.
        if (random.uniform(0.0,1.0) > jump_frequency):
            continue

        #If it is decided to perform an exchange attempt at this temperature, store the current annealer state as cooled_state.
        #This is the state/configuration that we will attempt to impose on the low temperature walker.
        cooled_state = anneal_sim.context.getState(getPositions = True, getVelocities = True, getEnergy = True, enforcePeriodicBox = True)

        #Set Tcwup, the probability for heating from the current intermediate temperature to the high temperature, and set annealer to the current low state.
        Tcwup = 0.0
        anneal_sim.context.setState(initial_low_state)
        anneal_sim.context.setVelocities(initial_low_state.getVelocities()*math.sqrt(temperatures[num_temps-cool_step-1]/temperatures[0]))

        #Propagate at the current temperature.
        anneal_sim.step(anneal_num)

        #Prepare to iterate through each intermediate temperature above the current one, to the high temperature.
        for heat_step in range(num_temps-cool_step, num_temps, 1):

            #Increase the velocity and propagate, as done for the cooling.
            anneal_sim.context.setVelocitiesToTemperature(temperatures[heat_step]*kelvin)
            anneal_integrator.setTemperature(temperatures[heat_step]*kelvin)
            anneal_num = 0
            while random.uniform(0.0,1.0) >= step_probability:
                anneal_num = anneal_num + 1    
            anneal_sim.step(anneal_num)

            #Calculate Tcwup for heating to the current temperature.
            energy1 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()+anneal_sim.context.getState(getEnergy = True).getKineticEnergy()
            anneal_sim.context.setVelocities(anneal_sim.context.getState(getVelocities = True).getVelocities()*math.sqrt(temperatures[heat_step-1]/temperatures[heat_step]))
            energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()+anneal_sim.context.getState(getEnergy = True).getKineticEnergy()
            Tcwup = Tcwup + (beta*(energy2/temperatures[heat_step-1] - energy1/temperatures[heat_step]))

        #Begin calculating the acceptance probability.
        acc = Tcwdown - Tcwup

        #Calculate the weight associated with having accepted the cool walking exchange.
        low_sim.context.setState(cooled_state)
        anneal_sim.context.setState(initial_low_state)
        energy1 = low_sim.context.getState(getEnergy = True).getPotentialEnergy()
        energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
        ener = beta*energy1/temperatures[num_temps-cool_step-1] + beta*energy2/temperatures[0]

        #Calculate the "before state," the weight associated with each configuration being at the temperature at which it was generated.
        low_sim.context.setState(initial_low_state)
        anneal_sim.context.setState(cooled_state)
        energy1 = low_sim.context.getState(getEnergy = True).getPotentialEnergy()
        energy2 = anneal_sim.context.getState(getEnergy = True).getPotentialEnergy()
        ener2 = beta*energy1/temperatures[0] + beta*energy2/temperatures[num_temps-cool_step-1]

        #Calculate the total acceptance probability.
        acc = acc + ener-ener2

        #Write out information to the countsfile to keep track of acceptances.
        s = repr(att_down) + ' ' + repr(acc_down) + ' ' + repr(totstep) + ' ' + repr(cool_step) + ' ' + repr(Tcwdown - Tcwup) + ' ' + repr(ener - ener2) + '\n'
        countsfile.write(s)

        #Check whether or not to exchange. If the exchange is rejected, then cooling will continue with the next lower temperature.
        if (acc > math.log(random.uniform(0.0,1.0))):

            #If accepted, increase acc_down, at set low_sim to the appropriate configuration and velocities and break out of the cooling run.
            acc_down = acc_down + 1
            low_sim.context.setState(cooled_state)
            low_sim.context.setVelocities(cooled_state.getVelocities()*math.sqrt(temperatures[0]/temperatures[num_temps-cool_step-1]))
            break

#Close countsfile after all production is finished.
countsfile.close()