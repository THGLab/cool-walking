# cool-walking
Temperature Cool Walking

Temperature Cool Walking (TCW) is an enhanced sampling method for biomolecular simulations. The code here is a top-level python script that can run TCW simulations using the OpenMM software package. The script ‘TCW_run.py’ uses the python argparse module to read in inputs and parameters as specified by the user. The required parameters are:
 - --pdb: a string to identify the .pdb input file.
 - --temperature_file: a txt file which lists the intermediate temperatures one temperature value per line in Kelvin
 - --forcefield and --watermodel: two strings that identify the desired OpenMM .xml files for the forcefield and water model (i.e. ‘amber99sb.xml’ and ‘tip3p.xml’)
 
Optionally, two OpenMM .xml restart files can be loaded in, one for the low temperature walker and one for the high temperature. These are --low_restart and --high_restart for the argument parser. All other inputs have default values, more information can be found using the --help option.

**NOTE: This assumes that you have OpenMM installed, and can run a simple MD simulation using OpenMM using a python script**

As set up, an example TCW simulation could be run from the command line as:

	python TCW_run.py --pdb ‘alanine_hydr.pdb’ --temperature_file ‘temperatures.txt’ --forcefield ‘amber99sb.xml’ --watermodel ‘tip3p.xml’

Example input files can be found in the examples directory.

The cool walking method is described in two papers:

1.	S. Brown & T. Head-Gordon (2003). Cool-walking: a new markov chain monte carlo sampling method. J. Comp. Chem. PAK Symposium 24, 68-76
2.	J. Lincoff, S. Sasmal, and T. Head-Gordon (2016). Comparing generalized ensemble methods for sampling of systems with many degrees of freedom. J. Chem. Phys. 145(17), 174107
