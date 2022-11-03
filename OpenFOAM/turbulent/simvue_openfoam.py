import os
import random
import sys
import time
import re

from simvue import Simvue

if __name__ == "__main__":
    # Regular expressions
    exp1 = re.compile("^(.+):  Solving for (.+), Initial residual = (.+), Final residual = (.+), No Iterations (.+)$")
    exp2 = re.compile("^Time = (.+)$")

    # Name of log file to collect metrics from
    LOG_FILE = 'log.pimpleFoam'
    
    # Check the log file for new entries at this interval (in secs)
    POLLING_INTERVAL = 5

    # Store these input files/directories
    INPUTS = ['0', 'constant', 'system']

    # Store these output files/directories
    OUTPUTS = ['log.blockMesh', 'log.checkMesh','log.pimpleFoam', 'log.renumbermesh', 'log.yPlus']

    run = Simvue()
    # name = 'vignesh-openfoam-test-%d-%d-%d' % (int(1000*random.random()), int(1000*random.random()), int(1000*random.random()))
    # run.init(name=name, tags=['OpenFOAM'])

    run.init(folder="/HPC-AI/turbulent", tags=['OpenFOAM', 'Turbulent'])

    running = True
    latest = []
    first = True
    cols = []
    file_pos = 0
    ttime = None
    metrics = {}


    # Save input files
    for input_file in INPUTS:
        if os.path.isfile(input_file):
            run.save(input_file, 'input', 'text/plain')
        elif os.path.isdir(input_file):
            run.save_directory(input_file, 'input', 'text/plain', preserve_path=True)
        else:
            print('ERROR: input file %s does not exist' % input_file)
            
            



    while running:
        # If log doesn't exist yet, wait for it
        if not os.path.isfile(LOG_FILE):
            time.sleep(POLLING_INTERVAL)
            continue
    
        # Read log file
        with open(LOG_FILE, 'r') as fh:
            fh.seek(file_pos)
            for line in fh.readlines():
                # Get time
                match = exp2.match(line)
                if match:
                    if metrics:
                        run.log_metrics(metrics)
                        metrics = {}

                    ttime = match.group(1)

                # Get metrics
                match = exp1.match(line)
                if match:
                    metrics['residuals.initial.%s' % match.group(2)] = match.group(3)
                    metrics['residuals.final.%s' % match.group(2)] = match.group(4)

            file_pos = fh.tell()
    
        # Check if application is still running
        if os.path.exists('.finished'):
            running = False
        else:
            time.sleep(POLLING_INTERVAL)
    
            
    # Save output files
    for output_file in OUTPUTS:
        if os.path.isfile(output_file):
            run.save(output_file, 'output', 'text/plain')
        elif os.path.isdir(output_file):
            run.save_directory(output_file, 'output', 'text/plain', preserve_path=True)   
        else:
            print('ERROR: output file %s does not exist' % output_file)
    
    run.close()