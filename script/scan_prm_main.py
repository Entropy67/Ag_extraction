import numpy as np
# import script.model.evolution_on_rate as evo
import script.scan as scan
import script.data as data
import json
import shutil
import logging
import os


def run(prm, mr, stop_condition=None):
    # Create the directory
    if not os.path.isdir(prm["dir"]):
        os.mkdir(prm["dir"])
    
    # Save the parameter
    with open(prm["dir"] + "/prm.json", "w") as fp:
        json.dump(prm, fp, indent = 4)
    
    # Save the script
    # shutil.copy2('run_sim_kon_ceilingAff.py', prm["dir"])
    
    # Write to log
    logging.basicConfig(filename=prm["dir"]+'/log',level=logging.DEBUG, format='%(asctime)s %(message)s')
    
    #mr3 = evo.manyRun(prm=prm)
    mr.num_run = prm["num_run"]
    
    double_prm = False
    if "double_prm" in prm:
        double_prm = prm["double_prm"]
    elif "prm2_name" in prm:
        double_prm = True
    
    cat3 = scan.Scaner(agent=mr, double_prm=double_prm)
    cat3.save_agent_data = True
    cat3.load_config(mr.prm)
    cat3.save=True
    cat3.run(stop_condition=stop_condition)
    cat3.close()
    return 
    
    
    
    
    
if __name__ == "__main__":
    main()