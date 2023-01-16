import os
import time

import numpy as np

if __name__ == "__main__":
    start_time = time.time()
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    
    #! Use draw.io to create figure 1 which is a flowchart
    
    #! Loop to create figures 2 - 11 
    for i in np.arange(2, 12):
        code_path = f"'{path}/fig{i}_code.py'"        
        os.system(f"python {code_path}")
        
    print(f"Total time: {time.time() - start_time:.2f} seconds")
        

