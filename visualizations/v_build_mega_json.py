# need to search through
#  Elite Ratio
#  Initial Population Size
#  Selection Pool Size
#  Selection Type
#  Mutation Probability
import time
import visualization_suite_1 as vsuite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import yaml
import pprint
import json
def search_configs(root_folder, search_criteria):
    matching_configs = []
    matching_paths = []
    
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                config_path = os.path.join(dirpath, filename)
                
                # Open and parse the config file as YAML
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                
                # Check if the config meets the search criteria
                match = all(config.get(key) == value for key, value in search_criteria.items())
                
                if match:
                    matching_configs.append(config_path)
                    matching_paths.append(dirpath)
                    # print(f"Match found: {config_path}")
                    # print(yaml.dump(config, default_flow_style=False))
                    # print('-' * 80)  # Separator between config files
    
    
    if not matching_configs:
        print("No matching configs found.")
    else:
        print(f"\nTotal matching configs: {len(matching_configs)}")
    
    return matching_configs, matching_paths

root_folder = r'C:\Users\Eli Brignac\OneDrive\Desktop\GGMD\visualizations\SC24_JTVAE_round_2'  # Replace with the path to your folder

reading_error = {
    0.1: [],
    0.2: [],
    0.3: [],
    0.4: [],
    0.5: [],
}


all_dfs = []
all_files = {}

count = 0

# Assuming 'mutate_prob' needs to match the current 'key' value
criteria = {"model_path": "/mnt/projects/ATOM/blackst/GMD_workspace/debugging/model.epoch-35"} 
configs, paths = search_configs(root_folder, criteria)  # Assuming search_criteria accepts dictionary as input

start = time.time()
with open('Fitness_results_of_SC24_JTVAE_round_2.json', 'w') as json_file:
    json_file.write("[\n")

    for count, (path, config) in enumerate(zip(paths, configs)):
        count += 1

        # Adjust the file path
        path = path + r'/data_all_generations.csv'
        raw_like_string = path.replace("\\", "/")

        # Read the CSV data
        df = pd.read_csv(raw_like_string, engine='python', on_bad_lines='skip')
        #print(df)
        df = df.dropna()
        # Perform analysis using vsuite.FitnessDataProcessor
        

        #print(df['List of generations molecule is present'])
        analysis = vsuite.FitnessDataProcessor(raw_like_string)
        avg_fitness = analysis.calculate_avg_fitness()
        max_fitness = analysis.calculate_max_fitness()
        top_50_avg = analysis.calculate_percentile_fitness()
        #diversity = analysis.calculate_avg_diversity()

        # Load config file
        with open(config, 'r') as file:
            config_params = yaml.safe_load(file)

        # Prepare data to save
        file_path = raw_like_string.split('/')[-2]
        config_path = config
        config_params = config_params
        df_path = path

        info_i_want_to_save = { 
            'config_path': config_path,
            'df_path': df_path,
            'config_params': config_params,
            'avg_fitness': avg_fitness.to_dict(),
            'max_fitness': max_fitness.to_dict(),
            'top_50_avg': top_50_avg.to_dict(),
            'df': df.to_dict(),  # Convert DataFrame to dict
            #'diversity': diversity.to_dict()
        }

        # Append each record to the JSON file, add a comma except for the last iteration
        json.dump(info_i_want_to_save, json_file, indent=4)
        json_file.write(",\n")


    # Write the closing bracket after the loop ends
        print(f'{count}/ {len(paths)} {time.time() - start}')
    json_file.write("\n]")
end = time.time()
pprint.pprint(all_files)

print(f"Time taken: {end - start} seconds")
