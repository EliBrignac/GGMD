import pandas as pd
import seaborn as sns
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from sklearn.metrics import r2_score
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
import ast
import py3Dmol
from IPython.display import display
from rdkit.Chem import rdMolDescriptors
import multiprocessing as mp
import concurrent.futures
import numpy as np
import cupy as cp

class FitnessDataProcessor:
    """Handles data processing for fitness calculations."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        self._process_data()

    
    def _split_string(self, s):
        # Check if s is None or not a string
        #print(s)
        if s is None or not isinstance(s, str):
            return []
        
        # Check for empty or invalid strings
        if s in ['[]', 'nan', 'NaN', 'None', 'none', '', ' ']:
            return []
        
        if any(i not in '0123456789,[] ' for i in s):
            return []
        s = s.strip('[]')
        return [int(x) for x in s.split(',')] if s else []


    def _process_data(self):
        """Prepare the data for analysis."""
        self.data['List of generations molecule is present'] = self.data['List of generations molecule is present'].apply(self._split_string)
        self.all_generations = self.data['List of generations molecule is present'].explode().dropna().astype(int)
        self.max_generation = self.all_generations.max() if not self.all_generations.empty else None

    def calculate_fitness_per_generation(self):
        """Calculate fitness per generation."""
        generation_fitness = {i: [] for i in range(self.max_generation + 1)}
        for _, row in self.data.iterrows():
            generations = row['List of generations molecule is present']
            fitness = row['fitness']
            for gen in generations:
                generation_fitness[gen].append(fitness)

        results_df = pd.DataFrame({
            'Generation': list(generation_fitness.keys()),
            'Fitness': list(generation_fitness.values())
        })
        return results_df

    def calculate_avg_fitness(self):
        """Calculate average fitness per generation."""
        fitness_per_gen = self.calculate_fitness_per_generation()
        fitness_df = fitness_per_gen.explode('Fitness').groupby('Generation')['Fitness'].agg(['mean', 'std']).reset_index()
        fitness_df = fitness_df.rename(columns={'mean': 'Average Fitness'})
        return fitness_df

    def calculate_max_fitness(self):
        """Calculate maximum fitness per generation."""
        fitness_per_gen = self.calculate_fitness_per_generation()
        return fitness_per_gen.explode('Fitness').groupby('Generation')['Fitness'].max().reset_index()
    
    def average_percentile(self, arr, lower, upper):
        if len(arr) == 0:
            return np.nan  # Return NaN if the array is empty
        
        # Calculate the 90th and 100th percentiles
        p90 = np.percentile(arr, lower)
        p100 = np.percentile(arr, upper)  # This is the maximum value in the array
        
        # Filter values between the 90th and 100th percentiles
        values_in_range = arr[(arr >= p90) & (arr <= p100)]
        
        # Return the average of the filtered values; return NaN if there are no values
        return np.mean(values_in_range) if len(values_in_range) > 0 else np.nan

    
    def calculate_percentile_fitness(self, lower=90, upper=100):
        fitness_per_gen = self.calculate_fitness_per_generation()
        #print(fitness_per_gen)
        #print(fitness_per_gen.explode('Fitness').groupby('Generation')['Fitness'].max().reset_index())
        fitness_per_gen['Fitness'] = fitness_per_gen['Fitness'].apply(lambda x: np.sort(x))
        fitness_per_gen['Fitness'] = fitness_per_gen['Fitness'].apply(lambda x: self.average_percentile(x, lower, upper))
        #print(fitness_per_gen)
        return fitness_per_gen
    

    def calculate_diversity_per_generation_para(self, df_gen, chunk_size=100):
        """Calculate average similarity (diversity) for a specific generation using parallel processing."""
        dum_list = df_gen['smiles'].tolist()
        fingerprints = self.precompute_fingerprints(dum_list)
        
        # Define function to calculate average similarity for a single chunk
        def calculate_chunk_avg_similarity(chunk):
            avg_sim = []
            for i, query_smiles in enumerate(chunk):
                query_fp = fingerprints.get(query_smiles)
                all_fps = [fingerprints.get(smiles) for smiles in dum_list if smiles != query_smiles]
                avg_sim.append(self.tanimoto_similarity(query_fp, all_fps))
            print(f"Processed chunk with {len(chunk)} molecules.")
            return np.mean(avg_sim)
        
        # Split the list of SMILES into chunks of the specified size
        chunks = [dum_list[i:i + chunk_size] for i in range(0, len(dum_list), chunk_size)]
        
        # Use ThreadPoolExecutor or ProcessPoolExecutor to parallelize
        with concurrent.futures.ThreadPoolExecutor() as executor:
            chunk_averages = list(executor.map(calculate_chunk_avg_similarity, chunks))
        
        # Calculate the final average diversity as 1 - average of the chunk similarities
        avg_diversity = 1 - np.mean(chunk_averages)
        return avg_diversity
    
    def div_gen_para_starter(self):
        """Calculate average similarity (diversity) per generation."""
        df_exploded = self.exploded(self.data)
        diversity = {'gen_num': [], 'diversity': []}
        df_groups = df_exploded.groupby(by='Gen_Num')
        
        for gen_num, df_gen in df_groups:
            print(f"Processing generation {gen_num}...")
            avg_sim = self.calculate_diversity_per_generation_para(df_gen)
            diversity['gen_num'].append(gen_num)
            diversity['diversity'].append(1 - avg_sim)

        return pd.DataFrame(diversity)
    
    


        

    @staticmethod
    def exploded(df_):
        """Safely evaluate and explode the 'Gen_Num' column."""
        if 'List of generations molecule is present' in df_.columns:
            df_.rename(columns={'List of generations molecule is present': 'Gen_Num'}, inplace=True)
        df_['Gen_Num'] = df_['Gen_Num'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return df_.explode('Gen_Num').reset_index(drop=True)

    @staticmethod
    def precompute_fingerprints(smiles_list):
        """Precompute Morgan fingerprints for SMILES strings."""
        fingerprints = {}
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fingerprints[smiles] = morgan_generator.GetFingerprint(mol)
        return fingerprints

    @staticmethod
    def tanimoto_similarity(query_fp, all_fps):
        """Calculate Tanimoto similarity using GPU with CuPy."""
        # Convert fingerprints to GPU arrays
        query_fp = cp.array(query_fp)
        all_fps = cp.array(all_fps)
        
        # Tanimoto similarity calculation on GPU
        numerator = (query_fp & all_fps).sum(axis=1)
        denominator = query_fp.sum() + all_fps.sum(axis=1) - numerator
        similarities = numerator / denominator
        return cp.asnumpy(similarities.mean()) 

    def calculate_diversity_per_generation(self, df_gen):
        """Calculate average similarity (diversity) for a specific generation."""
        dum_list = df_gen['smiles'].tolist()
        fingerprints = self.precompute_fingerprints(dum_list)
        avg_sim = []

        for i, query_smiles in enumerate(dum_list):
            query_fp = fingerprints.get(query_smiles)
            all_fps = [fingerprints.get(smiles) for j, smiles in enumerate(dum_list) if i != j]
            avg_sim.append(self.tanimoto_similarity(query_fp, all_fps))

        return np.mean(avg_sim)

    def calculate_avg_diversity(self):
        """Calculate average similarity (diversity) per generation."""
        df_exploded = self.exploded(self.data)
        diversity = {'gen_num': [], 'diversity': []}
        df_groups = df_exploded.groupby(by='Gen_Num')

        for gen_num, df_gen in df_groups:
            avg_sim = self.calculate_diversity_per_generation(df_gen)
            diversity['gen_num'].append(gen_num)
            diversity['diversity'].append(1 - avg_sim)

        return pd.DataFrame(diversity)



class FitnessPlotter:
    """Handles the plotting of fitness data."""

    def __init__(self, dfs):
        """Initializes with a list of DataFrames."""
        self.dfs = dfs
        sns.set(style="whitegrid", palette="muted")

    def general_plot(self, x_col, y_col, line_plot=True, regression=False, 
                    std_col=None, std=False, marker=None, spline_smoothing=False, 
                    scatter=False, ax=None, legend=True,
                    title=None, x_label=None, y_label=None):
        """
        General function for plotting fitness data on a provided axis.
        """
        if ax is None:
            ax = plt.gca()  # Use the current axis if none is provided

        np.random.seed(0)
        names = [df['name'] for df in self.dfs]
        data_frames = [df['data'] for df in self.dfs]
        colors = sns.color_palette("bright", len(data_frames))


        for df in data_frames:
            name = names.pop(0)
            x = df[x_col]
            y = df[y_col]
            color = colors.pop(0)

            if line_plot:
                sns.lineplot(x=x, y=y, marker=marker, alpha=0.8, color=color, 
                            linewidth=2.5, label=f"{name}", ax=ax)

            if spline_smoothing:
                x_new = np.linspace(x.min(), x.max(), 1000)
                spline = make_interp_spline(x, y, k=3)
                y_smooth = spline(x_new)
                ax.plot(x_new, y_smooth, linestyle="-", color=color, linewidth=1.5, 
                        label=f"{name}")

            if std and std_col:
                ax.errorbar(x, y, yerr=df[std_col], fmt='o', color=color, capsize=5, alpha=0.5)

            if regression:
                p = Polynomial.fit(x, y, deg=1)
                x_fit = np.linspace(x.min(), x.max())
                y_fit = p(x_fit)

                ax.plot(x_fit, y_fit, linestyle="--", color=color)#, label=f"Regression {name} p_val = {r2_score(y, p(x)):.4f}")
                if scatter:
                    ax.scatter(x, y, color=color, s=18)

        # Set the plot labels and title    
        if x_col == 'gen_num':
            x_col = 'Generation'

        x_col = x_col[0].upper() + x_col[1:]
        y_col = y_col[0].upper() + y_col[1:]

        if title:
            ax.set_title(title, fontsize=16, weight='bold')
        else:
            ax.set_title(f"{x_col} vs {y_col}", fontsize=16, weight='bold')
        if x_label:
            ax.set_xlabel(x_label, fontsize=14)
        else:
            ax.set_xlabel(x_col, fontsize=14)
        if y_label:
            ax.set_ylabel(y_label, fontsize=14)
        else:
            ax.set_ylabel(y_col, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        if legend:
            ax.legend(fontsize=12, loc='best')





class GGMDAnalysis:
    """High-level interface for performing fitness analysis."""

    def __init__(self, file_paths):
        self.file_paths = file_paths

    def plot_avg_fitness(self, line_plot=True, regression=False, std=False, 
                        spline_smoothing=False, scatter=False, ax=None, legend=True):
        dfs = [{'data': FitnessDataProcessor(file).calculate_avg_fitness(), 'name': file} for file in self.file_paths]
        plotter = FitnessPlotter(dfs)
        return plotter.general_plot(x_col='Generation', y_col='Average Fitness', line_plot=line_plot, std_col='std', 
                                    std=std, regression=regression, spline_smoothing=spline_smoothing, scatter=scatter, ax=ax, legend=legend)


    def plot_max_fitness(self, line_plot=True, regression=False, std=False, 
                        spline_smoothing=False, scatter=False, ax=None, legend=True):
        dfs = [{'data': FitnessDataProcessor(file).calculate_max_fitness(), 'name': file} for file in self.file_paths]
        plotter = FitnessPlotter(dfs)
        plotter.general_plot(x_col='Generation', y_col='Fitness', line_plot=line_plot, std_col=None, 
                                    std=std, regression=regression, spline_smoothing=spline_smoothing, scatter=scatter, ax=ax, legend=legend,
                                    title='Generation vs Max Fitness', x_label='Generation', y_label='Max Fitness')
        return 


    def plot_percentile_fitness(self, lower, upper, line_plot=True, regression=False, std=False, 
                        spline_smoothing=False, scatter=False, ax=None, legend=True):
        dfs = [{'data': FitnessDataProcessor(file).calculate_percentile_fitness(lower, upper), 'name': file} for file in self.file_paths]
        plotter = FitnessPlotter(dfs)
        return plotter.general_plot(x_col='Generation', y_col='Fitness', line_plot=line_plot, std_col=None, 
                                    std=std, regression=regression, spline_smoothing=spline_smoothing, scatter=scatter, ax=ax, legend=legend)


    def plot_avg_diversity(self, line_plot=True, regression=False, std=False, 
                        spline_smoothing=False, scatter=False, ax=None, legend=True):
        dfs = [{'data': FitnessDataProcessor(file).calculate_avg_diversity(), 'name': file} for file in self.file_paths]
        plotter = FitnessPlotter(dfs)
        return plotter.general_plot(x_col='gen_num', y_col='diversity', line_plot=line_plot, 
                                    std_col=None, std=std, regression=regression, spline_smoothing=spline_smoothing, scatter=scatter, ax=ax, legend=legend)




    def display_molecules(self, smiles_list, legends=None, shape=(3, 3)):
        """Display a list of molecules given their SMILES strings in 2D, with optional legends, in a specified grid shape (rows, columns)."""
        if smiles_list is None or smiles_list == []:
            return None
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        rows, cols = shape  # Extract rows and columns from the shape tuple
        max_molecules = rows * cols  # Maximum number of molecules to display

        # Limit the SMILES list to the number of molecules that fit in the grid
        smiles_list = smiles_list[:max_molecules]
        
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

        # If no custom legends are provided, default to SMILES strings as legends
        if legends is None:
            legends = smiles_list
        
        # Limit the legends to match the number of molecules
        legends = legends[:max_molecules]

        # Create the grid image with molecules and legends
        img = Chem.Draw.MolsToGridImage(mols, molsPerRow=cols, subImgSize=(400, 400), legends=legends)
        return img

    def display_top_n_molecules(self, n=9, shape=(3, 3)):
        """Display the top n molecules with the highest fitness for each DataFrame separately, with the specified grid shape."""
        
        if not isinstance(n, int):
            raise ValueError(f"Input to display_top_n_molecules must be an integer. You provided instance of type {type(n)}")
        
        if n < 1:
            raise ValueError(f"Input to display_top_n_molecules must be a positive integer. You provided {n}")
        
        dfs = [FitnessDataProcessor(file).data for file in self.file_paths]
        
        #names = ['Te', 'result_669.csv']
        for i, df in enumerate(dfs):
            # Get the top n molecules for the current DataFrame
            top_n = df.nlargest(n, 'fitness')
            # Create legends for the current top molecules
            legends = []
            for smiles, fitness, compound_id in zip(top_n['smiles'], top_n['fitness'], top_n['compound_id']):
                formula = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))
                legend = f"ID: {compound_id}  Formula: {formula} Fitness: {fitness:.4f}"
                legends.append(legend)
                
                        
            # Display the molecules for the current DataFrame
            print(f"Displaying top {n} molecules for File {i}:")
            display(self.display_molecules(top_n['smiles'].tolist(), legends, shape=shape))





    def display_molecules_3D(self, smile_string):
        """Display a molecule given their SMILES strings in 3D"""
        if not isinstance(smile_string, str):
            raise ValueError(f"Input to display_molecules_3D must be a single SMILES string. You provided instance of type {type(smile_string)}")

        viewer = py3Dmol.view(width=400, height=400)
        mol = Chem.MolFromSmiles(smile_string)
        block = Chem.MolToMolBlock(mol)
        viewer.addModel(block, 'mol')
        viewer.setStyle({'stick': {}})
        viewer.zoomTo()
        return viewer.show()
    
    

    def summary(self):
        """Print a summary of the analysis and display plots and molecules."""
        
        print("Summary of the analysis:")
        print(f"Number of files: {len(self.file_paths)}")
        print(f"Files: {self.file_paths}")

        print('\n\n')
        print("Fitness Plots:")
        # Set up the 2x3 plot for fitness metrics
        fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10))

        # Plot the average fitness
        #print("Plotting average fitness...")
        self.plot_avg_fitness(ax=axs1[0, 0])
        axs1[0, 0].set_title("Average Fitness")

        # Plot the maximum fitness
        #print("Plotting maximum fitness...")
        self.plot_max_fitness(ax=axs1[0, 1])
        axs1[0, 1].set_title("Max Fitness")

        # Plot the top 10% fitness
        #print("Plotting top 10% fitness...")
        self.plot_percentile_fitness(lower=90, upper=100, ax=axs1[0, 2])
        axs1[0, 2].set_title("Top 10% Fitness")

        # Plot average fitness with regression
        #print("Plotting average fitness regression...")
        self.plot_avg_fitness(ax=axs1[1, 0], line_plot=False, regression=True)
        axs1[1, 0].set_title("Avg Fitness Regression")

        # Plot maximum fitness with regression
        #print("Plotting max fitness regression...")
        self.plot_max_fitness(ax=axs1[1, 1], line_plot=False, regression=True)
        axs1[1, 1].set_title("Max Fitness Regression")

        # Plot top 10% fitness with regression
        #print("Plotting top 10% fitness regression...")
        self.plot_percentile_fitness(lower=90, upper=100, ax=axs1[1, 2], line_plot=False, regression=True)  # Assuming this method exists
        axs1[1, 2].set_title("Top 10% Fitness Regression")

        # Adjust layout for fitness plots
        plt.tight_layout()
        plt.show()


        
        print("Diversity Plots:")
        # Set up the 2x2 plot for diversity metrics
        fig2, axs2 = plt.subplots(1, 2, figsize=(18, 5))

        # Plot average diversity
        self.plot_avg_diversity(ax=axs2[0])
        axs2[0].set_title("Average Diversity")

        # Plot average diversity (repeated)
        self.plot_avg_diversity(ax=axs2[1], line_plot=False, regression=True)
        axs2[1].set_title("Average Diversity (Repeated)")

        # Adjust layout for diversity plots
        plt.tight_layout()
        plt.show()

        # Display molecules with top fitness scores
        print(f"Displaying molecules:")
        self.display_top_n_molecules(n=10)

