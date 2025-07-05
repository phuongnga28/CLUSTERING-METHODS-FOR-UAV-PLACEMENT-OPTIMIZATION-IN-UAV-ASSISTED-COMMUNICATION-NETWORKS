# Graduation Thesis Code

## 1. Program Framework

### 1.1. Description

- The program is written in Python and adapted from the original MATLAB version in the referenced research paper.

### 1.2. Installation

- Install the required libraries: `numpy`, `matplotlib`, `scipy`, `pandas`, `sklearn`

```bash
pip install -r requirements.txt
```

### 1.3. Usage

- Run the program:

```bash
python main.py
```

### 1.4. Main File Structure

- The program consists of 4 main parts:

  - **Part 1**: Initialize data and set input parameters  
  - **Part 2**: Use algorithms from libraries to determine the positions of UAVs  
  - **Part 3**: Calculate necessary parameters based on formulas from the research paper  
  - **Part 4**: Display results in the terminal and visualize with plots

### 1.5. Key Files

- `main.py`: The main script that runs the entire program  
- `generate_uniform_data.py`: Generates random data using a uniform distribution  
- `k_means_centroids.py`: Determines UAV positions using the KMeans algorithm  
- `k_medoids_centroids.py`: Determines UAV positions using the KMedoids algorithm  
- `minibatch_k_means_centroids.py`: Determines UAV positions using the MiniBatchKMeans algorithm  
- `optimize_pow_height_cluster.py`: Optimizes UAV parameters such as power, height, and other variables  
- `plot_all_UAV_ranges.py`: Plots UAV positions and their coverage areas  
- `plot_range_with_density.py`: Plots UAV coverage areas along with the density of data points

## 2. Notes

- `uniform_data.npy`: Random data based on a uniform distribution, used for more objective evaluation of algorithm performance and to avoid biased random data  
- `minibatch_draw.py`: Visualizes the clustering process and UAV positioning using the MiniBatchKMeans algorithm on a sample dataset  
- All steps are thoroughly explained in the graduation thesis document

## Made with passion by [Nga Phuong Nguyen]
