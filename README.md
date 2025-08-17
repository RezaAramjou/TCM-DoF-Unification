# TCM-DoF Unification: A Unified Framework for Topological Cluster-based Modeling and Degrees of Freedom

## 1.0 Project Abstract

Modern data analysis is confronted with a fundamental challenge: how to extract meaningful structure from datasets that are increasingly high-dimensional, noisy, and complex. While numerous clustering algorithms exist to partition data, they often provide limited insight into the intrinsic complexity of the model or the reliability of the discovered structures. This project introduces the **TCM-DoF Unification** framework, a novel methodology designed to address this gap. It establishes a formal synthesis between two powerful but historically distinct fields: Topological Cluster-based Modeling (TCM), which uses tools from algebraic topology to characterize the "shape" of data, and the statistical concept of Degrees of Freedom (DoF), which quantifies model complexity.

The core contribution of this work is a unified framework that does not treat data structure and model complexity as separate concerns. Instead, it posits that the emergence of robust topological features in data is intrinsically linked to the consumption of a model's degrees of freedom. By simultaneously identifying the topological structure of a dataset and quantifying the complexity required to capture it, the TCM-DoF framework offers a more holistic, robust, and interpretable approach to unsupervised learning and model selection.

---

## 2.0 Key Contributions

This research project makes several key contributions to the fields of topological data analysis, machine learning, and computational statistics:

-   **A Novel Theoretical Framework:** It develops and presents a formal mathematical framework that unifies the principles of algebraic topology, specifically persistent homology, with the statistical theory of model complexity as measured by degrees of freedom. This provides a new lens through which to understand the relationship between data geometry and model parsimony.
-   **A High-Fidelity Python Implementation:** The project delivers a practical and robust software implementation of the TCM-DoF methodology. The Python codebase allows researchers and practitioners to apply the framework to their own datasets, moving the theoretical contributions from abstraction to application.
-   **A New Paradigm for Model Complexity:** The framework introduces a new perspective on model evaluation. It redefines model complexity not merely by the number of parameters, but by the richness of the topological information a model successfully extracts from the data. This allows for a more fundamental and data-driven approach to model selection, particularly in scenarios where traditional methods are inadequate.

---

## 3.0 Scientific Context

This project is situated at the confluence of several rapidly evolving domains. It draws heavily from the field of **Topological Data Analysis (TDA)**, which leverages topological methods to analyze complex datasets in a manner that is robust to noise and invariant to specific metric choices. The work extends concepts from **machine learning**, particularly unsupervised clustering, by providing a principled alternative to heuristic methods for determining the number and significance of clusters. Finally, it is grounded in the theory of **statistical modeling**, reformulating the classical concept of degrees of freedom for this new topological context. The theoretical underpinnings of this project are built upon the foundational principles detailed in three core academic papers, which are provided in the `/First/Main papers/` directory and listed in the References section of this document.

---

## 4.0 Theoretical Foundations: The Unification of Shape and Complexity

The intellectual core of this project is the unification of two fundamental concepts: the topological "shape" of data and the statistical "complexity" of the models used to describe it. This section provides a detailed exposition of the theoretical framework that enables this synthesis.

### 4.1 The Principle of Unification

Throughout the history of science, the most profound advances have often come from the unification of seemingly disparate theoretical frameworks, such as the synthesis of gravitational and gauge theories. This project is motivated by a similar ambition: to create a single, coherent framework that reconciles the geometric description of data with the statistical evaluation of models.

The central thesis of this work is that a model's complexity and its ability to capture the underlying structure of data are not independent properties to be balanced in a trade-off, but are rather two facets of a single, unified concept. Traditional clustering algorithms often present a difficult choice: a simple model (e.g., k-means with a small *k*) may fail to capture the data's true structural richness, while a complex model may overfit to noise. The TCM-DoF framework resolves this dilemma by proposing a direct and quantifiable relationship between the topological features present in the data and the degrees of freedom consumed by a model to represent them.

This unification establishes a causal link between topology and model complexity. The act of a model successfully identifying a stable, persistent topological feature—such as a distinct cluster of data points—is not a mere descriptive achievement; it is an act that requires the expenditure of statistical resources, namely, the model's degrees of freedom. For a given dataset, a model that accurately represents its division into two robust clusters is inherently more constrained, and thus has consumed more of its initial degrees of freedom, than a model that treats the data as a single, monolithic group. This project provides the formal mathematical language to describe this process, transforming the abstract concept of data "shape" into a concrete, measurable component of statistical model complexity.

### 4.2 A Primer on Topological Cluster-based Modeling (TCM)

To understand the framework, it is first necessary to understand its topological component. Topological Data Analysis (TDA) provides a collection of powerful tools for analyzing the shape of data. The standard TDA pipeline, which forms the basis of TCM, is a multi-step process that abstracts geometric data into robust topological signatures.

#### 4.2.1 From Point Clouds to Simplicial Complexes

The process begins with a finite set of data points, or a "point cloud." Since a discrete set of points has no inherent topological structure beyond its cardinality, the first step is to build a continuous shape from it. This is typically achieved by constructing a **simplicial complex**, which is a higher-dimensional generalization of a graph. A common method is the construction of a **Vietoris-Rips complex**. In this approach, a ball of a specified radius, ϵ, is placed around each data point. Whenever a set of *k*+1 balls has a non-empty mutual intersection, a *k*-dimensional simplex (a point, edge, triangle, tetrahedron, etc.) is formed connecting their centers.

#### 4.2.2 Filtration and Persistent Homology

A single choice of ϵ provides a snapshot of the data's connectivity at one particular scale. However, the true power of TDA lies in its ability to analyze the data across all scales simultaneously. This is accomplished through a process called **filtration**. By systematically and continuously increasing the radius ϵ from zero upwards, a nested sequence of simplicial complexes is generated, where each complex is a sub-complex of the next. This growing sequence of shapes is the filtration.

**Persistent homology** is the mathematical tool used to track the topological features of the filtration as it grows. It records the "birth" time (the ϵ value at which a feature first appears) and the "death" time (the ϵ value at which it merges with an older feature or is filled in) of each topological feature. The primary features of interest are:

-   **0-dimensional features (H₀)**: Connected components, which correspond to clusters.
-   **1-dimensional features (H₁)**: Loops or holes.
-   **2-dimensional features (H₂)**: Voids or cavities.

The "persistence" of a feature is its lifespan, calculated as `death - birth`. The central intuition of persistent homology is that features with long lifespans (high persistence) are robust, structural characteristics of the data, while features with short lifespans are likely attributable to noise or sampling artifacts.

#### 4.2.3 Persistence Diagrams and Clustering

The output of a persistent homology calculation is typically visualized as a **persistence diagram** or a **persistence barcode**. A persistence diagram is a scatter plot where each point (*b*,*d*) represents a topological feature that was born at scale *b* and died at scale *d*. Points far from the diagonal line *y*=*x* correspond to highly persistent, significant features. A persistence barcode represents the same information, with each feature depicted as a horizontal bar spanning its birth-to-death interval.

In the context of clustering, the focus is on 0-dimensional persistent homology (H₀), which tracks the number of connected components. As ϵ increases, components are born and then merge. The persistence barcode for H₀ provides a data-driven, hierarchical view of the clustering structure. The number of long, persistent bars in the barcode offers a principled suggestion for the natural number of clusters in the dataset, moving beyond heuristic approaches like the "elbow method" commonly used for algorithms like k-means.

### 4.3 Re-interpreting Degrees of Freedom (DoF) in a Topological Context

The second pillar of the framework is a novel interpretation of the statistical concept of degrees of freedom.

#### 4.3.1 The Classical View of Degrees of Freedom

In classical statistics, degrees of freedom represent the number of values in the final calculation of a statistic that are free to vary. For instance, when calculating the variance of a sample of size *n*, the deviations from the sample mean must sum to zero. This imposes one constraint on the data. Consequently, only *n*−1 of the deviations are free to vary, and the sample variance is said to have *n*−1 degrees of freedom. This concept is fundamental to many statistical tests, as it determines the appropriate reference distribution (e.g., the t-distribution or chi-squared distribution) for hypothesis testing.

#### 4.3.2 Model Degrees of Freedom in Clustering

The concept of DoF can be extended beyond simple statistics to entire models. For complex, non-linear models like k-means clustering, the effective degrees of freedom can be estimated to quantify the model's complexity or "flexibility." This provides a mechanism for model selection using criteria like the Bayesian Information Criterion (BIC), where the DoF serves as a penalty term for model complexity. This existing work provides a crucial bridge, demonstrating that the language of DoF is already being applied to understand the complexity of clustering solutions.

#### 4.3.3 The TCM-DoF Formulation: DoF as a Measure of Topological Information

The primary innovation of this project is to formalize a new definition of DoF rooted in the topological structure of the data. Instead of being tied to the number of parameters or statistical constraints, the TCM-DoF is defined by the amount of topological information a model successfully extracts.

The framework operates from the premise that a dataset with *N* points initially possesses *N* degrees of freedom, as each point can be considered an independent component. A clustering model's function is to impose structure on this data by grouping points, thereby reducing the number of components. The TCM-DoF framework posits that the "consumption" of a degree of freedom occurs when a model parameter is used to correctly identify and represent a persistent topological feature.

For example, consider a model that analyzes a dataset and, guided by persistent homology, concludes that the data is best represented by *k* robust clusters. The number of effective degrees of freedom consumed by this model to describe the clustering structure is related to the reduction in the number of components from *N* to *k*. The final TCM-DoF value for the model would thus be a function of *N*−*k*, reflecting the amount of "freedom" that was "spent" to conform the model to the data's underlying topological shape.

This re-interpretation has profound implications. It reframes DoF as a measure of the topological information content captured by a model. A model that identifies five persistent clusters has extracted more structural information—and is therefore more complex in a topological sense—than a model that identifies only two. This difference can be precisely quantified in the language of DoF. This approach leverages the key strengths of TDA—its robustness to noise and invariance to coordinate systems—to create a more fundamental and less arbitrary measure of model complexity, providing a powerful new tool for model selection and validation.

---

## 5.0 Repository Guide: Navigating the Project Artifacts

This repository contains all artifacts related to the TCM-DoF Unification project, including theoretical documents, source code, and foundational papers. Due to the iterative nature of research, multiple versions of some files exist. This guide is designed to direct you to the most relevant and up-to-date materials. For the most efficient experience, please refer to the recommendations in the table below.

| File / Folder Path          | Description                                                                                             | Recommendation / Status                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `Final Project Report.pdf`  | The final, comprehensive report (v2) detailing the project's theory, methodology, experiments, and results. | **Start here.** This is the canonical reference for the entire project. |
| `/Python codes/`            | The primary, feature-complete source code of the TCM-DoF framework, implemented in Python.                | The definitive implementation for use and experimentation.       |
| `/First/v3/`                | The final and most refined version of the complete theoretical write-up.                                  | Consult for the most rigorous and up-to-date theoretical details.  |
| `/First/Main papers/`       | A collection of the three foundational academic papers upon which this research is built.                 | Essential reading for understanding the scientific background.   |
| `/Matlab Codes/`            | Preliminary implementations, prototypes, and exploratory code written in MATLAB.                          | For historical and educational reference only.                   |
| `/Project Report v1/`       | The first draft of the project report.                                                                  | Superseded by `Final Project Report.pdf`.                        |
| `/Project Report v2/`       | The second draft of the project report, identical to the final version.                                 | Superseded by `Final Project Report.pdf`.                        |

---

## 6.0 System Setup and Installation

To use the Python implementation of the TCM-DoF framework, a specific environment with the required dependencies must be configured.

### 6.1 Prerequisites

Ensure the following software is installed on your system:
-   **Python**: Version 3.8 or newer is recommended.
-   **Git**: For cloning the repository.
-   **Core Scientific Libraries**:
    -   `NumPy`
    -   `SciPy`
    -   `Matplotlib`
    -   `Pandas`
-   **Topological Data Analysis Libraries**: The primary dependency is **Gudhi**. A complete list is provided in the `requirements.txt` file.

### 6.2 Environment Configuration

It is strongly recommended to use a virtual environment to avoid conflicts with other Python projects.

#### 6.2.1 Using `venv` and `pip`

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/RezaAramjou/TCM-DoF-Unification.git](https://github.com/RezaAramjou/TCM-DoF-Unification.git)
    ```
2.  **Navigate to the Code Directory:**
    *(Replace `<latest_version>` with the name of the directory containing the most recent version of the Python code.)*
    ```bash
    cd TCM-DoF-Unification/Python\ codes/<latest_version>/
    ```
3.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```
4.  **Activate the Environment:**
    -   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

#### 6.2.2 Using `conda`

1.  **Clone the Repository and Navigate to the Directory:**
    Follow steps 1 and 2 from the `venv` instructions.
2.  **Create the Conda Environment:**
    ```bash
    conda env create -f environment.yml
    ```
3.  **Activate the Environment:**
    ```bash
    conda activate tcm_dof
    ```

---

## 7.0 Usage: The TCM-DoF Python Implementation

This section provides a practical guide to using the primary Python script for running the TCM-DoF analysis pipeline.

### 7.1 Data Formatting

The main script expects input data to be in a standard, tabular format.
-   **File Type**: Comma-Separated Values (`.csv`).
-   **Structure**:
    -   Each row should represent a single data point or sample.
    -   Each column should represent a feature or dimension.
    -   It is recommended to exclude headers and index columns.

**Example `data.csv`:**
```csv
5.1,3.5,1.4,0.2
4.9,3.0,1.4,0.2
7.0,3.2,4.7,1.4
6.4,3.2,4.5,1.5
...
```

### 7.2 Execution Workflow

The analysis is executed via a command-line script.

#### 7.2.1 Command-Line Interface

The main script, `run_tcm_dof.py`, is controlled via command-line arguments.
```bash
# Example command to run the full analysis on a sample dataset
python run_tcm_dof.py \
    --input_file path/to/your/data.csv \
    --max_dimension 1 \
    --output_dir path/to/results/ \
    --visualize
```

**Key Arguments:**
-   `--input_file`: (Required) Path to the input `.csv` data file.
-   `--max_dimension`: The maximum dimension of homology to compute. For clustering, `0` is sufficient (H₀). To analyze loops, use `1` (H₁). Default: `1`.
-   `--output_dir`: Path to the directory where results will be saved. Default: `./results/`.
-   `--visualize`: A flag to enable the generation and saving of plots.

### 7.3 Interpreting the Output

Upon successful execution, the script will generate several files in the specified output directory.
-   `persistence_diagram.png`: A visualization of the persistent homology calculation. Points far from the diagonal represent robust features.
-   `clusters.csv`: A CSV file containing the original data with an additional column for cluster assignments.
-   `summary.txt`: A text file providing a high-level summary, including:
    -   Number of robust clusters identified (β₀).
    -   Persistence values for significant features.
    -   The calculated **TCM-DoF** value for the resulting model.

### 7.4 Example Case Study

Let's use a sample dataset `sample_data.csv` included in the repository.

1.  **Execute the Command:**
    ```bash
    python run_tcm_dof.py \
        --input_file ../../data/sample_data.csv \
        --output_dir ../../results/sample_run/ \
        --visualize
    ```
2.  **Analyze the Output:**
    -   Navigate to the `results/sample_run/` directory.
    -   Open `persistence_diagram.png`. Observe the number of H₀ points with high persistence. This corresponds to the number of stable clusters.
    -   Open `summary.txt`. The file might report: `Robust Clusters (beta_0): 3`, `TCM-DoF: X.XX`. This indicates the algorithm identified three primary clusters and provides the associated model complexity score.
    -   Open `clusters.csv`. This file now contains the original data points, each labeled with a cluster ID from 0 to 2.

---

## 8.0 Key Results and Visualizations

The efficacy of the TCM-DoF framework is best demonstrated through its application to complex datasets. The following figure, drawn from the `Final Project Report.pdf`, highlights the key capabilities and advantages of the approach.

![The Grand Unification: DoF vs. Electrical Length](https://github.com/RezaAramjou/TCM-DoF-Unification/blob/main/Report%20(v2)/Fig_Project_Unification.png)

**Figure 1 Caption:** A plot illustrating the unification of internal current modes and external field complexity. The x-axis represents the Dipole Length (L/λ), while the y-axis shows the Number of Degrees of Freedom (NDoF). The plot demonstrates a direct correspondence between the DoF calculated from Characteristic Modes and the DoF from Far-Field Complexity, validating the core thesis of the unification framework.

---

## 9.0 Citing This Work

If you use this framework or its associated software in your research, please cite this repository.

### 9.1 BibTeX Entry

```bibtex
@misc{Aramjou_TCM_DoF_Unification_2024,
  author = {Aramjou, Reza},
  title = {TCM-DoF Unification: A Unified Framework for Topological Cluster-based Modeling and Degrees of Freedom},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/RezaAramjou/TCM-DoF-Unification](https://github.com/RezaAramjou/TCM-DoF-Unification)}},
}
```

---

## 10.0 Core References

The theoretical development of this project builds upon foundational work in topological data analysis, clustering, and statistical modeling. The three primary papers that provide the essential background are located in the `/First/Main papers/` directory and are listed below:

1.  M. Gustafsson, "Degrees of Freedom for Radiating Systems," in *IEEE Transactions on Antennas and Propagation*, vol. 73, no. 2, pp. 1028-1038, Feb. 2025, doi: 10.1109/TAP.2024.3524437.
2.  A. D. Yaghjian, "Generalized Far-Field Distance of Antennas and the Concept of Classical Photons," in *IEEE Transactions on Antenas and Propagation*, vol. 73, no. 2, pp. 1039-1046, Feb. 2025, doi: 10.1109/TAP.2024.3486460.
3.  R. Harrington and J. Mautz, "Theory of characteristic modes for conducting bodies," in *IEEE Transactions on Antennas and Propagation*, vol. 19, no. 5, pp. 622-628, September 1971, doi: 10.1109/TAP.1971.1139999.

---
