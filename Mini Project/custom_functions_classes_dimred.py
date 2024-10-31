import pandas as pd

def unique_column_content_check(df):
    unique_counts = {}
    for column in df.columns:
        unique_counts[column] = df[column].nunique()
    return unique_counts

import seaborn as sns
import matplotlib.pyplot as plt

def corr_matrix_dataframe(df):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
    return corr_matrix

from sklearn.feature_selection import mutual_info_classif

def make_mi_scores(X, y):
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(mi_scores):
    mi_scores = mi_scores.sort_values(ascending=True)
    plt.barh(mi_scores.index, mi_scores.values)
    plt.title("Mutual Information Scores")
    plt.show()

import pandas as pd

def calculate_skewness(df):
    # Calculate skewness for each column
    skewness_values = df.skew()
    
    # Function to label skewness
    def label_skewness(skewness):
        if skewness < -0.5:
            return "Left Skewed"
        elif skewness > 0.5:
            return "Right Skewed"
        else:
            return "Not Skewed"
    
    # Create a DataFrame to hold the skewness values and their labels
    skewness_df = pd.DataFrame({
        'Skewness': skewness_values,
        'Skewness Label': skewness_values.apply(label_skewness)
    })
    
    return skewness_df


from sklearn.base import TransformerMixin

class DropColumnTransformer(TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)

class CustomOutlierRemoverInterquartile(TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        return X[~((X < (Q1 - self.factor * IQR)) | (X > (Q3 + self.factor * IQR))).any(axis=1)]

from sklearn.preprocessing import MinMaxScaler

class CustomMinMaxScaler(TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_indices_relation(df, column_indices):
    # Determine the number of subplots needed
    num_plots = len(column_indices)
    
    # Calculate rows and columns for the subplot grid
    cols = 3  # Define the number of columns for the subplot grid
    rows = np.ceil(num_plots / cols).astype(int)  # Calculate the number of rows
    
    # Set up a grid for plotting
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()  # Flatten to easily index axes

    for ax, index in zip(axes, column_indices):
        column_name = df.columns[index]
        # Normalize the values to get the color mapping
        normalized_values = (df[column_name] - df[column_name].min()) / (df[column_name].max() - df[column_name].min())
        # Get colors from the magma colormap
        colors = cm.magma(normalized_values)

        ax.scatter(df.index, df[column_name], color=colors)
        ax.set_xlabel('Index')
        ax.set_ylabel(column_name)
        ax.set_title(f"{column_name} vs Index")

    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_violin_features(df):
    sns.violinplot(data=df)
    plt.title("Violin Plot of Features")
    plt.show()

def plot_violin_with_binary_hue(df, feature, hue):
    sns.violinplot(x=hue, y=feature, data=df)
    plt.title(f"Violin Plot of {feature} with {hue} Hue")
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms_nonbinary(df, figsize=(20, 8)):
    """
    Plots histograms for each non-binary column in the DataFrame with a KDE line.
    
    Parameters:
    - df: DataFrame containing the data.
    - figsize: Tuple specifying the size of the figure.
    """
    # Selecting non-binary columns
    nonbinary_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Setting up the plot
    num_columns = len(nonbinary_columns)
    nrows = (num_columns + 2) // 3  # Arranging plots in a grid of 3 columns
    
    plt.figure(figsize=figsize)
    
    # Using a color palette
    palette = sns.color_palette("magma", num_columns)  # Viridis palette for a colorful look
    
    for i, column in enumerate(nonbinary_columns):
        plt.subplot(nrows, 3, i + 1)
        
        # Plotting the histogram with Seaborn
        sns.histplot(df[column], bins=20, color=palette[i], kde=True, stat="density", linewidth=1)
        
        # Adding a title and labels
        plt.title(column, fontsize=14, weight='bold')
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        
        # Customizing the grid
        plt.grid(axis='y', alpha=0.7)
    
    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def  plot_histograms_nonbinary_logarithmic(df, columns, figsize=(10, 5)):
    """
    Plots histograms for specified non-binary columns in the DataFrame after applying log(1 + x) transformation.
    Overlays a normal distribution curve for comparison and fits a line.
    
    Parameters:
    - df: DataFrame containing the data.
    - columns: List of column names to plot.
    - figsize: Tuple specifying the size of the figure.
    """
    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")

    # Filter to ensure that only valid columns are plotted
    columns = [col for col in columns if col in df.columns]

    # Setting up the plot
    num_columns = len(columns)
    nrows = (num_columns + 2) // 3  # Arranging plots in a grid of 3 columns
    
    plt.figure(figsize=figsize)
    
    for i, column in enumerate(columns):
        plt.subplot(nrows, 3, i + 1)

        # Apply log(1 + x) transformation
        transformed_data = np.log1p(df[column].dropna())  # log(1 + x)

        # Calculate histogram for transformed data
        counts, bins = np.histogram(transformed_data, bins=30, density=True)  # Increase bins and normalize
        bin_centers = 0.5 * (bins[1:] + bins[:-1])  # Calculate bin centers for plotting

        # Plot the histogram with magma color palette
        plt.bar(bin_centers, counts, width=np.diff(bins), color=plt.cm.magma(0.6), edgecolor='black', alpha=0.8, label='Histogram')

        # Fit a normal distribution to the data
        mu, std = norm.fit(transformed_data)  # Fit a normal distribution to the data

        # Create a range of values for the fitted line
        x = np.linspace(bins[0], bins[-1], 100)
        p = norm.pdf(x, mu, std)  # Get the PDF for the fitted normal distribution

        # Plot the fitted normal distribution curve with a dashed line
        plt.plot(x, p, color='red', linestyle='--', linewidth=2, label='Fitted Normal Distribution')

        plt.title(f"Log-transformed Histogram of {column}")
        plt.xlabel(f"log(1 + {column})")
        plt.ylabel('Density')  # Change to density
        plt.legend()  # Add legend to identify histogram and fitted line
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add gridlines for better readability

    plt.tight_layout()
    plt.show()



def plot_pairplots_kde_hue(df, hue):
    sns.pairplot(df, hue=hue, kind='kde')
    plt.show()

def plot_class_distribution(y):
    sns.countplot(y)
    plt.title("Class Distribution")
    plt.show()

from mpl_toolkits.mplot3d import Axes3D

def plot_algo3d(df, x, y, z, hue):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x], df[y], df[z], c=df[hue])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()

def plot_explained_variance(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()

def show_pca_weights(pca, feature_names):
    weights = pd.DataFrame(pca.components_, columns=feature_names)
    print(weights)

def lda_transform_plot(X_lda, y):
    plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
    plt.xlabel("LDA1")
    plt.ylabel("LDA2")
    plt.title("LDA Projection")
    plt.show()
