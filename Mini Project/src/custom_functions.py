import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm as cm
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc



def unique_column_content_check(df):
    unique_counts = {}
    for column in df.columns:
        unique_counts[column] = df[column].nunique()
    return unique_counts


class corr_matrix_dataframe:
    def __init__(self, df):
        """
        Initializes the CorrelationAnalyzer with a DataFrame.

        Parameters:
            df (DataFrame): Input DataFrame for analysis.
        """
        self.df = df

    def corr_table(self):
        """
        Computes the correlation of the DataFrame and returns it as a simple table.

        Returns:
            DataFrame: A DataFrame containing correlation coefficients.
        """
        # Compute the correlation matrix
        corr_matrix = self.df.corr()

        # Convert to a DataFrame and reset the index to flatten the table
        correlation_table = corr_matrix.stack().reset_index()
        correlation_table.columns = ['Feature 1', 'Feature 2', 'Correlation']

        # Return the formatted correlation table
        return correlation_table


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



class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns_to_drop = columns  # Set the attribute correctly
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')  # Ignore errors if columns don't exist




class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the MinMaxScaler here without 'columns' argument
        self.scaler = MinMaxScaler()
        self.columns = None  # Add an attribute to store columns if needed

    def fit(self, X, y=None):
        # Store the columns to be scaled during fit if needed
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
        # Fit the scaler on the specified columns or all columns
        self.scaler.fit(X)  
        return self

    def transform(self, X):
        # Transform the data using the fitted scaler
        # Apply scaling only to the specified columns if columns were provided
        if self.columns is not None and isinstance(X, pd.DataFrame):
            X[self.columns] = self.scaler.transform(X[self.columns])
        else:
            X = self.scaler.transform(X)
        return X


def plot_indices_relation(df, column_indices):
    # Determine the number of subplots needed
    num_plots = len(column_indices)
    
    # Calculate rows and columns for the subplot grid
    cols = 2  # Define the number of columns for the subplot grid
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



def plot_histograms_nonbinary(df, base_width=10, row_height=4):
    """
    Plots histograms for each non-binary column in the DataFrame with a KDE line.
    
    Parameters:
    - df: DataFrame containing the data.
    - base_width: Base width for the figure.
    - row_height: Height of each row in the plot.
    """
    # Selecting non-binary columns
    nonbinary_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Setting up the plot
    num_columns = len(nonbinary_columns)
    ncols = 2  # 2 columns for each row
    nrows = np.ceil(num_columns / ncols).astype(int)  # Number of rows needed
    
    # Dynamically adjust figsize based on number of rows
    figsize = (base_width, nrows * row_height)
    plt.figure(figsize=figsize)
    
    # Using a color palette
    palette = sns.color_palette("magma", num_columns)
    
    for i, column in enumerate(nonbinary_columns):
        plt.subplot(nrows, ncols, i + 1)
        
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



def plot_histograms_nonbinary_logarithmic(df, columns, base_width=10, row_height=5):
    """
    Plots log-transformed histograms for specified non-binary columns in the DataFrame.
    
    Parameters:
    - df: DataFrame containing the data.
    - columns: List of column names to plot.
    - base_width, row_height: Dimensions of the figure.
    """
    sns.set(style="whitegrid")
    
    # Validating columns in DataFrame
    columns = [col for col in columns if col in df.columns]
    num_columns = len(columns)
    nrows = (num_columns + 1) // 2  # Arranging in a grid of 2 columns
    
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(base_width, row_height * nrows))
    axes = axes.flatten()  # Flatten to simplify indexing
    
    for i, column in enumerate(columns):
        ax = axes[i]
        
        # Log transformation and histogram calculation
        transformed_data = np.log1p(df[column].dropna())
        counts, bins = np.histogram(transformed_data, bins=30, density=True)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        
        # Plotting histogram and normal distribution fit
        ax.bar(bin_centers, counts, width=np.diff(bins), color=plt.cm.magma(0.6), edgecolor='black', alpha=0.8)
        mu, std = norm.fit(transformed_data)
        x = np.linspace(bins[0], bins[-1], 100)
        p = norm.pdf(x, mu, std)
        
        ax.plot(x, p, color='red', linestyle='--', linewidth=2)
        ax.set_title(f"Histogram - {column}")
        ax.set_xlabel(f"log(1 + {column})")
        ax.set_ylabel('Density')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Hide unused subplots if any
    for i in range(num_columns, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_class_distribution(column, column_name):
    # Calculate class counts and percentages
    class_counts = column.value_counts().reset_index()
    class_counts.columns = [column_name, 'Count']
    total = len(column)
    
    # Printing class distributions with percentages
    for class_value, count in class_counts.itertuples(index=False):
        percentage = (count / total) * 100
        print(f"Class={class_value}, n={count} ({percentage:.3f}%)")
    
    # Plotting with a magma color theme using hue
    plt.figure(figsize=(6, 4))
    sns.barplot(data=class_counts, x=column_name, y='Count', hue=column_name, palette="magma", dodge=False)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'Distribution of {column_name}')
    plt.legend(title=column_name)
    plt.show()


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        """
        Initializes the ModelEvaluator with a trained model, test features, and test labels.
        
        Parameters:
            model: Trained classification model with `predict_proba` method.
            X_test: Test features (array-like or DataFrame).
            y_test: True labels (array-like or Series).
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.fpr = None
        self.tpr = None
        self.roc_auc = None

    def calculate_roc_auc(self):
        """
        Calculates the ROC-AUC score and stores false positive rate (fpr), true positive rate (tpr),
        and AUC score (roc_auc).
        """
        try:
            # Get probability predictions for the positive class
            y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
            self.fpr, self.tpr, _ = roc_curve(self.y_test, y_pred_proba)
            self.roc_auc = auc(self.fpr, self.tpr)
            print(f"AUC Score: {self.roc_auc:.2f}")
        except AttributeError:
            print("Model does not support predict_proba. Ensure the model has this method.")

    def plot_roc_curve(self):
        """
        Plots the ROC Curve using the previously calculated false positive rate (fpr),
        true positive rate (tpr), and AUC score (roc_auc).
        """
        if self.fpr is None or self.tpr is None or self.roc_auc is None:
            print("ROC-AUC not calculated. Run `calculate_roc_auc()` first.")
            return

        plt.figure(figsize=(8, 6))
        plt.plot(self.fpr, self.tpr, color='#D55E00', lw=2, label=f'ROC curve (AUC = {self.roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # Random guessing line

        # Set plot limits and labels
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()



class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]




import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm

def plot_violin_relation(df, column_indices, cols=3):
    """
    Plots violin plots for specified columns in a grid layout with colors from magma colormap.
    
    Parameters:
    - df: DataFrame containing the data.
    - column_indices: List of column indices to plot.
    - cols: Number of columns in the subplot grid (default is 3).
    """
    # Number of plots and rows
    num_plots = len(column_indices)
    rows = np.ceil(num_plots / cols).astype(int)  # Calculate number of rows
    
    # Set up the plot grid
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()  # Flatten axes for easy indexing
    
    # Generate colors from the magma colormap
    colors = cm.magma(np.linspace(0.4, 0.8, num_plots))  # Avoid extremes for better contrast

    for i, (ax, index) in enumerate(zip(axes, column_indices)):
        column_name = df.columns[index]
        
        # Create the violin plot without a palette and set color manually
        sns.violinplot(data=df, y=column_name, ax=ax)
        
        # Set the color from the magma colormap
        ax.collections[0].set_facecolor(colors[i])
        
        # Add labels and title
        ax.set_ylabel(column_name)
        ax.set_title(f"Violin Plot of {column_name}")
    
    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
