# -*- coding: utf-8 -*-

# Import Data

# Load the dataset
data_path = "Data/HRVdata20201209.csv"  # Replace 'dataset_name.csv' with the actual file name
df = pd.read_csv(data_path)

# Now, df contains the dataset ready for analysis or modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from keras.layers import *
from sklearn.utils import class_weight
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

import scipy as sp
from scipy.stats import ttest_ind
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import optimizers, initializers, callbacks
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lime
import lime.lime_tabular
from lime import lime_tabular
import plotly.graph_objects as go
from sklearn.utils import resample
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight

METRICS_BINARY = [keras.metrics.BinaryAccuracy(name='accuracy')]

METRICS = [
      keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      keras.metrics.MeanSquaredError(name='Brier score'),
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'),
      keras.metrics.CategoricalAccuracy(name='categorical_accuracy') # precision-recall curve
]


df.columns = df.columns.str.replace(' ','')
print(df.shape)
df.head()

print(df["SI>1"].value_counts())
print(df["Sepsis3"].value_counts())

df=df.rename(columns={"LF.HF.ratio.LombScargle":"LF.HF.ratio.LS"
,"LF.Power.LombScargle":"LF.Power.LS"
, "HF.Power.LombScargle":"HF.Power.LS"
,"Power.Law.Y.Intercept.LombScargle":"Power.Law.Y.Intercept.LS"
,"Power.Law.Slope.LombScargle":"Power.Law.Slope.LS"
,"VLF.Power.LombScargle":"VLF.Power.LS"})

df.columns



# Assuming `df` is your original DataFrame

# List all the columns and remove 'Sepsis3' and 'SI>1' from the list
cols = df.columns.tolist()
cols.remove('Sepsis3')
cols.remove('SI>1')

# Add 'Sepsis3' at the beginning and 'SI>1' as the second column
new_order = ['Sepsis3', 'SI>1'] + cols

# Reindex the DataFrame with the new order of columns
new_df = df[new_order]

X_t, X_v, y_t, y_v = train_test_split(df.iloc[:,2:].values, df.iloc[:,1].values, test_size=0.2,
                                      stratify=df.iloc[:,1].values, random_state=0)

    X_v: The features for the validation (or test) set.
    y_t: The corresponding labels for the training set.
    y_v: The corresponding labels for the validation (or test) set.

X_t_means = np.mean(X_t, axis=0)
X_t_stds = np.std(X_t, axis=0)

X_t = (X_t - X_t_means) / X_t_stds
X_v = (X_v - X_t_means) / X_t_stds

def normalize_data(X, y, normalizer='StandardScaler'):
    if normalizer == 'StandardScaler':
        scaler = StandardScaler()
    elif normalizer == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif normalizer == 'RobustScaler':
        scaler = RobustScaler()
    else:
        raise ValueError(f'Invalid normalizer: {normalizer}')

    X_scaled = scaler.fit_transform(X)
    return X_scaled, y



all_features = cols
binary_dependent_variable = new_df.iloc[:, 0]
group1 = new_df[binary_dependent_variable == 0]
group2 = new_df[binary_dependent_variable == 1]



# Perform t-tests for all features
t_test_results = {}

for feature in all_features:

    if feature in df.columns:
        group1_data = group1[feature]
        group2_data = group2[feature]
        t_stat, p_value = ttest_ind(group1_data, group2_data, nan_policy='omit')  # 'omit' will ignore NaNs
        t_test_results[feature] = {'t_stat': t_stat, 'p_value': p_value}

# Convert the results dictionary to a dataframe
t_test_results_df = pd.DataFrame.from_dict(t_test_results, orient='index')


print(t_test_results_df)

# Reset the index to turn the index into a column
t_test_results_df_reset = t_test_results_df.reset_index()

# Rename the columns
t_test_results_df_reset.columns = ['Feature', 't_stat', 'p_value']

# Save the DataFrame with the feature name to a CSV file
output_file_path = 't_test_sepsis.csv'
t_test_results_df_reset.to_csv(output_file_path, index=False)




# Data for the sepsis features, t-statistics, and p-values copied from the CSV file saved
# from the previous step
data_sepsis_ttest = {
    'Feature': [
        'Mean.rate', 'Coefficient.of.variation', 'Poincar..SD1', 'Poincar..SD2',
        'LF.HF.ratio.LS', 'LF.Power.LS', 'HF.Power.LS',
        'DFA.Alpha.1', 'DFA.Alpha.2', 'Largest.Lyapunov.exponent',
        'Correlation.dimension', 'Power.Law.Slope.LS',
        'Power.Law.Y.Intercept.LS', 'DFA.AUC', 'Multiscale.Entropy',
        'VLF.Power.LS', 'Complexity', 'eScaleE', 'pR', 'pD',
        'dlmax', 'sedl', 'pDpR', 'pL', 'vlmax', 'sevl', 'shannEn',
        'PSeo', 'Teo', 'SymDp0_2', 'SymDp1_2', 'SymDp2_2', 'SymDfw_2',
        'SymDse_2', 'SymDce_2', 'formF', 'gcount', 'sgridAND', 'sgridTAU',
        'sgridWGT', 'aFdP', 'fFdP', 'IoV', 'KLPE', 'AsymI', 'CSI', 'CVI',
        'ARerr', 'histSI', 'MultiFractal_c1', 'MultiFractal_c2', 'SDLEalpha',
        'SDLEmean', 'QSE', 'Hurst.exponent', 'mean', 'median'
    ],
    't_stat': [
        10.87533968, -11.17612084, -15.06827204, -14.13526196, 4.821927812,
        -5.284034021, -3.757338241, 2.643342524, 8.110745265, 2.889384253,
        2.134250738, -15.19047639, -14.07471805, -17.33332099, -1.556410673,
        15.75747087, 11.83172259, 3.277459185, -3.917943493, 0.103044549,
        -2.165613643, 3.799061476, 5.700108122, -1.081799111, -7.298372658,
        0.932690088, 2.466739116, -15.46232999, -15.9452476, 10.73323268,
        -12.68348681, -6.680077953, 10.05798259, -12.08446134, -9.022648451,
        8.015913825, 17.53655772, 17.89333274, -10.76146314, -11.1841164,
        -23.74501457, -24.79094201, -11.98965286, 11.30652953, 7.828346451,
        7.984348855, -14.44243982, -7.646135744, 6.386629769, -7.431594267,
        2.689242047, -3.291488132, -3.037057735, -11.41088745, -11.42887849,
        -11.76222594, -10.2294111
    ],
    'p_value': [
        3.40E-27, 1.31E-28, 4.82E-50, 2.22E-44, 1.47E-06, 1.33E-07,
        0.000174, 0.008238899, 6.50E-16, 0.003879221, 0.032878509,
        8.27E-51, 5.05E-44, 4.06E-65, 0.119683856, 1.96E-54, 8.27E-32,
        0.001055756, 9.07E-05, 0.917932406, 0.030395372, 0.000147246,
        1.28E-08, 0.279402296, 3.45E-13, 0.351032224, 0.013673565,
        1.56E-52, 1.17E-55, 1.53E-26, 3.21E-36, 2.69E-11, 1.54E-23,
        4.35E-33, 2.71E-19, 1.40E-15, 1.47E-66, 3.97E-69, 1.14E-26,
        1.20E-28, 3.07E-117, 5.93E-127, 1.32E-32, 3.13E-29, 6.18E-15,
        1.80E-15, 3.30E-46, 2.54E-14, 1.88E-10, 1.29E-13, 0.007189008,
        0.001004609, 0.002403288, 9.82E-30, 8.03E-30, 1.84E-31, 2.77E-24
    ]
}

# Create the DataFrame
df_sepsis = pd.DataFrame(data_sepsis_ttest)

# Calculate -log10 of p-value for the plot
df_sepsis['-log10(p_value)'] = -np.log10(df_sepsis['p_value'])

#Sort in ascending order of p-value
df_sepsis = df_sepsis.sort_values(by='p_value', ascending=True)

# Select the top 15 features based on the smallest p-values
top_features = df_sepsis.head(15)  # This now contains the top 15 features

# Prepare the data for the heatmap - Use the 'top_features' subset for this
heatmap_data_p_values = top_features['-log10(p_value)'].to_frame().set_index(top_features['Feature'])
heatmap_data_t_stats = top_features['t_stat'].to_frame().set_index(top_features['Feature'])

# Set up the matplotlib figure with two plots side by side
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Plot the heatmaps for the top 15 features
sns.heatmap(heatmap_data_p_values, cmap='YlGnBu', cbar=True, ax=axes[0])
sns.heatmap(heatmap_data_t_stats, cmap='vlag', cbar=True, ax=axes[1])

# Customize the plots
axes[0].set_title('-log10(P-Values)', fontsize=16)
axes[0].set_ylabel('Features', fontsize=14)
axes[0].set_xlabel('')

axes[1].set_title('T-Statistics', fontsize=16)
axes[1].set_ylabel('')
axes[1].set_xlabel('')

plt.tight_layout()
#plt.show()
plt.savefig('SA1.png', bbox_inches='tight')

combined_data = pd.DataFrame({
    'Feature': np.tile(top_features['Feature'], 2),
    'Variable': np.concatenate((['-log10(p_value)']*15, ['T-Stat']*15)),
    'Value': np.concatenate((top_features['-log10(p_value)'], top_features['t_stat']))
})


heatmap_data = combined_data.pivot(index='Feature', columns='Variable', values='Value')


plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data, annot=True, fmt='.2g', cmap='vlag',
                 cbar_kws={'label': 'T-Stat / P-Value'})
ax.set_title('Top 15 Statistically Significant Features', fontsize=16)
plt.tight_layout()
plt.show()

top_features.head(15)

feature_list =top_features['Feature'].tolist()

print(feature_list)


# Assuming df is your original DataFrame and it contains the data
columns_to_keep = ['Sepsis3','fFdP', 'aFdP', 'sgridAND', 'gcount',
                   'DFA.AUC', 'Teo', 'VLF.Power.LS', 'PSeo',
                   'Power.Law.Slope.LS', 'Poincar..SD1', 'CVI',
                   'Poincar..SD2', 'Power.Law.Y.Intercept.LS',
                   'SymDp1_2', 'SymDse_2']

# Create a new DataFrame with only the selected columns
df_new = df[columns_to_keep]

# Compute the correlation matrix
corr = df_new.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10,n=9, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Save the figure with a higher resolution
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df_new.corr()[['Sepsis3']].sort_values(by='Sepsis3', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
#heatmap.set_title('Features Correlating with Peaceful', fontdict={'fontsize':18}, pad=16);
plt.savefig('correlation_features.png', dpi=300, bbox_inches='tight')

# Plotly graph
top_features = df_sepsis
fig = go.Figure(go.Bar(
    x=top_features['-log10(p_value)'],
    y=top_features['Feature'],
    orientation='h'
))


fig.update_yaxes(autorange="reversed")  # This reverses the bar order

fig.update_layout(
    title='Feature Significance for Sepsis based on p-value - Top 15 Features',
    xaxis_title='-log10(p-value)',
    yaxis_title='Features',
    xaxis=dict(
        title='-log10(p-value)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Features',
        titlefont_size=16,
        tickfont_size=14,
    )
)

fig.show()

# Plotly graph
fig = go.Figure(go.Bar(
    x=top_features['t_stat'][:15],
    y=top_features['Feature'][:15],
    orientation='h'
))


fig.update_yaxes(autorange="reversed")  # This reverses the bar order

fig.update_layout(
  #  title='Feature Significance for Sepsis based on t-statistic - Top 15 Features',
    xaxis_title='T-statistic',
    yaxis_title='Features',
    xaxis=dict(
        title='T-statistic',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Features',
        titlefont_size=16,
        tickfont_size=14,
    )
)

#fig.show()
fig.write_image("FS_tscore.png")


#Bootstrapping is a statistical technique that allows for estimating the distribution of a statistic by resampling with replacement from the original sample. By resampling many times, we can calculate a p-value using the Central Limit Theorem, which tells us that with a large enough sample size, the sampling distribution of the mean will be normally distributed, regardless of the distribution of the original data.

new_df.head(5)

new_df.shape

# The first column of the dataframe is the dependent binary variable -> Sepsis3
binary_dependent_variable = new_df.iloc[:, 0]

# Define two groups based on the binary dependent variable
group1 = new_df[binary_dependent_variable == 0]
group2 = new_df[binary_dependent_variable == 1]


features_all = new_df.columns[1:]

# Perform bootstrapping to get p-values
bootstrap_results = {}

for feature in features_all:
    # Data for this feature
    data1 = group1[feature].dropna()  # Removing NaNs
    data2 = group2[feature].dropna()  # Removing NaNs

    # Combine data to resample from
    combined_data = np.concatenate([data1, data2])
    diff_of_means = np.mean(data2) - np.mean(data1)  # The observed difference in means

    # Bootstrapping
    boot_diffs = []
    for _ in range(10000):  # Number of bootstraps
        boot_group1 = resample(combined_data, n_samples=len(data1))
        boot_group2 = resample(combined_data, n_samples=len(data2))
        boot_diffs.append(np.mean(boot_group2) - np.mean(boot_group1))

    # Calculate the p-value
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(diff_of_means))
    bootstrap_results[feature] = {'diff_of_means': diff_of_means, 'bootstrap_p_value': p_value}

# Convert the results dictionary to a dataframe for convenient viewing
bootstrap_results_df = pd.DataFrame.from_dict(bootstrap_results, orient='index')
print(bootstrap_results_df)

# Convert the results dictionary to a dataframe for convenient viewing
bootstrap_results_df = pd.DataFrame.from_dict(bootstrap_results, orient='index')

# Reset the index to turn the index into a column
bootstrap_results_df_reset = bootstrap_results_df.reset_index()

# Optionally rename the columns if desired
bootstrap_results_df_reset.columns = ['Feature', 't_stat', 'p_value']

# Save the DataFrame with the feature name to a CSV file
output_file_path = 'bootstrap_test_sepsis.csv'  # Replace with your desired file path
bootstrap_results_df_reset.to_csv(output_file_path, index=False)
"""

    Mean.rate
    Coefficient.of.variation
    Poincar..SD1
    Poincar..SD2
    LF.HF.ratio.LS
    LF.Power.LS
    HF.Power.LS
    DFA.Alpha.1
    DFA.Alpha.2
    Largest.Lyapunov.exponent
    Correlation.dimension
    Power.Law.Slope.LS
    Power.Law.Y.Intercept.LS
    DFA.AUC
    VLF.Power.LS
    Complexity
    eScaleE
    pR
    dlmax
    sedl
    pDpR
    vlmax
    shannEn
    PSeo
    Teo
    SymDp0_2
    SymDp1_2
    SymDp2_2
    SymDfw_2
    SymDse_2
    SymDce_2
    formF
    gcount
    sgridAND
    sgridTAU
    sgridWGT
    aFdP
    fFdP
    IoV
    KLPE
    AsymI
    CSI
    CVI
    ARerr
    histSI
    MultiFractal_c1
    MultiFractal_c2
    SDLEalpha
    SDLEmean
    QSE
    Hurst.exponent
    mean
    median

Non-Statistically Significant Variables (p ≥ 0.05):

    Multiscale.Entropy
    pD
    pL
    sevl
    SI>1
"""
## Plot Graphs for Paper using Bootstrap p_value and difference of mean

# List of feature names
data_sepsis_boot = {
'Feature' : [
    "Mean.rate", "Coefficient.of.variation", "Poincar..SD1", "Poincar..SD2",
    "LF.HF.ratio.LS", "LF.Power.LS", "HF.Power.LS",
    "DFA.Alpha.1", "DFA.Alpha.2", "Largest.Lyapunov.exponent",
    "Correlation.dimension", "Power.Law.Slope.LS",
    "Power.Law.Y.Intercept.LS", "DFA.AUC", "Multiscale.Entropy",
    "VLF.Power.LS", "Complexity", "eScaleE", "pR", "pD",
    "dlmax", "sedl", "pDpR", "pL", "vlmax", "sevl", "shannEn", "PSeo",
    "Teo", "SymDp0_2", "SymDp1_2", "SymDp2_2", "SymDfw_2", "SymDse_2",
    "SymDce_2", "formF", "gcount", "sgridAND", "sgridTAU", "sgridWGT",
    "aFdP", "fFdP", "IoV", "KLPE", "AsymI", "CSI", "CVI", "ARerr",
    "histSI", "MultiFractal_c1", "MultiFractal_c2", "SDLEalpha",
    "SDLEmean", "QSE", "Hurst.exponent", "mean", "median"
],
'Diff_of_Means' : [
    -8.240215055, 0.012150827, 0.006295956, 0.016895439, -0.62656606,
    0.014643639, 0.013033055, -0.04266945, -0.102439364, -0.004998493,
    -0.178662235, 0.353800698, 0.48438319, 0.364979128, 0.028043987,
    -0.118797602, -40.978954, -0.000202408, 1.063116032, -0.094215324,
    0.007624541, -0.148127828, -0.521441844, 1.132687025, 0.015824314,
    -0.04928707, -0.084818859, 0.000185576, 0.000317584, -0.072639148,
    0.052109216, 0.020529932, -2.395660718, 0.524943662, 0.150116842,
    -0.645462903, -0.690130082, -0.641160873, 0.120723224, 0.119677694,
    0.073051552, 0.131794189, 0.189963086, -0.014914318, -0.046139726,
    -0.914554157, 0.36458631, 0.002273078, -0.387731531, 0.050481701,
    -0.029034371, 0.072597277, 0.02734795, 0.344283311, 0.144613806,
    0.014397177, 0.01096766
]
,
'p_value' : [
    0, 0, 0, 0, 0, 0, 0.0002, 0.007, 0, 0.0037,
    0.0329, 0, 0, 0, 0.1137, 0, 0, 0.0011, 0.0002,
    0.9146, 0.0321, 0.0003, 0, 0.2861, 0, 0.3538,
    0.0152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.007, 0.0017,
    0.0028, 0, 0, 0, 0
]
}

# Create the DataFrame
df_sepsis_boot = pd.DataFrame(data_sepsis_boot)

#Sort in ascending order of p-value
df_sepsis_boot = df_sepsis_boot.sort_values(by='p_value', ascending=True)

# Normalize 'Diff_of_Means' based on the max absolute value to maintain direction
df_sepsis_boot['Max_Abs_Normalized_Diff_of_Means'] = df_sepsis_boot['Diff_of_Means'] / abs(df_sepsis_boot['Diff_of_Means']).max()

# Sort by p-value in ascending order and select the top 15 features again
top_15_features_max_abs = df_sepsis_boot.sort_values(by='p_value', ascending=True).head(15)

# Print or inspect the new DataFrame to see the result
print(top_15_features_max_abs[['Feature', 'Diff_of_Means', 'p_value', 'Max_Abs_Normalized_Diff_of_Means']])

# Select the top 15 features based on the smallest p-values
top_features_boot = df_sepsis_boot.head(15)  # This now contains the top 15 features
#top_features_boot = df_sepsis_boot
# Prepare the data for the heatmap - Use the 'top_features' subset for this
heatmap_data_p_values = top_features_boot['p_value'].to_frame().set_index(top_features_boot['Feature'][:15])
heatmap_data_diff_of_means = top_features_boot['Diff_of_Means'].to_frame().set_index(top_features_boot['Feature'][:15])

# Set up the matplotlib figure with two plots side by side
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Plot the heatmaps for the top 15 features
sns.heatmap(heatmap_data_p_values, cmap='YlGnBu', cbar=True, ax=axes[0])
sns.heatmap(heatmap_data_diff_of_means, cmap='vlag', cbar=True, ax=axes[1])

# Customize the plots
#axes[0].set_title('P-Values', fontsize=16)
#axes[0].set_ylabel('Features', fontsize=14)
#axes[0].set_xlabel('')

#axes[1].set_title('Difference_Of_Means', fontsize=16)
axes[1].set_ylabel('')
axes[1].set_xlabel('Difference of Means')

plt.tight_layout()
#plt.savefig("Fig3-diff-of means.png",format='png', dpi=300, bbox_inches='tight')


# Assuming df_sepsis_boot is already defined and loaded with data...

# Select the top 15 features based on the smallest p-values
top_features_boot = df_sepsis_boot.head(15)  # This now contains the top 15 features

# Prepare the data for the heatmap - Use the 'top_features' subset for this
heatmap_data_diff_of_means = top_features_boot['Diff_of_Means'].to_frame().set_index(top_features_boot['Feature'][:15])

# Set up the matplotlib figure
plt.figure(figsize=(4, 3))

# Plot the heatmap for the top 15 features' difference of means

sns.heatmap(heatmap_data_diff_of_means, cmap='vlag', cbar=True, annot=True)

# Customize the plot
#plt.title('Difference of Means', fontsize=16)
plt.ylabel('')  # Optional: define if you want a label for the y-axis


# Adjust layout for better visualization
plt.tight_layout()

# Save the figure
plt.xticks(ticks=[0.5], labels=['Difference of Means from Bootstrapping'])

# Show the plot

plt.savefig("Fig3-diff-of means.png",format='png', dpi=300, bbox_inches='tight')
plt.show()

combined_data_boot = pd.DataFrame({
    'Feature': np.tile(top_features_boot['Feature'], 2),
    'Variable': np.concatenate((['p-value']*15, ['Diff_of_Means']*15)),
    'Value': np.concatenate((top_features_boot['p_value'], top_features_boot['Diff_of_Means']))
})


heatmap_data_boot = combined_data_boot.pivot(index='Feature', columns='Variable', values='Value')


plt.figure(figsize=(10, 8))
ax = sns.heatmap(heatmap_data_boot, annot=True, fmt='.2g', cmap='vlag',
                 cbar_kws={'label': 'T-Stat / P-Value'})
ax.set_title('Top 15 Statistically Significant Features', fontsize=16)
plt.tight_layout()
#plt.show()
plt.savefig('significant_features.png',bbox_inches='tight')

feature_list_boot =top_features_boot['Feature'].tolist()

print(feature_list_boot)


# Plotly graph
fig = go.Figure(go.Bar(
    x=top_features_boot['Diff_of_Means'],
    y=top_features_boot['Feature'],
    orientation='h'
))


fig.update_yaxes(autorange="reversed")  # This reverses the bar order

fig.update_layout(
   # title='Diff of Means for Sepsis based on Bootstrapping - 15 Statistically Significant Feature',
    xaxis_title='Difference of Means',
    yaxis_title='Features',
    xaxis=dict(
        title='Difference of Means',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Features',
        titlefont_size=16,
        tickfont_size=14,
    )
)

fig.show()
#fig.write_image("Diff_Of_Means_15.png")

# Normalize the 'Diff_of_Means' column
df_sepsis_boot['Normalized_Diff_of_Means'] = (df_sepsis_boot['Diff_of_Means'] - df_sepsis_boot['Diff_of_Means'].min()) / (df_sepsis_boot['Diff_of_Means'].max() - df_sepsis_boot['Diff_of_Means'].min())

# Sort by 'p_value' in ascending order and select the top 15 features
top_15_features = df_sepsis_boot.sort_values(by='p_value', ascending=True).head(15)


# Calculate Z-score normalization for 'Diff_of_Means'
df_sepsis_boot['Z_Normalized_Diff_of_Means'] = (df_sepsis_boot['Diff_of_Means'] - df_sepsis_boot['Diff_of_Means'].mean()) / df_sepsis_boot['Diff_of_Means'].std()




# Plotly graph for the top 15 features based on p-value with normalized differences of means
fig = go.Figure(go.Bar(
    x=top_15_features['Z_Normalized_Diff_of_Means'],
    y=top_15_features['Feature'],
    orientation='h'
))

fig.update_yaxes(autorange="reversed")  # Reverse the bar order for better view

fig.update_layout(
    xaxis_title='Z Normalized Difference of Means',
    yaxis_title='Features',
    xaxis=dict(
        title='Z Normalized Difference of Means',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Features',
        titlefont_size=16,
        tickfont_size=14,
    )
)

fig.show()
fig.write_image("boot_strap_features.png", format='png', scale=3, width=1980, height=1080)

# Sort by p-value in ascending order and select the top 15 features
top_15_features = df_sepsis_boot.sort_values(by='p_value', ascending=True).head(15)



# Plotly graph for the top 15 features based on p-value with normalized differences of means
fig = go.Figure(go.Bar(
    x=top_15_features['Max_Abs_Normalized_Diff_of_Means'],
    y=top_15_features['Feature'],
    orientation='h'
))

fig.update_yaxes(autorange="reversed")  # Reverse the bar order for better view

fig.update_layout(
    xaxis_title='Z Normalized Difference of Means',
    yaxis_title='Features',
    xaxis=dict(
        title='Z Normalized Difference of Means',
        titlefont_size=16,
        tickfont_size=14,
    ),
    yaxis=dict(
        title='Features',
        titlefont_size=16,
        tickfont_size=14,
    )
)

fig.show()
fig.write_image("boot_strap_features.png", format='png', scale=3, width=1980, height=1080)




all_features = ['fFdP', 'aFdP', 'sgridAND', 'gcount', 'DFA.AUC', 'Teo', 'VLF.Power.LS', 'PSeo', 'Power.Law.Slope.LS', 'Poincar..SD1', 'CVI', 'Poincar..SD2', 'Power.Law.Y.Intercept.LS', 'SymDp1_2', 'SymDse_2', 'IoV']

# The second column of the dataframe is the dependent binary variable -> Sepsis3
binary_dependent_variable = df.iloc[:, 1]

# Define two groups based on the binary dependent variable
group1 = df[binary_dependent_variable == 0]
group2 = df[binary_dependent_variable == 1]

# Calculate number of rows and columns for the subplot grid
num_features = len(all_features)
ncols = 4  # Define number of columns for subplots
nrows = int(num_features / ncols) if num_features % ncols == 0 else (num_features // ncols) + 1

# Create figure with subplots
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 4, nrows * 3), sharex=False, sharey=False)
axes = axes.flatten()  # Flatten the 2D array of axes for easy iterating

# Plot histograms for all features
for i, feature in enumerate(all_features):
    ax = axes[i]
    if feature in df.columns:
        # Plot the histograms for the current feature
        ax.hist(group1[feature].dropna(), alpha=0.5, label='Group 1', bins=20)
        ax.hist(group2[feature].dropna(), alpha=0.5, label='Group 2', bins=20)
        #ax.hist(group1[feature].dropna(), alpha=0.5, label='Group 1 (Dependent variable = 0)', bins=20)
        #ax.hist(group2[feature].dropna(), alpha=0.5, label='Group 2 (Dependent variable = 1)', bins=20)
       # ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.set_title(feature,fontsize=18)
        ax.legend()

# Hide any empty subplots if the number of features is not a multiple of the number of columns
for i in range(num_features, nrows * ncols):
    axes[i].set_visible(False)

plt.tight_layout()  # Adjust subplot parameters for a clean layout
plt.savefig("Fig1-relative.png",format='png', dpi=300, bbox_inches='tight')


def get_boruta_features(X, y, TRIALS=100, p=0.5):


    def get_important_features(X, y):
        rf = RandomForestClassifier(max_depth=20)
        rf.fit(X,y)
        importances = {feature_name: f_importance for feature_name, f_importance in zip(X.columns, rf.feature_importances_)}
        only_shadow_feat_importance = {key:value for key,value in importances.items() if "shadow" in key}
        highest_shadow_feature = list(dict(sorted(only_shadow_feat_importance.items(), key=lambda item: item[1], reverse=True)).values())[0]
        selected_features = [key for key, value in importances.items() if value > highest_shadow_feature]
        return selected_features

    def get_tail_items(pmf):
        total = 0
        for i, x in enumerate(pmf):
            total += x
            if total >= 0.05:
                break
        return i

    def choose_features(feature_hits, TRIALS, thresh):
        green_zone_thresh = TRIALS - thresh
        blue_zone_upper = green_zone_thresh
        blue_zone_lower = thresh
        green_zone = [key for key, value in feature_hits.items() if    value >= green_zone_thresh]
        blue_zone = [key for key, value in feature_hits.items() if (value >= blue_zone_lower and value < blue_zone_upper)]
        return green_zone,blue_zone


    X = X.copy()  # Create a copy of X to avoid SettingWithCopyWarning
    for col in X.columns:
        X[f"shadow_{col}"] = X[col].sample(frac=1).reset_index(drop=True)

    feature_hits = {i:0 for i in X.columns}
    for _ in range(TRIALS):
        imp_features = get_important_features(X, y)
        for key, _ in feature_hits.items():
            if key in imp_features: feature_hits[key] += 1

    print(feature_hits)

    pmf = [sp.stats.binom.pmf(x, TRIALS, p) for x in range(TRIALS + 1)]

    thresh = get_tail_items(pmf)
    green, blue = choose_features(feature_hits, TRIALS, thresh)
    return green + blue

features_all = new_df.columns[1:]
print(features_all) #Including SI>1

features_all = new_df.columns[1:]
X = new_df[features_all]
y = new_df['Sepsis3']
features_boruta_sepsis3 = get_boruta_features(X, y, p=0.5)
print(features_boruta_sepsis3)
len(features_boruta_sepsis3)


def select_features(X, y, threshold, model_type, test_size=0.2,random_state=0):

    X_Scaled, y = normalize_data(X, y, 'RobustScaler')

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y, stratify=y, test_size=test_size, random_state=random_state)


    # Create a classifier
    if model_type == 'rf':
        clf = RandomForestClassifier(n_estimators=1000,
                                 random_state=0,
                                 n_jobs=-1,
                                 class_weight='balanced',
                                 min_samples_leaf=50,
                                 max_features='sqrt',
                                 max_depth=10)
    elif model_type == 'xgb':
        clf = XGBClassifier(
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=4,
        min_child_weight=1,
        gamma=0.9,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27,
        reg_alpha=0.01,
        reg_lambda=1,
        eval_metric='auc'
    ) # Use scale_pos_weight for imbalance data
    else:
        raise ValueError("Invalid model_type. Expected 'rf' or 'xgb'.")

    # Train the classifier
    clf.fit(X_train, y_train)

    # Create a selector object that will use the classifier to identify
    # features that have an importance of more than the threshold
    sfm = SelectFromModel(clf, threshold=threshold)

    # Train the selector
    sfm.fit(X_train, y_train)

    important_feature_indices = sfm.get_support(indices=True)

    # Get the names of the important features
    important_feature_names = [X.columns[i] for i in important_feature_indices]

    # Plot feature importances
    feature_importances = clf.feature_importances_
    sorted_idx = feature_importances.argsort() [::-1][:20]

    # Sort the feature importance names and scores
    sorted_importance_names = [X.columns[i] for i in sorted_idx]
    sorted_importance_scores = feature_importances[sorted_idx]

    # Plot using seaborn or matplotlib
    plt.figure(figsize=(7, 6))
    sns.barplot(x=sorted_importance_scores, y=sorted_importance_names, orient='h')
    sns.despine()
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature Names')
    plt.savefig(model_type+'.png',bbox_inches='tight')

    return important_feature_names

features_all = new_df.columns[1:]
X = new_df[features_all]
y = new_df['Sepsis3']
features_xgboost_sepsis3 = select_features(X, y, 0.01, 'xgb')
print(features_xgboost_sepsis3)


features_all = new_df.columns[1:]
X = new_df[features_all]
y = new_df['Sepsis3']
features_random_forest_sepsis3 = select_features(X, y, 0.01, 'rf')
print(features_random_forest_sepsis3)


boruta_features_sepsis3= ['Mean.rate', 'Coefficient.of.variation', 'Poincar..SD1', 'Poincar..SD2', 'LF.HF.ratio.LS', 'LF.Power.LS', 'HF.Power.LS', 'DFA.Alpha.1', 'DFA.Alpha.2', 'Correlation.dimension', 'Power.Law.Slope.LS', 'Power.Law.Y.Intercept.LS', 'DFA.AUC', 'Multiscale.Entropy', 'VLF.Power.LS', 'Complexity', 'vlmax', 'shannEn', 'PSeo', 'Teo', 'SymDp2_2', 'gcount', 'sgridAND', 'sgridWGT', 'aFdP', 'fFdP', 'IoV', 'KLPE', 'AsymI', 'CSI', 'CVI', 'ARerr', 'MultiFractal_c1', 'MultiFractal_c2', 'SDLEalpha', 'SDLEmean', 'QSE', 'Hurst.exponent', 'mean', 'median','formF', 'sgridTAU']

len(boruta_features_sepsis3)

boruta_bootstrap_features_sepsis3=['Mean.rate', 'Coefficient.of.variation', 'Poincar..SD1', 'Poincar..SD2', 'LF.HF.ratio.LS', 'LF.Power.LS', 'HF.Power.LS', 'DFA.Alpha.1', 'Correlation.dimension', 'Power.Law.Slope.LS', 'Power.Law.Y.Intercept.LS', 'DFA.AUC','VLF.Power.LS', 'Complexity', 'vlmax', 'shannEn', 'PSeo', 'Teo', 'SymDp2_2', 'gcount', 'sgridAND', 'aFdP', 'fFdP', 'IoV', 'KLPE', 'AsymI', 'CSI', 'CVI', 'ARerr', 'MultiFractal_c1', 'MultiFractal_c2', 'SDLEalpha', 'SDLEmean', 'QSE', 'Hurst.exponent', 'mean', 'median', 'DFA.Alpha.2', 'formF', 'sgridTAU', 'sgridWGT']

len(boruta_bootstrap_features_sepsis3)
# Removed Multiscale.Entropy it is not statistically significant in Bootstrapping

bootstrap_features_sepsis3 = [
    'Mean.rate',
    'Coefficient.of.variation',
    'Poincar..SD1',
    'Poincar..SD2',
    'LF.HF.ratio.LS',
    'LF.Power.LS',
    'HF.Power.LS',
    'DFA.Alpha.1',
    'DFA.Alpha.2',
    'Largest.Lyapunov.exponent',
    'Correlation.dimension',
    'Power.Law.Slope.LS',
    'Power.Law.Y.Intercept.LS',
    'DFA.AUC',
    'VLF.Power.LS',
    'Complexity',
    'eScaleE',
    'pR',
    'dlmax',
    'sedl',
    'pDpR',
    'vlmax',
    'shannEn',
    'PSeo',
    'Teo',
    'SymDp0_2',
    'SymDp1_2',
    'SymDp2_2',
    'SymDfw_2',
    'SymDse_2',
    'SymDce_2',
    'formF',
    'gcount',
    'sgridAND',
    'sgridTAU',
    'sgridWGT',
    'aFdP',
    'fFdP',
    'IoV',
    'KLPE',
    'AsymI',
    'CSI',
    'CVI',
    'ARerr',
    'histSI',
    'MultiFractal_c1',
    'MultiFractal_c2',
    'SDLEalpha',
    'SDLEmean',
    'QSE',
    'Hurst.exponent',
    'mean',
    'median'
]

xgboost_features_sepsis3=['SI>1', 'Mean.rate', 'Coefficient.of.variation', 'Poincar..SD1', 'Poincar..SD2', 'LF.HF.ratio.LS', 'LF.Power.LS', 'HF.Power.LS', 'DFA.Alpha.1', 'Correlation.dimension', 'Power.Law.Slope.LS', 'Power.Law.Y.Intercept.LS', 'DFA.AUC', 'Multiscale.Entropy', 'VLF.Power.LS', 'Complexity', 'pR', 'pD', 'dlmax', 'sedl', 'pDpR', 'vlmax', 'shannEn', 'PSeo', 'Teo', 'SymDp0_2', 'SymDp1_2', 'SymDp2_2', 'SymDfw_2', 'SymDse_2', 'SymDce_2', 'formF', 'gcount', 'sgridAND', 'sgridTAU', 'sgridWGT', 'aFdP', 'fFdP', 'IoV', 'KLPE', 'AsymI', 'CSI', 'CVI', 'ARerr', 'histSI', 'MultiFractal_c1', 'MultiFractal_c2', 'SDLEalpha', 'SDLEmean', 'QSE', 'Hurst.exponent', 'mean', 'median']

randomforest_features_sepsis3=['Mean.rate', 'Poincar..SD1', 'Poincar..SD2', 'LF.HF.ratio.LS', 'LF.Power.LS', 'HF.Power.LS', 'Power.Law.Slope.LS', 'Power.Law.Y.Intercept.LS', 'DFA.AUC', 'VLF.Power.LS', 'Complexity', 'shannEn', 'PSeo', 'Teo', 'gcount', 'sgridAND', 'aFdP', 'fFdP', 'IoV', 'KLPE', 'AsymI', 'CVI', 'ARerr', 'QSE', 'Hurst.exponent']

# Define the feature lists
feature_lists = {
     'boruta_features_sepsis3': boruta_features_sepsis3,
     'boruta_bootstrap_features_sepsis3':boruta_bootstrap_features_sepsis3,
     'bootstrap_features_sepsis3':bootstrap_features_sepsis3,
     'xgboost_features_sepsis3':xgboost_features_sepsis3,
     'randomforest_features_sepsis3':randomforest_features_sepsis3
}


# Declare a global dataframe
global df_results
df_results = pd.DataFrame(columns=['State','Predict', 'Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall','Beta'])

def preprocess_data (df, target_column, feature_names):
    X = df[feature_names].values
    y = df[target_column].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, stratify=y, random_state=0)

    # Standardize the features
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)

    X_train = (X_train - means) / stds
    X_val = (X_val - means) / stds

    return X_train, y_train, X_val, y_val

#adding epsilon to avoid divide by zero error
def find_optimal_threshold(precision, recall, thresholds, beta=2):
    epsilon = 1e-7  # small constant
    f_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + epsilon)
    index = np.argmax(f_score)
    return thresholds[index]

def train_models_threshold(state, predict, X_train, y_train, X_test, y_test, beta=2):
    global df_results

    # Define the models
    xgb = XGBClassifier(
              learning_rate=0.01,
              n_estimators=1000,
              max_depth=4,
              gamma=0.9,
              subsample=0.8,
              colsample_bytree=0.8,
              objective= 'binary:logistic',
              nthread=4,
              scale_pos_weight=1,
              seed=27,
              reg_alpha=0.01,
              reg_lambda=1,
              eval_metric='auc')

    rf = RandomForestClassifier(n_estimators=1000,
                             n_jobs=-1,
                             class_weight='balanced',
                             min_samples_leaf=50,
                             max_features='sqrt',
                             max_depth=10)

    # Train the models
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Make predictions
    xgb_preds = xgb.predict(X_test)
    rf_preds = rf.predict(X_test)

     # Make predictions on training data
    xgb_train_preds = xgb.predict(X_train)
    rf_train_preds = rf.predict(X_train)

    # Predict probabilities
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]

    # Get precision and recall values for different thresholds
    xgb_precision, xgb_recall, xgb_thresholds = precision_recall_curve(y_test, xgb_probs)
    rf_precision, rf_recall, rf_thresholds = precision_recall_curve(y_test, rf_probs)


    # Find optimal thresholds
    xgb_threshold = find_optimal_threshold(xgb_precision, xgb_recall, xgb_thresholds,beta=beta)
    rf_threshold = find_optimal_threshold(rf_precision, rf_recall, rf_thresholds,beta=beta)

    # Adjust predictions based on the new thresholds
    xgb_preds = [1 if prob > xgb_threshold else 0 for prob in xgb_probs]
    rf_preds = [1 if prob > rf_threshold else 0 for prob in rf_probs]

    # Print performance on Testing Data with Thresholds
    print("XGBoost Performance on Testing Data with Thresholds:")
    print("Accuracy: ", accuracy_score(y_test, xgb_preds))
    print("F1 Score: ", f1_score(y_test, xgb_preds))
    print("Precision: ", precision_score(y_test, xgb_preds,zero_division=1))
    print("Recall: ", recall_score(y_test, xgb_preds,zero_division=1))
    print("Confusion Matrix: ")
    plt.figure(figsize=(2, 2))
    sns.heatmap(confusion_matrix(y_test, xgb_preds), annot=True, fmt='d',cmap='Blues')
    plt.show()
    df_results = pd.concat([df_results, pd.DataFrame([{
        'State': state,
        'Predict':predict,
        'Model': 'XGBoost',
        'Accuracy': accuracy_score(y_test, xgb_preds),
        'F1 Score': f1_score(y_test, xgb_preds),
        'Precision': precision_score(y_test, xgb_preds,zero_division=1),
        'Recall': recall_score(y_test, xgb_preds,zero_division=1),
        'Beta':beta
    }], index=[0])], ignore_index=True)

    print("Random Forest Performance on Testing Data with Thresholds:")
    print("Accuracy: ", accuracy_score(y_test, rf_preds))
    print("F1 Score: ", f1_score(y_test, rf_preds))
    print("Precision: ", precision_score(y_test, rf_preds,zero_division=1))
    print("Recall: ", recall_score(y_test, rf_preds,zero_division=1))
    print("Confusion Matrix: ")
    plt.figure(figsize=(2, 2))
    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d',cmap='Blues')
    plt.show()
    df_results = pd.concat([df_results, pd.DataFrame([{
        'State': state,
        'Predict':predict,
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, rf_preds),
        'F1 Score': f1_score(y_test, rf_preds),
        'Precision': precision_score(y_test, rf_preds,zero_division=1),
        'Recall': recall_score(y_test, rf_preds,zero_division=1),
        'Beta':beta
    }], index=[0])], ignore_index=True)

    return xgb, rf, xgb_threshold, rf_threshold

def run_experiments(df, target_column, feature_lists, betas):
    results = []
    for feature_list_name, feature_list in feature_lists.items():
        for beta in betas:
            # Preprocess the data
            X_train, y_train, X_val, y_val = preprocess_data(df, target_column, feature_list)

            # Train the models and get thresholds
            xgb, rf, xgb_threshold, rf_threshold = train_models_threshold(
                feature_list_name, target_column, X_train, y_train, X_val, y_val, beta=beta
            )

            # Collect results
            results.append({
                'FeatureList': feature_list_name,
                'Beta': beta,
                'XGB_Threshold': xgb_threshold,
                'RF_Threshold': rf_threshold,
                'Results_DF': df_results
            })

    return results




# Define the betas you want to test
betas = [0.5, 1, 2]

# Run experiments with all feature lists and betas
results=experiment_results = run_experiments(new_df, 'Sepsis3', feature_lists, betas)

#print(results)


#Compare F1 score
df_results[(df_results['Predict'] == 'Sepsis3') & (df_results['Beta'] == 1.0)].sort_values(by='F1 Score', ascending=False)

#Compare Precisions
df_results[(df_results['Predict'] == 'Sepsis3')& (df_results['Beta'] == 0.5)].sort_values(by='Precision', ascending=False)

#Compare Precisions
df_results[(df_results['Predict'] == 'Sepsis3')& (df_results['Beta'] == 2.0)].sort_values(by='Recall', ascending=False)

sns.set(style='white')
cm = np.array([[924,20],[ 36,99]])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 50},
            cmap=sns.color_palette(['tomato', 'palegreen'], as_cmap=True), cbar=False,
            yticklabels=['No Sepsis', 'Sepsis'], xticklabels=['No Sepsis', 'Sepsis'], ax=ax)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=20, length=0)

#ax.set_title('Seaborn Confusion Matrix with labels', size=24, pad=20)
ax.set_xlabel('Predicted Values', size=20)
ax.set_ylabel('Actual Values', size=20)

additional_texts = ['(True Positive)', '(False Negative)', '(False Positive)', '(True Negative)']
for text_elt, additional_text in zip(ax.texts, additional_texts):
    ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=24)

plt.tight_layout()

# Save the figure before displaying it
plt.savefig("confusion_matrix.png", dpi=300)

# Then display it
plt.show()


#High Recall Model

X_train, y_train, X_val, y_val = preprocess_data (new_df, 'Sepsis3', boruta_bootstrap_features_sepsis3)
xgb_boruta_sep_recall, rf_borutasep_recall, xgb_threshold_sep_recall, rf_threshold_sep_recall =train_models_threshold('Boruta_Boot_Sepsis3', 'Sepsis3', X_train, y_train, X_val, y_val, beta=2.0)

print(X_val.shape)

print(rf_threshold_sep_recall)

pred_v = rf_borutasep_recall.predict_proba(X_val)[:, 1]
pred_v = pred_v >= rf_threshold_sep_recall

recall_score(y_val, pred_v)

# Add the recall prediction probability from Random Forest - Normalize the entire feature matrix

ensemble_sep_df=pd.DataFrame()

X_combined = np.concatenate((X_train, X_val), axis=0)
Y_combined = np.concatenate((y_train, y_val), axis=0)

X_sepsis_recall = rf_borutasep_recall.predict_proba(X_combined)[:,1]

ensemble_sep_df['pred_proba_recall']=X_sepsis_recall.flatten()
ensemble_sep_df['Actual_Value']=Y_combined


print(ensemble_sep_df.head())

#High Precision Model
X_train, y_train, X_val, y_val = preprocess_data (new_df, 'Sepsis3', bootstrap_features_sepsis3)
xgb_boruta_sep_precision, rf_boruta2_sep_precision, xgb_threshold2_sep_precision, rf_threshold2_sep_precision=train_models_threshold('Boot_Sepsis3', 'Sepsis3', X_train, y_train, X_val, y_val, beta=0.5)

print(xgb_threshold2_sep_precision)

pred_v2 = xgb_boruta_sep_precision.predict_proba(X_val)[:,1]
pred_v2 = pred_v2 >= xgb_threshold2_sep_precision

precision_score(y_v, pred_v2)

X_combined = np.concatenate((X_train, X_val), axis=0)
X_boruta_proba_precision = xgb_boruta_sep_precision.predict_proba(X_combined)[:,1]
ensemble_sep_df['pred_proba_precision']=X_boruta_proba_precision.flatten()

print(ensemble_sep_df.head())

ensemble_sep_df.shape

precision_threshold = xgb_threshold2_sep_precision
recall_threshold=rf_threshold_sep_recall

# Format prediction probability  (e.g., 6 decimal places)
ensemble_sep_df['pred_proba_recall'] = ensemble_sep_df['pred_proba_recall'].map('{:.6f}'.format).astype(float)
ensemble_sep_df['pred_proba_precision'] = ensemble_sep_df['pred_proba_precision'].map('{:.6f}'.format).astype(float)

#reordering columns
ensemble_sep_df = ensemble_sep_df[['pred_proba_recall', 'pred_proba_precision', 'Actual_Value']]


print(ensemble_sep_df.head())

print(ensemble_sep_df["Actual_Value"].value_counts())

ensemble_df = ensemble_sep_df[['pred_proba_recall', 'pred_proba_precision', 'Actual_Value']]

# Plot with color coded true labels
plt.scatter(ensemble_df['pred_proba_precision'], ensemble_df['pred_proba_recall'], c=ensemble_df['Actual_Value'],
            cmap='viridis', alpha=0.6)
#plt.title('2D Graph of Precision vs. Recall Model Predictions')
plt.xlabel('Predicted Precision Probability')
plt.ylabel('Predicted Recall Probability')
plt.colorbar(label='True Label')
plt.grid(True)
#plt.show()
plt.savefig('Fig6-2d plot.png',format='png', dpi=300, bbox_inches='tight', quality=95)

X = ensemble_df[['pred_proba_recall', 'pred_proba_precision']].values
y = ensemble_df['Actual_Value'].values

# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

X_train = X[:3451,:]
y_train = y[:3451]
X_test = X[3451:,:]
y_test = y[3451:]

# Standardize the input features.
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Initialize and train a logistic regression model.
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions on the test set.
y_pred = logreg.predict(X_test)

# Calculate precision and recall for the test set.
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1_score:.4f}')

# Split the dataset into training and testing sets.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

X_train = X[:3451,:]
y_train = y[:3451]
X_test = X[3451:,:]
y_test = y[3451:]


# Standardize the input features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train an SVM model.
svm_model = SVC(
    C=1.0,
    kernel='rbf',
    degree=4,
    gamma='scale',
    probability=True,
    class_weight='balanced',
    max_iter=-1,
    random_state=0
)

svm_model.fit(X_train_scaled, y_train)

svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set.
y_pred = svm_model.predict(X_test_scaled)

# Calculate precision and recall for the test set.
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 score: {f1_score:.4f}')

#Manual Check of Ensemble mode

# Define a function to create compound predictions based on the rules
def get_compound_prediction(row):
    if row['pred_binary_recall'] == 1 and row['pred_binary_precision'] ==1 and row['Actual_Value']== 1:
        return 1  # Both models predict 1
    elif row['pred_binary_recall'] == 0 and row['pred_binary_precision'] == 0 and row['Actual_Value']== 0:
        return 0  # Both models predict 0
    if row['pred_binary_recall'] == 1 and row['pred_binary_precision'] ==1 and row['Actual_Value']== 0:
        return 2  # Precision is affected
    if row['pred_binary_recall'] == 0 and row['pred_binary_precision'] ==0 and row['Actual_Value']== 1:
        return 3  # Recall is affected
    else:
        return -1  # Models disagree

# Create a new column for compound predictions
ensemble_sep_df['compound_prediction'] = ensemble_sep_df.apply(get_compound_prediction, axis=1)

ensemble_sep_df.shape

print(ensemble_sep_df["compound_prediction"].value_counts())

ensemble_sep_df_test=ensemble_sep_df[3451:]

ensemble_sep_df_test.shape

print(ensemble_sep_df_test["compound_prediction"].value_counts())


# Trying a split ratio of 75/25

#Random Forest Model High in F1score
X_train, y_train, X_val, y_val = preprocess_data (new_df, 'Sepsis3', bootstrap_features_sepsis3)
xgb_boruta_sep_precision, rf_boruta2_sep_precision, xgb_threshold2_sep_precision, rf_threshold2_sep_precision=train_models_threshold('Boot_Sepsis3', 'Sepsis3', X_train, y_train, X_val, y_val, beta=1.0)

# A split ratio of 80/20 is better for higher model outcome metrics.

#Random Forest Model High in F1score
X_train, y_train, X_val, y_val = preprocess_data (new_df, 'Sepsis3', bootstrap_features_sepsis3)
xgb_boruta_sep_precision, rf_boruta2_sep_precision, xgb_threshold2_sep_precision, rf_threshold2_sep_precision=train_models_threshold('Boot_Sepsis3', 'Sepsis3', X_train, y_train, X_val, y_val, beta=1.0)

print(xgb_threshold2_sep_precision)

pred_v2 = xgb_boruta_sep_precision.predict_proba(X_val)[:,1]
pred_v2 = pred_v2 >= xgb_threshold2_sep_precision

f1_score(y_v, pred_v2)




def explain_model_with_lime(X_train, X_test, model, feature_names, num_instances=5):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['0', '1'],
        verbose=True,
        mode='classification'
    )

    # Increase the figure size if there are many features
    plt.figure(figsize=(12, 8))  # Adjust the size accordingly

    # Explain multiple instances
    for idx_to_explain in range(num_instances):
        print(f"Explaining instance {idx_to_explain+1}/{num_instances}")
        lime_exp = explainer.explain_instance(X_test[idx_to_explain], model.predict_proba, num_features=25)

        # Show the explanation
        lime_exp.show_in_notebook(show_table=True, show_all=False)
        # print the explanation:
        print(lime_exp.as_list())

        # Generate and edit the plot
        fig = lime_exp.as_pyplot_figure()
        plt.xticks(rotation=45)  # Adjust rotation angle to make text readable, 45 or 90 degrees might be good choices
        plt.tight_layout()  # Advanced layouts adjustments

        plt.show()


explain_model_with_lime(X_train, X_val, xgb_boruta_sep_precision,bootstrap_features_sepsis3)

def explain_model_with_lime_and_summarize(X_train, X_val, model, feature_names, num_instances=5):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['0', '1'],
        verbose=True,
        mode='classification'
    )

    # Initialize an empty dictionary to store feature contributions
    feature_contributions = {}

    # Explain multiple instances
    for idx_to_explain in range(num_instances):
        print(f"Explaining instance {idx_to_explain+1}/{num_instances}")
        lime_exp = explainer.explain_instance(X_val[idx_to_explain], model.predict_proba,
                                              num_features=len(feature_names))

        # Collect feature contributions for this instance
        for feature, contribution in lime_exp.as_list():
            if feature not in feature_contributions:
                feature_contributions[feature] = []  # Initialize if feature name is encountered for the first time
            feature_contributions[feature].append(contribution)

        # Optionally visualize each explanation
        # lime_exp.show_in_notebook(show_table=True, show_all=False) # may need to disable or modify based on the environment

    # Summarize the feature contributions
    feature_summary = {feature: np.mean(contributions) for feature, contributions in feature_contributions.items()}

    # Sorting features by their average contribution magnitude
    sorted_features = sorted(feature_summary.items(), key=lambda x: abs(x[1]), reverse=True)

    # Displaying summarized feature importance
    for feature, importance in sorted_features:
        print(f"Feature: {feature}, Average Importance: {importance}")

    return sorted_features

# The function can then be called as before, ensuring X_train and X_val are compatible with the usage shown.
explanation_summary = explain_model_with_lime_and_summarize(X_train, X_val, xgb_boruta_sep_precision,bootstrap_features_sepsis3, num_instances=863)

data_lines_xboost_high_precision = """
Feature: fFdP > 0.24, Average Importance: 0.07929822659402844
Feature: fFdP <= -0.67, Average Importance: -0.07824706919540145
Feature: aFdP <= -0.73, Average Importance: -0.05370774439034065
Feature: aFdP > 0.56, Average Importance: 0.053176875924875716
Feature: KLPE > 0.51, Average Importance: -0.051993962955961405
Feature: KLPE <= -0.74, Average Importance: 0.0518216601374576
Feature: ARerr <= -0.79, Average Importance: 0.04130225288579106
Feature: SDLEalpha > 0.38, Average Importance: -0.0363829727603971
Feature: -0.31 < fFdP <= 0.24, Average Importance: 0.031138051144736167
Feature: -0.67 < fFdP <= -0.31, Average Importance: -0.030970008371782928
Feature: Mean.rate > 0.74, Average Importance: -0.029146153947962907
Feature: sgridAND > 0.71, Average Importance: -0.024763959660103203
Feature: IoV > 0.70, Average Importance: 0.024275565958796017
Feature: -0.28 < ARerr <= 0.57, Average Importance: -0.023674195107405777
Feature: ARerr > 0.57, Average Importance: -0.022079741407235752
Feature: SDLEalpha <= -0.20, Average Importance: 0.020962620520991224
Feature: LF.Power.LS <= -0.73, Average Importance: -0.019655067208224658
Feature: -0.06 < aFdP <= 0.56, Average Importance: 0.019227691734667936
Feature: -0.73 < aFdP <= -0.06, Average Importance: -0.01873330934665983
Feature: LF.HF.ratio.LS <= -0.59, Average Importance: -0.018499235574873922
Feature: MultiFractal_c2 <= -0.20, Average Importance: -0.018163131288312394
Feature: LF.HF.ratio.LS > 0.21, Average Importance: 0.017761583825461003
Feature: AsymI > 0.60, Average Importance: -0.017717513007650284
Feature: -0.74 < KLPE <= -0.25, Average Importance: 0.016522483911341004
Feature: -0.81 < Mean.rate <= -0.05, Average Importance: 0.016336789554648096
Feature: -0.05 < Mean.rate <= 0.74, Average Importance: 0.016283245325967373
Feature: -0.25 < KLPE <= 0.51, Average Importance: -0.016015659283939648
Feature: -0.20 < LF.Power.LS <= 0.53, Average Importance: 0.01590875965335867
Feature: mean <= -0.54, Average Importance: 0.0150795368236681
Feature: AsymI <= -0.60, Average Importance: 0.014418570182749576
Feature: pR > 0.18, Average Importance: 0.013784317071776808
Feature: HF.Power.LS <= -0.71, Average Importance: 0.013257022511782289
Feature: IoV <= -0.52, Average Importance: -0.013237104343359082
Feature: -0.37 < mean <= 0.04, Average Importance: -0.01302541864716416
Feature: -0.58 < pR <= -0.34, Average Importance: -0.012181909748638212
Feature: pR <= -0.58, Average Importance: -0.012108703360744075
Feature: VLF.Power.LS <= -0.65, Average Importance: 0.011556403201912634
Feature: -0.52 < IoV <= 0.10, Average Importance: -0.011044372391785424
Feature: -0.34 < pR <= 0.18, Average Importance: 0.01098442016759226
Feature: sgridAND <= -0.64, Average Importance: 0.010969744474180362
Feature: sedl <= -0.62, Average Importance: 0.010214390081268357
Feature: QSE > 0.73, Average Importance: -0.010014795646039762
Feature: QSE <= -0.75, Average Importance: 0.009814613829467899
Feature: LF.Power.LS > 0.53, Average Importance: 0.009700732797672293
Feature: -0.20 < SDLEalpha <= 0.17, Average Importance: 0.009587318785191246
Feature: 0.06 < DFA.AUC <= 0.79, Average Importance: -0.008960900673615558
Feature: -0.64 < sgridAND <= 0.16, Average Importance: 0.008582665288823645
Feature: gcount > 0.71, Average Importance: -0.008573685654940124
Feature: median > 0.10, Average Importance: 0.008571869173762495
Feature: shannEn <= -0.52, Average Importance: 0.008404433779846324
Feature: MultiFractal_c1 <= -0.72, Average Importance: 0.00834138174303439
Feature: DFA.AUC > 0.79, Average Importance: 0.008193788528858636
Feature: Power.Law.Slope.LS > 0.60, Average Importance: 0.007849859919087258
Feature: MultiFractal_c2 > 0.52, Average Importance: 0.007737830684294242
Feature: -0.21 < PSeo <= 0.05, Average Importance: -0.0074995780041870125
Feature: VLF.Power.LS > 0.67, Average Importance: -0.007302933777191413
Feature: DFA.AUC <= -0.78, Average Importance: 0.006896261160748276
Feature: -0.75 < QSE <= -0.02, Average Importance: 0.0068447451265734845
Feature: 0.22 < MultiFractal_c2 <= 0.52, Average Importance: 0.006807138264612494
Feature: -0.60 < AsymI <= -0.02, Average Importance: 0.006715993727908132
Feature: 0.26 < shannEn <= 0.74, Average Importance: -0.006628189675261688
Feature: -0.14 < Correlation.dimension <= 0.58, Average Importance: 0.006586596403467728
Feature: HF.Power.LS > 0.39, Average Importance: -0.006443295232767942
Feature: -0.02 < QSE <= 0.73, Average Importance: -0.006423184650901672
Feature: 0.06 < VLF.Power.LS <= 0.67, Average Importance: -0.006328288641223809
Feature: SymDp2_2 > 0.11, Average Importance: 0.006309865586365852
Feature: vlmax > 0.17, Average Importance: 0.006294417463175029
Feature: pDpR > 0.59, Average Importance: -0.006074189365316148
Feature: -0.78 < DFA.AUC <= 0.06, Average Importance: -0.0060525089697436195
Feature: sedl > 0.59, Average Importance: -0.005776590595644218
Feature: 0.16 < sgridAND <= 0.71, Average Importance: 0.005662217688178697
Feature: SDLEmean > 0.70, Average Importance: -0.005321488878158786
Feature: -0.73 < LF.Power.LS <= -0.20, Average Importance: -0.005261686378191057
Feature: Teo > 0.15, Average Importance: 0.0051953657036609485
Feature: -0.31 < HF.Power.LS <= 0.39, Average Importance: -0.0051866131758056735
Feature: 0.17 < SDLEalpha <= 0.38, Average Importance: 0.005032534339182654
Feature: Poincar..SD1 <= -0.77, Average Importance: 0.005024912476698458
Feature: SymDce_2 <= -0.79, Average Importance: -0.0050217603471283635
Feature: PSeo > 0.05, Average Importance: 0.004967884330084043
Feature: SymDp1_2 <= -0.76, Average Importance: -0.004942186236889682
Feature: eScaleE <= -0.38, Average Importance: 0.0048627716434343935
Feature: -0.64 < gcount <= 0.17, Average Importance: 0.004825964042114192
Feature: -0.79 < ARerr <= -0.28, Average Importance: 0.004419654776723433
Feature: -0.20 < MultiFractal_c2 <= 0.22, Average Importance: 0.004415711982417197
Feature: -0.72 < MultiFractal_c1 <= -0.14, Average Importance: -0.004408868943394453
Feature: Poincar..SD2 > 0.39, Average Importance: 0.004325894460949435
Feature: Coefficient.of.variation <= -0.69, Average Importance: 0.0040802094944864834
Feature: -0.56 < Teo <= -0.37, Average Importance: -0.003994079656518869
Feature: SymDp2_2 <= -0.56, Average Importance: -0.00389851211002729
Feature: Hurst.exponent > 0.78, Average Importance: 0.003822290251390713
Feature: Power.Law.Y.Intercept.LS <= -0.69, Average Importance: -0.003745206711729139
Feature: Correlation.dimension > 0.58, Average Importance: -0.0037220774816532614
Feature: -0.03 < sedl <= 0.59, Average Importance: -0.003715737055522543
Feature: Power.Law.Y.Intercept.LS > 0.70, Average Importance: 0.003707363059368843
Feature: SymDce_2 > 0.83, Average Importance: 0.0036360549117911297
Feature: dlmax <= -0.50, Average Importance: 0.0035258071042626785
Feature: DFA.Alpha.1 <= -0.79, Average Importance: 0.0035218562487920567
Feature: SDLEmean <= -0.35, Average Importance: 0.0034879994153057814
Feature: Hurst.exponent <= -0.77, Average Importance: -0.0034796448424766497
Feature: pDpR <= -0.71, Average Importance: 0.0034299181842854954
Feature: -0.31 < LF.HF.ratio.LS <= 0.21, Average Importance: 0.003351388186673777
Feature: shannEn > 0.74, Average Importance: -0.003227320481760045
Feature: -0.68 < Power.Law.Slope.LS <= -0.07, Average Importance: -0.0032063205980147683
Feature: Mean.rate <= -0.81, Average Importance: -0.0031472570487420903
Feature: gcount <= -0.64, Average Importance: 0.0031256989424738677
Feature: DFA.Alpha.2 <= -0.50, Average Importance: 0.0030798236163134584
Feature: eScaleE > 0.42, Average Importance: -0.003068488017742052
Feature: -0.54 < mean <= -0.37, Average Importance: -0.0030187254367728313
Feature: Correlation.dimension <= -0.75, Average Importance: -0.0030057582657815633
Feature: -0.02 < AsymI <= 0.60, Average Importance: -0.0029870374667518166
Feature: sgridTAU <= -0.62, Average Importance: 0.0029820873796271505
Feature: -0.27 < Poincar..SD1 <= 0.48, Average Importance: -0.002906743424496703
Feature: SymDp1_2 > 0.53, Average Importance: 0.0028753594290905986
Feature: -0.14 < MultiFractal_c1 <= 0.57, Average Importance: -0.002790795535725707
Feature: PSeo <= -0.29, Average Importance: 0.0027769651644272203
Feature: -0.59 < LF.HF.ratio.LS <= -0.31, Average Importance: -0.002746943603562305
Feature: 0.07 < DFA.Alpha.1 <= 0.81, Average Importance: -0.002671806270321548
Feature: histSI <= -0.39, Average Importance: 0.0026613155801794236
Feature: -0.28 < Poincar..SD2 <= 0.39, Average Importance: -0.002565728635210224
Feature: sgridTAU > 0.81, Average Importance: -0.0024869871295338792
Feature: Power.Law.Slope.LS <= -0.68, Average Importance: -0.002343760346078644
Feature: -0.67 < Poincar..SD2 <= -0.28, Average Importance: -0.00233790530735036
Feature: -0.56 < vlmax <= -0.32, Average Importance: -0.0022225581727215644
Feature: vlmax <= -0.56, Average Importance: -0.0022166219285175426
Feature: -0.65 < VLF.Power.LS <= 0.06, Average Importance: 0.002212794053696349
Feature: median <= -0.34, Average Importance: -0.0021122689768045427
Feature: dlmax > 0.06, Average Importance: -0.0020789505946941125
Feature: Coefficient.of.variation > 0.44, Average Importance: -0.002046354431890421
Feature: -0.39 < histSI <= 0.26, Average Importance: -0.002045262163884294
Feature: -0.07 < Power.Law.Slope.LS <= 0.60, Average Importance: -0.002014135116393444
Feature: -0.29 < Largest.Lyapunov.exponent <= 0.43, Average Importance: -0.001976888991636572
Feature: -0.75 < Complexity <= -0.29, Average Importance: -0.0019228782652108885
Feature: -0.29 < Complexity <= 0.50, Average Importance: 0.0019214327780777537
Feature: -0.38 < eScaleE <= 0.07, Average Importance: -0.0018790090057749084
Feature: Largest.Lyapunov.exponent > 0.43, Average Importance: 0.001850978424637048
Feature: CSI > 0.51, Average Importance: 0.0017923685235352413
Feature: -0.56 < SymDp2_2 <= -0.39, Average Importance: -0.0017628043613898568
Feature: -0.44 < SymDp1_2 <= 0.53, Average Importance: 0.0016914040506858384
Feature: Complexity <= -0.75, Average Importance: -0.0016862955447404795
Feature: Complexity > 0.50, Average Importance: 0.0016751105379340392
Feature: -0.79 < DFA.Alpha.1 <= 0.07, Average Importance: -0.0016362510888909916
Feature: formF > 0.47, Average Importance: 0.0016036883456656131
Feature: -0.52 < shannEn <= 0.26, Average Importance: 0.0015647982080854205
Feature: sgridWGT <= -0.59, Average Importance: 0.001550555928846083
Feature: 0.15 < DFA.Alpha.2 <= 0.67, Average Importance: -0.0015474052248425233
Feature: DFA.Alpha.2 > 0.67, Average Importance: -0.0015341403667522855
Feature: CVI <= -0.76, Average Importance: 0.001532781188187915
Feature: -0.77 < Poincar..SD1 <= -0.27, Average Importance: -0.0015319225533260445
Feature: formF <= -0.74, Average Importance: -0.0014329352334391313
Feature: -0.32 < vlmax <= 0.17, Average Importance: -0.0014301824969478881
Feature: -0.69 < Power.Law.Y.Intercept.LS <= 0.04, Average Importance: -0.0013337382754282357
Feature: 0.26 < histSI <= 0.69, Average Importance: -0.0013304980596023457
Feature: -0.39 < SymDp2_2 <= 0.11, Average Importance: -0.0013215490206885368
Feature: Poincar..SD1 > 0.48, Average Importance: -0.0013182777151143701
Feature: -0.71 < pDpR <= -0.04, Average Importance: 0.0012833231717438477
Feature: -0.35 < SDLEmean <= 0.43, Average Importance: 0.0011676516073498217
Feature: 0.26 < sgridWGT <= 0.77, Average Importance: -0.0011347871503895387
Feature: -0.69 < Coefficient.of.variation <= -0.25, Average Importance: -0.0010720987528481916
Feature: -0.25 < Coefficient.of.variation <= 0.44, Average Importance: -0.0010580681162020535
Feature: 0.17 < gcount <= 0.71, Average Importance: 0.0009865296368582075
Feature: -0.77 < Hurst.exponent <= -0.31, Average Importance: 0.0009826479145854919
Feature: -0.34 < median <= 0.10, Average Importance: -0.0009733553696759551
Feature: -0.79 < SymDce_2 <= 0.17, Average Importance: 0.0009496876923554654
Feature: 0.04 < Power.Law.Y.Intercept.LS <= 0.70, Average Importance: 0.0009268617640220936
Feature: 0.43 < SDLEmean <= 0.70, Average Importance: 0.0009203833365759023
Feature: -0.75 < CSI <= -0.20, Average Importance: -0.0009113591545971815
Feature: 0.24 < sgridTAU <= 0.81, Average Importance: -0.0009008925955692574
Feature: -0.04 < pDpR <= 0.59, Average Importance: 0.0008939269273172528
Feature: mean > 0.04, Average Importance: 0.0008687348241906599
Feature: -0.33 < dlmax <= 0.06, Average Importance: -0.0008303884640524576
Feature: SymDp0_2 <= -0.40, Average Importance: 0.0008028823261983312
Feature: SymDse_2 <= -0.85, Average Importance: -0.0007995763379996221
Feature: -0.75 < Correlation.dimension <= -0.14, Average Importance: 0.000790902373046225
Feature: DFA.Alpha.1 > 0.81, Average Importance: 0.0007645607605636334
Feature: 0.17 < SymDce_2 <= 0.83, Average Importance: 0.000729147558487716
Feature: Poincar..SD2 <= -0.67, Average Importance: 0.0007195144577986225
Feature: -0.74 < formF <= -0.20, Average Importance: -0.0006603839476210244
Feature: Largest.Lyapunov.exponent <= -0.72, Average Importance: 0.0006439955056104353
Feature: -0.31 < Hurst.exponent <= 0.78, Average Importance: -0.0006310443708043329
Feature: CSI <= -0.75, Average Importance: -0.0006084870863590622
Feature: -0.76 < SymDp1_2 <= -0.44, Average Importance: 0.000563723769204122
Feature: Teo <= -0.56, Average Importance: -0.0005574267230240274
Feature: -0.40 < SymDp0_2 <= 0.43, Average Importance: -0.0005555324729096453
Feature: -0.90 < SymDfw_2 <= 0.06, Average Importance: -0.0005438290057806571
Feature: sgridWGT > 0.77, Average Importance: 0.0005394660459325069
Feature: -0.71 < HF.Power.LS <= -0.31, Average Importance: -0.0004991631759760705
Feature: SymDp0_2 > 0.72, Average Importance: -0.0004908164961433084
Feature: -0.37 < Teo <= 0.15, Average Importance: -0.00046658834263425635
Feature: -0.76 < CVI <= 0.06, Average Importance: -0.0004605776018764656
Feature: MultiFractal_c1 > 0.57, Average Importance: -0.0004535291657168324
Feature: -0.20 < CSI <= 0.51, Average Importance: 0.0004265994284559588
Feature: -0.20 < formF <= 0.47, Average Importance: 0.000380770323435826
Feature: 0.06 < CVI <= 0.79, Average Importance: -0.0003793641713160611
Feature: -0.62 < sgridTAU <= 0.24, Average Importance: 0.0003778230290126111
Feature: 0.10 < IoV <= 0.70, Average Importance: 0.0003293168482877767
Feature: -0.50 < DFA.Alpha.2 <= 0.15, Average Importance: 0.00029835157699922583
Feature: CVI > 0.79, Average Importance: 0.00028652037581144
Feature: 0.43 < SymDp0_2 <= 0.72, Average Importance: -0.00028619399441936767
Feature: SymDfw_2 > 0.83, Average Importance: 0.0002708461766054973
Feature: -0.85 < SymDse_2 <= -0.37, Average Importance: 0.00025915804752289827
Feature: -0.29 < PSeo <= -0.21, Average Importance: -0.00023237744910413553
Feature: SymDse_2 > 0.72, Average Importance: 0.00013739279418185848
Feature: -0.72 < Largest.Lyapunov.exponent <= -0.29, Average Importance: 0.00011203070732347859
Feature: -0.59 < sgridWGT <= 0.26, Average Importance: 0.00010998570047299757
Feature: 0.06 < SymDfw_2 <= 0.83, Average Importance: 9.043102534753107e-05
Feature: -0.62 < sedl <= -0.03, Average Importance: -8.722190464843975e-05
Feature: -0.50 < dlmax <= -0.33, Average Importance: -7.446321598142514e-05
Feature: -0.37 < SymDse_2 <= 0.72, Average Importance: 4.998764434935809e-05
Feature: histSI > 0.69, Average Importance: 4.6128849292825665e-05
Feature: SymDfw_2 <= -0.90, Average Importance: 4.140030372440885e-05
Feature: 0.07 < eScaleE <= 0.42, Average Importance: -2.0010295143118298e-06
"""
# Desired number of top features to display
n = 25

# Initialize empty lists to hold the extracted data
features = []
average_importances = []

# Split the data into individual lines for processing
data_lines = data_lines_xboost_high_precision.strip().split("\n")

# Loop through each line and extract the relevant parts
for line in data_lines:
    # Split the line by comma to separate feature from its importance
    parts = line.split(", Average Importance: ")
    # Extract feature name and average importance and store them
    feature = parts[0].replace("Feature: ", "")
    importance = float(parts[1])
    features.append(feature)
    average_importances.append(importance)

# Calculate the absolute importances
abs_importances = [abs(imp) for imp in average_importances]

# Combine features, original importances, and absolute importances
combined_data = list(zip(features, average_importances, abs_importances))

# Sort by absolute importances and select the top "n"
top_combined_data = sorted(combined_data, key=lambda x: x[2], reverse=True)[:n]

# Extract the top "n" features and their importances for plotting
top_features, top_importances, _ = zip(*top_combined_data)

# Sort the top "n" features for better visualization (optional, depending on preference)
importances_sorted, features_sorted = zip(*sorted(zip(top_importances, top_features)))

# Plot
plt.figure(figsize=(10, 8))
colors = ['red' if imp < 0 else 'green' for imp in importances_sorted]
plt.barh(features_sorted, importances_sorted, color=colors)
plt.xlabel("Average Importance")
plt.ylabel("Features")
#plt.title(f"Top {n} Feature Importances for Sepsis Prediction")
plt.axvline(x=0, color='grey', linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Adding legend to explain the colors
red_bar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
green_bar = plt.Rectangle((0,0),1,1,fc='green', edgecolor = 'none')
plt.legend([red_bar, green_bar], ['Negative Importance (Non-sepsis)', 'Positive Importance (Sepsis)'], loc='best', ncol=1, prop={'size':10})

plt.tight_layout()

plt.savefig('Fig7-explain-xgboost plot.png',format='png', dpi=300, bbox_inches='tight')
plt.show()

    
    Low Frequency/High Frequency (LF/HF) Ratio: The average importance of LF.HF.ratio.LS > 0.21 being positive (0.01893907150929673) underlines the significance of this feature in predicting sepsis, reinforcing the sentiment about its vital role in differentiating physiological responses leading to sepsis. This affirms the model's ability to recognize autonomic imbalances implicated in sepsis.

    Mean Rate: Interestingly, Mean.rate > 0.74 has a negative average importance (-0.02794918553667844), whereas -0.81 < Mean.rate <= -0.05 shows positive importance (0.020790602692575554). This inverse relationship requires context regarding the model's interpretability regarding mean rate values; however, it still points towards the mean rate being a factor the model takes into account significantly.

    Hurst Exponent and Complexity Levels: In this dataset, the Hurst exponent (Hurst.exponent <= -0.77 with negative importance and Hurst.exponent > 0.78 with positive importance) and Complexity (Complexity <= -0.75 with negative importance and -0.29 < Complexity <= 0.50 with positive importance) do show up, indicating their influence in the model predictions, although their effects are highly nuanced. Notably, both higher and lower thresholds of the Hurst exponent are considered, suggesting the model captures a broad spectrum of autonomic system behaviors.

    Asymmetry Index (AsymI): The presence of both AsymI <= -0.60 with positive importance and AsymI > 0.60 with negative importance highlights the model's attention to asymmetry in the autonomic nervous system's response, which can be relevant in physiological stress responses such as sepsis.

    Other Relevant Features: Features such as fFdP > 0.24 with positive importance and KLPE > 0.51 with negative importance also indicate the model’s consideration of various HRV measures.

    fFdP: This could represent a feature related to frequency domain analysis of HRV. In frequency domain measures of HRV, parameters such as LF (Low Frequency), HF (High Frequency), and VLF (Very Low Frequency) are common. "fFdP" might be a specific marker derived from such frequency domain analyses, possibly indicating a particular power density or a ratio.

    KLPE: This could stand for a measure related to entropy or non-linear dynamics within HRV analysis. In the realm of non-linear HRV metrics, measures such as sample entropy, approximate entropy, and multiscale entropy are used to assess the complexity of heart rate dynamics. "KLPE" might indicate a specific type of entropy measure, possibly related to Kullback-Leibler divergence (a measure of how one probability distribution diverges from a second, expected probability distribution) combined with permutation entropy (a type of entropy measure that assesses the complexity of a time series) or another complexity metric.

# Explanability for Neural Network

def keras_predict_proba_wrapper(model):
    def predict_proba(X):
        # Use model's predict method to get the predicted probabilities
        preds = model.predict(X)

        if model.output_shape[-1] == 1:  # Binary classification model
            # For binary classification, 'predict' might output a single probability per instance (assuming positive class)
            # We need to convert this to a two-dimensional array with probabilities for both classes (0 and 1)
            return np.hstack([1-preds, preds])
        else:  # Multiclass classification model
            # For multiclass, 'predict' should already output probabilities per class
            return preds
    return predict_proba



# Assuming precision, recall, and f1 are your custom metric functions defined somewhere
# You must have the actual functions used during model training available.
# For example, if you previously defined them like so:
# def precision(y_true, y_pred): ...
# def recall(y_true, y_pred): ...
# def f1(y_true, y_pred): ...

custom_objects = {
    "precision": precision,  # reference to your precision function
    "recall": recall,        # reference to your recall function
    "f1": f1                 # reference to your f1 function
}

model_save_path = '/content/neural_net_80_85_76_f1_pre_recall.h5'
loaded_model = load_model(model_save_path, custom_objects=custom_objects)
loaded_model_predict_proba = keras_predict_proba_wrapper(loaded_model)

# Assuming loaded_model is your loaded Keras model
loaded_model_predict_proba = keras_predict_proba_wrapper(loaded_model)

# Random test input, replace with appropriate shape and data for your model

test_input = np.random.rand(1, X_train.shape[1])  # Update with correct input shape
test_output = loaded_model_predict_proba(test_input)
print(test_output)  # Verify this prints expected predictions

X = new_df[boruta_bootstrap_features_sepsis3]
y = new_df['Sepsis3']
X_scaled, y = normalize_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, stratify=y, random_state=0)

#!pip install lime


def explain_model_with_lime_and_summarize(X_train, X_val, model_predict_proba, feature_names, num_instances=5):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['0', '1'],
        verbose=True,
        mode='classification'
    )

    # Initialize an empty dictionary to store feature contributions
    feature_contributions = {}

    # Explain multiple instances
    for idx_to_explain in range(min(num_instances, len(X_val))):
        print(f"Explaining instance {idx_to_explain+1}/{num_instances}")
        # Directly use the wrapped model prediction function
        lime_exp = explainer.explain_instance(X_val[idx_to_explain], model_predict_proba,
                                              num_features=len(feature_names))

        # Collect feature contributions for this instance
        for feature, contribution in lime_exp.as_list():
            if feature not in feature_contributions:
                feature_contributions[feature] = []
            feature_contributions[feature].append(contribution)

        # Optionally visualize each explanation
        # lime_exp.show_in_notebook(show_table=True, show_all=False)

    # Summarize the feature contributions
    feature_summary = {feature: np.mean(contributions) for feature, contributions in feature_contributions.items()}

    # Sorting features by their average contribution magnitude
    sorted_features = sorted(feature_summary.items(), key=lambda x: abs(x[1]), reverse=True)

    # Displaying summarized feature importance
    for feature, importance in sorted_features:
        print(f"Feature: {feature}, Average Importance: {importance}")

    return sorted_features

explanation_neural_summary = explain_model_with_lime_and_summarize(X_train, X_test, loaded_model_predict_proba,feature_names=bootstrap_features_sepsis3, num_instances=1079)

data_lines_neural = """
Feature: vlmax > 0.55, Average Importance: 0.1253770647743344
Feature: vlmax <= -0.73, Average Importance: -0.08294103194914076
Feature: shannEn > 0.24, Average Importance: 0.06568411512388186
Feature: dlmax > 0.10, Average Importance: 0.052426435480218844
Feature: Teo > 0.49, Average Importance: -0.05190921958826374
Feature: Power.Law.Slope.LS <= -0.80, Average Importance: 0.04949392737740783
Feature: Teo <= -0.73, Average Importance: 0.047450603754658253
Feature: shannEn <= -0.67, Average Importance: -0.045808087380886764
Feature: Poincar..SD2 > 0.36, Average Importance: 0.045283299060985514
Feature: -0.73 < vlmax <= -0.07, Average Importance: -0.0438083470177313
Feature: Power.Law.Slope.LS > 0.77, Average Importance: -0.03963938084876098
Feature: SymDfw_2 > 0.54, Average Importance: -0.03471436934353653
Feature: Power.Law.Y.Intercept.LS <= -0.67, Average Importance: 0.033519879170684004
Feature: SymDfw_2 <= -0.80, Average Importance: 0.03289265110318026
Feature: Power.Law.Y.Intercept.LS > 0.67, Average Importance: -0.03166075609222324
Feature: LF.HF.ratio.LS > 0.18, Average Importance: 0.022850146974873572
Feature: DFA.Alpha.1 > 0.79, Average Importance: 0.022813190575984903
Feature: Poincar..SD2 <= -0.68, Average Importance: -0.022733173173002285
Feature: DFA.Alpha.1 <= -0.82, Average Importance: -0.022387713124343553
Feature: -0.67 < shannEn <= -0.31, Average Importance: -0.022218403165407265
Feature: dlmax <= -0.56, Average Importance: -0.021935396794038862
Feature: sgridAND > 0.70, Average Importance: 0.02062944533998314
Feature: -0.56 < dlmax <= -0.40, Average Importance: -0.020265892600485412
Feature: Poincar..SD1 > 0.47, Average Importance: -0.019022398426215458
Feature: KLPE <= -0.60, Average Importance: 0.018489256010189987
Feature: -0.73 < Teo <= -0.25, Average Importance: 0.017689764666603033
Feature: PSeo > 0.72, Average Importance: 0.0175603687985103
Feature: 0.03 < Power.Law.Slope.LS <= 0.77, Average Importance: -0.01742170803416831
Feature: formF <= -0.21, Average Importance: 0.016642414185720062
Feature: -0.68 < Poincar..SD2 <= -0.30, Average Importance: -0.016289088310663186
Feature: LF.Power.LS <= -0.74, Average Importance: -0.015953485044508373
Feature: SymDce_2 <= -0.21, Average Importance: -0.015646144051601814
Feature: pR > 0.13, Average Importance: 0.0153854368904062
Feature: LF.Power.LS > 0.51, Average Importance: 0.015343327308908025
Feature: LF.HF.ratio.LS <= -0.58, Average Importance: -0.014385303097377903
Feature: formF > 0.38, Average Importance: -0.013797142183309768
Feature: KLPE > 0.80, Average Importance: -0.013451390131854335
Feature: Poincar..SD1 <= -0.78, Average Importance: 0.013432772064648017
Feature: AsymI > 0.76, Average Importance: 0.01306887832583203
Feature: pDpR > 0.70, Average Importance: -0.012947735921386592
Feature: aFdP > 0.09, Average Importance: 0.012794816068867522
Feature: -0.25 < Teo <= 0.49, Average Importance: -0.01277830010301212
Feature: sgridAND <= -0.77, Average Importance: -0.012409139131511478
Feature: pDpR <= -0.66, Average Importance: 0.012394825510802237
Feature: AsymI <= -0.60, Average Importance: -0.012350894733106434
Feature: -0.80 < SymDfw_2 <= -0.30, Average Importance: 0.011202154581694924
Feature: -0.30 < SymDfw_2 <= 0.54, Average Importance: -0.010639721171894776
Feature: -0.40 < dlmax <= 0.10, Average Importance: -0.010462787304356945
Feature: PSeo <= -0.52, Average Importance: -0.010008015349786136
Feature: SymDce_2 > 0.53, Average Importance: 0.009965647060022183
Feature: sgridWGT > 0.04, Average Importance: 0.009488058821962277
Feature: DFA.Alpha.2 > 0.60, Average Importance: -0.009056310318651816
Feature: Largest.Lyapunov.exponent > 0.61, Average Importance: 0.008939335516988257
Feature: sgridTAU > 0.79, Average Importance: 0.008898338060758815
Feature: SymDp2_2 > 0.77, Average Importance: 0.008525443910769634
Feature: -0.52 < PSeo <= 0.11, Average Importance: -0.008403005382097194
Feature: -0.58 < LF.HF.ratio.LS <= -0.31, Average Importance: -0.008344607541711107
Feature: 0.06 < Power.Law.Y.Intercept.LS <= 0.67, Average Importance: -0.008311294318949931
Feature: SymDp2_2 <= -0.76, Average Importance: -0.008107100681368645
Feature: VLF.Power.LS > 0.16, Average Importance: -0.007883383830118192
Feature: 0.23 < KLPE <= 0.80, Average Importance: -0.007835037854264404
Feature: Largest.Lyapunov.exponent <= -0.69, Average Importance: -0.0078089774831292865
Feature: -0.79 < Mean.rate <= -0.04, Average Importance: 0.007685696735054537
Feature: -0.67 < Power.Law.Y.Intercept.LS <= 0.06, Average Importance: 0.007657598178796706
Feature: DFA.AUC > 0.50, Average Importance: -0.0075893071507261295
Feature: -0.77 < sgridAND <= -0.05, Average Importance: -0.007559515214477178
Feature: sedl <= -0.66, Average Importance: -0.007517526085124832
Feature: sgridTAU <= -0.75, Average Importance: -0.007330951043198079
Feature: -0.80 < Power.Law.Slope.LS <= 0.03, Average Importance: 0.00711279192318747
Feature: SymDp1_2 > 0.50, Average Importance: 0.006922284922837415
Feature: -0.82 < DFA.Alpha.1 <= 0.04, Average Importance: -0.006904510273128678
Feature: 0.04 < DFA.Alpha.1 <= 0.79, Average Importance: 0.006697019255228817
Feature: -0.78 < Poincar..SD1 <= -0.28, Average Importance: 0.006682988624087525
Feature: -0.30 < Poincar..SD2 <= 0.36, Average Importance: -0.006194803374820711
Feature: pR <= -0.56, Average Importance: -0.006090135529346875
Feature: HF.Power.LS <= -0.71, Average Importance: -0.005636198922744274
Feature: Correlation.dimension > 0.71, Average Importance: -0.005302964950719381
Feature: -0.22 < LF.Power.LS <= 0.51, Average Importance: 0.004995425223829084
Feature: SymDp1_2 <= -0.76, Average Importance: -0.004939752364632237
Feature: -0.56 < pR <= -0.37, Average Importance: -0.004873377624561873
Feature: Mean.rate > 0.75, Average Importance: -0.004871610516193585
Feature: 0.22 < SymDce_2 <= 0.53, Average Importance: 0.004579184675103034
Feature: -0.74 < LF.Power.LS <= -0.22, Average Importance: -0.004465574770003066
Feature: Mean.rate <= -0.79, Average Importance: -0.004344637452651578
Feature: VLF.Power.LS <= -0.55, Average Importance: 0.004293544498393551
Feature: aFdP <= -0.33, Average Importance: -0.00423652350645943
Feature: -0.37 < pR <= 0.13, Average Importance: -0.004187148339042517
Feature: Complexity > 0.73, Average Importance: -0.004160804737501656
Feature: SymDse_2 > 0.59, Average Importance: 0.0040539851992305595
Feature: -0.28 < DFA.AUC <= 0.50, Average Importance: 0.003938174112837765
Feature: DFA.Alpha.2 <= -0.75, Average Importance: 0.003924007184780448
Feature: eScaleE > 0.03, Average Importance: -0.003918164026713625
Feature: Correlation.dimension <= -0.70, Average Importance: 0.0038529022290372215
Feature: -0.31 < shannEn <= 0.24, Average Importance: 0.0037211037005940722
Feature: 0.25 < AsymI <= 0.76, Average Importance: 0.0036345718889629025
Feature: -0.37 < sgridWGT <= 0.04, Average Importance: -0.0036146010105223753
Feature: -0.60 < AsymI <= 0.25, Average Importance: -0.003532260000898876
Feature: 0.17 < sedl <= 0.71, Average Importance: 0.003464986242988104
Feature: HF.Power.LS > 0.42, Average Importance: 0.003392043628680594
Feature: Complexity <= -0.54, Average Importance: 0.0033824933641221936
Feature: 0.15 < pDpR <= 0.70, Average Importance: -0.003345843039533469
Feature: -0.76 < SymDp2_2 <= 0.04, Average Importance: -0.003342872660323451
Feature: -0.07 < vlmax <= 0.55, Average Importance: 0.003214269168338275
Feature: -0.69 < Largest.Lyapunov.exponent <= -0.08, Average Importance: -0.003202449944112828
Feature: -0.75 < sgridTAU <= -0.30, Average Importance: -0.003173005579262572
Feature: fFdP > 0.67, Average Importance: -0.003134930076410565
Feature: -0.75 < DFA.Alpha.2 <= -0.14, Average Importance: 0.00312067779669261
Feature: -0.55 < VLF.Power.LS <= -0.32, Average Importance: 0.0030539324171276073
Feature: -0.74 < DFA.AUC <= -0.28, Average Importance: 0.0030192342472826375
Feature: -0.04 < Mean.rate <= 0.75, Average Importance: 0.002856582584794878
Feature: -0.30 < HF.Power.LS <= 0.42, Average Importance: 0.0028408855979420245
Feature: gcount <= -0.38, Average Importance: 0.002835718896331606
Feature: -0.53 < sgridWGT <= -0.37, Average Importance: -0.0028348102446688794
Feature: -0.66 < sedl <= 0.17, Average Importance: 0.0028092948885863962
Feature: sedl > 0.71, Average Importance: 0.002777796576930499
Feature: -0.66 < pDpR <= 0.15, Average Importance: 0.0027443029412525933
Feature: -0.29 < eScaleE <= -0.22, Average Importance: 0.002568253113443491
Feature: -0.60 < KLPE <= 0.23, Average Importance: 0.0025120952406094167
Feature: 0.16 < formF <= 0.38, Average Importance: -0.0023842984993593668
Feature: -0.70 < Correlation.dimension <= 0.03, Average Importance: 0.002182681953318282
Feature: sgridWGT <= -0.53, Average Importance: -0.0020791566680228744
Feature: -0.54 < Complexity <= 0.24, Average Importance: 0.0018331184165848804
Feature: eScaleE <= -0.29, Average Importance: 0.001632344712986447
Feature: -0.32 < VLF.Power.LS <= 0.16, Average Importance: 0.0016227005126558075
Feature: -0.50 < fFdP <= 0.15, Average Importance: 0.0015627924759911184
Feature: -0.28 < Poincar..SD1 <= 0.47, Average Importance: -0.0015348420125250826
Feature: fFdP <= -0.50, Average Importance: 0.0014597786806463402
Feature: -0.76 < SymDp1_2 <= -0.21, Average Importance: -0.001458881168693399
Feature: -0.30 < sgridTAU <= 0.79, Average Importance: 0.001456193561431758
Feature: -0.71 < HF.Power.LS <= -0.30, Average Importance: -0.001283600635235393
Feature: DFA.AUC <= -0.74, Average Importance: 0.001260956491946543
Feature: -0.22 < eScaleE <= 0.03, Average Importance: 0.0012549058758781948
Feature: 0.04 < SymDp2_2 <= 0.77, Average Importance: 0.0012527004466815958
Feature: -0.14 < DFA.Alpha.2 <= 0.60, Average Importance: 0.0012136684367033334
Feature: -0.21 < SymDce_2 <= 0.22, Average Importance: 0.0012078120475910205
Feature: -0.72 < SymDse_2 <= -0.13, Average Importance: -0.001174091587525308
Feature: gcount > 0.69, Average Importance: -0.0011567646987280075
Feature: SymDse_2 <= -0.72, Average Importance: -0.0011549498430252333
Feature: Coefficient.of.variation <= -0.69, Average Importance: 0.0011524624460828331
Feature: -0.31 < LF.HF.ratio.LS <= 0.18, Average Importance: -0.0010026596325997982
Feature: Coefficient.of.variation > 0.42, Average Importance: -0.0008710852943953131
Feature: -0.08 < Largest.Lyapunov.exponent <= 0.61, Average Importance: 0.000839825501269028
Feature: SymDp0_2 > 0.60, Average Importance: -0.0008317203947574415
Feature: 0.24 < Complexity <= 0.73, Average Importance: -0.0008227501302215453
Feature: 0.03 < Correlation.dimension <= 0.71, Average Importance: -0.000732665428641749
Feature: 0.11 < PSeo <= 0.72, Average Importance: 0.0006641988522635256
Feature: -0.38 < gcount <= 0.42, Average Importance: -0.0006553830225352877
Feature: -0.26 < Coefficient.of.variation <= 0.42, Average Importance: 0.0006505228699415441
Feature: -0.60 < SymDp0_2 <= -0.03, Average Importance: 0.0005466672934150132
Feature: -0.75 < IoV <= -0.21, Average Importance: 0.000533691100276781
Feature: -0.69 < Coefficient.of.variation <= -0.26, Average Importance: 0.0005239649899785489
Feature: -0.03 < SymDp0_2 <= 0.60, Average Importance: 0.0004957362562212113
Feature: 0.42 < gcount <= 0.69, Average Importance: -0.0004771960897561983
Feature: IoV <= -0.75, Average Importance: 0.00046594616666347394
Feature: -0.13 < SymDse_2 <= 0.59, Average Importance: -0.0004584642960215515
Feature: -0.21 < IoV <= 0.47, Average Importance: -0.0003966450640551144
Feature: -0.05 < sgridAND <= 0.70, Average Importance: 0.0003254859051035749
Feature: -0.21 < SymDp1_2 <= 0.50, Average Importance: -0.0003060341764406546
Feature: -0.33 < aFdP <= 0.09, Average Importance: 0.0003018018182293396
Feature: -0.21 < formF <= 0.16, Average Importance: 8.6663512871893e-05
Feature: SymDp0_2 <= -0.60, Average Importance: -5.746637792458876e-05
Feature: IoV > 0.47, Average Importance: 5.583008474426159e-05
Feature: 0.15 < fFdP <= 0.67, Average Importance: -2.207412864753984e-05
"""

# Desired number of top features to display
n = 25

# Initialize empty lists to hold the extracted data
features = []
average_importances = []

# Split the data into individual lines for processing
data_lines = data_lines_neural.strip().split("\n")

# Loop through each line and extract the relevant parts
for line in data_lines:
    # Split the line by comma to separate feature from its importance
    parts = line.split(", Average Importance: ")
    # Extract feature name and average importance and store them
    feature = parts[0].replace("Feature: ", "")
    importance = float(parts[1])
    features.append(feature)
    average_importances.append(importance)

# Calculate the absolute importances
abs_importances = [abs(imp) for imp in average_importances]

# Combine features, original importances, and absolute importances
combined_data = list(zip(features, average_importances, abs_importances))

# Sort by absolute importances and select the top "n"
top_combined_data = sorted(combined_data, key=lambda x: x[2], reverse=True)[:n]

# Extract the top "n" features and their importances for plotting
top_features, top_importances, _ = zip(*top_combined_data)

# Sort the top "n" features for better visualization (optional, depending on preference)
importances_sorted, features_sorted = zip(*sorted(zip(top_importances, top_features)))

# Plot
plt.figure(figsize=(10, 8))
colors = ['red' if imp < 0 else 'green' for imp in importances_sorted]
plt.barh(features_sorted, importances_sorted, color=colors)
plt.xlabel("Average Importance")
plt.ylabel("Features")
#plt.title(f"Top {n} Feature Importances for Sepsis Prediction [Neural Network Model]")
plt.axvline(x=0, color='grey', linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Adding legend to explain the colors
red_bar = plt.Rectangle((0,0),1,1,fc="red", edgecolor = 'none')
green_bar = plt.Rectangle((0,0),1,1,fc='green', edgecolor = 'none')
plt.legend([red_bar, green_bar], ['Negative Importance (Non-sepsis)', 'Positive Importance (Sepsis)'], loc='best', ncol=1, prop={'size':10})

plt.tight_layout()

plt.savefig("Fig8-Explain_Neural_Network.png" ,format='png', dpi=300, bbox_inches='tight')
plt.show()

X_test.shape


X = new_df[boruta_bootstrap_features_sepsis3]
y = new_df['Sepsis3']
X_scaled, y = normalize_data(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.75, stratify=y, random_state=0)

def precision(y_true, y_pred):
  y_pred = tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.float32)
  tp = tf.cast(tf.logical_and(y_true == 1, y_pred == 1), tf.float32)
  fp = tf.cast(tf.logical_and(y_true == 0, y_pred == 1), tf.float32)
  return tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fp) + 1e-4)

def recall(y_true, y_pred):
  y_pred = tf.cast(tf.math.greater_equal(y_pred, 0.5), tf.float32)
  tp = tf.cast(tf.logical_and(y_true == 1, y_pred == 1), tf.float32)
  fn = tf.cast(tf.logical_and(y_true == 1, y_pred == 0), tf.float32)
  return tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fn) + 1e-4)

def f1(y_true, y_pred):
  return 2 * precision(y_true, y_pred) * recall(y_true, y_pred) / (precision(y_true, y_pred) + recall(y_true, y_pred))

# Simple Neural Network Model (NN)
# Create the NN and fit it to the training data
hidden_units = 64
dropout_rate = 0.4

layers_dimensions = [hidden_units,hidden_units,hidden_units,hidden_units]
initializer = initializers.GlorotNormal(seed=2)
model_nn = keras.Sequential([
                            BatchNormalization(input_shape=[X_train.shape[1]]),
                            Dense(layers_dimensions[0],kernel_initializer=initializer,bias_initializer='zeros'),
                            BatchNormalization(),
                            Activation("relu"),
                            Dropout(rate=dropout_rate),
                            Dense(layers_dimensions[1],kernel_initializer=initializer,bias_initializer='zeros'),
                            BatchNormalization(),
                            Activation("relu"),
                            Dropout(rate=dropout_rate),
                            Dense(layers_dimensions[2],kernel_initializer=initializer,bias_initializer='zeros'),
                            BatchNormalization(),
                            Activation("relu"),
                            Dropout(rate=dropout_rate),


                            Dense(layers_dimensions[3],kernel_initializer=initializer,bias_initializer='zeros'),
                            BatchNormalization(),
                            Activation("relu"),



                            Dense(1,kernel_initializer=initializer,bias_initializer='zeros'),


                            Activation("sigmoid"),


])

optimizer = optimizers.Adam(learning_rate=5e-3)

model_nn.compile(optimizer="adam",loss="binary_crossentropy", metrics=[precision, recall, f1])



model_checkpoint = ModelCheckpoint(
    'best_model.h5', # Specify the path to save your model
    monitor='val_f1',                    # Monitor the validation F1 score
    mode='max',                          # The mode 'max' means the training will save the model when there's an increase
    save_best_only=True,                 # Only save a model when 'val_f1' has improved
    verbose=1)                           # Prints out messages when a model is being saved


# Calculate the weights for each class
weights = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)
class_weights = {i: weights[i] for i in range(len(weights))}

early_stopping = callbacks.EarlyStopping(monitor="val_f1", mode="max", min_delta=0.001, patience=550, restore_best_weights=True)

history_nn=model_nn.fit(
    x=X_train,
    y=y_train,
    batch_size=64,
    epochs=1400,
    verbose=2,
    validation_data = [X_test,y_test],
    callbacks=[early_stopping],
    class_weight=class_weights
)

history_nn_pd = pd.DataFrame(history_nn.history)
plt.plot(history_nn_pd["loss"])
plt.plot(history_nn_pd["val_loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(history_nn_pd["val_f1"])
plt.plot(history_nn_pd["val_f1"])
plt.xlabel("Epochs")
plt.ylabel("F1 score")
plt.show()

y_pred = model_nn.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred = y_pred >= 0.5
print("Neural Network Performance on Training Data:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Confusion Matrix: ")
plt.figure(figsize=(2, 2))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',cmap='Blues')
plt.show()


pd.DataFrame(history_nn_pd).to_csv("neural_net_80_85_76.csv")
"""
Neural Network Performance on Training Data:
Accuracy:  0.953660797034291
F1 Score:  0.8046874999999999
Precision:  0.8512396694214877
Recall:  0.762962962962963
Confusion Matrix : 926, 18, 32, 103
"""
# Save the model to a file
model_save_path = 'neural_net_80_85_76.h5' # After 400 epoch
model_nn.save(model_save_path)



# Load the CSV file into a pandas DataFrame
df_nn_history = pd.read_csv("neural_net_80_85_76.csv")

# Display the first few rows of the DataFrame
print(df_nn_history.head())



def plot_metrics(df, window_size=10):
    # Calculate F1 Scores for training and validation
    df['f1_score_training'] = (2 * df['precision'] * df['recall']) / (df['precision'] + df['recall'])
    df['f1_score_validation'] = (2 * df['val_precision'] * df['val_recall']) / (df['val_precision'] + df['val_recall'])

    # Apply smoothing with a rolling window
    df['loss_smooth'] = df['loss'].rolling(window=window_size).mean()
    df['val_loss_smooth'] = df['val_loss'].rolling(window=window_size).mean()
    df['f1_score_training_smooth'] = df['f1_score_training'].rolling(window=window_size).mean()
    df['f1_score_validation_smooth'] = df['f1_score_validation'].rolling(window=window_size).mean()

    # Creating plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Defining custom colors and line thickness
    color_train_loss = 'darkgreen' # Dark green for training loss
    color_val_loss = 'darkred'     # Dark red for validation loss
    color_train_f1 = 'tab:blue'
    color_val_f1 = 'tab:orange'
    line_thickness = 1.5  # Line thickness

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', color='k', fontweight='bold')  # Axis label in black for visibility
    ax1.plot(df.index, df['loss_smooth'], label='Training Loss', color=color_train_loss, linewidth=line_thickness)
    ax1.plot(df.index, df['val_loss_smooth'], label='Validation Loss', linestyle='--', color=color_val_loss, linewidth=line_thickness)
    ax1.tick_params(axis='y')

    # Creating a twin of ax1 for F1 Score with shared x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.plot(df.index, df['f1_score_training_smooth'], label='Training F1 Score', color=color_train_f1, linewidth=line_thickness)
    ax2.plot(df.index, df['f1_score_validation_smooth'], label='Validation F1 Score', linestyle='--', color=color_val_f1, linewidth=line_thickness)
    ax2.tick_params(axis='y')

    # Adding legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # Adding grid with dark color
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5, color='dimgray')

    # Adding title with bold
   # plt.title('Training/Validation Loss and F1 Score Across Epochs (Smoothed)', fontweight='bold')

    # Displaying the plot
    plt.savefig("Fig6-Training_NN.png",format='png', dpi=300, bbox_inches='tight')
    plt.show()


from keras.models import load_model

# Save the model to a file
model_save_path = 'neural_net_80_85_76.h5' # After 400 epoch
model_nn.save(model_save_path)


# Load the model from the file
loaded_model = load_model(model_save_path)

# Assuming X_test is available
y_pred = loaded_model.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred = y_pred >= 0.5