import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
clcc ={111:'cont. urban fabric',
 112:'disc urban fabric',
 121:'industrial or commercial units',
 122:'road and rail',
 123:'port areas',
 124:'airports',
 131:'mineral extraction sites',
 132:'dump sites',
 133:'construction sites',
 141:'green urban areas',
 142:'sport and leasure',
 211:'non irregated arable land',
 212:'permenant irregated land',
 213:'rice fields',
 221:'vine yards',
 223:'olive groves',
 231:'pastures',
 241:'annual with perm. crops',
 242:'complex cultivation patters',
 243:'land principally occupied by agriculture',
 244:'agro forest areas',
 311:'broad leaved forest',
 312:'conferous forest',
 313:'mixed forest',
 321:'natural grassland',
 322:'moors and heathland',
 323: 'scierohllous vegitation',
 324:'transitional woodland shrub',
 331: 'beaches dunes and sand plains',
 332:'bare rock',
 333:'sparsely vegetated areas',
 334:'burnt areas',
 335:'glaciers and perpetual snow',
 411:'inland marshes',
 412:'peat bogs',
 421:'salt marshes',
 422:'salines',
 423:'intertidal flats',
 511:'water courses',
 512:'water bodies',
 521:'costal lagoons',
 522:'estuaries',
 523:'sea and ocean'}

# Assuming you have loaded the training data into X_train and target labels into y_train
feature_arr = np.load('feature_array.npy', allow_pickle=True)
target_arr = np.load('target_labels.npy', allow_pickle= True)
feature_arr.shape
X_train = feature_arr.reshape(-1,43)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
y_train = target_arr.reshape(-1)
# Create the logistic regression model for binary classification
logreg_binary = LogisticRegression()

# Train the model on the training data
logreg_binary.fit(X_train, y_train)

# Get the coefficients for class labeled as 1 (positive class)
# Get the coefficients for class labeled as 1 (positive class)
coefficients_class_1 = logreg_binary.coef_[0]
feature_names = []
for feature, coefficient in zip(clcc.values(), coefficients_class_1):
    feature_names.append(feature)
# Get the coefficients for class labeled as 0 (negative class)


# Get the feature names (assuming you have them stored in a list called feature_names)
# Sort the coefficients in descending order
sorted_indices = np.argsort(coefficients_class_1)[::-1]
sorted_coefficients = coefficients_class_1[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

# Get the top 3 features with the highest positive impact
top_3_positive_features = sorted_feature_names[:3]
top_3_positive_coefficients = sorted_coefficients[:3]

# Get the top 3 features with the highest negative impact
top_3_negative_features = sorted_feature_names[-3:]
top_3_negative_coefficients = sorted_coefficients[-3:]

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
def create_bar_plot(features, coefficients, title):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefficients, y=features, palette='viridis')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title(title)
    plt.grid(axis='x')
    
    # Convert the Seaborn plot to a Plotly figure
    plotly_fig = go.Figure(data=[go.Bar(x=coefficients, y=features, orientation='h', marker=dict(color=coefficients, colorscale='viridis'))])
    return plotly_fig

# Define the layout of the dashboard with the background image container

# Define the layout of the dashboard with the background image container
app.layout = html.Div(children=[
    html.Div(style={'background-image': 'url("/assets/wilderness.jpg")', 'height': '100vh', 'width': '100%'},
             children=[
                 html.H1("Logistic Regression Coefficients Dashboard", style={'text-align': 'center', 'padding-top': '30px'}),
                 
                 html.H2("Top 3 Features contributing to wilderness:", style={'text-align': 'center'}),
                 dcc.Graph(
                     figure=create_bar_plot(top_3_positive_features, top_3_positive_coefficients, 'Top 3 Features - Positive Impact')
                 ),
                 
                 html.H2("Top 3 Features contributing to Non-wilderness:", style={'text-align': 'center'}),
                 dcc.Graph(
                     figure=create_bar_plot(top_3_negative_features, top_3_negative_coefficients, 'Top 3 Features - Negative Impact')
                 ),
             ]
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)