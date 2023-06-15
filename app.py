from dash import Dash, Output, Input, html, dcc, callback, dash_table
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns

# Import data
file_path ='Data_Solar.csv'
df = pd.read_csv(file_path,sep=';', parse_dates=['Timestamp'])

# Create a dictionary to map old column names to new names
new_names = {
    '00886A - VNM - Nafoco - Inverter 1 - Current AC [A] - I_AC': 'I_AC',
    '00886A - VNM - Nafoco - Inverter 1 - Direct Current [A] - I_DC': 'I_DC',
    '00886A - VNM - Nafoco - Inverter 1 - AC Voltage [V] - U_AC': 'U_AC',
    '00886A - VNM - Nafoco - Inverter 1 - Voltage DC [V] - U_DC': 'U_DC',
    '00886A - VNM - Nafoco - Basics - Irradiance on module plane [W/m²] - G_M0': 'Irrandiance',
    '00886A - VNM - Nafoco - Basics - Module Temperature [°C] - T_MODULE': 'T_MODULE'
}

# Rename the columns using the dictionary
df = df.rename(columns=new_names)

# Transform data
df = df.dropna()
df['P_DC'] = df['I_DC']*df['U_DC']
df['P_AC'] = df['I_AC']*df['U_AC']

df['hourfloat'] = df.Timestamp.dt.hour + df.Timestamp.dt.minute/60.0
df['hour_min_sin'] = np.sin(2.*np.pi*df.hourfloat/24.)
df['hour_min_cos'] = np.cos(2.*np.pi*df.hourfloat/24.)
df = df.drop(['hourfloat'], axis=1)

# Split train and test 
df_train_set, df_test_set = train_test_split(df, test_size=0.1, random_state=2023)
df2_train_set = df_train_set.drop(['Timestamp'], axis=1)

# Standardize data based on mean & sd of whole data
# Transform data train
df2_train = df_train_set.drop(['Timestamp'], axis=1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df2_train)
df2_train_std = pd.DataFrame(scaled_data, columns=df2_train.columns)

# Transform data test 
df2_test = df_test_set.drop(['Timestamp'], axis=1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df2_test)
df2_test_std = pd.DataFrame(scaled_data, columns=df2_test.columns)

one_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma=0.4)
one_svm.fit(df2_train_std)

pre_test_outliers = one_svm.predict(df2_test_std)
df_test_set['Predict'] = pre_test_outliers
df_test_outliers = df_test_set[(df_test_set['Predict'] == -1)]
df_test_outliers = df_test_outliers.drop(['hour_min_cos', 'hour_min_sin', 'Predict'], axis=1)

# Define the outliers_indices based on the given conditions
outliers_indices = (df_test_set['Predict'] == -1)
outliers = df_test_set[outliers_indices & (df_test_set['Irrandiance'] > 200) & (df_test_set['P_AC'] < 50000)]

# Visuzlize outliers chart
fig_outlier = go.Figure()
fig_outlier.add_trace(go.Scatter(
    x=df_test_set['P_AC'],
    y=df_test_set['Irrandiance'],
    mode='markers',
    name='Data',
    text=df_test_set['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
))
fig_outlier.add_trace(go.Scatter(
    x=outliers['P_AC'],
    y=outliers['Irrandiance'],
    mode='markers',
    name='Outliers',
    marker=dict(color='red'),
    text=outliers['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
))

# Update layout options
fig_outlier.update_layout(
    hovermode='closest',
    title='Outliers Chart',
    xaxis_title='Irrandiance',
    yaxis_title='P_AC'
)

# Set hover text template
fig_outlier.update_traces(hovertemplate='Timestamp: %{text}<br>Irrandiance: %{y}<br>P_AC: %{x}')

# Visuzlize chart by date
if 'Hour' not in df.columns:
    df['Hour'] = df['Timestamp'].dt.time
    df = df[['Hour'] + list(df.columns[:-1])]

# Filter the DataFrame by date
date_selected = '2022-01-01'
df_chart = df[df['Timestamp'].dt.date == pd.to_datetime(date_selected).date()]
# Convert Hour column to string
df_chart['Hour'] = df_chart['Hour'].astype(str)
# Subset the hour values
subset_hour_values = df_chart['Hour'].unique()[::15]
# Create traces for each column
traces = []
for column in df_chart.columns:
    if column != 'Timestamp':
        trace = go.Scatter(
            x=df_chart['Hour'],
            y=df_chart[column],
            mode='lines',
            name=column
        )
        traces.append(trace)
# Set the layout for the chart
layout = go.Layout(
    # title=date_selected,
    xaxis=dict(
        title='Hour',
        ticktext=subset_hour_values,
        tickvals=subset_hour_values,
        tickangle=45
    ),
    yaxis=dict(
        title='Value',
        tickfont=dict(size=15)
    ),
    showlegend=True
)
# Create the figure
fig = go.Figure(data=traces, layout=layout)


# Initialize the app
app = Dash(__name__)

# Set up the Dash application layout
app.layout = html.Div(children=[
    html.H3("Select a date:"),
    dcc.DatePickerSingle(
        id='date-picker',
        date=date_selected
    ),
    dcc.Graph(
        id='chart',
        figure=fig
    ),
    dcc.Graph(
        id='scatter-plot',
        figure=fig_outlier
    )
])

# Define callback for updating the chart based on the selected date
@app.callback(
    Output('chart', 'figure'),
    [Input('date-picker', 'date')]
)
def update_chart(date):
    # Filter the DataFrame by date
    df_chart = df[df['Timestamp'].dt.date == pd.to_datetime(date).date()]
    # Convert Hour column to string
    df_chart['Hour'] = df_chart['Hour'].astype(str)
    # Subset the hour values
    subset_hour_values = df_chart['Hour'].unique()[::15]
    # Create traces for each column
    traces = []
    for column in df_chart.columns:
        if column != 'Timestamp':
            trace = go.Scatter(
                x=df_chart['Hour'],
                y=df_chart[column],
                mode='lines',
                name=column
            )
            traces.append(trace)
    # Set the layout for the chart
    layout = go.Layout(
        title='Chart on ' + date,
        xaxis=dict(
            title='Hour',
            ticktext=subset_hour_values,
            tickvals=subset_hour_values,
            # tickangle=45
        ),
        yaxis=dict(
            title='Value',
            tickfont=dict(size=15)
        ),
        showlegend=True
    )
    # Create the updated figure
    fig = go.Figure(data=traces, layout=layout)
    return fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True,host='0.0.0.0', port='8050')