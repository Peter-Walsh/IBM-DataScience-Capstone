# Import required libraries
import pandas as pd
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()
# launch_sites = set(spacex_df['Launch Site'].to_list())


def get_all_scatterplot():
    return px.scatter(data_frame=spacex_df.astype({'Payload Mass (kg)': np.int32}), 
    x='Payload Mass (kg)', y='class', color='Booster Version Category')


def get_one_scatterplot(launch_site_name):
    scatter_data = spacex_df[spacex_df['Launch Site'] == launch_site_name]
    return px.scatter(data_frame=scatter_data.astype({'Payload Mass (kg)': np.int32}), 
    x='Payload Mass (kg)', y='class', color='Booster Version Category')


def get_all_pie_charts():
    pie_data = spacex_df.groupby(by='Launch Site').sum()['class'].reset_index()
    return px.pie(data_frame=pie_data, values='class', names='Launch Site')


def get_one_pie_chart(launch_site_name):
    launch_site = spacex_df['Launch Site'] == launch_site_name
    pie_data = spacex_df[launch_site].groupby(by='class').count().reset_index()
    return px.pie(data_frame=pie_data, values="Flight Number", names="class")


# Create a dash application
app = dash.Dash(__name__)

app_title = html.H1('SpaceX Launch Records Dashboard', style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40})

launch_options = [
    {"label": "All", "value": "ALL"},
    {"label": "CCAFS LC-40", "value": "CCAFS LC-40"},
    {"label": "CCAFS SLC-40", "value": "CCAFS SLC-40"},
    {"label": "KSC LC-39A", "value": "KSC LC-39A"},
    {"label": "VAFB SLC-4E", "value": "VAFB SLC-4E"}
]

launch_dropdown_dropdown = dcc.Dropdown(id='site-dropdown',
                                          options=launch_options, 
                                          placeholder='Select a Launch Site',
                                          style={'width':'80%', 
                                                 'padding':'3px', 
                                                 'font-size': '20px', 
                                                 'text-align-last' : 'center'})

launch_dropdown_title = html.Div([html.H2('Launch Site:')])

launch_dropdown = html.Div(
    [launch_dropdown_title, launch_dropdown_dropdown, html.Br(),
    html.Div([
        html.Div([ ], id='success-pie-chart')
    ], style={'display': 'flex'})])
        

range_slider = dcc.RangeSlider(id='payload-slider', min=0, max=10000,
                step=1000, value=(min_payload, max_payload))

scatter_graph = html.Div([
    html.Div([ ], id='success-payload-scatter-chart')
])

payload_graph = html.Div([range_slider, html.Br(), scatter_graph])

# Create an app layout
app.layout = html.Div(children=[app_title, launch_dropdown, 
html.Br(), payload_graph])
    

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(
    [Output(component_id='success-pie-chart', component_property='children')],
    [Input(component_id='site-dropdown', component_property='value')],
)
def get_graph(launch_site):
    if launch_site == 'ALL':
        pie_fig = get_all_pie_charts()
    else:
        pie_fig = get_one_pie_chart(launch_site)
    
    return [dcc.Graph(figure=pie_fig)]

# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(
    [Output(component_id='success-payload-scatter-chart', component_property='children')],
    [Input(component_id='site-dropdown', component_property='value'),
    Input(component_id='payload-slider', component_property='value')],
    # Input(component_id='payload-slider', component_property='value')],
)
def get_scatterplot(launch_site, payload_mass):
    if launch_site == 'ALL':
        scatter_fig = get_all_scatterplot()
    else:
        scatter_fig = get_one_scatterplot(launch_site)

    scatter_fig.update_xaxes(range=[payload_mass[0], payload_mass[1]])
    
    return [dcc.Graph(figure=scatter_fig)]

# Run the app
if __name__ == '__main__':
    app.run_server()
