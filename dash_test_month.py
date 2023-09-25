import pandas as pd
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output  # pip install dash (version 2.0.0 or higher)
import calendar


app = Dash(__name__)

# -- Import and clean data (importing csv into pandas)
# df = pd.read_csv("intro_bees.csv")
#df = pd.read_csv("https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Other/Dash_Introduction/intro_bees.csv")

# #prepare dataframe
# MyColumns = ['State','Year','Week','Cases_per_week','Cum_Cases','SevereCases_per_week','Cum_SevereCases']
# ## load 2014 data
# Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2014.csv'
# dataframe_USA_2014 = pd.read_csv(Path)
# dataframe_USA_2014 = dataframe_USA_2014[['Reporting Area', 'MMWRYear','MMWRWeek','Dengue Fever†, Current week', 'Dengue Fever†, Cum 2014','Dengue Hemorrhagic Fever§, Current week', 'Dengue Hemorrhagic Fever§, Cum 2014']]
# dataframe_USA_2014.columns = MyColumns
# #dataframe_USA_2014

# ## 2015
# Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2015.csv'
# dataframe_USA_2015 = pd.read_csv(Path)
# #dataframe_USA_2015.columns
# dataframe_USA_2015 = dataframe_USA_2015[['Reporting Area', 'MMWRYear','MMWRWeek','Dengue§, Current week', 'Dengue§, Cum 2015','Dengue Severe, Current week', 'Dengue Severe, Cum 2015']]
# dataframe_USA_2015.columns = MyColumns
# #dataframe_USA_2015

# ## 2016
# Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2016.csv'
# dataframe_USA_2016 = pd.read_csv(Path)
# #dataframe_USA_2016.columns
# dataframe_USA_2016 = dataframe_USA_2016[['Reporting Area', 'MMWR Year','MMWR Week','Dengue§, Current week', 'Dengue§, Cum 2016','Dengue Severe, Current week', 'Dengue Severe, Cum 2016']]
# dataframe_USA_2016.columns = MyColumns
# #dataframe_USA_2016

# ## 2017
# Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2017.csv'
# dataframe_USA_2017 = pd.read_csv(Path)
# #dataframe_USA_2017.columns
# dataframe_USA_2017 = dataframe_USA_2017[['Reporting Area', 'MMWRYear','MMWRWeek','Dengue Virus Infection, Dengue§, Current week', 'Dengue Virus Infection, Dengue§, Cum 2017','Dengue Virus Infection, Severe Dengue, Current week', 'Dengue Virus Infection, Severe Dengue, Cum 2017']]
# dataframe_USA_2017.columns = MyColumns
# #dataframe_USA_2017

# ## 2018
# Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2018.csv'
# dataframe_USA_2018 = pd.read_csv(Path)
# #dataframe_USA_2018.columns
# dataframe_USA_2018 = dataframe_USA_2018[['Reporting Area', 'MMWRYear','MMWRWeek','Dengue Virus Infections, Dengue†, Current week', 'Dengue Virus Infections, Dengue†, Cum 2018','Dengue Virus Infections, Severe Dengue, Current week', 'Dengue Virus Infections, Severe Dengue, Cum 2018']]
# dataframe_USA_2018.columns = MyColumns
# #dataframe_USA_2018

# ## 2019
# Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2019.csv'
# dataframe_USA_2019 = pd.read_csv(Path)
# #dataframe_USA_2019.columns
# dataframe_USA_2019 = dataframe_USA_2019[['Reporting Area', 'MMWR Year','MMWR Week','Dengue virus infections§ , Dengue, Current week', 'Dengue virus infections§ , Dengue, Cum 2019†','Dengue virus infections§ , Severe dengue, Current week', 'Dengue virus infections§ , Severe dengue,  Cum 2019†']]
# dataframe_USA_2019.columns = MyColumns
# #dataframe_USA_2019

# ## 2020
# Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2020.csv'
# dataframe_USA_2020 = pd.read_csv(Path)
# #dataframe_USA_2020.columns
# dataframe_USA_2020 = dataframe_USA_2020[['Reporting Area', 'MMWR Year','MMWR Week','Dengue virus infections, Dengue, Current week', 'Dengue virus infections, Dengue, Cum 2020†','Dengue virus infections, Severe dengue, Current week', 'Dengue virus infections, Severe dengue,  Cum 2020†']]
# dataframe_USA_2020.columns = MyColumns
# #dataframe_USA_2020

# ## 2021
# Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2021.csv'
# dataframe_USA_2021 = pd.read_csv(Path)
# #dataframe_USA_2021.columns
# dataframe_USA_2021 = dataframe_USA_2021[['Reporting Area', 'MMWR Year','MMWR Week','Dengue virus infections, Dengue, Current week', 'Dengue virus infections, Dengue, Cum 2021†','Dengue virus infections, Severe dengue, Current week', 'Dengue virus infections, Severe dengue,  Cum 2021†']]
# dataframe_USA_2021.columns = MyColumns
# #dataframe_USA_2021

# ## 2022
# #Path = r'C:\Users\chateaux.m\Documents\DESU data science\Projet\Data\USA\2022.csv'
# #dataframe_USA_2022 = pd.read_csv(Path)
# #dataframe_USA_2022.columns
# #dataframe_USA_2022 = dataframe_USA_2022[['Reporting Area', 'MMWR Year','MMWR Week','Dengue virus infections, Dengue, Current week', 'Dengue virus infections, Dengue, Cum 2022†','Dengue virus infections, Severe dengue, Current week', 'Dengue virus infections, Severe dengue,  Cum 2022†']]
# #dataframe_USA_2022.columns = MyColumns
# #dataframe_USA_2022

# dataframe_USA = pd.concat([dataframe_USA_2014,dataframe_USA_2015,dataframe_USA_2016,dataframe_USA_2017,dataframe_USA_2018,dataframe_USA_2019,dataframe_USA_2020,dataframe_USA_2021], axis = 0)

# #remplacer les NaNs par des 0
# dataframe_USA_filled = dataframe_USA.fillna(0)
# dataframe_USA_filled

# #keep states informations only
# us_states = [
#     "ALABAMA", "ARIZONA", "ARKANSAS", "CALIFORNIA",
#     "COLORADO", "CONNECTICUT", "DELAWARE", "FLORIDA", "GEORGIA",
#     "IDAHO", "ILLINOIS", "INDIANA", "IOWA",
#     "KANSAS", "KENTUCKY", "LOUISIANA", "MAINE", "MARYLAND",
#     "MASSACHUSETTS", "MICHIGAN", "MINNESOTA", "MISSISSIPPI", "MISSOURI",
#     "MONTANA", "NEBRASKA", "NEVADA", "NEW HAMPSHIRE", "NEW JERSEY",
#     "NEW MEXICO", "NEW YORK", "NORTH CAROLINA", "NORTH DAKOTA", "OHIO",
#     "OKLAHOMA", "OREGON", "PENNSYLVANIA", "RHODE ISLAND", "SOUTH CAROLINA", 
#     "SOUTH DAKOTA", "TENNESSEE", "TEXAS", "UTAH", "VERMONT", "VIRGINIA",
#     "WASHINGTON", "WEST VIRGINIA", "WISCONSIN", "WYOMING",
#     "DISTRICT OF COLUMBIA"
# ]

# filtered_df = dataframe_USA_filled[dataframe_USA_filled['State'].isin(us_states)]
# #filtered_df.dtypes


# Year_dengue = filtered_df.groupby(['State', 'Year']).apply(lambda group: group[group['Week'] == group['Week'].max()])
# Year_dengue.reset_index(drop = True, inplace=True)

# #filtered2_df.columns[4] = 'Cases_per_year'
# #Year_dengue.drop(['Week','Cases_per_week','SevereCases_per_week'],axis = 1,inplace= True)
# Year_dengue.drop(['Week','Cases_per_week','SevereCases_per_week','Cum_SevereCases'],axis = 1,inplace= True)
# #Year_dengue.columns = ['State','Year','Cases','Severe_Cases']
# Year_dengue.columns = ['State','Year','Cases']
# #Year_dengue #penser à enlever 2022 car seulement 5 semaines prises en compte donc on ne peut pas dire qu'à la fin de la 5eme semaine on a les cas sur l'année

# #ajouter le code de l'état pour la représentation graphique

# # Dictionnaire de correspondance des noms d'États avec les codes
# state_codes = {
#     "ALABAMA":"AL", 
#     "ARIZONA":"AZ", 
#     "ARKANSAS":"AR", 
#     "CALIFORNIA":"CA",
#     "COLORADO":"CO",
#     "CONNECTICUT":"CT", 
#     "DELAWARE":"DE", 
#     "FLORIDA":"FL",
#     "GEORGIA":"GA",
#     "IDAHO":"ID", 
#     "ILLINOIS":"IL", 
#     "INDIANA":"IN", 
#     "IOWA":"IA",
#     "KANSAS":"KS", 
#     "KENTUCKY":"KY", 
#     "LOUISIANA":"LA", 
#     "MAINE":"ME", 
#     "MARYLAND":"MD",
#     "MASSACHUSETTS":"MA", 
#     "MICHIGAN":"MI", 
#     "MINNESOTA":"MN", 
#     "MISSISSIPPI":"MS", 
#     "MISSOURI":"MO",
#     "MONTANA":"MT", 
#     "NEBRASKA":"NE", 
#     "NEVADA":"NV", 
#     "NEW HAMPSHIRE":"NH", 
#     "NEW JERSEY":"NJ",
#     "NEW MEXICO":"NM", 
#     "NEW YORK":"NY", 
#     "NORTH CAROLINA":"NC", 
#     "NORTH DAKOTA":"ND", 
#     "OHIO":"OH",
#     "OKLAHOMA":"OK", 
#     "OREGON":"OR", 
#     "PENNSYLVANIA":"PA", 
#     "RHODE ISLAND":"RI", 
#     "SOUTH CAROLINA":"SC", 
#     "SOUTH DAKOTA":"SD",
#     "TENNESSEE":"TN", 
#     "TEXAS":"TX", 
#     "UTAH":"UT", 
#     "VERMONT":"VT", 
#     "VIRGINIA":"VA",
#     "WASHINGTON":"WA", 
#     "WEST VIRGINIA":"WV", 
#     "WISCONSIN":"WI", 
#     "WYOMING":"WY",
#     "DISTRICT OF COLUMBIA":"DC" 
# }

# # Ajouter une nouvelle colonne contenant les codes d'États correspondants
# Year_dengue['state_code'] = Year_dengue['State'].map(state_codes)


#Load previously save file instead of preparing it :

USA = pd.read_csv(r'C:\Users\chateaux.m\Documents\DESU_DS\Projet\Data\USA\USA_with_pred.csv')

#ajouter le code de l'état pour la représentation graphique

# Dictionnaire de correspondance des noms d'États avec les codes
state_codes = {
    "ALABAMA":"AL",
    "ALASKA":"AK", 
    "ARIZONA":"AZ", 
    "ARKANSAS":"AR", 
    "CALIFORNIA":"CA",
    "COLORADO":"CO",
    "CONNECTICUT":"CT", 
    "DELAWARE":"DE", 
    "FLORIDA":"FL",
    "GEORGIA":"GA",
    "HAWAII":"HI",
    "IDAHO":"ID", 
    "ILLINOIS":"IL", 
    "INDIANA":"IN", 
    "IOWA":"IA",
    "KANSAS":"KS", 
    "KENTUCKY":"KY", 
    "LOUISIANA":"LA", 
    "MAINE":"ME", 
    "MARYLAND":"MD",
    "MASSACHUSETTS":"MA", 
    "MICHIGAN":"MI", 
    "MINNESOTA":"MN", 
    "MISSISSIPPI":"MS", 
    "MISSOURI":"MO",
    "MONTANA":"MT", 
    "NEBRASKA":"NE", 
    "NEVADA":"NV", 
    "NEW HAMPSHIRE":"NH", 
    "NEW JERSEY":"NJ",
    "NEW MEXICO":"NM", 
    "NEW YORK":"NY", 
    "NORTH CAROLINA":"NC", 
    "NORTH DAKOTA":"ND", 
    "OHIO":"OH",
    "OKLAHOMA":"OK", 
    "OREGON":"OR", 
    "PENNSYLVANIA":"PA", 
    "RHODE ISLAND":"RI", 
    "SOUTH CAROLINA":"SC", 
    "SOUTH DAKOTA":"SD",
    "TENNESSEE":"TN", 
    "TEXAS":"TX", 
    "UTAH":"UT", 
    "VERMONT":"VT", 
    "VIRGINIA":"VA",
    "WASHINGTON":"WA", 
    "WEST VIRGINIA":"WV", 
    "WISCONSIN":"WI", 
    "WYOMING":"WY",
    "DISTRICT OF COLUMBIA":"DC" 
}

# Ajouter une nouvelle colonne contenant les codes d'États correspondants
USA['state_code'] = USA['State'].map(state_codes)


# print(np.sum(pd.isna(USA['state_code'])))

# Ajouter une nouvelle colonne contenant les codes ANSI correspondants
# state_ANSI = {
#     "ALABAMA":"01", 
#     "ARIZONA":"04", 
#     "ARKANSAS":"05", 
#     "CALIFORNIA":"06",
#     "COLORADO":"08",
#     "CONNECTICUT":"09", 
#     "DELAWARE":"10", 
#     "FLORIDA":"12",
#     "GEORGIA":"13",
#     "IDAHO":"16", 
#     "ILLINOIS":"17", 
#     "INDIANA":"18", 
#     "IOWA":"19",
#     "KANSAS":"20", 
#     "KENTUCKY":"21", 
#     "LOUISIANA":"22", 
#     "MAINE":"23", 
#     "MARYLAND":"24",
#     "MASSACHUSETTS":"25", 
#     "MICHIGAN":"26", 
#     "MINNESOTA":"27", 
#     "MISSISSIPPI":"28", 
#     "MISSOURI":"29",
#     "MONTANA":"30", 
#     "NEBRASKA":"31", 
#     "NEVADA":"32", 
#     "NEW HAMPSHIRE":"33", 
#     "NEW JERSEY":"34",
#     "NEW MEXICO":"35", 
#     "NEW YORK":"36", 
#     "NORTH CAROLINA":"37", 
#     "NORTH DAKOTA":"38", 
#     "OHIO":"39",
#     "OKLAHOMA":"40", 
#     "OREGON":"41", 
#     "PENNSYLVANIA":"42", 
#     "RHODE ISLAND":"44", 
#     "SOUTH CAROLINA":"45", 
#     "SOUTH DAKOTA":"46",
#     "TENNESSEE":"47", 
#     "TEXAS":"48", 
#     "UTAH":"49", 
#     "VERMONT":"50", 
#     "VIRGINIA":"51",
#     "WASHINGTON":"53", 
#     "WEST VIRGINIA":"54", 
#     "WISCONSIN":"55", 
#     "WYOMING":"56",
#     "DISTRICT OF COLUMBIA":"11" 
# }

# Year_dengue['ANSI'] = Year_dengue['State'].map(state_ANSI)
# USA['Normalized_cases'] = USA['Cases_per_month']/USA['Demographic']
# global_min = USA['Normalized_cases'].min()
# global_max = USA['Normalized_cases'].max()
global_min = USA['Cases_per_month'].min()
global_max = USA['Cases_per_month'].max()
month_names = list(calendar.month_name)[1:]


# Year_dengue_grouped = Year_dengue.groupby(['State', 'ANSI', 'Year','state_code']).sum()
# Year_dengue_grouped.reset_index(drop = False, inplace=True)
#print(Year_dengue_grouped[:1])

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),

    # dcc.Dropdown(id="slct_year",
    #              options=[
    #                  {"label": "2014", "value": 2014},
    #                  {"label": "2015", "value": 2015},
    #                  {"label": "2016", "value": 2016},
    #                  {"label": "2017", "value": 2017},
    #                  {"label": "2018", "value": 2018},
    #                  {"label": "2019", "value": 2019},
    #                  {"label": "2020", "value": 2020},
    #                  {"label": "2021", "value": 2021}],
    #              multi=False,
    #              value=2014,
    #              style={'width': "40%"}
    #              ),

    dcc.Slider(
            id='slct_year',
            min=USA['Year'].min(),
            max=USA['Year'].max(),
            step=1,
            marks={str(year): str(year) for year in range(USA['Year'].min(), USA['Year'].max() + 1)},
            value=2014,  # Initial value (you can change this to a default year)
        ),

    dcc.Slider(
        id='slct_month',
        min=USA['Month'].min(),
        max=USA['Month'].max(),
        step=1,
        # marks={str(month): str(month) for month in range(USA['Month'].min(), USA['Month'].max() + 1)}, 
        marks = {i: month_names[i - 1] for i in range(1, 13)},
        value=1,  # Initial value (you can change this to a default month)
        ),

    # html.Div([
    #     dcc.Slider(
    #         id='slct_year',
    #         min=USA['Year'].min(),
    #         max=USA['Year'].max(),
    #         step=1,
    #         marks={str(year): str(year) for year in range(USA['Year'].min(), USA['Year'].max() + 1)},
    #         value=2014,  # Initial value (you can change this to a default year)
    #     ),
    #     dcc.Slider(id='slct_month',
    #             min=1,
    #             max=12,
    #             step=1,
    #             marks={i: str(i) for i in range(1, 13)},
    #             value=1,  # Initial value (you can change this to a default month)
    #             ),
    # ], style={'display': 'flex', 'justify-content': 'space-between'}),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_dengue_map', figure={})

])


# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_dengue_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value'),
     Input(component_id='slct_month', component_property='value')]
)
def update_graph(option_slctd,month_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The year chosen by user was: {}".format(option_slctd)

    # dff = Year_dengue_grouped.copy()
    dff = USA.copy()
    dff = dff[(dff["Year"] == option_slctd) & (dff["Month"] == month_slctd)]
    #dff = dff[dff["Affected by"] == "Varroa_mites"]

    # Plotly Express
    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Cases_per_month',
        range_color=[0, 20],
        hover_data=['State', 'Cases_per_month'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Cases_per_month': 'Number of cases'},
        template='plotly_dark'
    )

    # Plotly Graph Objects (GO)
    # fig = go.Figure(
    #     data=[go.Choropleth(
    #         locationmode='USA-states',
    #         locations=dff['state_code'],
    #         z=dff["Cases"].astype(float),
    #         colorscale='Reds',
    #     )]
    # )
    #
    # fig.update_layout(
    #     title_text="Bees Affected by Mites in the USA",
    #     title_xanchor="center",
    #     title_font=dict(size=24),
    #     title_x=0.5,
    #     geo=dict(scope='usa'),
    # )

    return container, fig


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
