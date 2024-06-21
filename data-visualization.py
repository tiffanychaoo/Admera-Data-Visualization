#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import statements
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback, no_update
from functools import reduce     

import base64
import io

#import json
#import dash_html_components as html
#from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform


# In[2]:


def file_to_dataframe(file):
    if 'txt' in file:
        df = pd.read_csv(file, sep='\t\s*', header=0, engine='python', index_col=False)
    else:
        df = pd.read_csv(file, header=0, index_col=False)
    return df


# In[3]:


# cleaning strings of quotations marks
def strip_quotes(dataframe):
    if 'cell_barcode' in dataframe.columns:
        dataframe['cell_barcode'] = dataframe['cell_barcode'].str.strip('\"')
    if 'cell_type' in dataframe.columns:
        dataframe['cell_type'] = dataframe['cell_type'].str.strip('\"')
        
    return dataframe


# In[4]:


def rehead(dataframe):   
    for column in dataframe.columns:
        if column =='samples':
            dataframe.rename(columns={'samples':'sample'}, inplace=True)

        start = 1 if column.startswith('\"') else 0
        end = -1 if column.endswith('\"') else len(column)
        start += 5 if column[start:end].startswith('data.') else 0
        end -= 2 if column[start:end].endswith('.x') or column.endswith('.y') else 0

        if column[start:end] =='Row.names' or column[start:end]=='Unnamed: 0':
            dataframe.rename(columns={column: "cell_barcode"}, inplace=True)
        else:
            dataframe.rename(columns={column: column[start:end]}, inplace=True)

    dataframe = dataframe.loc[:,~dataframe.columns.duplicated()].copy()
    
    return strip_quotes(dataframe)


# In[5]:


#file1 = 'testYF.coordinate.umap.txt'
#file2 = 'cell_type_barcode_percent.mt-meta.data.csv'
#file3 = 'testYF.sample.txt'
#file4 = 'percent.mt.sample.csv'
#file5 = '23132-01-aggrGEX-cell_type_barcode_percent.mt-meta.data-CombineCoordinates.csv'
#file6 = '23132-01-all-markers.txt'

#df1 = file_to_dataframe(file1)
#df2 = file_to_dataframe(file2)
#df3 = file_to_dataframe(file3)
#df4 = file_to_dataframe(file4)
#df5 = file_to_dataframe(file5)
##df6 = file_to_dataframe(file6)

#df3 = rehead(df3)
#df4 = rehead(df4)
#df5 = rehead(df5)
#df6 = rehead(df6)

#df_list = [df1, df2]
#df_list2 = [df3, df4]


# In[6]:


# combining multiple dataframes
def combine_dataframes(list_of_dataframes):
    dataframes = []
    for dataframe in list_of_dataframes:
        reheaded_df = rehead(dataframe)
        dataframes.append(reheaded_df)

    # combining the dataframes (unsure if this works with more than two dataframes) 
    merged_df = reduce(lambda left,right: pd.merge(left,right, how='inner'), dataframes)
    return merged_df


# In[7]:


# grouping by specific category
def group_samples(dataframe, group='sample'): 
    grouped_dict = dict(tuple(dataframe.groupby(by=group)))
    grouped_dict['all'] = dataframe
    return grouped_dict


# In[8]:


def recombine_samples(dataframe, keys):
    grouped = group_samples(dataframe)
    if len(keys) == 1:
        return grouped[keys[0]]
    else:
        dataframes = []
        for key in keys:
            dataframes.append(grouped[key])
        concatenated_df = pd.concat(dataframes, axis=0)
        return concatenated_df


# In[9]:


# creating plotly express scatterplot
def create_scatter(dataframe, keys, scatter_type, sort_by):
    temp_df = recombine_samples(dataframe, keys).sort_values(by=[sort_by], ascending=True)
    temp_df[sort_by] = dataframe[sort_by].astype(str)
    
    proj_id = temp_df['orig.ident'].unique()[0]
    
    xaxis = 'tSNE_1' if scatter_type == 'tSNE' else 'UMAP_1'
    yaxis = 'tSNE_2' if scatter_type == 'tSNE' else 'UMAP_2'
    
    fig = px.scatter(temp_df, 
                     x=xaxis, 
                     y=yaxis, 
                     color=sort_by, 
                     color_discrete_map=dict(zip(temp_df[sort_by].unique(), px.colors.qualitative.Light24)),
                     opacity=0.75, 
                     title= scatter_type + ' (samples: ' + ', '.join(str(key) for key in keys) + ') <br><sup> Project ID: ' + proj_id + '<br> Click or hover on a point for more information </sup>',
                     hover_name='cell_barcode', 
                     hover_data=[column for column in dataframe.columns], 
                     custom_data=[sort_by, 'sample'])
                     #category_orders={sort_by: temp if sort_by != 'cell_type' else np.sort(temp_df[sort_by].unique())}) # this is not sorting the legend items numerically
    fig.update_traces(marker={'size':4})
    fig.update_layout(legend= {'itemsizing': 'constant', 
                               'itemclick':'toggle', 
                               'itemdoubleclick':'toggleothers'},
                      font_family='helvetica',
                      #hoverlabel= {'bordercolor':'white'},
                      clickmode="event+select")
    fig.update_xaxes(categoryorder='category ascending') # this is not sorting the legend items numerically
    
    return fig


# In[10]:


# creating plotly express boxplot
def create_boxplot(dataframe, keys, dependent_var, c_dataframe, sort_by):
    temp_df = recombine_samples(dataframe, keys).sort_values(by=[sort_by], ascending=True)
    temp_df[sort_by] = dataframe[sort_by].astype(str)
    
    color_df = recombine_samples(c_dataframe, keys).sort_values(by=[sort_by], ascending=True)
    color_df[sort_by] = color_df[sort_by].astype(str)
    
    proj_id = temp_df['orig.ident'].unique()[0]
    
    fig = px.box(
        data_frame = temp_df, 
                 x=dependent_var, 
                 y=sort_by, 
                 color=sort_by,
                 color_discrete_map=dict(zip(color_df[sort_by].unique(), px.colors.qualitative.Light24)),
                 orientation='h', 
                 boxmode='overlay', 
                 title= str(dependent_var) + ' (samples: ' + ', '.join(str(key) for key in keys) + ') <br><sup> Project ID: ' + proj_id + '</sup>',
                 hover_name='cell_barcode', 
                 hover_data=[column for column in dataframe.columns],
                 category_orders={sort_by : np.sort(dataframe[sort_by].unique())},
                 points="all"
                )
    fig.update_traces(marker={'size':4})
    fig.update_layout(legend= {'itemsizing': 'constant', 
                               'itemclick':'toggle', 
                               'itemdoubleclick':'toggleothers'}, 
                      font_family='helvetica',
                      #hoverlabel= {'bordercolor':'white'},
                      clickmode="event+select")
    fig.update_xaxes(categoryorder='category ascending')

    return fig


# In[11]:


def create_bar(dataframe, sort_by):
    temp_df = dataframe.sort_values(by=[sort_by], ascending=True)
    temp_df[sort_by] = dataframe[sort_by].astype(str)
    categories = temp_df[sort_by].unique()
    grouped_df = group_samples(temp_df)
    
    barchart = go.Figure()

    for sample_num in dataframe['sample'].unique():
        category_percentages = []
        for category in categories:
            numerator = len(grouped_df[sample_num][grouped_df[sample_num][sort_by] == category])
            denominator = len(grouped_df[sample_num])
            percentage = numerator/denominator
            category_percentages.append(percentage)

        barchart.add_trace(go.Bar(x=categories, y=category_percentages, name='Sample ' + str(sample_num)))

    barchart.update_layout(
        title=sort_by + ' percentages by sample',
        yaxis_title='percentage',
        font_family='helvetica')
           
    return barchart

#create_bar(df5, 'cell_type')


# In[12]:


# creating plotly graph objects table (for displaying information about selected point(s)
def create_table(dataframe, c_dataframe, sort_by):
    temp_df = dataframe.sort_values(by=[sort_by], ascending=True)
    temp_df[sort_by] = dataframe[sort_by].astype(str)
    
    color_df = c_dataframe.sort_values(by=[sort_by], ascending=True)
    color_df[sort_by] = color_df[sort_by].astype(str)
    
    c = dict(zip(color_df[sort_by].unique(), px.colors.qualitative.Light24))
    fig = go.Figure(
        data = [go.Table(
            columnwidth=[3, 2.5, 1.8, 1.8, 2.7, 0.75, 1, 1, 1.5, 2, 2, 2, 2],
            header=dict(values=list(dataframe.columns),
                        fill_color=c[temp_df.head(1)[sort_by][temp_df.head(1)[sort_by].index[0]]],
                        #font_color='ghostwhite',
                        font_family='helvetica',
                        align='center'),
            cells=dict(values=[dataframe[value] for value in dataframe],
                       fill_color='light grey',
                       font_family='helvetica',
                       align='center'))])
    return fig


# In[13]:


def create_pies(dataframe, keys, sort_by):
    temp_df = recombine_samples(dataframe, keys).sort_values(by=[sort_by], ascending=True)
    temp_df[sort_by] = dataframe[sort_by].astype(str)
    
    grouped_sample = group_samples(temp_df)
    
    sector_color = dict(zip(temp_df[sort_by].unique(), px.colors.qualitative.Light24))

    data = []
    for i in range(0, len(keys)):
        #labels = grouped_sample[keys[i]][sort_by].astype(int).sort_values(),
        labels = grouped_sample[keys[i]][sort_by]
        trace = go.Pie(labels= labels,
                       name= 'Sample ' + str(keys[i]), 
                       title='Sample ' + str(keys[i]),
                       direction='clockwise',
                       marker_colors=labels.map(sector_color),
                       textfont_family = 'helvetica',
                       textfont_color = 'white',
                       textposition ='inside',
                       domain=dict(x=[0, 0.5] if i%2==0 else [0.5, 1], y=[0, 0.5] if i//2<1 else [0.5, 1]),
                       sort=True)
        data.append(trace)
        
    layout = go.Layout(title=sort_by + ' percentages by sample', font_family='helvetica')    
    pie = go.Figure(data=data, layout=layout)

    return pie


# In[14]:


# APP LAYOUT
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, title='Data Visualization', external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.Hr(),
    html.Div([
        # INTRODUCTION & INSTRUCTIONS
        dcc.Markdown('''
            ### UMAP and tSNE Visualization
            Upload CSV, TXT, or Excel (XLS) files. 
            When uploading multiple files, they must be uploaded all at once. 
            This can be done by clicking and dragging the mouse over a group of files, 
            or alternatively, by pressing and holding shift or control then selecting multiple files.
            Ensure that the cell barcodes match and that there are as many column names in the header as 
            there are columns of data (for txt files only).
            
        '''),
    
        # FILE UPLOAD AREA
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '95%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=True
        ) 
    ]),

    html.Hr(),
    
    # UPLOAD OUTPUT: store and uploaded filenames
    html.Div(id='upload-data-output'),

    # STORED DATAFRAME
    dcc.Store(id='upload-data-store', storage_type='session'),

    
    # PERCENTAGE CHARTS
    html.Div([
        # PIE CHART (L)
        html.Div([
            dcc.Graph(id='pie-chart')
        ], style={'width': '48%', 'float':'left'}),
        
        # BAR CHART (R)
        html.Div([
            dcc.Graph(id='bar-chart'),
        ], style={'width': '48%', 'float': 'right'}),

        
    ], style={'width':'95%', 'display':'inline-block'}),
    
    html.Hr(),
    
    # SECOND HEADER
    dcc.Markdown('''
        #### Crosslinked Graphs
        Click or hover over points on the graphs for more information.
        '''),

    html.Hr(),
    
    # VARIABLE SELECTION
    # SORT BY DROPDOWN
    html.Div([
        html.Label('Sort by: ', style={'display':'inline-block', 'float':'left', 'padding-bottom':'10px'}),
        html.Div(id='sort-by-dropdown', style={'width': '95%', 'padding-bottom':'10px'}),
        html.Br(),
    ]),
    
    # CHECKLIST FOR SAMPLE TYPES
    html.Div([
        html.Label('Sample #', style={'display':'inline-block', 'float':'left', 'padding-bottom':'10px'}),
        html.Div(id='linked-sample-type', style={'width': '60%', 'float': 'left', 'padding-bottom':'10px'}),
        html.Br(),
    ]),
    
    html.Div([
         # DROPDOWN MENU FOR SCATTER
        html.Div(id='dropdown-menu-L', style={'width': '45%', 'float': 'left'}),
        
        # DROPDOWN MENU FOR BOXPLOT
        html.Div(id='dropdown-menu-R', style={'width': '45%', 'float': 'right'}),
        
    ], style={'width':'95%', 'display':'inline-block'}),
    
   
    
    # GRAPHS
    html.Div([
        # SCATTERPLOT GRAPH (L)
        html.Div([
            dcc.Graph(id='linked-scatterplot')
        ], style={'width': '48%', 'float':'left'}),
        
        # BOXPLOT GRAPH (R)
        html.Div([
            dcc.Graph(id='linked-boxplot'),
        ], style={'width': '48%', 'float': 'right'}),

       
    ], style={'width':'95%', 'display':'inline-block'}),

    
    # TABLE FOR SELECTED POINT
    html.Div([
            dcc.Graph(id='linked-table'),
        ], style={'width': '95%', 'float':'left'}),
    
    
], style={'width': '95%', 'padding-right' : 20, 'padding-left' : 20})


# In[15]:


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), header=0)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t\s*', header=0, engine='python')
    except Exception as e: ## TODO not sure that this works 
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return filename, df


# In[16]:


@app.callback(
    Output('upload-data-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'))

def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        list_of_dataframes = [parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]
        
        if len(list_of_dataframes) > 1:
            dataframes = []
            for pair in list_of_dataframes:
                dataframes.append(pair[1])
            unjsonified_df = combine_dataframes(dataframes)
        else:
            unjsonified_df = rehead(list_of_dataframes[0][1])
        
        return unjsonified_df.to_json(orient="split", index=False, path_or_buf=None)


# In[17]:


# CREATE SORT BY DROPDOWN
@app.callback(
    Output('sort-by-dropdown', 'children'),
    Input('upload-data-store', 'data'))

def create_sort_dropdown(jsonified_df):
    df = pd.read_json(jsonified_df, orient='split')
    sort_options = []
    
    for column_name in df.columns:
        if column_name in ['cell_type', 'seurat_clusters', 'Cluster', 'cluster']:
            sort_options.append(column_name)
    
    return dcc.Dropdown(
            options=[{'label' : option, 'value' : option} for option in sort_options],
            value=sort_options[0],
            id='sort-option',
        )


# In[18]:


# CREATE SAMPLE TYPE CHECKBOXES
@app.callback(
    Output('linked-sample-type', 'children'),
    Input('upload-data-store', 'data'))

def create_checkbox(jsonified_df):
    df = pd.read_json(jsonified_df, orient='split')
    sample_types = df['sample'].unique()
    sample_types.sort()
    
    return dcc.Checklist(
            options=[{'label' : str(sample_type), 'value' : int(sample_type)} for sample_type in sample_types],
            value = sample_types,
            labelStyle={'display':'inline-block', 'marginTop': '8px'},
            style={'display':'inline'},
            id='sample-type'
        )


# In[19]:


# CREATE LEFT DROPDOWN MENU
@app.callback(
    Output('dropdown-menu-L', 'children'),
    Input('upload-data-store', 'data'))

def create_dropdown_L(jsonified_df):
    df = pd.read_json(jsonified_df, orient='split')
    columns = []
    if 'UMAP_1' in df.columns:
        columns.append('UMAP')
    if 'tSNE_1' in df.columns:
        columns.append('tSNE')
    
    return dcc.Dropdown(
            options=[{'label' : column, 'value' : column} for column in columns],
            value=columns[0],
            id='scatter-map-type',
        )


# In[20]:


# CREATE RIGHT DROPDOWN MENU
@app.callback(
    Output('dropdown-menu-R', 'children'),
    Input('upload-data-store', 'data'))

def create_dropdown_R(jsonified_df):
    df = pd.read_json(jsonified_df, orient='split')
    columns = []
    
    for column_name in df.columns:
        if column_name not in ['UMAP_1', 'UMAP_2', 'tSNE_1', 'tSNE_2', 'orig.ident', 'cell_barcode', 'cell_type', 'sample']:
            columns.append(column_name)
    
    return dcc.Dropdown(
            options=[{'label' : column, 'value' : column} for column in columns],
            value=columns[0],
            id='boxplot-dependent-variable',
        )


# In[21]:


# CREATE PLOTS
@app.callback(
    Output('linked-scatterplot', 'figure', allow_duplicate=True),
    Output('linked-boxplot', 'figure', allow_duplicate=True),
    Input('upload-data-store', 'data'), # jsonified df
    Input('sample-type', 'value'), # sample types
    Input('boxplot-dependent-variable', 'value'), # boxplot dependent variable
    Input('scatter-map-type', 'value'), # scatter map type (UMAP or tSNE)
    Input('sort-option', 'value'), # color sort
    prevent_initial_call=True)

def update_sample_type(jsonified_df, sample_keys, dependent_var, scatter_type, color_sort):
    df = pd.read_json(jsonified_df, orient='split')

    if len(sample_keys) == len(df['sample'].unique()):
        scatter = create_scatter(df, ['all'], scatter_type, color_sort)
        boxplot = create_boxplot(df, ['all'], dependent_var, df, color_sort)
    else:
        scatter = create_scatter(df, sample_keys, scatter_type, color_sort)
        boxplot = create_boxplot(df, sample_keys, dependent_var, df, color_sort)

    return scatter, boxplot


# In[22]:


# updating boxplot based on dependent variable selected
@app.callback(
    Output('linked-boxplot', 'figure', allow_duplicate=True),
    Input('upload-data-store', 'data'),
    Input('boxplot-dependent-variable', 'value'),
    State('sort-option', 'value'),
    prevent_initial_call=True)

def update_dependent_var(jsonified_df, dependent_var, color_sort):
    df = pd.read_json(jsonified_df, orient='split')
    
    if dependent_var=='sample': # fix this
        return create_bar(df, color_sort)
    else:
        return create_boxplot(df, ['all'], dependent_var, df, color_sort)


# In[23]:


# updating boxplot based on point clicked
@app.callback(
    Output('linked-boxplot', 'figure', allow_duplicate=True),
    Output('linked-table', 'figure'),
    Input('linked-scatterplot', 'clickData'),
    Input('upload-data-store', 'data'),
    State('boxplot-dependent-variable', 'value'),
    State('sort-option', 'value'),
    prevent_initial_call=True)

def display_click_data(clickData, jsonified_df, dependent_var, color_sort):
    df = pd.read_json(jsonified_df, orient='split')
    
    cell_barcode = clickData['points'][0]['hovertext']
    sort_type = clickData['points'][0]['customdata'][0]
    sample_type = clickData['points'][0]['customdata'][1]
    
    if sort_type.isdecimal():
        dff = df[df[color_sort] == int(sort_type)] 
    else:
        dff = df[df[color_sort] == sort_type]
    
    dfff = df[df['cell_barcode'] == cell_barcode]    
    cell_info_table = create_table(dfff, df, color_sort)
  
    selected_boxplot = create_boxplot(dff, [sample_type], dependent_var, df, color_sort)
    selected_boxplot.update_traces(marker={'size':8})
    
    return selected_boxplot, cell_info_table


# In[24]:


@app.callback(
    Output('pie-chart', 'figure'),
    Output('bar-chart', 'figure'),
    Input('upload-data-store', 'data'),
    Input('sort-option', 'value'))

def display_percentage_charts(jsonified_df, color_sort):
    df = pd.read_json(jsonified_df, orient='split')
    samples = [key for key in group_samples(df).keys()]
    return create_pies(df, samples, color_sort), create_bar(df, color_sort)


# In[ ]:


if __name__=='__main__':
    app.run(port=8050)

