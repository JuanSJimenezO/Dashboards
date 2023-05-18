# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:49:52 2023

@author: Saint90
"""
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
import plotly.express as px
import statsmodels.api as sm
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import datetime as dt
from jupyter_dash import JupyterDash


df = pd.read_csv(r"C:/Users/Saint90/.kaggle/sample-sales-data/sales_data_sample.csv", encoding='latin1')
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'].apply(lambda x: pd.to_datetime(x).strftime('%Y-%m')))
    
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = JupyterDash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Sales Dashboard 2005'),
    html.Div(children='''
        Select a Country:
    '''),
    dcc.Dropdown(
        id='region-dropdown',
        options=[{'label': i, 'value': i} for i in df['COUNTRY'].unique()],
        value='USA'
    ),
    html.Div(children='''
        Select a City:
    '''),
    dcc.Dropdown(
        id='category-dropdown',
        # options=[{'label': i, 'value': i} for i in df['CITY'].unique()],
        value='NYC'
    ),
    html.Div([
        html.Div(
        className="row",
        children=[
            html.Div(
                className="six columns",
                children=[
                    html.Div(
                        children=dcc.Graph(id='sales-graph')
                    )
                ]
            ),
            html.Div(
                className="six columns",
                children=html.Div(
                    children=dcc.Graph(id='correlation-heatmap'),
                )
            )
        ]
    ),
    html.Div(
        className="row",
        children=[
            html.Div(
                className="twelve columns",
                children=[
                    html.Div(
                        children=dcc.Graph(id='bottom-bar-graph')
                    )
                ]
            )
        ]
    )
    ])
])
             
             
@app.callback(
    Output('category-dropdown', 'options'),
    Input('region-dropdown', 'value')
)
def update_category_dropdown(selected_region):
    # Obtener las opciones del segundo dropdown basado en la selección del primero
    filtered_cities = df[df['COUNTRY'] == selected_region]['CITY'].unique()
    options = [{'label': i, 'value': i} for i in filtered_cities]
    return options
             
             
                      
@app.callback(
    Output('sales-graph', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('category-dropdown', 'value')])

def update_graph(region, category):
    print(region, category)
    filtered_df = df[(df['COUNTRY'] == region) & (df['CITY'] == category)]
    sales_by_month = filtered_df.groupby('ORDERDATE')['SALES'].sum()
    
    # Fit ARIMA model to time series data
    model = ARIMA(sales_by_month, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=6)
    
    start_date = sales_by_month.index.max() + relativedelta(months=1)
    end_date =  start_date + relativedelta(months=len(forecast)-1)
    
    # Create date range for the forecast period
    range_dates = pd.date_range(start=start_date, end=end_date, freq='MS').tolist()
    
    # Combine actual and predicted sales data
    actual_sales = go.Scatter(
        x=sales_by_month.index,
        y=sales_by_month.values,
        name='Actual',
        marker=dict(color='blue')
    )
    predicted_sales = go.Scatter(
        x=range_dates,
        y=forecast,
        name='Predicted',
        marker=dict(color='red')
    )
    
    data = [actual_sales, predicted_sales]
    layout = go.Layout(
        title='Total Sales by Month',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Sales ($)')
    )
    
    return {'data': data, 'layout': layout}


@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('category-dropdown', 'value')])

def update_heatmap(region, category):
    filtered_df = df[(df['COUNTRY'] == 'USA') & (df['CITY'] ==  'NYC')]
   
    
    temp_df = pd.DataFrame()
    temp_df['Sales_product'] = filtered_df[['PRODUCTLINE','SALES']].groupby(['PRODUCTLINE'])['SALES'].sum()
    temp_df = temp_df.sort_values(by= 'Sales_product', ascending=False)
    temp_df = temp_df.index.tolist()
    for new_index in ['SALES', 'Small','Medium','Large']:
        temp_df.append(new_index)
   
    
    numeric_columns = temp_df
    
    
    filtered_df = pd.get_dummies(filtered_df, columns=['PRODUCTLINE','DEALSIZE'], prefix = '',prefix_sep='')
    
    # numeric_columns = filtered_df.select_dtypes(include=np.number).columns
    
    corr = filtered_df[numeric_columns].corr()
    
    fig = go.Figure(
        data=go.Heatmap(
            x=corr.columns,
            y=corr.columns,
            z=corr.values,
            colorscale='PuBu', 
            zmin=-1,
            zmax=1
        )
    )
    
    fig.update_layout(
        height=600,
        xaxis=dict(side='top')
    )
    
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8049)
