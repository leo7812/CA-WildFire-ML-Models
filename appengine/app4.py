# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

from google.cloud import storage
import os
from io import StringIO


# need to update this for website
# BLOB is an acronym for "Binary Large Object". It's a data type that stores binary data, such as images, videos, and audio.
def get_csv_from_gcs(bucket_name, source_blob_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    data = blob.download_as_text()
    return pd.read_csv(StringIO(data))


# Incorporate data
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

# Initialize the app
app = Dash()

# the underlying Flask server instance that Dash uses to run the application
# Many WSGI servers (e.g., Gunicorn, uWSGI) expect a server object.
# https://dash.plotly.com/deployment
server = app.server

# Cloud Storage demo
BUCKET_NAME = os.environ.get("BUCKET_NAME")
df2 = get_csv_from_gcs(BUCKET_NAME, 'customers-100-simple.csv')

# App layout
app.layout = html.Div([
    html.Div(children='My First App with Data, Graph, and Controls'),
    html.Hr(),
    dcc.RadioItems(options=['pop', 'lifeExp', 'gdpPercap'], value='lifeExp', id='controls-and-radio-item'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=6),
    dcc.Graph(figure={}, id='controls-and-graph'),
    html.Hr(),
    # this is a separate table for google cloud storage demo
    dash_table.DataTable(data=df.to_dict('records'), page_size=6) 
])

# Add controls to build the interaction
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_graph(col_chosen):
    fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
    return fig


# Run the app
if __name__ == '__main__':
    app.run(debug=True)