from flask import Flask, request, jsonify, render_template, session
import pickle
import pandas as pd
from sklearn.impute import SimpleImputer
import dash
from dash import dcc, html
import plotly.express as px

# Load the preprocessing steps
with open('preprocessing_model_final.pkl', 'rb') as file1:
    preprocessing_steps = pickle.load(file1)

def preprocess_input(input_data):
    # Convert 'Distance(mi)' column to numeric
    input_data['Distance(km)'] = pd.to_numeric(input_data['Distance(mi)'], errors='coerce') * 1.60934
    input_data['Temperature(C)'] = ((pd.to_numeric(input_data['Temperature(F)'], errors='coerce') - 32) * 5 / 9).round(1)
    input_data['Wind_Chill(C)'] = ((pd.to_numeric(input_data['Wind_Chill(F)'], errors='coerce') - 32) * 5 / 9).round(1)
    input_data['Visibility(km)'] = (pd.to_numeric(input_data['Visibility(mi)'], errors='coerce') * 1.60934).apply(lambda x: f'{x:.2f}')
    input_data['Wind_Speed(kph)'] = (pd.to_numeric(input_data['Wind_Speed(mph)'], errors='coerce') * 1.60934).apply(lambda x: f'{x:.2f}')
    
    # Add new columns and preprocess them
    input_data['Start_Lat'] = pd.to_numeric(input_data['Start_Lat'], errors='coerce')
    input_data['Start_Lng'] = pd.to_numeric(input_data['Start_Lng'], errors='coerce')
    input_data['Start_Hour'] = pd.to_numeric(input_data['Start_Hour'], errors='coerce')
    input_data['End_Hour'] = pd.to_numeric(input_data['End_Hour'], errors='coerce')
    input_data['Weather_Encoded'] = pd.to_numeric(input_data['Weather_Encoded'], errors='coerce')
     
    # Drop the original columns
    columns_to_drop = ['Distance(mi)', 'Temperature(F)', 'Wind_Chill(F)', 'Visibility(mi)', 'Wind_Speed(mph)']
    input_data.drop(columns=columns_to_drop, inplace=True) 
    # Include boolean columns
    boolean_columns = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    for col in boolean_columns:
        # If column is not present, assume it's unchecked (0)
        input_data[col] = input_data.get(col, 0)

    # Convert 'Traffic_Signal' to numerical (0 or 1)
    input_data['Traffic_Signal'] = input_data['Traffic_Signal'].apply(lambda x: 1 if x.lower() == 'on' else 0)
    
    time_mapping = {'Day': 1, 'Night': 0}
    time_columns = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
    for col in time_columns:
        input_data[col] = input_data[col].map(time_mapping)

    input_data.reset_index(drop=True, inplace=True)
    return input_data

# Load the trained model
with open('prediction_model_final.pkl', 'rb') as file2:
    model = pickle.load(file2)

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Initialize the Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')

# Define Dash layout
# Update the Dash layout in your Flask app
dash_app.layout = html.Div([
    html.H1("USA Map"),
    dcc.Graph(id='map-graph', figure={}),
    html.Div(id='hidden-div', style={'display': 'none'})  # Hidden div to trigger callback
])

# Define Dash callback to update map
@dash_app.callback(
    dash.dependencies.Output('map-graph', 'figure'),
    [dash.dependencies.Input('hidden-div', 'children')]
)
def update_map(_):
    lat = session.get('latitude')
    lon = session.get('longitude')
    if lat is not None and lon is not None:
        # Create a DataFrame with latitude and longitude
        map_data = {'latitude': [float(lat)], 'longitude': [float(lon)]}
        map_df = pd.DataFrame(map_data)

        # Create map figure using Plotly Express
        fig = px.scatter_mapbox(map_df, lat='latitude', lon='longitude', zoom=3 , color_discrete_sequence=['red'])
        fig.update_layout(mapbox_style="carto-positron")  # Change map style if needed
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})  # Remove margins
        return fig
    else:
        # Return empty figure if no coordinates are provided
        return {'data': [], 'layout': {}}

# Define route for home page
@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    input_data = request.form.to_dict()

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess input data
    input_df = preprocess_input(input_df.copy())

    # Select relevant features
    X_input = input_df[['Start_Lat', 'Start_Lng', 'Start_Hour', 'End_Hour', 'Weather_Encoded','Temperature(C)', 'Wind_Chill(C)', 'Visibility(km)', 'Wind_Speed(kph)', 'Humidity(%)', 'Pressure(in)', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Precipitation(in)', 'Distance(km)']]

    # Impute missing values with the mean
    # Identify numerical columns
    numerical_columns = ['Distance(km)', 'Temperature(C)', 'Wind_Chill(C)', 'Visibility(km)', 'Wind_Speed(kph)']

    # Select numerical and non-numerical columns
    numerical_data = input_df[numerical_columns]
    non_numerical_data = input_df.drop(columns=numerical_columns)

    # Impute missing values with the mean for numerical columns
    imputer = SimpleImputer(strategy='mean')
    numerical_data_imputed = pd.DataFrame(imputer.fit_transform(numerical_data), columns=numerical_data.columns)

    # Concatenate imputed numerical columns with non-numerical columns
    X_input_imputed = pd.concat([non_numerical_data, numerical_data_imputed], axis=1)

    # Make prediction
    prediction = model.predict(X_input_imputed)

    # Store latitude and longitude in session variables
    session['latitude'] = input_data['Start_Lat']
    session['longitude'] = input_data['Start_Lng']

    # Return the result along with latitude and longitude
    return render_template('result2.html', prediction=prediction, latitude=input_data['Start_Lat'], longitude=input_data['Start_Lng'])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
