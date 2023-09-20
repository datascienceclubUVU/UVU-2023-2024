import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from pandasql import sqldf

st.set_page_config(page_title='UC Price Recommendation Engine')

@st.cache_data(ttl=3600)
def load_data(file_path):
    data = pd.read_parquet(file_path)
    return data

data = load_data('preprocessed_car_listings.parquet')


st.title('Used Car Price Recommendation Engine')

st.markdown(body="<h4>Description</h4><p>Welcome to the World's Most Basic AI-Driven Web App!</p>This Web App allows you to input your vehicle's features and you will receive a recommended listing price. Enjoy! :)", unsafe_allow_html=True)
st.text('')
st.text('')
st.header('Vehicle Description')
col1, col2, col3, col4 = st.columns([1,1,1,1])

with col1:
    year_selector = st.number_input(label='Model Year', min_value=2005, max_value=2024, value=2016)

with col2:
    makes = sqldf(f"SELECT DISTINCT make_x FROM data WHERE model_year_x = {year_selector}")
    make_selector = st.selectbox(label='Make', options=['Acura', 'Audi', 'BMW', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'Ford', 'GMC', 'Honda', 'Hyundai', 'INFINITI', 'Jeep', 'Kia', 'Lexus', 'Lincoln', 'MINI', 'Mazda', 'Mercedes-Benz', 'Mitsubishi', 'Nissan', 'Ram', 'Subaru', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo'], index=23)

with col3:
    models = sqldf(f"SELECT DISTINCT model_x FROM data WHERE model_year_x = {year_selector} AND make_x = '{make_selector}' ORDER BY model_x;")
    models = pd.Series(models['model_x'])
    models = models.tolist()
    model_selector = st.selectbox(label='Model', options=[model for model in models], index=1)

with col4:
    trims = sqldf(f"SELECT DISTINCT trim_x FROM data WHERE model_year_x = {year_selector} AND make_x = '{make_selector}' AND model_x = '{model_selector}' ORDER BY trim_x;")
    trims = pd.Series(trims['trim_x'])
    trims = trims.tolist()
    trim_selector = st.selectbox(label='Trim', options=[trim for trim in trims], index=1)

st.text('')
st.text('')
st.header('Technical Specifications')

col5, col6, col7, col8 = st.columns([1,1,1,1])

with col5:
    mileage_input = st.number_input(label='Mileage', min_value=0, max_value=170000, step=1000, value=96150)

with col6:
    accident_input = st.number_input(label="Number of Accidents", min_value=0, max_value=7, value=1)

with col7:
    owners_input = st.number_input(label='Number of Owners', min_value=1, max_value=9, value=2)

with col8:
    useage_selector = st.selectbox(label='Use Type', options=['Personal', 'Fleet'], index=0)

st.text('')
st.text('')
st.header('Additional Details')

col9, col10, col11, col12 = st.columns([1,1,1,1])

with col9:
    ext_color_selector = st.selectbox(label='Exterior Color', options=['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Red', 'Silver', 'White'], index=1)

with col10:
    int_color_selector = st.selectbox(label='Interior Color', options=['Black', 'Blue', 'Brown', 'Gray', 'Green', 'Red', 'White'], index=0)
# NH, ND, WY, ME
with col11:
    state_selector = st.selectbox(label='State', options=['AK', 'AL', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI'], index=37)

with col12:
    cities = sqldf(f"SELECT DISTINCT city_x FROM data WHERE state_x = '{state_selector}' ORDER BY city_x")
    cities = pd.Series(cities['city_x'])
    cities = cities.tolist()
    city_selector = st.selectbox(label='City', options=[city for city in cities], index=16)

@st.cache_resource
def build_model():
    X = data.drop(columns=['key_0', 'price_x', 'model_year_x', 'make_x', 'model_x', 'trim_x', 'mileage_x', 'num_accidents_x', 'num_owners_x', 'exterior_color_x', 'interior_color_x', 'city_x', 'state_x', 'price_y', 'usage_type_x'])
    y = data['price_x']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    used_car_predictor = DecisionTreeRegressor()
    used_car_predictor.fit(X_train, y_train)
    return used_car_predictor

make = sqldf(f"SELECT DISTINCT make_y FROM data WHERE make_x = '{make_selector}'")
model = sqldf(f"SELECT DISTINCT model_y FROM data WHERE model_x = '{model_selector}'")
trim = sqldf(f"SELECT DISTINCT trim_y FROM data WHERE trim_x = '{trim_selector}'")
usage = sqldf(f"SELECT DISTINCT usage_type_y FROM data WHERE usage_type_x LIKE '%{useage_selector}%'")
exterior = sqldf(f"SELECT DISTINCT exterior_color_y FROM data WHERE exterior_color_x LIKE '%{ext_color_selector}%'")
interior = sqldf(f"SELECT DISTINCT interior_color_y FROM data WHERE interior_color_x LIKE '%{int_color_selector}%'")
state = sqldf(f"SELECT DISTINCT state_y FROM data WHERE state_x LIKE '%{state_selector}%'")
city = sqldf(f"SELECT DISTINCT city_y FROM data WHERE city_x = '{city_selector}'")

a = [year_selector, pd.Series(make['make_y'])[0], pd.Series(model['model_y'])[0], pd.Series(trim['trim_y'])[0], mileage_input, accident_input, owners_input, pd.Series(usage['usage_type_y'])[0], pd.Series(exterior['exterior_color_y'])[0], pd.Series(interior['interior_color_y'])[0], pd.Series(city['city_y'])[0], pd.Series(state['state_y'])[0]]
a = np.array(a) # convert to a numpy array
a = np.expand_dims(a, 0)
predicted_price = build_model().predict(a)
predicted_price = int(predicted_price[0])

col11, col12, col13, col14 = st.columns([2,1,1,1])
with col11:
    prediction_btn = st.button(label='Get Recommendation')
with col14:
    clear_btn = st.button(label='Clear Output')

if prediction_btn:
    st.markdown("<h3>Recommended Listing Price: <br><br> <h2>${:,}</h2>".format(predicted_price), unsafe_allow_html=True)
elif clear_btn:
    st.markdown("")