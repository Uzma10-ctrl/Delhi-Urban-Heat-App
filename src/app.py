import streamlit as st
import pandas as pd
import joblib
import os
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import seaborn as sns
import requests

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


# ---------- CONFIG ----------
st.set_page_config(page_title="Delhi Urban Heat Smart App", layout="wide")

# ---------- TITLE ----------
st.markdown("""
<h1 style="text-align:center; font-size:50px; font-weight:800;">
🌇 Delhi Urban Heat Smart App
</h1>
<p style="text-align:center; font-size:18px;">
Live Heat Risk Prediction • AQI • Maps • Smart Insights
</p>
""", unsafe_allow_html=True)

st.divider()

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "delhi_heat_aqi_satellite.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "heat_aqi_model.pkl")

# ---------- LOAD ----------
data = load_data(DATA_PATH)
model = load_model(MODEL_PATH)


area_list = data["area"].unique().tolist()

features = [
    "day_temp",
    "night_temp",
    "humidity",
    "wind_speed",
    "built_up",
    "green_cover",
    "aqi",
    "ndvi",
    "lst"
]


# ---------- SIDEBAR ----------
if "view" not in st.session_state:
    st.session_state.view = "Prediction"

st.sidebar.markdown(
    "<h3 style='text-align:center; font-weight:800;'>🔎DASHBOARD </h3>",
    unsafe_allow_html=True
)


if st.sidebar.button("🔮 Prediction", use_container_width=True):
    st.session_state.view = "Prediction"

if st.sidebar.button("🗺 Map", use_container_width=True):
    st.session_state.view = "Map"

if st.sidebar.button("📊 Charts", use_container_width=True):
    st.session_state.view = "Charts"

if st.sidebar.button("🌡 Heat Retention", use_container_width=True):
    st.session_state.view = "About Heat Retention"

if st.sidebar.button("🛡 Precautions", use_container_width=True):
    st.session_state.view = "Precautions"

if st.sidebar.button("🚨 Emergency", use_container_width=True):
    st.session_state.view = "Emergency"

view = st.session_state.view




# ---------- LIVE LOCATION ----------
def get_live_aqi(lat, lon):
    try:
        url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token=demo"
        r = requests.get(url, timeout=4).json()
        return r["data"]["aqi"]
    except:
        return None


# ---------- PREDICTION ----------
if view == "Prediction":
    st.subheader("📍 Select Your Area for Heat Prediction")

    # Dropdown for live location tracking
    selected_area = st.selectbox("Choose Area:",area_list)
    loc_data = data[data["area"] == selected_area].iloc[0]

    st.write(f"**Region:** {loc_data['region']}")
    st.write(f"**Coordinates:** {loc_data['latitude']}, {loc_data['longitude']}")

    # Live AQI
    aqi_live = get_live_aqi(loc_data['latitude'], loc_data['longitude'])
    if aqi_live:
        st.success(f"Live AQI detected: {aqi_live}")
    else:
        st.warning("Live AQI unavailable. Using dataset value.")
        aqi_live = loc_data['aqi']

    st.subheader("🎚 Adjust Parameters (Optional)")
    c1, c2, c3 = st.columns(3)
    with c1:
        day_temp = st.slider("Day Temp (°C)", 30, 50, int(loc_data['day_temp']))
        night_temp = st.slider("Night Temp (°C)", 20, 40, int(loc_data['night_temp']))
        humidity = st.slider("Humidity (%)", 30, 80, int(loc_data['humidity']))
    with c2:
        wind_speed = st.slider("Wind Speed (m/s)", 0.5, 5.0, float(loc_data['wind_speed']))
        built_up = st.slider("Built-up Index", 0.0, 1.0, float(loc_data['built_up']))
        green_cover = st.slider("Green Cover", 0.0, 1.0, float(loc_data['green_cover']))
    with c3:
        ndvi = st.slider("NDVI", 0.0, 1.0, float(loc_data['ndvi']))
        lst = st.slider("LST (°C)",min_value=30.0,max_value=50.0,value=float(loc_data["lst"]),step=0.1)
        aqi = st.slider("AQI", 50, 500, int(aqi_live))

    if st.button("🔍 Predict Heat Risk"):
        input_data = [[day_temp, night_temp, humidity, wind_speed, built_up, green_cover, aqi, ndvi, lst]]
        pred = model.predict(input_data)[0]

        result = {0:("Low Heat Risk 🟢","#2ecc71"),
                  1:("Moderate Heat Risk 🟡","#f1c40f"),
                  2:("High Heat Risk 🔴","#e74c3c")}
        


        proba = model.predict_proba(input_data)[0]
        st.progress(float(max(proba)))
        st.caption(f"Prediction confidence: {max(proba)*100:.1f}%")


        st.markdown(f"""
        <div style="background:{result[pred][1]};
        padding:20px;border-radius:12px;
        text-align:center;font-size:26px;color:white;">
        {result[pred][0]}
        </div>
        """, unsafe_allow_html=True)





# ---------- MAP ----------
elif view == "Map":
    st.subheader("🗺 Delhi Heat Distribution")

    m = folium.Map(location=[28.64, 77.21], zoom_start=11)

    for _, r in data.iterrows():
        folium.CircleMarker(
            [r["latitude"], r["longitude"]],
            radius=10,
            popup=f"{r['area']} | Heat Level: {r['heat_retention_level']}",
            color=["green","orange","red"][r["heat_retention_level"]],
            fill=True
        ).add_to(m)

    st_folium(m, width=1000, height=520)

# ---------- CHARTS ----------
elif view == "Charts":
    st.subheader("📊 Explore Heat Data Charts")

    chart_type = st.selectbox(
        "Select Chart Type:",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart"]
    )

    if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
        x_axis = st.selectbox("X-axis:", ["area", "aqi", "built_up", "green_cover", "ndvi", "lst", "day_temp", "night_temp"])
        y_axis = st.selectbox("Y-axis:", ["heat_retention_level", "day_temp", "night_temp", "aqi", "ndvi"])

    fig, ax = plt.subplots(figsize=(10,5))

    if chart_type == "Bar Chart":
        sns.barplot(x=x_axis, y=y_axis, data=data, ax=ax)
        ax.set_title(f"{y_axis} vs {x_axis}", fontsize=16)
        ax.set_xlabel(x_axis, fontsize=12)
        ax.set_ylabel(y_axis, fontsize=12)
        plt.xticks(rotation=45)
    elif chart_type == "Line Chart":
        sns.lineplot(x=x_axis, y=y_axis, data=data, marker="o", ax=ax)
        ax.set_title(f"{y_axis} vs {x_axis}", fontsize=16)
        ax.set_xlabel(x_axis, fontsize=12)
        ax.set_ylabel(y_axis, fontsize=12)
        plt.xticks(rotation=45)
    elif chart_type == "Scatter Plot":
        sns.scatterplot(x=x_axis, y=y_axis, data=data, hue="heat_retention_level", palette=["green","orange","red"], s=100, ax=ax)
        ax.set_title(f"{y_axis} vs {x_axis}", fontsize=16)
        ax.set_xlabel(x_axis, fontsize=12)
        ax.set_ylabel(y_axis, fontsize=12)
        plt.xticks(rotation=45)
    elif chart_type == "Pie Chart":
        pie_col = st.selectbox("Select Column for Pie Chart:", ["heat_class", "region", "heat_retention_level"])
        pie_data = data[pie_col].value_counts()
        ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=["#2ecc71","#f1c40f","#e74c3c"])
        ax.set_title(f"Distribution of {pie_col}", fontsize=16)

    st.pyplot(fig)


# ---------- ABOUT HEAT RETENTION ----------
elif view == "About Heat Retention":
    st.subheader("🌡 What is Urban Heat Retention?")

    st.markdown("""
    **Urban Heat Retention** refers to the ability of urban areas to 
    **absorb and trap heat during the day and release it slowly at night**.
    
    This happens mainly due to:
    - Concrete buildings  
    - Roads and asphalt  
    - Lack of trees and green spaces  
    """)



    st.markdown("""
    ### 🔥 Why is Heat Retention High in Delhi?
    - Dense construction and high **built-up index**
    - Increasing **air pollution (AQI)**
    - Reduced **green cover**
    - Heat-absorbing materials like concrete and tar roads
    """)

    st.markdown("""
    ### 📊 How This App Helps
    This dashboard:
    - Predicts **heat risk level** using Machine Learning  
    - Uses **temperature, AQI, vegetation & satellite indicators**  
    - Shows **area-wise heat patterns on maps and charts**
    """)

    st.markdown("""
    ### 🟢 Low Heat  
    Areas with good greenery and airflow  

    ### 🟡 Moderate Heat  
    Mixed residential and commercial zones  

    ### 🔴 High Heat  
    Dense construction, low greenery, high pollution
    """)

    st.info("🌿 Increasing green spaces and reducing pollution can significantly lower heat retention.")


# ---------- PRECAUTIONS ----------
elif view == "Precautions":
    st.subheader("🛡 Heat Safety Guidelines")
    st.markdown("""
    • Stay hydrated  
    • Avoid outdoor work 12–4 PM  
    • Wear light cotton clothes  
    • Use sunscreen & sunglasses  
    • Avoid heavy physical activity  
    """)

# ---------- EMERGENCY ----------
elif view == "Emergency":
    st.subheader("📞 Emergency Contacts")
    st.markdown("""
    **Ambulance:** 108  
    **Fire:** 101  
    **Disaster Helpline:** 1077  
    **Delhi Pollution Control:** 011-2235-1234  
    """)
