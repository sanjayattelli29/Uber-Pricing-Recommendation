import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from app import HyderabadRidePricePrediction
import folium
from streamlit_folium import folium_static
import time
import math

# Set page configuration
st.set_page_config(
    page_title="Hyderabad Ride Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stSelectbox {
        border-radius: 10px;
    }
    .reportview-container {
        background: #FAFAFA
    }
    .sidebar .sidebar-content {
        background: #FFFFFF
    }
    h1 {
        color: #1E1E1E;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2 {
        color: #4A4A4A;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h3 {
        color: #717171;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize the predictor
@st.cache_resource
def get_predictor():
    return HyderabadRidePricePrediction()

predictor = get_predictor()

# Header with animation
def load_animation():
    with st.spinner("🚗 Starting engines..."):
        time.sleep(1)
    st.success("🎉 Welcome to Hyderabad Ride Price Prediction!")

# Main header
st.title("🚕 Hyderabad Ride Price Prediction")
st.markdown("---")

# Show sample predictions in sidebar
with st.sidebar:
    st.subheader("📊 Sample Predictions")
    sample_routes = [
        ("Hitech City", "Secunderabad", {}),
        ("Banjara Hills", "Charminar", {"weather": "Rainy", "traffic": "Heavy"}),
        ("Ameerpet", "Madhapur", {"time_of_day": "Evening", "service_type": "Premium"})
    ]
    
    for i, (origin, destination, params) in enumerate(sample_routes):
        prediction = predictor.predict_ride_price(origin, destination, **params)
        with st.expander(f"Sample {i+1}"):
            st.write(f"**Route:** {origin} → {destination}")
            st.write(f"**Price:** {prediction['formatted_price']}")
            st.write(f"**Distance:** {prediction['distance']:.1f} km")
            st.write(f"**Duration:** {prediction['duration']} mins")

# Create three columns for better layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📍 Location Details")
    origin = st.selectbox(
        "Select Pickup Location",
        options=predictor.locations,
        index=predictor.locations.index("Hitech City")
    )
    
    destination = st.selectbox(
        "Select Drop Location",
        options=predictor.locations,
        index=predictor.locations.index("Secunderabad")
    )

with col2:
    st.subheader("⏰ Time & Date")
    use_current_time = st.checkbox("Use Current Time", value=True)
    
    if use_current_time:
        current_hour = datetime.now().hour
        hour = current_hour
        st.info(f"Current Time: {datetime.now().strftime('%I:%M %p')}")
    else:
        hour = st.slider(
            "Select Hour",
            min_value=0,
            max_value=23,
            value=datetime.now().hour,
            format="%I %p"
        )

    # Day type selection with better UI
    day_type = st.radio(
        "Select Day Type",
        options=["Weekday", "Weekend", "Holiday"],
        horizontal=True
    )

with col3:
    st.subheader("🌤️ Conditions")
    weather = st.selectbox(
        "Weather Conditions",
        options=["Clear", "Cloudy", "Light Rain", "Rainy", "Thunderstorm"],
        index=0
    )
    
    traffic = st.select_slider(
        "Traffic Conditions",
        options=["Light", "Moderate", "Heavy"],
        value="Moderate"
    )

# Service type selection with custom styling
st.markdown("---")
st.subheader("🚘 Choose Your Ride")
service_cols = st.columns(3)

selected_service = None
for i, service in enumerate(["Economy", "Standard", "Premium"]):
    with service_cols[i]:
        if st.button(
            f"{service}\n{'💰' if service == 'Economy' else '🚗' if service == 'Standard' else '🏎️'}",
            key=service,
            help=f"Select {service} service"
        ):
            selected_service = service
        
        if service == "Economy":
            st.caption("Most affordable option")
        elif service == "Standard":
            st.caption("Balance of comfort & cost")
        else:
            st.caption("Premium comfort & service")

if selected_service is None:
    selected_service = "Standard"

# Predict button with loading animation
st.markdown("---")
if st.button("🚀 Predict Ride Price", use_container_width=True):
    with st.spinner("Calculating your fare..."):
        time.sleep(1)
        
        # Get prediction
        result = predictor.predict_ride_price(
            origin=origin,
            destination=destination,
            time_of_day="Morning" if 5 <= hour < 12 else "Afternoon" if 12 <= hour < 17 else "Evening" if 17 <= hour < 21 else "Night",
            hour=hour,
            weather=weather,
            day_type=day_type,
            traffic=traffic,
            service_type=selected_service
        )
        
        if "error" in result:
            st.error(result["error"])
            st.stop()
        
        # Display results in a nice layout
        st.markdown("---")
        st.subheader("🎯 Ride Details")
        
        # Create three columns for the results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col1:
            st.metric("Estimated Price", result['formatted_price'])
            
        with res_col2:
            st.metric("Distance", f"{result['distance']:.1f} km")
            
        with res_col3:
            st.metric("Duration", f"{result['duration']} mins")
            
        # Show route on map
        st.markdown("---")
        st.subheader("🗺️ Route Map")
        
        # Create map
        if origin in predictor.location_coords and destination in predictor.location_coords:
            origin_coords = predictor.location_coords[origin]
            dest_coords = predictor.location_coords[destination]
            
            m = folium.Map(
                location=[(origin_coords[0] + dest_coords[0])/2, 
                         (origin_coords[1] + dest_coords[1])/2],
                zoom_start=12
            )
            
            # Add markers
            folium.Marker(
                origin_coords,
                popup=origin,
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
            
            folium.Marker(
                dest_coords,
                popup=destination,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
            
            # Draw line between points
            folium.PolyLine(
                locations=[origin_coords, dest_coords],
                weight=2,
                color='blue',
                opacity=0.8
            ).add_to(m)
            
            # Display map
            folium_static(m)
            
            # Price comparison chart
            st.markdown("---")
            st.subheader("💰 Price Comparison")
            
            # Create bar chart with Plotly
            alternatives = result['alternatives']
            fig = go.Figure(data=[
                go.Bar(
                    x=list(alternatives.keys()),
                    y=list(alternatives.values()),
                    marker_color=['#76C893' if k == selected_service else '#94D2BD' for k in alternatives.keys()],
                    text=[f"₹{v}" for v in alternatives.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Service Type Comparison",
                xaxis_title="Service Type",
                yaxis_title="Price (₹)",
                plot_bgcolor='white',
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show savings tip
            min_price = min(alternatives.values())
            min_service = [k for k, v in alternatives.items() if v == min_price][0]
            
            if selected_service != min_service:
                savings = alternatives[selected_service] - min_price
                st.info(f"💡 Tip: Choose '{min_service}' service to save ₹{savings}")
            else:
                st.success("💡 You've selected the most economical option!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Made with ❤️ for Hyderabad | Data updated regularly for accurate predictions</p>
    </div>
    """,
    unsafe_allow_html=True
) 