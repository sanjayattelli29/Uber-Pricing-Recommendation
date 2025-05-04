import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import math
import matplotlib.pyplot as plt
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from datetime import datetime, timedelta

class HyderabadRidePricePrediction:
    def __init__(self):
        """Initialize the ride price prediction system with Hyderabad-specific data"""
        # Define popular locations in Hyderabad
        self.locations = [
            "Hitech City", "Gachibowli", "Banjara Hills", "Jubilee Hills",
            "Secunderabad", "Begumpet", "Madhapur", "Kukatpally",
            "Ameerpet", "Mehdipatnam", "LB Nagar", "KPHB",
            "Lingampally", "Charminar", "Abids", "Uppal",
            "Film Nagar", "Shamshabad Airport", "Tank Bund", "Paradise Circle",
            "Miyapur", "Dilsukhnagar", "Koti", "Lakdikapul",
            "Botanical Garden", "Financial District", "Inorbit Mall", "Golconda Fort"
        ]

        # Approximate distances between locations in kilometers (straight-line distances)
        # This is a simplified distance matrix for demonstration
        self.location_coords = {
            "Hitech City": (17.4474, 78.3762),
            "Gachibowli": (17.4401, 78.3489),
            "Banjara Hills": (17.4156, 78.4309),
            "Jubilee Hills": (17.4239, 78.4071),
            "Secunderabad": (17.4399, 78.4983),
            "Begumpet": (17.4442, 78.4618),
            "Madhapur": (17.4477, 78.3920),
            "Kukatpally": (17.4849, 78.4088),
            "Ameerpet": (17.4372, 78.4461),
            "Mehdipatnam": (17.3949, 78.4405),
            "LB Nagar": (17.3466, 78.5549),
            "KPHB": (17.4933, 78.3915),
            "Lingampally": (17.4913, 78.3166),
            "Charminar": (17.3616, 78.4747),
            "Abids": (17.3930, 78.4752),
            "Uppal": (17.4058, 78.5586),
            "Film Nagar": (17.4105, 78.4170),
            "Shamshabad Airport": (17.2403, 78.4294),
            "Tank Bund": (17.4236, 78.4738),
            "Paradise Circle": (17.4417, 78.4947),
            "Miyapur": (17.5020, 78.3720),
            "Dilsukhnagar": (17.3689, 78.5247),
            "Koti": (17.3824, 78.4791),
            "Lakdikapul": (17.4041, 78.4539),
            "Botanical Garden": (17.4521, 78.3920),
            "Financial District": (17.4143, 78.3431),
            "Inorbit Mall": (17.4350, 78.3871),
            "Golconda Fort": (17.3833, 78.4011)
        }

        # Initialize models and encoders
        self.model = None
        self.location_encoder = None
        self.scaler = None

        # Set Hyderabad-specific pricing factors
        self.base_fare = 30  # Base fare in INR
        self.per_km_rate = 12  # Rate per kilometer in INR
        self.per_minute_rate = 1  # Rate per minute in INR
        self.minimum_fare = 60  # Minimum fare in INR

        # Create or load the model
        self.create_or_load_model()

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great circle distance between two points in kilometers"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(min(1, math.sqrt(a)))  # Ensure a is not greater than 1
        r = 6371  # Radius of earth in kilometers
        return c * r

    def calculate_distance(self, origin, destination):
        """Calculate the distance between two locations in Hyderabad"""
        if origin not in self.location_coords or destination not in self.location_coords:
            # Default distance if locations not in our database
            return 10

        # Get coordinates
        origin_coords = self.location_coords[origin]
        dest_coords = self.location_coords[destination]

        # Calculate direct distance
        direct_distance = self.haversine_distance(
            origin_coords[0], origin_coords[1],
            dest_coords[0], dest_coords[1]
        )

        # Add 30% to account for road routes vs straight line
        road_distance = direct_distance * 1.3

        return round(road_distance, 2)

    def calculate_trip_duration(self, distance, traffic):
        """Estimate trip duration in minutes based on distance and traffic conditions"""
        # Average speeds in km/h based on traffic
        speeds = {
            "Light": 35,
            "Moderate": 25,
            "Heavy": 15
        }

        # Calculate time in hours then convert to minutes
        speed = speeds.get(traffic, 25)  # Default to moderate if traffic level not found
        duration_hours = distance / speed
        duration_minutes = duration_hours * 60

        # Add 5 minutes as buffer
        return round(duration_minutes + 5)

    def create_or_load_model(self):
        """Create a new model or load an existing one"""
        model_file = "hyderabad_ride_model.joblib"
        encoder_file = "hyderabad_location_encoder.joblib"
        scaler_file = "hyderabad_feature_scaler.joblib"

        # Check if model already exists
        if os.path.exists(model_file) and os.path.exists(encoder_file) and os.path.exists(scaler_file):
            try:
                self.model = joblib.load(model_file)
                self.location_encoder = joblib.load(encoder_file)
                self.scaler = joblib.load(scaler_file)
                print("Model loaded successfully")
                return
            except:
                print("Error loading model files, creating new ones")

        # Generate sample data and train model if not exists
        self.generate_sample_data()

    def generate_sample_data(self):
        """Generate realistic sample data for Hyderabad ride prices"""
        print("Generating realistic Hyderabad ride data...")

        # Number of samples
        n_samples = 8000

        # Generate random data
        np.random.seed(42)
        data = []

        for _ in range(n_samples):
            # Select random origin and destination
            origin = np.random.choice(self.locations)
            available_dests = [loc for loc in self.locations if loc != origin]
            destination = np.random.choice(available_dests)

            # Calculate distance
            distance = self.calculate_distance(origin, destination)

            # Generate time of day with realistic distribution
            hour = np.random.choice(range(24), p=[
                0.01, 0.01, 0.01, 0.01, 0.02, 0.04,  # 0-5 AM
                0.06, 0.08, 0.07, 0.05, 0.04, 0.05,  # 6-11 AM
                0.06, 0.05, 0.04, 0.05, 0.06, 0.07,  # 12-5 PM
                0.08, 0.06, 0.04, 0.02, 0.01, 0.01   # 6-11 PM
            ])

            if 5 <= hour < 12:
                time_of_day = "Morning"
            elif 12 <= hour < 17:
                time_of_day = "Afternoon"
            elif 17 <= hour < 21:
                time_of_day = "Evening"
            else:
                time_of_day = "Night"

            # Time factor: evening and night rides cost more
            time_factor = 1.0
            if time_of_day == "Morning" and 7 <= hour < 10:  # Morning rush hour
                time_factor = 1.2
            elif time_of_day == "Evening" and 17 <= hour < 20:  # Evening rush hour
                time_factor = 1.3
            elif time_of_day == "Night" and (hour >= 22 or hour < 5):  # Late night
                time_factor = 1.2

            # Weather with seasonal probabilities (monsoon vs dry season)
            is_monsoon = np.random.random() < 0.3  # 30% chance of being in monsoon season
            if is_monsoon:
                weather = np.random.choice(["Clear", "Rainy", "Thunderstorm"], p=[0.3, 0.5, 0.2])
            else:
                weather = np.random.choice(["Clear", "Cloudy", "Light Rain"], p=[0.7, 0.25, 0.05])

            # Weather factor
            weather_factor = 1.0
            if weather == "Rainy":
                weather_factor = 1.25
            elif weather == "Thunderstorm":
                weather_factor = 1.4
            elif weather == "Cloudy":
                weather_factor = 1.05
            elif weather == "Light Rain":
                weather_factor = 1.15

            # Day type (weekday/weekend/holiday)
            day_type = np.random.choice(["Weekday", "Weekend", "Holiday"], p=[0.71, 0.27, 0.02])

            # Day factor
            day_factor = 1.0
            if day_type == "Weekend":
                day_factor = 1.1
            elif day_type == "Holiday":
                day_factor = 1.25

            # Traffic conditions
            if (day_type == "Weekday" and (7 <= hour < 10 or 17 <= hour < 20)):
                # Rush hour on weekdays
                traffic = np.random.choice(["Moderate", "Heavy"], p=[0.3, 0.7])
            elif day_type == "Weekend" and 11 <= hour < 19:
                # Weekend shopping hours
                traffic = np.random.choice(["Light", "Moderate", "Heavy"], p=[0.2, 0.5, 0.3])
            else:
                traffic = np.random.choice(["Light", "Moderate", "Heavy"], p=[0.6, 0.3, 0.1])

            # Traffic factor
            traffic_factor = 1.0
            if traffic == "Moderate":
                traffic_factor = 1.15
            elif traffic == "Heavy":
                traffic_factor = 1.3

            # Calculate duration in minutes
            duration = self.calculate_trip_duration(distance, traffic)

            # Calculate base price using Hyderabad's realistic pricing model
            # Base fare + per km charge + per minute charge
            base_price = self.base_fare + (distance * self.per_km_rate) + (duration * self.per_minute_rate)

            # Apply surge pricing factors
            price = base_price * time_factor * weather_factor * day_factor * traffic_factor

            # Ensure minimum fare
            price = max(price, self.minimum_fare)

            # Add some random noise (±5%)
            price = price * (0.95 + 0.1 * np.random.random())

            # Round to nearest rupee
            price = round(price)

            # Different service tiers
            service_type = np.random.choice(["Economy", "Standard", "Premium"])
            if service_type == "Economy":
                service_factor = 0.85
            elif service_type == "Standard":
                service_factor = 1.0
            else:  # Premium
                service_factor = 1.3

            final_price = round(price * service_factor)

            data.append({
                "origin": origin,
                "destination": destination,
                "distance_km": distance,
                "duration_min": duration,
                "time_of_day": time_of_day,
                "hour": hour,
                "weather": weather,
                "day_type": day_type,
                "traffic": traffic,
                "service_type": service_type,
                "price": final_price
            })

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save data
        df.to_csv("hyderabad_ride_data.csv", index=False)

        # Show some statistics
        print(f"Generated {n_samples} ride samples with realistic Hyderabad pricing")
        print(f"Average ride price: ₹{df['price'].mean():.2f}")
        print(f"Price range: ₹{df['price'].min()} - ₹{df['price'].max()}")

        # Train model
        self.train_model(df)

        return df

    def train_model(self, data):
        """Train RandomForest model on the data"""
        print("Training model on Hyderabad ride data...")

        # Create LabelEncoder for locations
        self.location_encoder = LabelEncoder()
        all_locations = np.concatenate([data["origin"].values, data["destination"].values])
        self.location_encoder.fit(all_locations)

        # Create encoders for other categorical features
        weather_encoder = LabelEncoder()
        weather_encoder.fit(data["weather"])

        service_encoder = LabelEncoder()
        service_encoder.fit(data["service_type"])

        # Prepare features
        X = pd.DataFrame({
            "origin": self.location_encoder.transform(data["origin"]),
            "destination": self.location_encoder.transform(data["destination"]),
            "distance_km": data["distance_km"],
            "duration_min": data["duration_min"],
            "hour": data["hour"],
            "time_morning": (data["time_of_day"] == "Morning").astype(int),
            "time_afternoon": (data["time_of_day"] == "Afternoon").astype(int),
            "time_evening": (data["time_of_day"] == "Evening").astype(int),
            "time_night": (data["time_of_day"] == "Night").astype(int),
            "weather": weather_encoder.transform(data["weather"]),
            "day_weekday": (data["day_type"] == "Weekday").astype(int),
            "day_weekend": (data["day_type"] == "Weekend").astype(int),
            "day_holiday": (data["day_type"] == "Holiday").astype(int),
            "traffic_light": (data["traffic"] == "Light").astype(int),
            "traffic_moderate": (data["traffic"] == "Moderate").astype(int),
            "traffic_heavy": (data["traffic"] == "Heavy").astype(int),
            "service": service_encoder.transform(data["service_type"])
        })

        # Target
        y = data["price"]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        # Save model
        joblib.dump(self.model, "hyderabad_ride_model.joblib")
        joblib.dump(self.location_encoder, "hyderabad_location_encoder.joblib")
        joblib.dump(self.scaler, "hyderabad_feature_scaler.joblib")

        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Model R² score - Train: {train_score:.4f}, Test: {test_score:.4f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\nTop 10 important features:")
        print(feature_importance.head(10))

        return self.model

    def predict_ride_price(self, origin, destination, time_of_day=None, hour=None,
                          weather="Clear", day_type="Weekday", traffic="Moderate",
                          service_type="Standard"):
        """Predict ride price based on user inputs"""
        # Validate inputs
        if origin == destination:
            return {
                "error": "Origin and destination cannot be the same",
                "price": None,
                "distance": None,
                "duration": None
            }

        # Set current hour if not provided
        if hour is None:
            current_hour = datetime.now().hour
            hour = current_hour

        # Set time of day based on hour if not provided
        if time_of_day is None:
            if 5 <= hour < 12:
                time_of_day = "Morning"
            elif 12 <= hour < 17:
                time_of_day = "Afternoon"
            elif 17 <= hour < 21:
                time_of_day = "Evening"
            else:
                time_of_day = "Night"

        # Calculate distance
        distance = self.calculate_distance(origin, destination)

        # Calculate duration
        duration = self.calculate_trip_duration(distance, traffic)

        try:
            # Transform origin and destination
            origin_encoded = self.location_encoder.transform([origin])[0]
            destination_encoded = self.location_encoder.transform([destination])[0]

            # Create feature vector
            X = pd.DataFrame({
                "origin": [origin_encoded],
                "destination": [destination_encoded],
                "distance_km": [distance],
                "duration_min": [duration],
                "hour": [hour],
                "time_morning": [1 if time_of_day == "Morning" else 0],
                "time_afternoon": [1 if time_of_day == "Afternoon" else 0],
                "time_evening": [1 if time_of_day == "Evening" else 0],
                "time_night": [1 if time_of_day == "Night" else 0],
                "weather": [0],  # Default encoding for "Clear"
                "day_weekday": [1 if day_type == "Weekday" else 0],
                "day_weekend": [1 if day_type == "Weekend" else 0],
                "day_holiday": [1 if day_type == "Holiday" else 0],
                "traffic_light": [1 if traffic == "Light" else 0],
                "traffic_moderate": [1 if traffic == "Moderate" else 0],
                "traffic_heavy": [1 if traffic == "Heavy" else 0],
                "service": [1 if service_type == "Standard" else 0 if service_type == "Economy" else 2]
            })

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Predict price for standard service
            standard_price = self.model.predict(X_scaled)[0]

            # Calculate prices for different service types
            if service_type == "Standard":
                predicted_price = standard_price
            elif service_type == "Economy":
                predicted_price = standard_price * 0.85
            else:  # Premium
                predicted_price = standard_price * 1.3

            # Round to nearest rupee
            predicted_price = round(predicted_price)

            # Calculate alternative service prices
            alternatives = {
                "Economy": round(standard_price * 0.85),
                "Standard": round(standard_price),
                "Premium": round(standard_price * 1.3)
            }

            return {
                "price": predicted_price,
                "currency": "INR",
                "formatted_price": f"₹{predicted_price}",
                "distance": distance,
                "duration": duration,
                "service_type": service_type,
                "alternatives": alternatives
            }

        except Exception as e:
            print(f"Error in prediction: {e}")

            # Fallback to direct calculation if model fails
            base_price = self.base_fare + (distance * self.per_km_rate) + (duration * self.per_minute_rate)

            # Apply service type factor
            if service_type == "Economy":
                service_factor = 0.85
            elif service_type == "Standard":
                service_factor = 1.0
            else:  # Premium
                service_factor = 1.3

            predicted_price = round(max(base_price * service_factor, self.minimum_fare))

            return {
                "price": predicted_price,
                "currency": "INR",
                "formatted_price": f"₹{predicted_price}",
                "distance": distance,
                "duration": duration,
                "service_type": service_type,
                "note": "Calculated using fallback method",
                "alternatives": {
                    "Economy": round(max(base_price * 0.85, self.minimum_fare)),
                    "Standard": round(max(base_price, self.minimum_fare)),
                    "Premium": round(max(base_price * 1.3, self.minimum_fare))
                }
            }

# Helper function to determine time of day from hour
def get_time_of_day(hour):
    """Convert hour to time of day category"""
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

# Create interactive widgets for Colab
def create_prediction_ui(predictor):
    """Create interactive UI for ride price prediction"""
    print("\n🚗 Hyderabad Ride Price Prediction System 🚗\n")

    # Origin dropdown
    origin_dropdown = widgets.Dropdown(
        options=predictor.locations,
        value="Hitech City",
        description='Origin:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    # Destination dropdown
    destination_dropdown = widgets.Dropdown(
        options=predictor.locations,
        value="Secunderabad",
        description='Destination:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    # Current time checkbox
    use_current_time = widgets.Checkbox(
        value=True,
        description='Use current time',
        style={'description_width': 'initial'}
    )

    # Time selection (hour)
    hour_selector = widgets.IntSlider(
        value=datetime.now().hour,
        min=0,
        max=23,
        step=1,
        description='Hour (24h):',
        disabled=True,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    # Time of day label (automatically updated)
    time_of_day_label = widgets.Label(value=f"Time of day: {get_time_of_day(hour_selector.value)}")

    # Weather dropdown
    weather_dropdown = widgets.Dropdown(
        options=["Clear", "Cloudy", "Light Rain", "Rainy", "Thunderstorm"],
        value="Clear",
        description='Weather:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    # Day type dropdown
    day_dropdown = widgets.Dropdown(
        options=["Weekday", "Weekend", "Holiday"],
        value="Weekday",
        description='Day Type:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    # Traffic dropdown
    traffic_dropdown = widgets.Dropdown(
        options=["Light", "Moderate", "Heavy"],
        value="Moderate",
        description='Traffic:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='300px')
    )

    # Service type selection
    service_type = widgets.RadioButtons(
        options=['Economy', 'Standard', 'Premium'],
        value='Standard',
        description='Service Type:',
        style={'description_width': 'initial'},
        layout={'width': 'max-content'}
    )

    # Output area
    output = widgets.Output()

    # Update time of day when hour changes
    def update_time_of_day(change):
        time_of_day_label.value = f"Time of day: {get_time_of_day(change.new)}"

    hour_selector.observe(update_time_of_day, names='value')

    # Toggle hour selector based on checkbox
    def toggle_hour_selector(change):
        hour_selector.disabled = change.new
        if change.new:  # If using current time
            hour_selector.value = datetime.now().hour

    use_current_time.observe(toggle_hour_selector, names='value')

    # Predict button
    def on_predict_button_clicked(b):
        with output:
            clear_output()

            # Get input values
            origin = origin_dropdown.value
            destination = destination_dropdown.value
            hour = datetime.now().hour if use_current_time.value else hour_selector.value
            time_of_day = get_time_of_day(hour)
            weather = weather_dropdown.value
            day_type = day_dropdown.value
            traffic = traffic_dropdown.value
            service = service_type.value

            # Make prediction
            result = predictor.predict_ride_price(
                origin=origin,
                destination=destination,
                time_of_day=time_of_day,
                hour=hour,
                weather=weather,
                day_type=day_type,
                traffic=traffic,
                service_type=service
            )

            if "error" in result:
                print(f"Error: {result['error']}")
                return

            # Current time
            current_time = datetime.now().strftime("%H:%M")
            estimated_arrival = (datetime.now() + timedelta(minutes=result['duration'])).strftime("%H:%M")

            # Display ride details
            print(f"\n🚗 {service} Ride: {origin} to {destination}")
            print(f"🕒 Current time: {current_time} ({time_of_day})")
            print(f"🛣 Distance: {result['distance']:.1f} km")
            print(f"⏱ Estimated duration: {result['duration']} minutes (arrival ~{estimated_arrival})")
            print(f"💰 Predicted Price: {result['formatted_price']}")

            # Display conditions
            print(f"\nConditions: {weather} weather, {day_type}, {traffic} traffic")

            # Show alternatives
            print("\nAvailable options:")
            alternatives = result['alternatives']

            # Plot the options
            plt.figure(figsize=(10, 5))
            services = list(alternatives.keys())
            prices = list(alternatives.values())

            colors = ['#4CAF50', '#2196F3', '#FF9800']
            selected_idx = services.index(service)

            bar_colors = ['lightgray'] * len(services)
            bar_colors[selected_idx] = colors[selected_idx]

            plt.bar(services, prices, color=bar_colors)

            # Add price labels on top of bars
            for i, price in enumerate(prices):
                plt.text(i, price + 5, f"₹{price}", ha='center', fontweight='bold')

                # Add "SELECTED" label to chosen service
                if i == selected_idx:
                    plt.text(i, price/2, "SELECTED", ha='center', color='white', fontweight='bold', rotation=90)

            plt.title('Ride Price Comparison')
            plt.ylabel('Price (₹)')
            plt.ylim(0, max(prices) * 1.2)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.show()

            # Show economical recommendation
            min_price = min(alternatives.values())
            min_service = [k for k, v in alternatives.items() if v == min_price][0]

            if service != min_service:
                savings = alternatives[service] - min_price
                print(f"\n💡 Tip: Choose '{min_service}' to save ₹{savings}")
            else:
                print(f"\n💡 You've selected the most economical option!")

    predict_button = widgets.Button(
        description='Predict Price',
        button_style='info',
        icon='car'
    )
    predict_button.on_click(on_predict_button_clicked)

    # Map button to show the route
    def on_map_button_clicked(b):
        with output:
            clear_output()
            origin = origin_dropdown.value
            destination = destination_dropdown.value

            if origin == destination:
                print("Error: Origin and destination cannot be the same")
                return

            # Get coordinates
            if origin in predictor.location_coords and destination in predictor.location_coords:
                origin_coords = predictor.location_coords[origin]
                dest_coords = predictor.location_coords[destination]

                # Generate Google Maps URL
                maps_url = f"https://www.google.com/maps/dir/{origin_coords[0]},{origin_coords[1]}/{dest_coords[0]},{dest_coords[1]}"

                print(f"Route from {origin} to {destination}")
                print(f"\nOpen this URL in your browser to view the route:")
                print(maps_url)

                # Display a simple map visualization
                plt.figure(figsize=(12, 8))

                # Plot all locations
                x_coords = [coords[1] for coords in predictor.location_coords.values()]
                y_coords = [coords[0] for coords in predictor.location_coords.values()]
                plt.scatter(x_coords, y_coords, color='gray', alpha=0.5, s=30)

                # Add location names
                for loc, coords in predictor.location_coords.items():
                    plt.annotate(loc, (coords[1], coords[0]), fontsize=8, alpha=0.7)

                # Highlight origin and destination
                origin_coords = predictor.location_coords[origin]
                dest_coords = predictor.location_coords[destination]

                plt.scatter([origin_coords[1]], [origin_coords[0]], color='green', s=100, label='Origin')
                plt.scatter([dest_coords[1]], [dest_coords[0]], color='red', s=100, label='Destination')

                # Draw line between origin and destination
                plt.plot([origin_coords[1], dest_coords[1]], [origin_coords[0], dest_coords[0]], 'b-', alpha=0.7)

                plt.title(f'Route from {origin} to {destination}')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

                # Show distance
                distance = predictor.calculate_distance(origin, destination)
                print(f"\nDirect distance: {distance:.2f} km")
            else:
                print("Location coordinates not available for mapping")

    map_button = widgets.Button(
        description='Show Map',
        button_style='success',
        icon='map'
    )
    map_button.on_click(on_map_button_clicked)

    # Layout UI elements
    ui_title = widgets.HTML("<h2>🚕 Hyderabad Ride Price Prediction</h2>")
    location_box = widgets.VBox([origin_dropdown, destination_dropdown])

    time_box = widgets.VBox([
        use_current_time,
        hour_selector,
        time_of_day_label
    ])

    conditions_box = widgets.VBox([
        weather_dropdown,
        day_dropdown,
        traffic_dropdown
    ])

    options_box = widgets.VBox([widgets.HTML("<h4>Ride Options</h4>"), service_type])

    inputs_box = widgets.HBox([
        widgets.VBox([widgets.HTML("<h4>Locations</h4>"), location_box]),
        widgets.VBox([widgets.HTML("<h4>Time</h4>"), time_box]),
        widgets.VBox([widgets.HTML("<h4>Conditions</h4>"), conditions_box]),
        options_box
    ])

    button_box = widgets.HBox([predict_button, map_button])

    # Display the UI
    display(ui_title)
    display(inputs_box)
    display(button_box)
    display(output)

    return {
        'origin': origin_dropdown,
        'destination': destination_dropdown,
        'hour': hour_selector,
        'weather': weather_dropdown,
        'day_type': day_dropdown,
        'traffic': traffic_dropdown,
        'service_type': service_type,
        'predict': predict_button,
        'map': map_button,
        'output': output
    }


def main():
    """Main function to run the prediction system"""
    print("Starting Hyderabad Ride Price Prediction System...")

    # Initialize the predictor
    predictor = HyderabadRidePricePrediction()

    # Show some sample predictions
    print("\nGenerating sample predictions...")
    sample_predictions = [
        predictor.predict_ride_price("Hitech City", "Secunderabad"),
        predictor.predict_ride_price("Banjara Hills", "Charminar",
                                    weather="Rainy", traffic="Heavy"),
        predictor.predict_ride_price("Ameerpet", "Madhapur",
                                    time_of_day="Evening", service_type="Premium")
    ]

    for i, prediction in enumerate(sample_predictions):
        print(f"\nSample Prediction {i+1}:")
        print(f"Price: {prediction['formatted_price']}")
        print(f"Distance: {prediction['distance']:.1f} km")
        print(f"Duration: {prediction['duration']} minutes")

    # Create and display the UI
    print("\nCreating interactive UI...")
    ui_components = create_prediction_ui(predictor)

    print("\nSystem ready! Use the UI above to make predictions.")
    return predictor


# Helper function to run the system
def run_hyderabad_ride_prediction():
    """Run the Hyderabad Ride Price Prediction System"""
    return main()

# Run the system when the cell is executed
predictor = main()  # Call main() directly to run the system
