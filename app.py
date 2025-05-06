import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import math
import matplotlib.pyplot as plt
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
            price = price * (1 + np.random.uniform(-0.05, 0.05))

            # Round to nearest rupee
            price = round(price)

            # Add to data
            data.append({
                'origin': origin,
                'destination': destination,
                'distance': distance,
                'duration': duration,
                'hour': hour,
                'time_of_day': time_of_day,
                'weather': weather,
                'day_type': day_type,
                'traffic': traffic,
                'price': price
            })

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Train the model
        self.train_model(df)

    def train_model(self, data):
        """Train the model on the generated data"""
        print("Training model...")

        # Encode categorical features
        self.location_encoder = LabelEncoder()
        data['origin_encoded'] = self.location_encoder.fit_transform(data['origin'])
        data['destination_encoded'] = self.location_encoder.transform(data['destination'])

        # Encode other categorical features
        time_of_day_encoder = LabelEncoder()
        weather_encoder = LabelEncoder()
        day_type_encoder = LabelEncoder()
        traffic_encoder = LabelEncoder()

        data['time_of_day_encoded'] = time_of_day_encoder.fit_transform(data['time_of_day'])
        data['weather_encoded'] = weather_encoder.fit_transform(data['weather'])
        data['day_type_encoded'] = day_type_encoder.fit_transform(data['day_type'])
        data['traffic_encoded'] = traffic_encoder.fit_transform(data['traffic'])

        # Prepare features
        features = [
            'origin_encoded', 'destination_encoded', 'distance', 'duration',
            'hour', 'time_of_day_encoded', 'weather_encoded',
            'day_type_encoded', 'traffic_encoded'
        ]

        X = data[features]
        y = data['price']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Save model and encoders
        joblib.dump(self.model, "hyderabad_ride_model.joblib")
        joblib.dump(self.location_encoder, "hyderabad_location_encoder.joblib")
        joblib.dump(self.scaler, "hyderabad_feature_scaler.joblib")

        print("Model trained and saved successfully")

    def predict_ride_price(self, origin, destination, time_of_day=None, hour=None,
                          weather="Clear", day_type="Weekday", traffic="Moderate",
                          service_type="Standard"):
        """Predict the price for a ride between two locations"""
        try:
            # Calculate distance and duration
            distance = self.calculate_distance(origin, destination)
            duration = self.calculate_trip_duration(distance, traffic)

            # If time_of_day is not provided, determine it from hour
            if time_of_day is None and hour is not None:
                if 5 <= hour < 12:
                    time_of_day = "Morning"
                elif 12 <= hour < 17:
                    time_of_day = "Afternoon"
                elif 17 <= hour < 21:
                    time_of_day = "Evening"
                else:
                    time_of_day = "Night"

            # Encode features
            origin_encoded = self.location_encoder.transform([origin])[0]
            destination_encoded = self.location_encoder.transform([destination])[0]

            # Create feature dictionary with correct feature names
            features = {
                'origin': origin_encoded,
                'destination': destination_encoded,
                'distance_km': distance,
                'duration_min': duration,
                'hour': hour if hour is not None else 12,
                'time_morning': 1 if time_of_day == "Morning" else 0,
                'time_afternoon': 1 if time_of_day == "Afternoon" else 0,
                'time_evening': 1 if time_of_day == "Evening" else 0,
                'time_night': 1 if time_of_day == "Night" else 0,
                'weather': 0 if weather == "Clear" else 1 if weather == "Cloudy" else 2 if weather == "Light Rain" else 3 if weather == "Rainy" else 4,
                'day_weekday': 1 if day_type == "Weekday" else 0,
                'day_weekend': 1 if day_type == "Weekend" else 0,
                'day_holiday': 1 if day_type == "Holiday" else 0,
                'traffic_light': 1 if traffic == "Light" else 0,
                'traffic_moderate': 1 if traffic == "Moderate" else 0,
                'traffic_heavy': 1 if traffic == "Heavy" else 0
            }

            # Convert to DataFrame
            X = pd.DataFrame([features])

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make prediction
            base_price = self.model.predict(X_scaled)[0]

            # Apply service type multiplier
            service_multipliers = {
                "Economy": 0.8,
                "Standard": 1.0,
                "Premium": 1.3
            }

            price = base_price * service_multipliers.get(service_type, 1.0)

            # Ensure minimum fare
            price = max(price, self.minimum_fare)

            # Round to nearest rupee
            price = round(price)

            # Calculate alternatives
            alternatives = {
                "Economy": round(base_price * service_multipliers["Economy"]),
                "Standard": round(base_price * service_multipliers["Standard"]),
                "Premium": round(base_price * service_multipliers["Premium"])
            }

            # Format price with Indian Rupee symbol
            formatted_price = f"₹{price:,}"

            return {
                "price": price,
                "formatted_price": formatted_price,
                "distance": distance,
                "duration": duration,
                "alternatives": alternatives
            }

        except Exception as e:
            return {"error": f"Error predicting price: {str(e)}"}

def get_time_of_day(hour):
    """Get time of day based on hour"""
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"
