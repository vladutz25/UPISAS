import requests
import numpy as np
from strategy import Strategy


class AdvancedWildfireAdaptationStrategy(Strategy):
    def __init__(self, monitor_api, execute_api, adaptation_schema, max_uav_speed=2, collision_distance=10):
        super().__init__()
        self.monitor_api = monitor_api
        self.execute_api = execute_api
        self.adaptation_schema = adaptation_schema
        self.max_uav_speed = max_uav_speed
        self.collision_distance = collision_distance
        self.observation_radius = 8  # Initial default

    def monitor(self):
        """
        Fetch data from the /monitor API endpoint and preprocess it for decision-making.
        """
        response = requests.get(self.monitor_api)
        if response.status_code != 200:
            raise Exception(f"Monitor API Error: {response.status_code}")
        data = response.json()

        # Extract constants and dynamic values
        constants = data.get("constants", {})
        dynamic_values = data.get("dynamicValues", {})

        # Extract UAV details, wind, and simulation parameters
        uav_details = dynamic_values.get("uavDetails", [])
        fire_zones = dynamic_values.get("fire_zones", [])  # Assuming fire zones are included
        wind = {
            "active": constants.get("activateWind", False),
            "direction": constants.get("windDirection", "none"),
            "velocity": constants.get("windVelocity", 0),
        }
        smoke_active = constants.get("activateSmoke", False)
        fire_spread_speed = constants.get("fireSpreadSpeed", 2)

        return {
            "uav_details": uav_details,
            "fire_zones": fire_zones,
            "wind": wind,
            "smoke_active": smoke_active,
            "fire_spread_speed": fire_spread_speed,
        }

    def analyze(self, data):
        """
        Analyze the current state and determine necessary adjustments.
        """
        uav_details = data["uav_details"]
        fire_zones = data["fire_zones"]
        wind = data["wind"]
        fire_spread_speed = data["fire_spread_speed"]

        # Step 1: Prioritize fire zones
        prioritized_zones = self.prioritize_fire_zones(fire_zones, uav_details)

        # Step 2: Predict fire spread
        predicted_zones = self.predict_fire_spread(wind, fire_zones)

        # Step 3: Detect potential collisions
        collision_warnings = self.detect_collisions(uav_details, self.collision_distance)

        # Step 4: Adjust observation radius dynamically
        self.adjust_observation_radius(fire_spread_speed)

        return {
            "prioritized_zones": prioritized_zones,
            "predicted_zones": predicted_zones,
            "collision_warnings": collision_warnings,
        }

    def plan(self, analysis_results):
        """
        Plan UAV movements and configurations to optimize wildfire tracking and collision avoidance.
        """
        uav_details = analysis_results.get("uav_details")
        prioritized_zones = analysis_results.get("prioritized_zones")
        collision_warnings = analysis_results.get("collision_warnings")

        # Allocate UAVs to prioritized fire zones
        assignments = self.allocate_uavs(prioritized_zones, uav_details)

        # Resolve collisions
        collision_resolutions = []
        for collision in collision_warnings:
            collision_resolutions.extend(self.resolve_collision(collision[0], collision[1]))

        # Combine assignments and collision resolutions
        return assignments + collision_resolutions

    def execute(self, adjustments):
        """
        Execute the planned adjustments by sending them to the /execute API endpoint.
        """
        for adj in adjustments:
            response = requests.put(self.execute_api, json=adj)
            if response.status_code != 200:
                raise Exception(f"Execution API Error: {response.status_code}")

    def prioritize_fire_zones(self, fire_zones, uav_positions):
        """
        Prioritize fire zones based on intensity, proximity, and UAV coverage.
        """
        priorities = []
        for zone in fire_zones:
            distance_to_nearest_uav = min(
                np.linalg.norm(np.array([zone["x"], zone["y"]]) - np.array([uav["x"], uav["y"]]))
                for uav in uav_positions
            )
            priority_score = zone["intensity"] / (distance_to_nearest_uav + 1)  # Avoid division by zero
            priorities.append({"zone": zone, "priority": priority_score})

        return sorted(priorities, key=lambda x: x["priority"], reverse=True)

    def predict_fire_spread(self, wind, fire_zones):
        """
        Predict the next fire spread zone based on wind direction.
        """
        predicted_zones = []
        for zone in fire_zones:
            if wind["direction"] == "north":
                predicted_zones.append({"x": zone["x"], "y": zone["y"] - 1})
            elif wind["direction"] == "south":
                predicted_zones.append({"x": zone["x"], "y": zone["y"] + 1})
            elif wind["direction"] == "east":
                predicted_zones.append({"x": zone["x"] + 1, "y": zone["y"]})
            elif wind["direction"] == "west":
                predicted_zones.append({"x": zone["x"] - 1, "y": zone["y"]})
        return predicted_zones

    def adjust_observation_radius(self, fire_spread_speed):
        """
        Adjust the UAV observation radius based on fire spread speed.
        """
        if fire_spread_speed > 3:
            self.observation_radius = max(self.observation_radius - 1, 5)  # Reduce radius for faster fires
        else:
            self.observation_radius = min(self.observation_radius + 1, 15)  # Increase for slower fires

    def allocate_uavs(self, prioritized_zones, uav_positions):
        """
        Allocate UAVs to prioritized fire zones.
        """
        assignments = []
        for zone in prioritized_zones:
            nearest_uav = min(
                uav_positions,
                key=lambda uav: np.linalg.norm(np.array([zone["zone"]["x"], zone["zone"]["y"]]) - np.array([uav["x"], uav["y"]]))
            )
            assignments.append({
                "id": nearest_uav["id"],
                "action": "move",
                "target": [zone["zone"]["x"], zone["zone"]["y"]],
                "speed": self.max_uav_speed,
            })
        return assignments

    def detect_collisions(self, uav_details, security_distance):
        """
        Detect potential UAV collisions based on their proximity.
        """
        collisions = []
        for i, uav1 in enumerate(uav_details):
            for uav2 in uav_details[i + 1:]:
                distance = np.sqrt((uav1["x"] - uav2["x"]) ** 2 + (uav1["y"] - uav2["y"]) ** 2)
                if distance < security_distance:
                    collisions.append((uav1, uav2))
        return collisions

    def resolve_collision(self, uav1, uav2):
        """
        Generate movement adjustments to resolve UAV collisions.
        """
        adjustments = []
        for uav in [uav1, uav2]:
            adjustments.append({
                "id": uav["id"],
                "action": "move",
                "target": [
                    uav["x"] + np.random.choice([-1, 1]),
                    uav["y"] + np.random.choice([-1, 1]),
                ],
                "speed": self.max_uav_speed / 2,  # Slow down during collision resolution
            })
        return adjustments
