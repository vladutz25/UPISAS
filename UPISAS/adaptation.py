import requests
import numpy as np
from strategy import Strategy


class AdvancedWildfireAdaptationStrategy(Strategy):
    def __init__(self, monitor_api, execute_api, adaptation_schema, max_uav_speed=2,
                 collision_distance=10):
        super().__init__()
        self.monitor_api = monitor_api
        self.execute_api = execute_api
        self.adaptation_schema = adaptation_schema
        self.max_uav_speed = max_uav_speed
        self.collision_distance = collision_distance

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
        wind = {
            "active": constants.get("activateWind", False),
            "direction": constants.get("windDirection", "none"),
            "velocity": constants.get("windVelocity", 0),
        }
        smoke_active = constants.get("activateSmoke", False)
        observation_radius = constants.get("observationRadius", 8)
        security_distance = constants.get("securityDistance", 10)

        return {
            "uav_details": uav_details,
            "wind": wind,
            "smoke_active": smoke_active,
            "observation_radius": observation_radius,
            "security_distance": security_distance,
            "fire_spread_speed": constants.get("fireSpreadSpeed", 2),
        }

    def analyze(self, data):
        """
        Analyze the current state and determine necessary adjustments.
        """
        uav_details = data["uav_details"]
        wind = data["wind"]

        # Step 1: Detect potential collisions
        collision_warnings = self.detect_collisions(uav_details, data["security_distance"])

        # Step 2: Plan adjustments for UAV movements
        adjustments = self.plan_adjustments(uav_details, collision_warnings, wind,
                                            data["observation_radius"])

        return adjustments

    def plan_adjustments(self, uav_details, collision_warnings, wind, observation_radius):
        """
        Plan UAV movements and configurations to optimize wildfire tracking and collision avoidance.
        """
        adjustments = []

        for uav in uav_details:
            # Example movement based on current position
            new_x = uav["x"] + np.random.choice([-1, 0, 1])  # Random step
            new_y = uav["y"] + np.random.choice([-1, 0, 1])  # Random step

            # Ensure UAVs maintain observation radius
            adjustments.append({
                "id": uav["id"],
                "action": "move",
                "target": [new_x, new_y],
                "speed": self.max_uav_speed,
            })

        # Resolve collisions by adding collision-specific adjustments
        for collision in collision_warnings:
            adjustments.extend(
                self.resolve_collision(collision[0], collision[1], observation_radius))

        return adjustments

    def execute(self, adjustments):
        """
        Execute the planned adjustments by sending them to the /execute API endpoint.
        """
        for adj in adjustments:
            response = requests.put(self.execute_api, json=adj)
            if response.status_code != 200:
                raise Exception(f"Execution API Error: {response.status_code}")

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

    def resolve_collision(self, uav1, uav2, observation_radius):
        """
        Generate movement adjustments to resolve UAV collisions.
        """
        adjustments = []
        for uav in [uav1, uav2]:
            # Example re-routing to resolve collisions
            adjustments.append({
                "id": uav["id"],
                "action": "move",
                "target": [
                    uav["x"] + np.random.choice([-observation_radius, observation_radius]),
                    uav["y"] + np.random.choice([-observation_radius, observation_radius]),
                ],
                "speed": self.max_uav_speed / 2,  # Slow down during collision resolution
            })
        return adjustments
