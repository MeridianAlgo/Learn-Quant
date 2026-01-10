"""
Kalman Filter for Financial Time Series
---------------------------------------
This utility implements a basic 1-Dimensional Kalman Filter.
In quantitative finance, Kalman Filters are often used for:
1. Moving Average smoothing with less lag.
2. Estimating the dynamic hedge ratio between two assets (beta).
3. Extracting signal from noisy market data.

The system assumes:
x_k = A * x_{k-1} + B * u_k + w_k  (State transition)
z_k = H * x_k + v_k                (Measurement)
"""


class KalmanFilter1D:
    """
    A simple 1D Kalman Filter implementation.

    Attributes:
        process_variance (float): The variance of the process noise (Q).
        measurement_variance (float): The variance of the measurement noise (R).
        estimated_value (float): The current best estimate of the state (x).
        estimation_error (float): The current covariance (error) of the estimation (P).
    """

    def __init__(
        self,
        process_variance,
        measurement_variance,
        estimated_value=0,
        estimation_error=1,
    ):
        """
        Initializes the Kalman Filter.

        Args:
            process_variance (float): 'Q' - How much the system changes on its own.
                                      volatility/uncertainty in the model.
            measurement_variance (float): 'R' - How much noise is in the measurement.
            estimated_value (float): Initial guess for the value.
            estimation_error (float): Initial guess for the error (P).
        """
        self.process_variance = process_variance  # Q
        self.measurement_variance = measurement_variance  # R
        self.estimated_value = estimated_value  # x
        self.estimation_error = estimation_error  # P

        # In a simple 1D random walk model for price, A=1, B=0, H=1 usually
        self.kalman_gain = 0

    def update(self, measurement):
        """
        The "Correction" or "Update" phase.
        Updates the estimate based on the new measurement.

        Args:
            measurement (float): The new observed data point (z).

        Returns:
            float: The updated estimated value.
        """
        # Calculate Kalman Gain: K = P / (P + R)
        self.kalman_gain = self.estimation_error / (
            self.estimation_error + self.measurement_variance
        )

        # Update estimate: x = x + K * (z - x)
        self.estimated_value = self.estimated_value + self.kalman_gain * (
            measurement - self.estimated_value
        )

        # Update error covariance: P = (1 - K) * P
        self.estimation_error = (1 - self.kalman_gain) * self.estimation_error

        return self.estimated_value

    def predict(self, u=0):
        """
        The "Prediction" phase.
        Predicts the next state. For a random walk price, we just assume x_next = x_curr.
        So P_next = P_curr + Q.

        Args:
            u (float): Optional control input (usually 0 for prices).
        """
        # Prediction of state (x) assumes constant model for Random Walk (A=1)
        # x = x + u (if we had control input)
        # Prediction of error covariance (P)
        # P = P + Q
        self.estimation_error = self.estimation_error + self.process_variance


if __name__ == "__main__":
    # Example Usage: Tracking a noisy constant value
    # True value = 100
    # Measurements are noisy around 100

    true_value = 100.0
    kf = KalmanFilter1D(
        process_variance=1e-5, measurement_variance=0.1**2, estimated_value=98.0
    )

    print(f"Initial Estimate: {kf.estimated_value}")

    # Simulate some noisy measurements
    measurements = [99.5, 100.2, 99.8, 100.5, 100.1]

    for i, z in enumerate(measurements):
        kf.predict()
        estimate = kf.update(z)
        print(
            f"Step {i+1}: Measured={z}, Estimate={estimate:.4f}, ErrorEst={kf.estimation_error:.4f}"
        )
