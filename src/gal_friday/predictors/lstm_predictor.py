"""Implementation of the PredictorInterface for LSTM models (TensorFlow/Keras or PyTorch)."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

# Import TensorFlow or PyTorch as needed, e.g.:
# import tensorflow as tf
# import torch
from ..interfaces.predictor_interface import PredictorInterface


class LSTMPredictor(PredictorInterface):
    """Implementation of PredictorInterface for LSTM models."""

    def __init__(
        self, model_path: str, model_id: str, config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the LSTMPredictor.

        Args:
            model_path: Path to the LSTM model file (e.g., .h5, .keras, .pt).
            model_id: Unique identifier for this model.
            config: Additional configuration parameters. Expected to contain
                    'scaler_path': Optional[str] and 'model_feature_names': List[str].
                    May also contain framework-specific configs like 'batch_size'.
        """
        super().__init__(model_path, model_id, config)
        # Add any LSTM specific initializations, e.g., framework choice
        self.framework = self.config.get("framework", "tensorflow").lower()  # or "pytorch"
        self.logger.info(
            f"LSTMPredictor initialized for model {self.model_id} using {self.framework} framework."
        )

    def load_assets(self) -> None:
        """
        Load an LSTM model and its associated scaler.
        Sets self.model and self.scaler.
        """
        self.logger.info(f"Loading assets for LSTM model: {self.model_id}")

        # --- Load Model ---
        try:
            if not Path(self.model_path).exists():
                self.logger.error(f"LSTM Model file not found: {self.model_path}")
                raise FileNotFoundError(f"LSTM Model file not found: {self.model_path}")

            if self.framework == "tensorflow":
                # Example for TensorFlow/Keras
                # Ensure tensorflow is installed: pip install tensorflow
                import tensorflow as tf

                self.model = tf.keras.models.load_model(self.model_path)
                self.logger.info(
                    f"TensorFlow/Keras LSTM model loaded successfully from {self.model_path}"
                )
            elif self.framework == "pytorch":
                # Example for PyTorch
                # Ensure torch is installed: pip install torch
                # import torch
                # self.model = torch.load(self.model_path)
                # self.model.eval() # Set to evaluation mode
                self.logger.warning("PyTorch LSTM model loading not fully implemented yet.")
                raise NotImplementedError(
                    "PyTorch LSTM loading needs specific model class and state_dict loading."
                )
            else:
                self.logger.error(f"Unsupported LSTM framework: {self.framework}")
                raise ValueError(f"Unsupported LSTM framework: {self.framework}")

        except FileNotFoundError:
            raise
        except ImportError as e:
            self.logger.error(f"Import error for {self.framework}: {e}. Is it installed?")
            raise RuntimeError(f"Required ML framework ({self.framework}) not installed.") from e
        except Exception:
            self.logger.exception(f"Failed to load LSTM model from {self.model_path}")
            raise

        # --- Load Scaler ---
        if self.scaler_path:
            try:
                if not Path(self.scaler_path).exists():
                    self.logger.error(f"Scaler file not found: {self.scaler_path}")
                    raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info(f"Scaler loaded successfully from {self.scaler_path}")
            except FileNotFoundError:
                raise
            except Exception:
                self.logger.exception(f"Failed to load scaler from {self.scaler_path}")
                raise
        else:
            self.logger.info(
                "No scaler_path provided for LSTM model. Proceeding without a scaler."
            )
            self.scaler = None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate predictions using the LSTM model.
        LSTM models often require input features in a specific 3D shape:
        (n_samples, n_timesteps, n_features_per_timestep).
        This method expects a 1D array of features for ONE sample at ONE point in time.
        The `run_inference_in_process` method will need to handle historical
        feature buffering and reshaping if the LSTM model is stateful or expects sequences.
        For now, this predict method assumes the input is for a single timestep prediction
        and the model is set up to handle that, or that reshaping is done by the runner.
        """
        if self.model is None:
            self.logger.error("LSTM Model not loaded. Cannot predict.")
            raise TypeError("LSTM model is not loaded.")

        if features.ndim != 1:
            self.logger.error(
                f"Input features must be a 1D array for LSTMPredictor, got {features.ndim}D."
            )
            raise ValueError("Input features must be a 1D array for LSTMPredictor.")

        features_2d = features.reshape(1, -1)  # (1, n_features_per_timestep)

        if self.scaler:
            try:
                processed_features_2d = self.scaler.transform(features_2d)
            except Exception as e:
                self.logger.exception("Error applying scaler transform to LSTM features.")
                raise ValueError(f"Error during LSTM feature scaling: {e!s}") from e
        else:
            processed_features_2d = features_2d

        # LSTM specific reshaping: (n_samples, n_timesteps, n_features_per_timestep)
        # This is a placeholder. Actual reshaping depends on model architecture (n_timesteps).
        # The `run_inference_in_process` should manage feature history if sequence > 1.
        # If model takes (1, 1, n_features), for a single timestep:
        n_features_per_timestep = processed_features_2d.shape[1]
        model_input = processed_features_2d.reshape(1, 1, n_features_per_timestep)

        try:
            if self.framework == "tensorflow":
                raw_predictions = self.model.predict(
                    model_input, verbose=0
                )  # verbose=0 for less Keras logs
                # Output shape might be (1, 1, n_outputs) or (1, n_outputs)
                return np.asarray(raw_predictions, dtype=np.float32).flatten()

            if self.framework == "pytorch":
                # import torch
                # with torch.no_grad():
                #     torch_input = torch.from_numpy(model_input).float()
                #     # Add device handling: torch_input = torch_input.to(self.device)
                #     raw_predictions = self.model(torch_input)
                # return raw_predictions.cpu().numpy().flatten().astype(np.float32)
                raise NotImplementedError("PyTorch LSTM prediction not fully implemented.")
            # Should have been caught in load_assets
            raise ValueError(f"Unsupported framework {self.framework} for prediction")

        except Exception as e:
            self.logger.exception("LSTM prediction failed.")
            raise ValueError(f"LSTM prediction error: {e!s}") from e

    @property
    def expected_feature_names(self) -> Optional[List[str]]:
        """Return the list of feature names the model expects from config."""
        # LSTMs usually care about the order and number of features, names less so
        # if the input layer is defined by number of features.
        return self.config.get("model_feature_names")

    @classmethod
    def run_inference_in_process(
        cls,
        model_id: str,
        model_path: str,
        scaler_path: Optional[str],
        feature_sequence: np.ndarray,  # Renamed: Expects 2D raw feature sequence (n_timesteps, n_features_per_step)
        model_feature_names: List[str],  # From model config (features *per timestep*)
        predictor_specific_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Load LSTM model and scaler, preprocess, predict. Executed in a separate process.
        This method now expects `feature_sequence` to be the complete 2D sequence.
        """
        logger = logging.getLogger(f"{cls.__name__}:{model_id}.run_inference_in_process")
        logger.debug(
            f"Starting LSTM inference for model {model_id} in separate process with input shape {feature_sequence.shape if feature_sequence is not None else 'None'}."
        )

        # This is a simplified instantiation. A real TF/PyTorch setup in a new process
        # might require careful GPU memory management, session clearing (TF1), etc.
        # For TF2/Keras, tf.keras.models.load_model should be mostly self-contained.

        framework = predictor_specific_config.get("framework", "tensorflow").lower()

        # --- Minimal Model & Scaler Loading (No class instance here) ---
        model_asset: Any = None
        scaler_asset: Any = None

        try:
            # Load Model
            if not Path(model_path).exists():
                return {"error": f"Model file not found: {model_path}", "model_id": model_id}

            if framework == "tensorflow":
                try:
                    import tensorflow as tf

                    # Suppress TensorFlow logging for cleaner output from worker process
                    tf.get_logger().setLevel("ERROR")
                    # Or, more selectively: logging.getLogger('tensorflow').setLevel(logging.ERROR)
                    model_asset = tf.keras.models.load_model(model_path)
                    logger.debug(f"TF Model {model_id} loaded from {model_path}")
                except Exception as e_load:
                    return {
                        "error": f"Failed to load TF model {model_path}: {e_load!s}",
                        "model_id": model_id,
                    }
            elif framework == "pytorch":
                # PyTorch model loading needs the class definition.
                # This is more complex to do generically in a separate process without shared code or
                # pickling the model class, which is often not feasible.
                # A common pattern is to save state_dict and re-instantiate the model.
                logger.warning(
                    "PyTorch model loading in run_inference_in_process is a placeholder."
                )
                return {
                    "error": "PyTorch inference in separate process not fully implemented.",
                    "model_id": model_id,
                }
            else:
                return {
                    "error": f"Unsupported framework {framework} for LSTM inference.",
                    "model_id": model_id,
                }

            # Load Scaler
            if scaler_path:
                if not Path(scaler_path).exists():
                    return {"error": f"Scaler file not found: {scaler_path}", "model_id": model_id}
                try:
                    scaler_asset = joblib.load(scaler_path)
                    logger.debug(f"Scaler for {model_id} loaded from {scaler_path}")
                except Exception as e_scale_load:
                    return {
                        "error": f"Failed to load scaler {scaler_path}: {e_scale_load!s}",
                        "model_id": model_id,
                    }

            # --- Prepare features ---
            # feature_sequence is already the 2D sequence (timesteps, features_per_timestep)
            if feature_sequence.ndim != 2:
                return {
                    "error": f"Feature sequence must be 2D for LSTM processing, got {feature_sequence.ndim}D.",
                    "model_id": model_id,
                }

            # Scaler is applied to the 2D sequence.
            # Assumes scaler was fit on data of shape (n_samples * n_timesteps, n_features_per_timestep)
            # or (n_samples, n_features_per_timestep) if scaling is applied per-feature independently of timestep structure.
            # For typical time-series scaling, you scale each feature column across all its timesteps.
            # If scaler expects (samples, features), and our sequence is (timesteps, features), it should work if interpreted correctly.
            # Or, scaler might need to be applied to a reshaped version if it was fit on (N, features_per_timestep)
            # This part needs careful alignment with how scalers are trained for sequence data.
            # A common approach: scaler.fit(all_flattened_features_of_one_type), then apply to that column in the sequence.
            # For simplicity now, assume scaler.transform works on (timesteps, features_per_timestep)

            processed_sequence = feature_sequence  # Placeholder if no scaler
            if scaler_asset:
                try:
                    # If scaler was fit on (samples, features_per_timestep), and sequence is (timesteps, features_per_timestep)
                    # this should correctly scale each feature across the timesteps.
                    processed_sequence = scaler_asset.transform(feature_sequence)
                    logger.debug("Feature sequence scaled.")
                except Exception as e_scale_apply:
                    return {
                        "error": f"Error applying scaler to sequence: {e_scale_apply!s}",
                        "model_id": model_id,
                    }
            else:
                logger.debug("No scaler used for sequence.")

            # --- LSTM specific reshaping for model input ---
            # Model expects (batch_size, timesteps, features_per_timestep)
            # Here, batch_size is 1 as we process one sequence at a time.
            # processed_sequence is already (timesteps, features_per_timestep)
            model_input = processed_sequence.reshape(
                1, processed_sequence.shape[0], processed_sequence.shape[1]
            )

            # No longer need these, as feature_sequence is already the sequence:
            # n_timesteps = predictor_specific_config.get("n_timesteps", 1)
            # if not isinstance(n_timesteps, int) or n_timesteps < 1:
            #     n_timesteps = 1
            # if processed_features_2d.shape[1] % n_timesteps != 0:
            #      return {"error": f"Total features ({processed_features_2d.shape[1]}) not divisible by n_timesteps ({n_timesteps}).", "model_id": model_id}

            logger.debug(f"LSTM model input shape: {model_input.shape}")

            # --- Predict ---
            prediction_output: np.ndarray
            raw_probabilities: Optional[np.ndarray] = (
                None  # To store multi-class probabilities if available
            )

            if framework == "tensorflow":
                raw_prediction_tensor = model_asset.predict(model_input, verbose=0)
                # raw_prediction_tensor shape could be (1, n_outputs) or (1, 1, n_outputs)
                prediction_output = np.asarray(raw_prediction_tensor).flatten()  # Flatten to 1D
                raw_probabilities = prediction_output  # If output is already probabilities
            # PyTorch prediction would go here
            # elif framework == "pytorch":
            #     # prediction_output = ...
            #     # raw_probabilities = ...
            else:
                # Should have been caught earlier, but defensive
                return {
                    "error": f"Unsupported framework {framework} in predict step.",
                    "model_id": model_id,
                }

            # Determine final_prediction_value and confidence_float
            final_prediction_value: float
            confidence_float: Optional[float] = None

            if prediction_output.size == 0:
                return {"error": "Prediction output array is empty.", "model_id": model_id}

            # Assuming for classification, the model outputs probabilities for each class.
            # If it's a binary classifier (output_size=1, sigmoid activation), prediction_output[0] is P(class_1)
            # If it's multi-class (output_size > 1, softmax activation), prediction_output contains P(class_0), P(class_1), ...

            is_multiclass_output = (
                prediction_output.size > 1
                and predictor_specific_config.get("output_activation", "") == "softmax"
            )
            is_binary_sigmoid_output = (
                prediction_output.size == 1
            )  # Often implies sigmoid output for P(class_1)

            if is_multiclass_output:
                # For multi-class with softmax, prediction is the class with max probability
                # We might want to return the probability of the *target positive class* if defined,
                # or simply the max probability as confidence and its class index as prediction.
                # For simplicity here, let's assume the task is to get the max probability as a general signal.
                final_prediction_value = float(np.max(prediction_output))  # Max probability
                confidence_float = final_prediction_value
                # If you needed the predicted class index: predicted_class_index = np.argmax(prediction_output)
            elif is_binary_sigmoid_output:
                # This is likely P(class_1)
                final_prediction_value = float(prediction_output[0])
                confidence_float = final_prediction_value
            else:  # Single regression output or unknown structure
                final_prediction_value = float(prediction_output[0])
                # Confidence for regression is not straightforward from output value alone
                confidence_float = None

            logger.debug(
                f"LSTM Prediction for {model_id} successful: {final_prediction_value}, Confidence: {confidence_float}"
            )
            return {
                "prediction": final_prediction_value,
                "confidence": confidence_float,
                "model_id": model_id,
            }

        except Exception as e:
            logger.error(
                f"Generic error during LSTM inference for {model_id}: {e!s}", exc_info=True
            )
            return {"error": f"LSTM Inference failed: {e!s}", "model_id": model_id}
