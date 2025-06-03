# File: gal_friday/prediction_service.py
# Original TODO: Line 974 - Implement confidence floor in future versions
# Context: Within _apply_ensembling_and_publish method, or a preceding/succeeding helper method.

# --- Existing Code Context (Simplified from prediction_service.py around line 974) ---
# ensemble_strategy = self._service_config.get("ensemble_strategy", "none").lower()
# ensemble_weights = self._service_config.get("ensemble_weights", {})
# confidence_floor_config = float(self._service_config.get("confidence_floor", 0.0)) # Default to 0.0 (no filtering)
#
# # Group predictions by target for ensembling
# predictions_by_target: dict[str, list[dict[str, Any]]] = {}
# for pred_data in successful_predictions: # successful_predictions is a list of dicts
#     # pred_data typically contains: "model_id", "prediction_value", "prediction_target", "confidence", "config"
#     target = pred_data["prediction_target"]
#     if target not in predictions_by_target:
#         predictions_by_target[target] = []
#     predictions_by_target[target].append(pred_data)
#
# final_predictions_to_publish: list[dict[str, Any]] = []
#
# IF ensemble_strategy == "none" or not predictions_by_target:
#     # Apply confidence floor BEFORE deciding to publish individual predictions
#     FOR pred_data IN successful_predictions:
#         current_confidence = pred_data.get("confidence")
#         IF current_confidence IS None OR current_confidence >= confidence_floor_config:
#             final_predictions_to_publish.append(pred_data)
#         ELSE:
#             LOG info f"Prediction from {pred_data['model_id']} for target {pred_data['prediction_target']} dropped due to low confidence ({current_confidence} < {confidence_floor_config})."
#         ENDIF
#     ENDFOR
# ELSE:
#     # Apply ensembling for each target group
#     FOR target, preds_for_target IN predictions_by_target.items():
#         IF NOT preds_for_target:
#             CONTINUE
#         ENDIF
#
#         # Option 1: Apply confidence floor to individual models BEFORE ensembling
#         filtered_preds_for_ensembling = []
#         FOR pred_data IN preds_for_target:
#             current_confidence = pred_data.get("confidence")
#             IF current_confidence IS None OR current_confidence >= confidence_floor_config:
#                 filtered_preds_for_ensembling.append(pred_data)
#             ELSE:
#                 LOG info f"Prediction from {pred_data['model_id']} for target {target} excluded from ensemble due to low confidence ({current_confidence} < {confidence_floor_config})."
#             ENDIF
#         ENDFOR
#
#         IF NOT filtered_preds_for_ensembling:
#             LOG info f"No predictions for target {target} met confidence floor for ensembling. No ensemble prediction will be generated."
#             CONTINUE # Skip ensembling for this target if no models meet confidence
#         ENDIF
#
#         # Proceed with ensembling using 'filtered_preds_for_ensembling'
#         # ... (existing ensembling logic: _apply_average_ensembling, _apply_weighted_average_ensembling) ...
#         # The ensembling methods would then add their result to final_predictions_to_publish
#
#         # Option 2: Apply confidence floor AFTER ensembling (to the ensemble's confidence)
#         # This would require the ensembling methods to also calculate/propagate an ensemble confidence.
#         # IF ensembled_prediction.confidence IS None OR ensembled_prediction.confidence >= confidence_floor_config:
#         #     final_predictions_to_publish.append(ensembled_prediction)
#         # ELSE:
#         #     LOG info f"Ensembled prediction for target {target} dropped due to low ensemble confidence ({ensembled_prediction.confidence} < {confidence_floor_config})."
#         # ENDIF
#     ENDFOR
# ENDIF
#
# # Publish all final predictions that made it to final_predictions_to_publish
# FOR pred IN final_predictions_to_publish:
#     # ... (existing publishing logic using PredictionEvent) ...
# ENDFOR

# --- Detailed Pseudocode for Confidence Floor Implementation ---

# 1. Configuration:
#    - Ensure `confidence_floor` is configurable in `config.yaml` under `prediction_service`.
#      - It could be a single global value (e.g., `0.7`).
#      - Or, more flexibly, per `prediction_target` (e.g., `confidence_floors: {"price_direction_1m": 0.75, "volatility_change": 0.6}`).
#      - Or even per `model_id` if very granular control is needed (less common for a general floor).
#    - `confidence_floor_value = ConfigManager.get_confidence_floor_for_target(target, default=0.0)`
#      (assuming per-target configuration is chosen).

# 2. Application Point:
#    The pseudocode above outlines two main options for applying the floor when ensembling is active:
#    * **Option A: Filter individual model predictions *before* ensembling.** (Shown as default in the context above)
#        * Pros: Simpler, ensures only confident models contribute to the ensemble.
#        * Cons: Might discard too much information if many models are slightly below the floor but collectively could produce a good ensemble.
#    * **Option B: Filter the *ensembled* prediction based on its calculated confidence.**
#        * Pros: Allows all models to contribute; the ensemble's confidence might be more robust.
#        * Cons: Requires ensembling methods to reliably calculate and return a meaningful ensemble confidence score. This can be tricky (e.g., simple average of confidences might not be ideal).

#    If `ensemble_strategy` is "none":
#    * The floor is applied directly to each individual model's prediction before it's added to `final_predictions_to_publish`.

# 3. Logic for Applying the Floor (Generic, applied at the chosen point):
#    FUNCTION apply_confidence_floor(prediction_data_list, config_confidence_floor):
#        passed_predictions = []
#        FOR prediction IN prediction_data_list:
#            model_confidence = prediction.get("confidence") # Assume confidence is a float between 0.0 and 1.0, or None
#
#            IF model_confidence IS None:
#                LOG warning f"Prediction from {prediction.get('model_id')} for target {prediction.get('prediction_target')} has no confidence score. Assuming it passes the floor."
#                # Policy decision:
#                #   Option 1: Assume it passes (as shown here)
#                #   Option 2: Assume it fails (safer, if confidence is expected)
#                #   Option 3: Have a separate config for how to treat missing confidence.
#                passed_predictions.append(prediction)
#                CONTINUE
#            ENDIF
#
#            IF model_confidence >= config_confidence_floor:
#                passed_predictions.append(prediction)
#            ELSE:
#                LOG info f"Prediction from {prediction.get('model_id')} for target {prediction.get('prediction_target')} (confidence: {model_confidence}) dropped. Below floor of {config_confidence_floor}."
#                # Optionally, publish an event for dropped predictions if detailed tracking is needed.
#            ENDIF
#        ENDFOR
#        RETURN passed_predictions
#    ENDFUNCTION

# 4. Integration:
#    - If Option A (filter before ensembling):
#      `predictions_for_target = apply_confidence_floor(predictions_for_target, confidence_floor_for_this_target)`
#      Then, if `predictions_for_target` is not empty, proceed to ensembling.
#
#    - If Option B (filter after ensembling):
#      `ensembled_prediction = perform_ensembling(...)`
#      `final_ensembled_list = apply_confidence_floor([ensembled_prediction], confidence_floor_for_this_target)`
#      Add contents of `final_ensembled_list` to `final_predictions_to_publish`.
#
#    - If no ensembling:
#      `final_predictions_to_publish = apply_confidence_floor(successful_predictions, global_confidence_floor_or_target_specific)`

# --- Considerations ---
# - Missing Confidence: How to handle predictions where `confidence` is `None`?
#   - Current pseudocode logs a warning and assumes it passes. This should be a deliberate policy choice.
#   - Alternative: Treat `None` as 0.0 confidence, effectively filtering it out unless the floor is 0.0.
# - Granularity of Floor: Global vs. per-target vs. per-model_id. Per-target seems like a good balance.
# - Ensemble Confidence Calculation: If filtering *after* ensembling, how is the ensemble's confidence determined?
#   - Simple average of contributing models' confidences?
#   - Weighted average based on model weights?
#   - More sophisticated measure (e.g., variance of predictions)?
#   This needs to be defined in the ensembling methods themselves.
# - Logging & Monitoring: Ensure clear logs for predictions dropped due to the confidence floor. This could be a metric to monitor.
# - Impact on Ensembling: If filtering before ensembling, consider the case where too few models pass the floor, potentially making the ensemble less reliable or impossible to compute. The logic should handle cases where `filtered_preds_for_ensembling` becomes empty.