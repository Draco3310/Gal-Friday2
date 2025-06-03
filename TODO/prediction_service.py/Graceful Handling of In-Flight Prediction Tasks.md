# File: gal_friday/prediction_service.py
# Original TODO: Line 123 - Implement graceful handling of in-flight prediction tasks
# Context: Within the PredictionService class, specifically in the shutdown/cleanup logic

# --- Dependencies/Collaborators ---
# - asyncio: For task management and cancellation
# - LoggerService: For logging task states and cleanup progress
# - PredictionTask: Class representing an in-flight prediction task
# - PredictionResult: Class representing a prediction result
# - PubSubManager: For publishing prediction completion/failure events

# --- Implementation Notes ---
# 1. Track all in-flight prediction tasks
# 2. During shutdown:
#    a. Prevent new predictions from starting
#    b. Allow in-flight predictions to complete with a timeout
#    c. Cancel any remaining tasks after the timeout
#    d. Clean up resources

# --- Pseudocode Implementation ---

# In PredictionService.__init__:
#     self._in_flight_tasks: dict[str, asyncio.Task] = {}  # task_id -> Task
#     self._shutting_down = asyncio.Event()
#     self._shutdown_timeout = config.get("prediction_service", {}).get("shutdown_timeout_seconds", 30)

async def shutdown(self) -> None:
    """Gracefully shut down the prediction service."""
    if self._shutting_down.is_set():
        return

    self._shutting_down.set()
    self.logger.info("Initiating graceful shutdown of prediction service...")
    
    # 1. Prevent new predictions
    self.logger.debug("Preventing new prediction tasks from starting...")
    
    # 2. Get list of in-flight task IDs
    task_ids = list(self._in_flight_tasks.keys())
    if not task_ids:
        self.logger.info("No in-flight prediction tasks to wait for.")
        return
        
    self.logger.info(f"Waiting for {len(task_ids)} in-flight prediction tasks to complete...")
    
    # 3. Wait for tasks to complete with timeout
    done, pending = await asyncio.wait(
        list(self._in_flight_tasks.values()),
        timeout=self._shutdown_timeout,
        return_when=asyncio.ALL_COMPLETED
    )
    
    # 4. Handle completed tasks
    for task in done:
        try:
            await task  # Re-raise any exceptions
        except asyncio.CancelledError:
            self.logger.debug("Prediction task was cancelled during shutdown")
        except Exception as e:
            self.logger.error(f"Prediction task failed during shutdown: {e}")
    
    # 5. Cancel any remaining tasks
    if pending:
        self.logger.warning(f"Cancelling {len(pending)} prediction tasks that didn't complete in time")
        for task in pending:
            task.cancel()
        
        # Wait a short time for cancellation to propagate
        await asyncio.sleep(0.1)
    
    self.logger.info("Prediction service shutdown complete")

async def _create_prediction_task(self, features: dict) -> str:
    """Create a new prediction task with proper tracking."""
    if self._shutting_down.is_set():
        raise RuntimeError("Cannot start new prediction: service is shutting down")
    
    task_id = str(uuid.uuid4())
    task = asyncio.create_task(
        self._predict_async(features, task_id),
        name=f"prediction-{task_id[:8]}"
    )
    
    # Add callback to clean up the task when done
    task.add_done_callback(lambda t: self._in_flight_tasks.pop(task_id, None))
    
    # Store the task
    self._in_flight_tasks[task_id] = task
    return task_id

async def _predict_async(self, features: dict, task_id: str) -> PredictionResult:
    """Wrapper around the actual prediction logic with proper error handling."""
    try:
        # Your existing prediction logic here
        result = await self._make_prediction(features)
        return result
    except asyncio.CancelledError:
        self.logger.warning(f"Prediction task {task_id} was cancelled")
        raise
    except Exception as e:
        self.logger.error(f"Prediction failed: {e}")
        raise

# --- Usage Example ---
# In the main application shutdown:
# async def stop(self):
#     await self.prediction_service.shutdown()
#     # ... shutdown other services ...