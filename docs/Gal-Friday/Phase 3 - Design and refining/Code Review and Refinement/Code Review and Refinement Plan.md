Code Review & Refinement Plan (Task 3.18)
1. Setup & Automated Checks:
[x] Configure Tools: Ensure flake8, black, and mypy are installed (preferably in the project's virtual environment) and configured appropriately (e.g., in pyproject.toml or dedicated config files) to match project standards (line length, specific checks).
[x] Run black: Apply automatic formatting to the entire codebase (src/, scripts/, tests/ if present) for initial consistency.
[x] Run flake8: Execute the linter across the codebase. Document or immediately fix reported issues (e.g., unused imports, undefined names, style violations).
[x] Run mypy: Execute the type checker across the codebase. Document or fix type errors and inconsistencies. Pay special attention to Any types that could be more specific and issues related to Optional.
2. Manual Review Checklist (Apply to each module/key file):
A. Correctness & Logic:
[ ] Does the module's implementation align with its responsibilities defined in the architecture documents (architecture_concept_*.md, interface_definitions_*.md)?
[ ] Is the core logic within functions/methods correct based on requirements/design? (e.g., feature calculations, risk checks, order translation).
[ ] Are there any obvious logical flaws, edge cases missed, or potential race conditions (especially in async handlers)?
[ ] Are calculations (e.g., portfolio value, position sizing, feature values) handled correctly, especially regarding data types (Decimal vs. float)?
B. Error Handling & Robustness:
[ ] Are try...except blocks used appropriately to catch potential exceptions (e.g., KeyError, ValueError, TypeError, network errors, API errors, asyncio.CancelledError)?
[ ] Are exception handlers specific enough? Avoid overly broad except Exception:.
[ ] Is error logging clear and informative (using the integrated LoggerService)? Is exc_info=True used correctly for unexpected errors?
[ ] How does the module behave on invalid or unexpected input (e.g., malformed API responses, unexpected event payloads)?
[ ] Is there adequate validation for external data (config values, API responses, event payloads)?
C. asyncio Usage:
[ ] Are async and await used correctly?
[ ] Are there any potentially blocking synchronous calls within async def functions that could stall the event loop? (Consider I/O, heavy computation). If necessary, ensure they are properly offloaded (e.g., using loop.run_in_executor for CPU-bound tasks, like already done in PredictionService).
[ ] Is task management correct? (e.g., tasks created with asyncio.create_task, awaited or cancelled appropriately, especially during shutdown).
[ ] If shared state is accessed concurrently (even if unlikely in current MVP structure), are appropriate synchronization primitives (asyncio.Lock, etc.) used correctly? (e.g., check PortfolioManager._lock).
D. Dependencies & Imports:
[ ] Are imports organized correctly (standard library, third-party, application-specific)?
[ ] Are there any unnecessary imports? Any potential circular dependencies?
[ ] Is the use of if TYPE_CHECKING: blocks correct for handling placeholder types?
E. Configuration & Hardcoding:
[ ] Is configuration accessed consistently via ConfigManager?
[ ] Are there magic numbers or hardcoded strings that should be constants or configuration values? (Check API URLs, default parameters, thresholds, event type strings).
[ ] Are default configuration values sensible?
F. Logging:
[ ] Is LoggerService correctly injected and used in all modules?
[ ] Are log levels appropriate (DEBUG, INFO, WARNING, ERROR, CRITICAL)?
[ ] Are log messages clear, concise, and provide sufficient context? Is source_module passed correctly?
[ ] Is sensitive information (API keys, raw PII if applicable later) avoided in logs?
G. Readability & Style:
[ ] Is the code generally easy to follow and understand?
[ ] Are variable and function names clear and descriptive?
[ ] Is the code complexity reasonable? Are there overly long functions/methods that could be broken down?
[ ] Are comments used effectively for non-obvious logic, assumptions, or TODOs? Remove redundant/obvious comments.
[ ] Is PEP 8 style generally followed (enforced by black/flake8, but check logical aspects)?
H. Resource Management:
[ ] aiohttp.ClientSession: Ensure it's created and closed properly (e.g., in ExecutionHandler.start/stop).
[ ] ProcessPoolExecutor: Ensure it's managed correctly (typically by main.py) and shutdown gracefully.
[ ] asyncpg.Pool: Ensure it's created, closed, and connections acquired/released correctly (within LoggerService and potentially other DB-interacting modules later).
[ ] File Handles (if any): Ensure they are closed properly (e.g., using with open(...)).
I. Docstrings & Type Hinting:
[ ] Do public modules, classes, and methods have docstrings explaining their purpose?
[ ] Are type hints present and accurate for function signatures and key variables? (mypy helps verify).
3. Module-Specific Focus Areas:
main.py: Review application lifecycle (startup, shutdown), dependency injection logic, signal handling, process pool management, root logging setup.
logger_service.py: Review handler configuration, DB pool management (asyncpg), log formatting, async handling (_process_queue).
config_manager.py: Review file loading, key access methods, error handling for missing keys/files.
data_ingestor.py: Review WebSocket connection/reconnection logic, message parsing (JSON, checksum), L2 book maintenance, event publishing structure. Check indentation issues previously noted.
portfolio_manager.py: Review state update logic (funds, positions), equity/drawdown calculations, use of Decimal, handling of ExecutionReportEvent, synchronous get_current_state interface. Check logic changes made during logger integration.
risk_manager.py: Review risk limit checks (drawdown, per-trade), position size calculation, interaction with PortfolioManager's synchronous state.
feature_engine.py: Review data caching (L2, OHLCV deques), feature calculation logic (L2, TA using pandas-ta), DataFrame creation/handling, event triggering logic.
prediction_service.py: Review model loading path handling, interaction with ProcessPoolExecutor (run_in_executor), feature preparation (_prepare_features_for_model), handling of inference results/errors, task cancellation.
strategy_arbitrator.py: Review strategy logic implementation (thresholds), SL/TP calculation logic (use of feature data), construction of TradeSignalProposedEvent.
execution_handler.py: Review API interaction (aiohttp), signature generation, request/response handling, error mapping, order parameter translation (_translate_signal_to_kraken_params), HALT state check, exchange info loading.
simulated_execution_handler.py: Review fill simulation logic (market/limit), slippage/commission calculation, interaction with HistoricalDataService placeholder/interface.
monitoring_service.py: Review HALT state management (is_halted, trigger_halt, trigger_resume), periodic check implementation, interaction with PortfolioManager.
cli_service.py: Review stdin handling (add_reader/remove_reader), command parsing, interaction with MonitoringService and MainAppController.
4. Refinement & Verification:
[ ] Apply Fixes: Address issues identified during manual review and remaining tool reports. Prioritize correctness, robustness, and critical async/resource issues.
[ ] Re-run Tools: Run black, flake8, mypy again to ensure fixes are effective and no new issues were introduced.
[ ] Commit Changes: Commit refined code with clear messages explaining the changes made during the review.
Prioritization for MVP:
High: Correctness of core trading logic (signal -> risk -> execution flow), robustness against common errors (API errors, disconnects), proper async/await usage, resource cleanup, critical configuration loading, accurate logging integration.
Medium: Type hinting accuracy, docstring coverage for public APIs, consistency in naming and style, refining complex logic for clarity.
Low: Minor style nits not caught by tools, adding comments for obvious code, deep performance optimizations (unless an obvious bottleneck is found).
Deliverable: A codebase where automated checks pass cleanly, and manual review items (prioritized for MVP) have been addressed, leading to a more stable and maintainable foundation for Phase 4 (Testing).
