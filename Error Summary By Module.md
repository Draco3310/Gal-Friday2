src/gal_friday/core/events.py (7 errors):
7 x [misc]: Attributes without default follow attributes with one (lines 56, 58, 75, 76, 79, 81, 82).
src/gal_friday/simulated_market_price_service.py (2 errors):
[import-untyped]: Missing stubs for pandas (line 6).
[no-untyped-def]: Missing return type (line 118).
src/gal_friday/logger_service.py (31 errors):
[import-untyped]: Skipping asyncpg (line 10).
[import-not-found]: Cannot find gal_friday.utils.config, gal_friday.core.pubsub, pythonjsonlogger (lines 12, 13, 16).
[var-annotated]: Need type annotation for _queue (line 56).
[no-untyped-def]: Multiple functions missing return types.
[attr-defined]: AsyncPostgresHandler has no attribute formatException (line 88).
[attr-defined]: Callable[...] has no attribute warning/error (lines 504, 506).
src/gal_friday/event_bus.py (20 errors):
[no-untyped-def]: Multiple functions missing return types.
[attr-defined]: Event has no attribute event_type (line 41).
[var-annotated]: Need type annotation for queue (line 108).
[call-arg]: Unexpected keyword source_module for logger.error (line 225).
[misc]: 3 x Attributes without default follow attributes with one (lines 269, 270, 275).
[arg-type]: Multiple errors passing incorrect event types to subscribe/publish/unsubscribe.
src/gal_friday/simulated_execution_handler.py (15 errors):
[import-untyped]: Missing stubs for pandas (line 9).
[attr-defined]: gal_friday.core.events missing TradeSignalApprovedEvent, ExecutionReportEvent (line 17).
[import-not-found]: Cannot find gal_friday.core.pubsub, gal_friday.historical_data_service (lines 23, 26).
[attr-defined]: ConfigManager has no attribute get_decimal (lines 58, 62, 70, 74).
[no-untyped-def]: Multiple functions missing return types.
[no-any-return]: Returning Any instead of bool (lines 375, 377).
[call-arg]: Missing logger_service in call to SimulatedExecutionHandler (line 489).
src/gal_friday/portfolio_manager.py (11 errors):
[attr-defined]: gal_friday.core.events missing ExecutionReportEvent (line 15).
[import-not-found]: Cannot find gal_friday.core.pubsub, gal_friday.market_price_service (lines 16, 19).
[no-untyped-def]: Multiple functions missing return types.
[call-arg]: Missing logger_service in call to PortfolioManager (line 645).
[no-any-return]: Returning Any instead of Decimal | None (line 687).
src/gal_friday/monitoring_service.py (14 errors):
[import-not-found]: Cannot find gal_friday.core.pubsub (line 11).
[no-untyped-def]: Multiple functions missing return types.
[call-arg]: Unexpected keyword source for SystemStateEvent (line 228).
[call-arg]: Missing arguments in call to PortfolioManager (line 352).
[call-arg]: Missing logger_service in call to MonitoringService (line 354).
[assignment]: Incompatible types (MockPortfolioManagerHighDrawdown vs PortfolioManager) (line 385).
src/gal_friday/data_ingestor.py (25 errors):
[import-untyped]: Skipping sortedcontainers (line 10).
[misc]: 3 x Attributes without default... (lines 47, 68, 81).
[var-annotated]: Need annotation for _subscriptions, _l2_books, event_bus (lines 138, 143, 1123).
[no-untyped-def]: Multiple functions missing return types.
[assignment]: Multiple incompatible type assignments (lines 284, 285, 345, 350).
[attr-defined]: Missing InvalidStatusCode (line 297), None.send (line 323), None.__aiter__ (line 349).
[arg-type]: Incompatible type for interval (line 1092), DataIngestor args (line 1169).
src/gal_friday/feature_engine.py (13 errors):
[import-untyped]: Missing stubs for pandas (line 5).
[import-not-found]: Cannot find pandas_ta (line 6).
[misc]: Attributes without default... (line 37).
[var-annotated]: Need annotation for _latest_l2_data, _ohlcv_history, _latest_features (lines 89, 93, 97).
[no-untyped-def]: Multiple functions missing return types.
[assignment]: Incompatible types (Task[Any], str vs list | None) (lines 114, 171).
[attr-defined]: BaseEvent has no attribute event_type (line 272).
src/gal_friday/execution_handler.py (17 errors):
[attr-defined]: gal_friday.core.events missing TradeSignalApprovedEvent, ExecutionReportEvent (line 13).
[import-not-found]: Cannot find gal_friday.core.pubsub (line 18).
[no-untyped-def]: Multiple functions missing return types.
[union-attr]: None has no attribute get (line 176).
[no-any-return]: Multiple functions returning Any instead of specific types.
[attr-defined]: ConfigManager has no attribute get_list (line 203).
[attr-defined]: Module has no attribute binascii (line 298).
src/gal_friday/cli_service.py (5 errors):
[no-untyped-def]: Functions missing return types (lines 184, 202, 209).
[call-arg]: Missing arguments in call to MonitoringService (line 190).
[call-arg]: Missing logger_service in call to CLIService (line 192).
[method-assign]: Cannot assign to a method (line 213).
src/gal_friday/prediction_service.py (8 errors):
[attr-defined]: gal_friday.core.events missing BaseEvent (line 11).
[misc]: Attributes without default... (line 37).
[var-annotated]: Need annotation for _active_inference_tasks (line 145).
[no-untyped-def]: Multiple functions missing return types.
[assignment]: Incompatible types (Task[Any]) (line 157).
scripts/train_initial_model.py (10 errors):
[import-untyped]: Missing stubs for pandas, sklearn.metrics, joblib (lines 12, 14, 15).
[attr-defined]: ConfigManager missing get_list, get_int (lines 40, 129, 141, 209).
[attr-defined]: list[Any] has no attribute tolist (line 148).
[no-untyped-def]: Functions missing return types (lines 278, 298, 320).
src/gal_friday/strategy_arbitrator.py (11 errors):
[misc]: Attributes without default... (line 41).
[return-value]: Incompatible return type (line 168).
[attr-defined]: PredictionPayload missing attributes (lines 180, 181).
[no-untyped-def]: Multiple functions missing return types.
[assignment]: Incompatible types (Task[Any]) (line 279).
src/gal_friday/risk_manager.py (34 errors):
[misc]: 4 x Attributes without default... (lines 45, 67, 83, 97).
[no-untyped-def]: Multiple functions missing return types.
[assignment]: Incompatible types (Task[Any]) (lines 192, 193).
[arg-type]: Incompatible types for function args (lines 602, 607).
[misc]: None object is not iterable (line 614).
[index]: Value of type dict | None is not indexable (lines 627, 638, 676).
[call-arg]: Unexpected keyword source for SystemHaltPayload (line 912).
[attr-defined]: TradeSignalProposedPayload missing attributes (lines 954, 956, 957).
[attr-defined]: RiskManager missing max_drawdown, max_total_exposure (lines 1006, 1009, 1015, 1026, 1029, 1035).
[attr-defined]: PortfolioManager missing get_current_equity, get_open_positions (lines 1067, 1114).
[has-type]: Cannot determine type of peak_equity (line 1070).
[no-any-return]: Multiple functions returning Any instead of specific types.
src/gal_friday/main.py (46 errors):
[misc]/[assignment]: Multiple errors assigning None to type placeholders.
[import-not-found]: Cannot find gal_friday.core.pubsub (line 32).
[truthy-function]: Multiple checks on types/functions instead of instances.
[no-untyped-def]: Multiple functions missing return/type annotations.
[valid-type]: Using runtime variables as type hints.
[attr-defined]/[union-attr]: Attributes not found on types/unions (often involving None or placeholder types).
[call-arg]: Multiple errors with missing/unexpected arguments in function calls.
[arg-type]: Multiple errors passing incompatible argument types.
src/gal_friday/backtesting_engine.py (20 errors):
[import-untyped]/[import-not-found]: Missing stubs/modules (pandas, pandas_ta, gal_friday.core.pubsub) (lines 4, 5, 40).
[attr-defined]: gal_friday.core.events missing attributes (line 15).
[no-untyped-def]: Functions missing type annotations (lines 26, 741).
[arg-type]: Incompatible type for float() (line 170).
[operator]: Unsupported operand types (line 172).
[attr-defined]: ConfigManager missing methods (lines 262, 264, 641, 1120).
[call-arg]/[arg-type]: Multiple errors with missing/incompatible arguments in calls to services (lines 694-701).