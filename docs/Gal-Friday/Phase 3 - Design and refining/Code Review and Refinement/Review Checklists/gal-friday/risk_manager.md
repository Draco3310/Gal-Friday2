# Risk Manager Module Code Review Checklist

## Module Overview
The `risk_manager.py` module is responsible for evaluating proposed trade signals, performing pre-trade risk checks, calculating appropriate position sizes, and either approving or rejecting trades. It handles:
- Consuming trade signal proposals from the Strategy Arbitrator
- Querying current portfolio state from the Portfolio Manager
- Enforcing risk limits (drawdown, exposure, consecutive losses)
- Calculating position sizes using fixed fractional risk method
- Publishing approved or rejected trade signals
- Monitoring overall portfolio risk for HALT conditions

## Module Importance
This module is **critically important** as it represents the primary risk control layer in the system. Errors in risk management can lead directly to excessive losses or violation of risk parameters.

## Architectural Context
According to the [architecture_concept](../../../../Phase%201%20-%20Requirements%20Analysis%20%26%20Planning/architecture_concept_gal_friday_v0.1.md), the `RiskManager` receives trade signal proposals from the `StrategyArbitrator`, synchronously queries the `PortfolioManager` for current state, and produces either approved or rejected trade signals for the `ExecutionHandler`. It also monitors overall risk to potentially trigger HALTs via the `MonitoringService`.

## Review Checklist

### A. Correctness & Logic

- [ ] Verify that the implementation conforms to the `RiskManager` interface defined in section 2.6 of the [interface_definitions](../../../../Phase%201%20-%20Requirements%20Analysis%20%26%20Planning/interface_definitions_gal_friday_v0.1.md) document
- [ ] Check that the module correctly consumes trade signal proposal events from the event bus
- [ ] Verify that portfolio state is retrieved synchronously from the Portfolio Manager per section 4.1 of [inter_module_comm](../../../../Phase%201%20-%20Requirements%20Analysis%20%26%20Planning/inter_module_comm_gal_friday_v0.1.md)
- [ ] Ensure proper enforcement of maximum drawdown limits (Total: 15%, Daily: 2%, Weekly: 5%) as specified in FR-503
- [ ] Check implementation of the consecutive losing trades limit per FR-504
- [ ] Verify correct position size calculation using the Fixed Fractional method as described in FR-505
- [ ] Ensure all required pre-trade checks are implemented per FR-506:
  - [ ] Maximum percentage of equity per asset
  - [ ] Maximum total portfolio exposure
  - [ ] Sufficient balance check
  - [ ] Maximum order size sanity check
  - [ ] "Fat finger" check on proposed entry price
- [ ] Verify that approved trade signals contain all required information (position size, confirmed SL/TP) and are properly formatted per section 3.6 of [inter_module_comm](../../../../Phase%201%20-%20Requirements%20Analysis%20%26%20Planning/inter_module_comm_gal_friday_v0.1.md)
- [ ] Check that rejected trade signals include appropriate reason codes per section 3.7 of [inter_module_comm](../../../../Phase%201%20-%20Requirements%20Analysis%20%26%20Planning/inter_module_comm_gal_friday_v0.1.md)
- [ ] Verify implementation of portfolio-level risk monitoring with HALT triggering per FR-905

### B. Error Handling & Robustness

- [ ] Check for proper handling of invalid trade signal proposals
- [ ] Verify error handling when portfolio state cannot be retrieved
- [ ] Ensure that calculation errors don't crash the module
- [ ] Check handling of edge cases in portfolio state (zero equity, extreme drawdown)
- [ ] Verify that the system can recover from temporary error conditions
- [ ] Ensure proper handling of unexpected data formats
- [ ] Check that errors are logged with appropriate context for debugging

### C. asyncio Usage

- [ ] Verify correct usage of asyncio patterns for event handling
- [ ] Ensure that synchronous calls to Portfolio Manager are handled appropriately within the async context
- [ ] Verify proper handling of CancelledError during shutdown
- [ ] Check that trade signal publishing follows proper async patterns
- [ ] Ensure proper cleanup of resources and tasks during the stop method
- [ ] Verify that periodic risk monitoring tasks are properly managed

### D. Dependencies & Imports

- [ ] Verify that imports are well-organized according to project standards
- [ ] Check appropriate handling of the dependency on PortfolioManager
- [ ] Ensure proper import and usage of the event bus/subscription mechanism
- [ ] Verify proper use of decimal for financial calculations
- [ ] Check that dependencies are used efficiently and appropriately

### E. Configuration & Hardcoding

- [ ] Verify that all risk parameters are configurable:
  - [ ] Maximum drawdown limits (total, daily, weekly)
  - [ ] Consecutive loss limit
  - [ ] Risk percentage per trade
  - [ ] Maximum exposure percentages
  - [ ] Maximum order size
  - [ ] Price deviation thresholds for fat finger check
- [ ] Check that no critical risk parameters are hardcoded
- [ ] Ensure that configuration values are validated during initialization

### F. Logging

- [ ] Verify appropriate logging of all risk decisions (approvals and rejections)
- [ ] Ensure rejection reasons are clearly logged
- [ ] Check for logging of position size calculations and risk parameters used
- [ ] Verify that risk limit breaches are logged at appropriate severity
- [ ] Ensure logging of portfolio risk metrics
- [ ] Check that HALT triggers are prominently logged

### G. Readability & Style

- [ ] Verify clear, descriptive method and variable names
- [ ] Check for well-structured code organization, especially for different risk checks
- [ ] Ensure complex risk calculations are well-commented
- [ ] Verify reasonable method length and complexity
- [ ] Check for helpful comments explaining risk management logic and thresholds

### H. Resource Management

- [ ] Verify efficient management of any data structures needed for risk tracking
- [ ] Check for cleanup of any resources upon shutdown
- [ ] Ensure background monitoring tasks are properly tracked and cancelled during shutdown
- [ ] Verify that memory usage is appropriate for tracking of risk metrics

### I. Docstrings & Type Hinting

- [ ] Ensure comprehensive docstrings for the class and all methods
- [ ] Verify accurate type hints for method parameters and return values
- [ ] Check that risk calculation algorithms are well-documented
- [ ] Ensure trade signal structures (approved/rejected) are well-documented
- [ ] Verify that public methods have complete parameter and return value documentation

### J. Risk-Specific Considerations

- [ ] Verify that risk calculations use Decimal for precision where appropriate
- [ ] Check that risk calculations are mathematically correct
- [ ] Ensure that rounding of position sizes is conservative (erring on the smaller side)
- [ ] Verify that position sizing respects exchange-specific minimum and maximum order sizes
- [ ] Check that multiple risk violations are handled appropriately (reporting all violations)
- [ ] Ensure that risk monitoring frequency is appropriate
- [ ] Verify that the module handles different market conditions appropriately

### K. Performance Considerations

- [ ] Verify that risk checks are performed efficiently
- [ ] Ensure that synchronous calls to Portfolio Manager don't cause bottlenecks
- [ ] Check that the module meets the latency requirements in NFR-502 (under 50ms for order placement)
- [ ] Verify that periodic risk monitoring doesn't impact critical path performance

## Improvement Suggestions

- [ ] Consider implementing graduated risk levels (normal, caution, restricted, halt)
- [ ] Evaluate adding dynamic risk adjustment based on recent performance
- [ ] Consider implementing more sophisticated position sizing strategies
- [ ] Evaluate adding correlation-based portfolio risk assessment
- [ ] Consider implementing time-of-day risk profile adjustments
- [ ] Assess adding real-time risk visualization tools
- [ ] Consider implementing risk-based trade filtering (quality scores)
- [ ] Evaluate adding stress testing for risk models
