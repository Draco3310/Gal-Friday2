# Implement Trading Pair Split via ConfigManager

## Task ID
**POS-MANAGER-002**

## Priority
**Medium**

## Epic
**Configuration Management & Symbol Processing**

## Story
As a developer working with position management, I need proper trading pair splitting functionality implemented via ConfigManager so that symbol processing is consistent and configurable across the system.

## Problem Statement
Line 372 in `gal_friday/portfolio/position_manager.py` contains example code for trading pair splitting that needs to be implemented properly using ConfigManager or removed if unnecessary.

## Acceptance Criteria
- [ ] Review current trading pair split example code
- [ ] Determine if trading pair splitting is needed for position management
- [ ] If needed, implement proper splitting logic using ConfigManager
- [ ] If not needed, remove example code and document the decision
- [ ] Ensure consistent symbol format handling across the system
- [ ] Add proper error handling for invalid trading pairs
- [ ] Create unit tests for trading pair splitting scenarios

## Technical Requirements
- Review line 372 in `gal_friday/portfolio/position_manager.py`
- Implement trading pair splitting using ConfigManager configuration
- Handle various trading pair formats (e.g., "BTCUSD", "BTC/USD", "BTC-USD")
- Ensure consistent symbol normalization
- Add validation for supported trading pairs
- Follow existing configuration patterns in the system

## Definition of Done
- [ ] Decision made on whether trading pair splitting is needed
- [ ] If implemented: proper splitting logic using ConfigManager
- [ ] If implemented: support for multiple trading pair formats
- [ ] If removed: example code cleaned up with documentation
- [ ] Error handling for invalid or unsupported pairs
- [ ] Unit tests cover splitting logic and edge cases
- [ ] Integration tests verify symbol processing consistency
- [ ] Code review completed and approved

## Dependencies
- ConfigManager implementation and configuration patterns
- Understanding of supported trading pair formats
- Symbol processing requirements across the system

## Estimated Effort
**Story Points: 3**

## Risk Assessment
**Low-Medium Risk** - Symbol processing changes could affect order management and risk calculations

## Implementation Notes
```python
# Example implementation using ConfigManager
def split_trading_pair(self, symbol: str) -> Tuple[str, str]:
    """Split trading pair into base and quote currencies.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSD", "BTC/USD")
        
    Returns:
        Tuple of (base_currency, quote_currency)
        
    Raises:
        ValueError: If symbol format is not supported
    """
    pair_config = self.config_manager.get('trading_pairs', {})
    separators = pair_config.get('separators', ['/', '-', ''])
    
    for separator in separators:
        if separator and separator in symbol:
            base, quote = symbol.split(separator, 1)
            return base.upper(), quote.upper()
    
    # Handle concatenated pairs like "BTCUSD"
    known_pairs = pair_config.get('known_pairs', {})
    if symbol in known_pairs:
        return known_pairs[symbol]['base'], known_pairs[symbol]['quote']
    
    raise ValueError(f"Unable to split trading pair: {symbol}")
```

## Related Files
- `gal_friday/portfolio/position_manager.py` (line 372)
- ConfigManager implementation
- Symbol processing utilities
- Order management modules that handle trading pairs 