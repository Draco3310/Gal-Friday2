# Gal-Friday2: Action Plan - Next Steps

## 🎯 Day 1-2: Remove All Placeholders
**Goal**: Eliminate all mock/placeholder code

### Step 1: Delete placeholder file
```bash
rm gal_friday/core/placeholder_classes.py
```

### Step 2: Fix monitoring_service.py
1. Open `gal_friday/monitoring_service.py`
2. Delete lines 46-159 (all Mock* classes)
3. Update any references to use real services via dependency injection

### Step 3: Fix risk_manager.py  
1. Open `gal_friday/risk_manager.py`
2. Delete lines 63-132 (mock implementations)
3. Add proper dependency injection in constructor

### Step 4: Fix dashboard files
1. `gal_friday/monitoring/dashboard_backend.py`:
   - Line 162: Replace `99.9` with `self.calculate_uptime()`
   - Line 209: Replace `0` with `self.calculate_correlation_risk()`
2. `gal_friday/monitoring/dashboard_service.py`:
   - Line 57: Connect to `self.portfolio_manager`

---

## 🎯 Day 3: Fix Hardcoded Values
**Goal**: Make everything configurable

### Step 1: data_ingestor.py
```python
# Line 1400 - Change from:
exchange="kraken",  # Hardcoded for now

# To:
exchange=self.config.get("data_ingestion.default_exchange", "kraken"),
```

### Step 2: strategy_arbitrator.py
```python
# Line 86 - Remove MVP comment and implement:
strategy = self._select_best_strategy(available_strategies)
```

### Step 3: backtesting_engine.py
```python
# Line 279 - Change from:
atr_value = Decimal("20.0")  # Default fallback

# To:
atr_value = Decimal(self.config.get("backtesting.default_atr", "20.0"))
```

---

## 🎯 Day 4-5: Algorithm Improvements
**Goal**: Implement better algorithms where noted

### Priority Improvements:
1. **retraining_pipeline.py:284** - Implement KL divergence:
   ```python
   from scipy.stats import entropy
   kl_divergence = entropy(old_distribution, new_distribution)
   ```

2. **portfolio_manager.py:1145** - Add relative thresholds:
   ```python
   threshold = balance * Decimal("0.001")  # 0.1% of balance
   ```

---

## 🚀 Quick Win Commands

### Find all "for now" comments:
```bash
grep -r "for now" gal_friday/ --include="*.py" | grep -v test
```

### Find all placeholders:
```bash
grep -r "placeholder\|mock\|TODO" gal_friday/ --include="*.py" -i | grep -v test
```

### Run tests after changes:
```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

---

## ✅ Definition of Done

- [ ] No files contain "placeholder" in class/function names
- [ ] No hardcoded exchange names or magic numbers
- [ ] All "MVP" references removed
- [ ] All mock classes replaced with real implementations
- [ ] Tests pass after all changes
- [ ] No "for now" comments in critical paths

**Estimated Time**: 5 days for all changes  
**Minimum Time**: 2 days for production blockers only 