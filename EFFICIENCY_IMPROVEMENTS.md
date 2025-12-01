# CUGA Efficiency Improvements

## Problem: 87 Steps for a Simple Task

The original implementation took **87 steps** to complete the Venmo friends sync task due to:
1. Rigid 3-phase cycling (planning → code_planning → execution → repeat)
2. Empty planning turns (26+ wasted steps)
3. "think: Continue" loops (7+ wasted steps)
4. Overly strict completion blocking (3+ wasted steps)
5. Code Planner outputting entire task plan repeatedly
6. Serial API doc lookups

## Solutions Implemented

### 1. ✅ Adaptive Phase Selection

**File:** `src/agent/appworld_generalist_agent.py`

**Changes:**
- Added `_should_skip_planning()` method (lines 1176-1190)
- Added `_needs_api_discovery()` method (lines 1192-1208)
- Added `_determine_next_phase()` method (lines 1234-1267)
- Replaced rigid cycling with adaptive phase selection (line 1531)

**How it works:**
```python
def _determine_next_phase(self) -> str:
    # Fast-track API discovery without planning overhead
    if self._needs_api_discovery():
        return "execution"
    
    # Initial planning at start
    if self._coordination_step == 0:
        return "planning"
    
    # Re-plan after failures
    if self._last_failed:
        return "planning"
    
    # Skip planning if just said "Continue"
    if self._should_skip_planning():
        return "execution"
    
    # Normal mini-cycle when coordination needed
    ...
```

**Impact:**
- Eliminates empty planning cycles
- Skips coordination when execution is progressing smoothly
- Detects "Continue" and skips re-planning
- **Expected reduction: ~30 wasted steps**

### 2. ✅ Smarter Completion Authorization

**File:** `src/agent/appworld_generalist_agent.py`

**Changes:**
- Enhanced `_maybe_block_completion()` method (lines 1385-1432)
- Added verification detection
- Added progress-based auto-authorization
- Added retry detection

**How it works:**
```python
# Auto-authorize if verification was done
if any("verify" in recent_history):
    return code  # Allow completion

# Auto-authorize after substantial progress
if self._coordination_step > 20:
    return code  # Allow completion

# Block only if truly premature (< 10 steps)
if self._coordination_step < 10:
    return blocked_message
```

**Impact:**
- Reduces completion blocking cycles from 3+ to 0-1
- Smart detection of when task is actually complete
- **Expected reduction: ~3 wasted steps**

### 3. ✅ Batched API Doc Lookups

**File:** `src/agent/appworld_generalist_agent.py`

**Changes:**
- Added `_generate_batched_api_discovery_code()` method (lines 1210-1232)
- Added fast-track discovery in `act()` method (lines 1533-1554)

**How it works:**
```python
def _generate_batched_api_discovery_code(self) -> str | None:
    if not self._global_apps_listed:
        return "print(apis.api_docs.show_app_descriptions())"
    
    missing_apps = self._missing_required_apps()
    if missing_apps:
        # Batch up to 3 apps at once
        code = []
        for app in missing_apps[:3]:
            code.append(f"print(apis.api_docs.show_api_descriptions(app_name='{app}'))")
        return "\n".join(code)
```

**Impact:**
- Discovers 3 apps per step instead of 1
- Skips planning/code_planning overhead during discovery
- **Expected reduction: ~8 steps in discovery phase**

### 4. ✅ Incremental Code Planning

**File:** `src/agent/appworld_generalist_agent.py`

**Changes:**
- Updated Code Planner system prompt (lines 611-643)
- Changed from "plan entire task" to "plan next 1-3 steps"

**Old prompt:**
```
"Break the planner's guidance into concrete steps that the Code Agent can execute."
[Result: 17-step JSON plans repeated endlessly]
```

**New prompt:**
```
"Generate the NEXT immediate step(s) (1-3 only) that the Code Agent should execute."

Example good outputs:
- ['Call supervisor.show_account_passwords() to get credentials']
- ['Login to phone app', 'Fetch first page of contacts']

Example BAD output:
- [17 steps describing entire task] ❌
```

**Impact:**
- Code Planner generates concise, actionable plans
- Plans adapt based on execution results
- Reduces token usage and planning time
- **Expected reduction: ~10 steps from shorter planning cycles**

### 5. ✅ Failure Tracking for Adaptive Behavior

**File:** `src/agent/appworld_generalist_agent.py`

**Changes:**
- Added `_last_failed` tracking in `observe()` (lines 1706-1712)
- Initialize in `reset()` (line 1152) and `_initialize_api_discovery_state()` (line 1183)

**How it works:**
```python
# In observe()
if obs:
    failure_indicators = ["execution failed", "traceback", "error", "exception"]
    self._last_failed = any(indicator in obs.lower() for indicator in failure_indicators)

# Used in _determine_next_phase()
if self._last_failed:
    return "planning"  # Re-plan after errors
```

**Impact:**
- Agent detects failures and replans intelligently
- Avoids continuing execution after errors
- **Improved error recovery**

## Expected Performance Improvements

| Component | Before | After | Savings |
|-----------|--------|-------|---------|
| API Discovery | 12 steps (4 APIs × 3-phase cycle) | 4 steps (batched, no cycle) | **-8 steps** |
| Empty Planning | 26+ steps (overhead) | 0-2 steps (only when needed) | **-24 steps** |
| "Continue" loops | 7+ steps (wasted cycles) | 0 steps (detected & skipped) | **-7 steps** |
| Completion blocking | 3+ steps (retry cycles) | 0-1 steps (smart auth) | **-2 steps** |
| Code Planning | 29 cycles (entire task) | 10-15 cycles (incremental) | **-14 steps** |
| **Total** | **87 steps** | **~22-30 steps** | **~57-65 steps saved (65-75% reduction)** |

## Key Design Principles

1. **Adaptive over Rigid**: Phase selection based on state, not fixed cycling
2. **Fast-track Discovery**: Skip coordination overhead during API discovery
3. **Smart Authorization**: Detect readiness for completion, don't block arbitrarily
4. **Incremental Planning**: Plan next steps, not entire task
5. **Failure-aware**: Track errors and adapt behavior accordingly

## Testing Recommendations

1. **Run the Venmo sync task again** - should complete in ~25 steps instead of 87
2. **Monitor logs** for:
   - "Using batched API discovery (fast-track)"
   - "Auto-authorizing completion: verification detected"
   - "Skipping planning: continuing execution"
3. **Verify behaviors**:
   - No empty planning cycles with just "Continue"
   - Code Planner outputs 1-3 steps, not 17
   - API discovery batched (3 apps per step)
   - Completion not blocked after verification

## Files Modified

- `src/agent/appworld_generalist_agent.py`:
  - Lines 1176-1267: Adaptive phase selection methods
  - Lines 1385-1432: Smart completion authorization
  - Lines 1210-1232: Batched API discovery
  - Lines 611-643: Incremental code planning prompt
  - Lines 1533-1554: Fast-track discovery integration
  - Lines 1706-1712: Failure tracking
  - Lines 1152-1153: Reset updates

## Status

✅ **All improvements implemented**
✅ **No linting errors**
✅ **Ready for testing**

