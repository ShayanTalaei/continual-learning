# Implementation Summary: Deduplication & API Doc Checking

## Overview
Implemented two critical improvements to the CUGA agent system to address inefficiencies observed in the Venmo friends sync task trajectory.

## Issue 2: Deduplication Logic ✅

### Changes Made

#### 1. Code Planner System Prompt (lines 578-581)
**Added explicit deduplication guidance:**
```
- DEDUPLICATION: When adding/removing multiple items by identifier (email, phone, user_id):
  * Plan explicit deduplication steps: first resolve all identifiers to canonical form, then deduplicate using a set
  * Example: 'Resolve all phone numbers to emails by searching → Store unique emails in a set → Iterate over unique set for additions'
  * This prevents duplicate API calls and 'already exists' errors
```

**Impact:** 
- Code Planner will now include deduplication as an explicit step in execution plans
- Plans will guide the Code Agent to: resolve → deduplicate → process
- Prevents the agent from attempting to add the same user 7+ times

#### 2. Code Agent System Prompt (lines 731-735)
**Added deduplication as responsibility #7:**
```
7. DEDUPLICATION: Prevent duplicate operations
   - When processing multiple identifiers (emails, phone numbers, IDs), deduplicate first
   - Pattern: resolve identifiers → collect in set → iterate over unique values
   - Example: unique_emails = set(); for phone in phones: user = search(phone); if user: unique_emails.add(user['email'])
   - Track processed items to avoid 'already exists' errors: processed = set()
```

**Impact:**
- Code Agent has concrete pattern to follow: `resolve → set() → iterate`
- Example code shows how to implement deduplication
- Agent will track processed items to avoid duplicate operations

### Expected Improvement
**Before:** 
- Multiple phone numbers → same email → 8 API calls (7 failures)
- Pattern: `for identifier in identifiers: add(identifier)` ❌

**After:**
- Multiple phone numbers → same email → deduplicate → 1 API call ✅
- Pattern: `unique = set(resolve(identifiers)); for item in unique: add(item)` ✅

---

## Issue 3: Strengthen API Doc Checking ✅

### Changes Made

#### 1. Improved Doc Lookup Tracking (lines 521-542)
**Enhanced `AppWorldCodeAgent.observe()` method:**

**Before:** Only tracked supervisor APIs poorly
```python
if "login" in api_name.lower() or "show_" in api_name.lower():
    self._doc_lookups_done.add(f"supervisor.{api_name}")
```

**After:** Comprehensive pattern matching
```python
# Look for API doc output patterns: "app_name": "...", "api_name": "..."
doc_patterns = [
    r'"app_name"\s*:\s*"(\w+)".*?"api_name"\s*:\s*"(\w+)"',
    r"'app_name'\s*:\s*'(\w+)'.*?'api_name'\s*:\s*'(\w+)'",
]
for pattern in doc_patterns:
    matches = re.findall(pattern, obs, re.DOTALL)
    for app_name, api_name in matches:
        api_key = f"{app_name}.{api_name}"
        self._doc_lookups_done.add(api_key)
```

**Impact:**
- Tracks ALL API docs correctly, not just supervisor
- Parses actual execution output to detect show_api_doc results
- Also tracks show_api_descriptions for common APIs

#### 2. Strengthened Guardrails (lines 433-456)
**Enhanced `_violates_guardrails()` method:**

**Improvements:**
- Excludes `api_docs` calls from doc requirement (they don't need docs for themselves)
- Lists ALL missing docs in error message, not just first one
- Better error message format: `"APIs used without prior documentation lookup: phone.login, venmo.add_friend"`
- Only counts non-doc API calls against the rate limit

**Before:**
```python
if api_key not in self._doc_lookups_done:
    return f"API '{api_key}' used without prior documentation lookup."
```

**After:**
```python
missing_docs = []
for app_name, api_name in non_doc_apis:  # Excludes api_docs
    api_key = f"{app_name}.{api_name}"
    if api_key not in self._doc_lookups_done:
        missing_docs.append(api_key)

if missing_docs:
    missing_str = ", ".join(missing_docs)
    return f"APIs used without prior documentation lookup: {missing_str}. Always call apis.api_docs.show_api_doc(...)..."
```

#### 3. Improved Repair Function (lines 460-495)
**Enhanced `_repair_action()` method:**

**Improvements:**
- Batches doc lookups for ALL missing APIs (not just first one)
- Extracts missing APIs from violation message
- Falls back to extracting from original code
- Limits to 5 APIs to avoid overflow
- Better comments and structure

**Before:** Single API repair
```python
api_match = re.search(r"apis\.(\w+)\.(\w+)\(", original_action)
if api_match:
    app_name, api_name = api_match.groups()
    repair_code = f"print(apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name}'))"
```

**After:** Batched repair for all missing APIs
```python
missing_apis = re.findall(r"(\w+)\.(\w+)", violation)
if missing_apis:
    repair_lines = ["# Looking up API documentation before use"]
    for app_name, api_name in missing_apis[:5]:
        repair_lines.append(
            f"print(apis.api_docs.show_api_doc(app_name='{app_name}', api_name='{api_name}'))"
        )
    repair_lines.append("# Review the documentation above before proceeding")
```

#### 4. Elevated API Doc Checking Priority (lines 752-756)
**Moved API documentation from responsibility #2 to #1 with stronger language:**

**Before:**
```
2. API DOCUMENTATION: Look up API specs before calling
   - apis.api_docs.show_api_doc(app_name='...', api_name='...')
   - Check required parameters and response structure
```

**After:**
```
1. API DOCUMENTATION (REQUIRED): ALWAYS look up API specs before calling
   - MANDATORY: apis.api_docs.show_api_doc(app_name='...', api_name='...') before each new API
   - Check required parameters, types, constraints, and response structure
   - Never guess parameter names or formats - always verify from docs
   - Example: Before calling phone.login(), first print(apis.api_docs.show_api_doc(app_name='phone', api_name='login'))
```

**Impact:**
- Doc checking is now the FIRST responsibility, not second
- Language changed from suggestion to requirement: "REQUIRED", "ALWAYS", "MANDATORY"
- Concrete example shows the pattern
- Emphasizes "Never guess" - always verify

### Expected Improvement

**Before:**
- Agent tries `phone.login(username='email@...')` → FAILS 401
- Agent retries with `phone.login(username='phone_number')` → SUCCESS
- Wasted 1 turn

**After:**
- Agent checks docs first: `print(apis.api_docs.show_api_doc('phone', 'login'))`
- Sees username expects phone number
- Calls correctly on first try: `phone.login(username='phone_number')` → SUCCESS
- Saves 1 turn per API

---

## Integration with CUGA Principles

These changes follow CUGA's core principles:

### 1. **API-Grounded Planning** ✅
- Strengthened doc checking ensures all operations are grounded in actual API specs
- No more guessing parameter formats or types

### 2. **Hierarchical Decomposition** ✅
- Code Planner now explicitly plans deduplication steps
- Better guidance flows from Planner → Code Planner → Code Agent

### 3. **Error Prevention** ✅
- Deduplication prevents "already exists" errors
- Doc checking prevents "invalid credentials" and parameter errors
- Guardrails catch and repair violations automatically

### 4. **Efficiency** ✅
- Batched doc lookups reduce round trips
- Deduplication reduces wasted API calls
- Pre-flight checks prevent failed attempts

---

## Expected Impact on Venmo Sync Task

### Trajectory Comparison

**Original (Inefficient):**
```
Step 1: Get phone friends → 8 friends
Step 2: Get Venmo friends → 10 friends
Step 3: Remove 6 Venmo friends → SUCCESS
Step 4: Try add stmcco@gmail.com → SUCCESS
Step 5: Try add stmcco@gmail.com → FAIL (already exists)
Step 6: Try add stmcco@gmail.com → FAIL (already exists)
...
Step 11: Try add stmcco@gmail.com → FAIL (already exists)
Total: ~54 steps with multiple failures
```

**Expected (Efficient):**
```
Step 1: Check phone.search_contacts docs
Step 2: Get phone friends → 8 friends
Step 3: Check venmo APIs docs
Step 4: Get Venmo friends → 10 friends
Step 5: Deduplicate and remove 6 Venmo friends → SUCCESS
Step 6: Resolve phone numbers to emails → get unique set
Step 7: Add unique emails (deduplicated) → SUCCESS for all
Step 8: Verify sync by re-fetching both lists
Step 9: Complete task
Total: ~20-25 steps, no failures
```

### Key Improvements
- **-7 failed API calls** (stmcco@gmail.com duplicates)
- **-1 failed login attempt** (wrong username format)
- **+verification step** (ensures correctness)
- **~50% reduction in total steps**

---

## Testing Recommendations

1. **Run Venmo Sync Task Again**
   - Should see explicit deduplication in code: `unique_emails = set(...)`
   - Should see doc lookups before first API use
   - Should complete in ~20-25 steps instead of 54

2. **Check Logs For**
   - "Registered API doc lookup: phone.login" messages
   - No "already in your friend list" errors
   - Code contains `set()` for deduplication

3. **Monitor Guardrails**
   - Fewer guardrail violations
   - If violations occur, repairs should batch multiple doc lookups
   - No violations for api_docs calls themselves

---

## Files Modified

- `src/agent/appworld_generalist_agent.py`:
  - Lines 578-581: Code Planner deduplication guidance
  - Lines 731-735: Code Agent deduplication responsibility
  - Lines 521-542: Improved doc lookup tracking
  - Lines 433-456: Strengthened guardrails
  - Lines 460-495: Enhanced repair function
  - Lines 752-756: Elevated API doc checking priority

## Status
✅ **All changes implemented successfully**
✅ **No linting errors**
✅ **Ready for testing**

