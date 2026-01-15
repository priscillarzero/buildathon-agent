# HVAC Domain Knowledge Base

## What is ODCV (Occupancy-Driven Control Ventilation)?

ODCV dynamically adjusts outside air (OA) ventilation based on real-time occupancy data from sensors (like R-Zero). This saves energy by:
- Reducing ventilation when spaces are unoccupied
- Scaling ventilation proportionally to actual occupancy vs. design maximum
- Enabling aggressive setbacks during nights/weekends

**Expected Savings**: up to 30% reduction in HVAC energy for spaces with variable occupancy.

---

## Key Metrics Reference

### Airflow Metrics
| Metric | Description | Typical Range | Red Flags |
|--------|-------------|---------------|-----------|
| OA-CFM | Outside air flow rate | 0-5000 CFM | Negative values, flat when occupancy changes |
| SA-CFM | Supply air flow | 0-10000 CFM | Should exceed OA-CFM |
| RA-CFM | Return air flow | 0-10000 CFM | Should roughly equal SA-CFM |
| SAFLOW-SET | Flow setpoint | Varies | Should reset to minimum when unoccupied |

### Temperature Metrics
| Metric | Description | Typical Range | Red Flags |
|--------|-------------|---------------|-----------|
| OA-T | Outside air temp | -20 to 120°F | Values outside physical reality |
| MA-T | Mixed air temp | Between OA-T and RA-T | Impossible if outside this range |
| SA-T | Supply air temp | 52-58°F (cooling) | Deviation >5°F from setpoint |
| RA-T | Return air temp | 68-78°F | Indicates space conditions |

### Control Metrics
| Metric | Description | Typical Range | Red Flags |
|--------|-------------|---------------|-----------|
| OA-DPR | OA damper position | 0-100% | Stuck at one value |
| SF-VFD | Supply fan speed | 0-100% | At 100% constantly = undersized |
| AHU-STATE | Operating mode | 1-6 | See modes below |

**AHU-STATE Modes:**
- 1 = Satisfied (no heating/cooling needed)
- 2 = Economizer (free cooling with outside air)
- 3 = Economizer + Mechanical Cooling
- 4 = Mechanical Cooling Only
- 5 = Heating
- 6 = Morning Warmup/Cooldown

### IAQ Metrics
| Metric | Description | Typical Range | Red Flags |
|--------|-------------|---------------|-----------|
| CO2 | Carbon dioxide | 400-1200 ppm | >1000 ppm inadequate ventilation |
| Occupancy | People count | 0-max capacity | Flat = sensor issue |

---

## Expected Relationships (Critical for Root Cause Analysis)

### 1. OA-CFM vs Occupancy
**Expected**: Linear relationship. More people = more outside air.
```
OA-CFM = (7.5 CFM/person × occupancy) + (0.06 CFM/sqft × area) + minimum_OA
```

**Anomaly Signatures:**
- OA-CFM flat while occupancy varies → Damper stuck or control disabled
- OA-CFM spikes with zero occupancy → Schedule override or sensor fault
- OA-CFM too low for occupancy → Damper issue or fan at capacity

### 2. Damper Position vs Airflow
**Expected**: Higher damper % = higher OA-CFM (roughly proportional)

**Anomaly Signatures:**
- High OA-DPR + Low OA-CFM → Damper actuator disconnected, duct blockage
- Low OA-DPR + High OA-CFM → Sensor calibration issue
- OA-DPR changes but OA-CFM constant → Stuck damper (actuator reads position but blade doesn't move)

### 3. CO2 vs Occupancy/Ventilation
**Expected**: CO2 rises with occupancy, falls with increased OA-CFM

**Anomaly Signatures:**
- High CO2 + High OA-CFM → Sensor location issue, or OA actually recirculated
- Low CO2 + Low occupancy → Normal
- Rising CO2 trend over days → Gradual sensor drift or system degradation

### 4. Energy vs Operating Conditions
**Expected**:
- Higher fan speed (VFD) = more energy
- Economizer mode (state 2) = lower energy than mechanical cooling (state 3,4)
- Unoccupied periods = minimal energy

**Anomaly Signatures:**
- High energy + Low occupancy → System not responding to occupancy
- Energy flat across day → Metering issue or no setback
- Energy spikes at night → Cleaning schedule, or control issue

---

## Common Failure Modes & Signatures

### 1. Stuck Damper
**Signature**: OA-DPR command changes but OA-CFM stays constant
**Impact**: Over/under ventilation, wasted energy
**Verification**: Check if damper physically moves during command change

### 2. Sensor Drift
**Signature**: Gradual offset from expected values over weeks/months
**Impact**: Incorrect control decisions
**Verification**: Compare to independent measurement or other sensors

### 3. Schedule Override
**Signature**: System doesn't reduce during expected unoccupied times
**Impact**: Running 24/7 when should setback
**Verification**: Check BMS schedules, look for manual overrides

### 4. Economizer Fault
**Signature**: Mechanical cooling runs when OA-T is favorable (<65°F)
**Impact**: Wasted energy (free cooling not used)
**Verification**: Check AHU-STATE during mild weather

### 5. Hunting/Oscillation
**Signature**: Rapid cycling of damper/VFD every few minutes
**Impact**: Wear on actuators, energy waste, comfort issues
**Verification**: Look for sinusoidal patterns in trend data

### 6. Occupancy Sensor Fault
**Signature**: Occupancy reads constant (0 or max), or unrealistic jumps
**Impact**: ODCV can't save energy without accurate occupancy
**Verification**: Compare to badge data or manual counts

---

## Seasonal Considerations

### Summer (Cooling Season)
- OA-T typically 75-100°F
- System in cooling mode most of day
- Economizer rarely useful
- Energy dominated by cooling load

### Winter (Heating Season)
- OA-T typically 30-60°F (Arizona mild)
- Economizer very useful for free cooling
- Watch for: morning warmup spikes, humidity issues
- Minimum OA may conflict with heating

### Shoulder Seasons (Spring/Fall)
- Best savings potential - lots of economizer hours
- Watch for: mode hunting between heating/cooling
- OA damper should be wide open often

---

## Analysis Framework

When investigating an anomaly, follow this structure:

1. **Quantify the Issue**
   - What is the actual value/behavior?
   - What was expected?
   - What is the magnitude of deviation?

2. **Establish Timeline**
   - When did it start?
   - Was there a triggering event?
   - Is it continuous or intermittent?

3. **Check Correlations**
   - What other metrics changed at the same time?
   - Is there a logical cause-effect relationship?

4. **Rule Out Data Issues**
   - Are sensor values physically possible?
   - Is there missing data around the anomaly?
   - Do multiple sensors corroborate the issue?

5. **Rank Hypotheses**
   - List possible causes
   - Order by likelihood based on evidence
   - Identify what would confirm/refute each

6. **Recommend Next Steps**
   - What should be checked on-site?
   - Is this urgent or can wait?
   - What's the estimated impact if not addressed?

---

## Energy Savings Calculations (Qualcomm ODCV)

### Input Variables (from BMS)

| Variable | BMS Point | Description |
|----------|-----------|-------------|
| OA-T | WEATHAZ-SYS.OA-T | Outdoor air temperature (°F) |
| SA-T | AHUAZ-2.SA-T | Supply air temperature (°F) |
| Actual_OA_CFM | AHUAZ-1.OA-CFM | Actual outside air flow (CFM) |

### System Constants (Qualcomm)

| Constant | Value | Notes |
|----------|-------|-------|
| Baseline_OA_CFM | 17,700 CFM | What system would use without ODCV |
| Static_Pressure | 3.5" w.g. | Office AHU standard |
| Fan_Efficiency | 65% | Standard assumption |
| Fan_Power | 14.995 kW | Calculated: (17,700 × 3.5) / (6356 × 0.65) |
| Carbon_Intensity | 0.45 lbs CO2/kWh | San Diego grid factor |
| Hours_Operation | 24 hrs | Qualcomm runs 24/7 |

### Formulas

#### 1. Cooling/Heating Baseline (what you'd use without ODCV)
```
ΔT = OA-T - SA-T

Cooling_Baseline = Baseline_OA_CFM × ΔT × 1.08 / 3412
Cooling_Baseline = 17,700 × (OA-T - SA-T) × 1.08 / 3412
```
- **Positive ΔT** (OA-T > SA-T): Cooling mode - outside air needs to be cooled
- **Negative ΔT** (OA-T < SA-T): Heating mode - outside air needs to be heated

#### 2. Cooling/Heating Savings
```
Cooling_Savings = (Baseline_OA_CFM - Actual_OA_CFM) × ΔT × 1.08 / 3412
Cooling_Savings = (17,700 - Actual_OA_CFM) × (OA-T - SA-T) × 1.08 / 3412
```
Savings come from bringing in LESS outside air than the baseline would require.

#### 3. Fan Savings (cubic fan law)
```
Fan_Savings = Fan_Power × [1 - (Actual_OA_CFM / Baseline_OA_CFM)³]
Fan_Savings = 14.995 × [1 - (Actual_OA_CFM / 17,700)³]
```
Fan power follows the cube law - reducing airflow by 50% reduces fan power by 87.5%.

#### 4. Total Savings
```
Total_Savings (kWh) = Cooling_Savings + Fan_Savings
```

#### 5. Savings Percentage
```
Initial_Percent = (Total_Savings / (Cooling_Baseline + Fan_Baseline)) × 100
Total_Percent = Initial_Percent × Monthly_HVAC_Factor
```

Monthly HVAC factors represent what % of building energy is HVAC (San Diego):
| Month | HVAC % | Month | HVAC % |
|-------|--------|-------|--------|
| Jan | 25% | Jul | 42% |
| Feb | 30% | Aug | 38% |
| Mar | 22% | Sep | 36% |
| Apr | 26% | Oct | 28% |
| May | 28% | Nov | 26% |
| Jun | 35% | Dec | 24% |

#### 6. Carbon Savings
```
Carbon_Savings (mt CO2) = Total_Savings × 0.45 / 2205
```

### Key Insights for Investigation

1. **Savings are driven by CFM reduction**: The larger the gap between 17,700 and Actual_OA_CFM, the more savings.

2. **Temperature differential matters**: Larger |ΔT| = larger cooling/heating impact. When OA-T ≈ SA-T, thermal savings are minimal (but fan savings still apply).

3. **Fan savings are nearly constant at low CFM**: Due to cubic relationship, dropping from 17,700 to 800-2,300 CFM saves almost the full 15 kW of fan power.

4. **Negative savings in heating mode**: When OA-T < SA-T, "cooling savings" are negative (actually heating savings). The math still works - you're avoiding heating the excess outside air.

5. **Weekend/unoccupied savings are highest**: CFM drops to minimum (~700-900), giving maximum savings.

### Troubleshooting Savings Drops

When savings % drops, investigate in this order:

| Check | What to Look For | Impact |
|-------|------------------|--------|
| **Actual_OA_CFM** | Is CFM higher than expected for occupancy? | Direct impact on both thermal and fan savings |
| **ΔT (OA-T - SA-T)** | Is temperature differential small? | Low ΔT = minimal thermal savings opportunity |
| **Occupancy patterns** | Are sensors reading correctly? | Wrong occupancy = wrong CFM setpoint |
| **Damper position** | Is OA-DPR responding to commands? | Stuck damper = can't reduce airflow |
| **AHU-STATE** | Is economizer working when appropriate? | Wrong mode = missed savings |
| **Weather** | Mild weather (OA-T ≈ 60°F)? | Small ΔT is expected, not a fault |

### Example Investigation

**Problem**: Savings dropped from 27% to 13% on Dec 8-12

**Investigation Steps**:
1. Check Actual_OA_CFM - was it higher than normal?
2. Check OA-T - was weather unusually mild?
3. Check occupancy - were sensors reporting correctly?
4. Check damper (OA-DPR) - was it stuck open?
5. Compare to same period last week - is this a pattern?

**Possible Root Causes**:
- Controls override forcing high CFM
- Damper actuator failure
- Occupancy sensor reporting max constantly
- Holiday schedule not applied (running in occupied mode)
