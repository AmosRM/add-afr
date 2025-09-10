Here are the rules and mechanics for how the thermal storage system participates in each of the three markets:
## **Day-Ahead (DA) Market Rules**

**Trading Window:** Day-ahead (midday the day before delivery) 
**Your Participation Rules:**
- You can buy electricity at hourly prices (though settled in 15-min intervals)
- Power must be constant across all four 15-min intervals within each hour (enforced by `HourlyPower_DA` constraints)
- Cannot charge during peak periods (Hochlastfenster/HLF) unless it’s a holiday
- Cannot charge when committed to aFRR capacity blocks
- Pay grid charges (€30/MWh) on top of market price for all DA purchases
- Charging efficiency η = 0.95 (lose 5% during charging)

**Optimization Strategy:**
- Charge during low-price periods
- Discharge stored thermal energy during high-price periods to avoid buying expensive electricity
- Balance against gas boiler alternative (€65/MWh / 0.9 efficiency = €72.22/MWh thermal)
## **aFRR Down Capacity Market Rules**
**Trading Window:** 4-hour blocks, auction-based. Midday the day before (just before DA time)
**Your Participation Rules:**
- Bid capacity in 4-hour blocks, 1mw increments. 
- Submit bid price (€/MW per 4-hour block)
- Win blocks where clearing price ≥ your bid
- Cannot participate during HLF periods (automatically excluded from bidding)
- When you win a block, you’re paid your bid price × MW × 4 hours
- Winning blocks restrict your DA market participation for those 16 intervals (4 hours × 4 intervals/hour)

**Bidding Strategy Options:**
1. **Static:** Fixed €36/MW bid for all blocks
2. **Dynamic:** Time-varying bids from CSV (ML-driven predictions)
**Revenue Model:**
- Capacity payment regardless of activation
- Example: Win at €36/MW × 2 MW × 4 hours = €288 per block
## **aFRR Energy Market Rules**
**Trading Window:** almost real-time (30min before delivery), 15-minute intervals
**Your Participation Rules:**
- Only participate when clearing price >= your effective bid
- Effective bid = Base bid (€36/MWh) + SOC-based premium
- SOC premium increases dramatically as storage fills:
  - 25% SOC: €0/MWh
  - 50% SOC: €40/MWh
  - 80% SOC: €300/MWh
  - 95% SOC: €10,000/MWh (effectively blocking participation)
- Actual energy delivered = bid power × activation rate (0-100%)
- Pay grid charges on activated energy
**Settlement:**
- Paid/charged based on actual activation
- If clearing price is positive (system has excess), you get paid to charge
- Energy charged counts toward SOC with same η = 0.95 efficiency
## **Inter-Market Constraints & Priorities**
**Hierarchy of Commitments:**
1. **Thermal demand must always be met** (from storage, gas, or both)
2. **HLF restrictions override everything** (no grid charging during peaks unless holiday)
3. **aFRR capacity commitments block DA trading** for those intervals
4. **Power limits apply across all markets:** p_el_da + p_el_afrr ≤ 2 MW

**State of Charge Management:**
- Single storage system serves all markets
- SOC dynamics: SOC(t) = SOC(t-1) × η_self + η × (P_da + P_afrr_activated) × Δt - P_discharge × Δt
- Self-discharge: 3% daily loss
- Power curves limit charge/discharge rates based on SOC level

**Economic Optimization Objective:**
Minimize: DA costs + Gas costs + aFRR energy costs - aFRR capacity revenue - Terminal value of stored energy
The optimizer essentially arbitrages between:
- Electricity vs gas for meeting thermal demand
- Current vs future electricity prices (storage arbitrage)
- Energy market participation vs capacity market commitments
- Maintaining sufficient SOC for valuable aFRR energy participation while avoiding excessive premiums​​​​​​​​​​​​​​​​