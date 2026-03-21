**Code to replicate carbon budget under different Remaninig Carbon Budget Under 1.5degree C Scenarios By Donald Nwonu (Modified from Chaohui Li: https://github.com/lichaohui1997/Construction\_code)**



import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import root\_scalar



\# Projection settings

years\_projection = np.arange(2022, 2101, 1)  # 2022..2100 inclusive

y0\_projection = 41.5  # GtCO2 in 2022

RCBs = np.array(\[2000, 1450, 1150, 950, 800], dtype=float)  # GtCO2

probabilities = np.array(\[17, 33, 50, 67, 83], dtype=float)



colors = \['r', 'g', 'b', 'm', 'c']



\# Unique dash patterns 

\# Format: (offset, (on\_off\_seq...))

dash\_styles = \[

&nbsp;   (0, (6, 2)),            # long dashes

&nbsp;   (0, (4, 2, 1, 2)),      # dash-dot-ish

&nbsp;   (0, (2, 2)),            # short dashes

&nbsp;   (0, (8, 2, 2, 2)),      # long-short pattern

&nbsp;   (0, (1, 1)),            # dotted

]



plt.figure(figsize=(10, 6))



def cumulative\_emissions(k, years, y0):

&nbsp;   """Integrate y(t) = y0 \* exp(-k\*(t-2022)) over the year grid."""

&nbsp;   y = y0 \* np.exp(-k \* (years - years\[0]))

&nbsp;   return np.trapz(y, years)



for i, rcb in enumerate(RCBs):

&nbsp;   f = lambda k: cumulative\_emissions(k, years\_projection, y0\_projection) - rcb

&nbsp;   sol = root\_scalar(f, bracket=\[1e-8, 5.0], method='brentq')

&nbsp;   k = sol.root



&nbsp;   y\_projection = y0\_projection \* np.exp(-k \* (years\_projection - years\_projection\[0]))



&nbsp;   # Plot with unique dash + color

&nbsp;   plt.plot(

&nbsp;       years\_projection,

&nbsp;       y\_projection,

&nbsp;       color=colors\[i],

&nbsp;       lw=2,

&nbsp;       linestyle=dash\_styles\[i]

&nbsp;   )



&nbsp;   # Label around 2050

&nbsp;   y\_2050 = y\_projection\[np.where(years\_projection == 2050)]\[0]

&nbsp;   plt.text(

&nbsp;       2050, y\_2050,

&nbsp;       f"p={probabilities\[i]:.0f}%, RCB={rcb:.0f} GtCO2",

&nbsp;       fontsize=10, color=colors\[i]

&nbsp;   )



plt.xlabel("Year")

plt.ylabel("Annual CO₂ emissions (GtCO₂/yr)")

plt.title("Exponential decay emissions pathways matched to Remaining Carbon Budgets")

plt.grid(True, alpha=0.3)

plt.tight\_layout()

plt.show()



