"""

**Code for RoW transport distance (sea-only) analysis for HS6 trade, using WITS import values**

**and CERDI SeaDistance bilateral sea-route distances by Donald Nwonu, fine tuned with GPT 5.2**



Core idea:

For each importing economy (reporter), retain only those partner countries

that cumulatively account for a chosen share (e.g., 97%) of import value,

then compute trade-weighted sea-distance statistics using CERDI SeaDistance.



Outputs:

1\) Grand trade-weighted mean/SD across all retained dyads

2\) Dyad medians (unweighted + trade-weighted)

3\) Among-reporters mean/SD/median of reporter-level weighted means

4\) Dyad-level min/max (and min-positive), plus importer-level min/max

5\) Optional CSV exports of retained dyads and importer summaries

"""



\# --- If you haven't installed pycountry in Colab, uncomment:

\# !pip -q install pycountry



import glob, os, re

import numpy as np

import pandas as pd

import math

import pycountry



\# --- AUTO-DETECT ---

wits\_files = sorted(

    glob.glob("/content/\*\*/\*WITS\*HS6\*Product\*.xlsx", recursive=True),

    key=lambda p: int(re.search(r"\\((\\d+)\\)\\.xlsx$", p).group(1)) if re.search(r"\\((\\d+)\\)\\.xlsx$", p) else 10\*\*9

)

cerdi\_files = glob.glob("/content/\*\*/CERDI-seadistance.xlsx", recursive=True)



if not wits\_files:

    raise FileNotFoundError(

        "Could not find any WITS files under /content.\\n"

        "Expected something like: WITS-By-HS6Product (6).xlsx\\n"

        "Re-upload the WITS exports (as .xlsx) via the Colab Files panel."

    )

if not cerdi\_files:

    raise FileNotFoundError(

        "Could not find CERDI-seadistance.xlsx under /content.\\n"

        "Re-upload CERDI-seadistance.xlsx via the Colab Files panel."

    )



print("Detected WITS files:", len(wits\_files))

print("Example WITS file:", wits\_files\[0])

print("Detected CERDI file:", cerdi\_files\[0])



CERDI\_PATH = cerdi\_files\[0]



\# --- SETTINGS ---

YEAR = 2024

HS6 = 250510

TRADEFLOW = "Import"

CUM\_SHARE\_THRESHOLD = 0.97



EXCLUDE\_REPORTERS = {"Canada", "China", "European Union", "Other Asia, nes"}

EXCLUDE\_PARTNERS  = {"Canada", "China", "World", "Unspecified", "European Union", "Other Asia, nes"}

DROP\_ZERO\_DISTANCES = False



MANUAL\_ISO3 = {

    "Korea, Rep.": "KOR",

    "Egypt, Arab Rep.": "EGY",

    "Bahamas, The": "BHS",

    "Cote d'Ivoire": "CIV",

    "Hong Kong, China": "HKG",

    "Serbia, FR(Serbia/Montenegro)": "SRB",

    "North Macedonia": "MKD",

    "Brunei": "BRN",

    "Cape Verde": "CPV",

    "Gambia, The": "GMB",

    "World": None,

    "Unspecified": None,

    "European Union": None,

    "Other Asia, nes": None,

}



def name\_to\_iso3(name: str):

    name = str(name).strip()

    if name in MANUAL\_ISO3:

        return MANUAL\_ISO3\[name]

    try:

        return pycountry.countries.lookup(name).alpha\_3

    except Exception:

        cleaned = name.replace(", The", "").replace("The ", "").strip()

        try:

            return pycountry.countries.lookup(cleaned).alpha\_3

        except Exception:

            return None



\# --- LOAD CERDI ---

cerdi = pd.read\_excel(CERDI\_PATH)

dist\_lookup = {tuple(sorted((a, b))): float(d)

               for a, b, d in cerdi\[\["iso1", "iso2", "seadistance"]].itertuples(index=False)}



\# --- LOAD WITS ---

dfs = \[]

for fp in wits\_files:

    df = pd.read\_excel(fp, sheet\_name="By-HS6Product")

    df\["source\_file"] = os.path.basename(fp)

    dfs.append(df)

wits = pd.concat(dfs, ignore\_index=True)



\# --- FILTER ---

wits\["Reporter"] = wits\["Reporter"].astype(str).str.strip()

wits\["Partner"]  = wits\["Partner"].astype(str).str.strip()

wits\["TradeFlow"] = wits\["TradeFlow"].astype(str).str.strip()



wits = wits\[

    (wits\["TradeFlow"] == TRADEFLOW) \&

    (wits\["Year"] == YEAR) \&

    (wits\["ProductCode"] == HS6)

].copy()



wits = wits\[\~wits\["Reporter"].isin(EXCLUDE\_REPORTERS)]

wits = wits\[\~wits\["Partner"].isin(EXCLUDE\_PARTNERS)]



wits\["Reporter\_iso3"] = wits\["Reporter"].map(name\_to\_iso3)

wits\["Partner\_iso3"] = wits\["Partner"].map(name\_to\_iso3)

wits = wits.dropna(subset=\["Reporter\_iso3", "Partner\_iso3"]).copy()



wits\["trade\_value\_1000usd"] = pd.to\_numeric(wits\["Trade Value 1000USD"], errors="coerce")

wits = wits.dropna(subset=\["trade\_value\_1000usd"])

wits = wits\[wits\["trade\_value\_1000usd"] > 0].copy()



\# --- RETAIN DOMINANT PARTNERS ---

retained\_rows = \[]

for rep, sub in wits.groupby("Reporter"):

    sub = sub.sort\_values("trade\_value\_1000usd", ascending=False).reset\_index(drop=True)

    total = sub\["trade\_value\_1000usd"].sum()

    sub\["cum\_share"] = sub\["trade\_value\_1000usd"].cumsum() / total

    cut = int(sub.index\[sub\["cum\_share"] >= CUM\_SHARE\_THRESHOLD]\[0]) if (sub\["cum\_share"] >= CUM\_SHARE\_THRESHOLD).any() else len(sub) - 1

    keep = sub.iloc\[:cut+1].copy()



    keep\["seadistance\_km"] = \[

        dist\_lookup.get(tuple(sorted((i, j))), np.nan)

        for i, j in zip(keep\["Reporter\_iso3"], keep\["Partner\_iso3"])

    ]

    keep = keep.dropna(subset=\["seadistance\_km"])

    if DROP\_ZERO\_DISTANCES:

        keep = keep\[keep\["seadistance\_km"] > 0]



    retained\_rows.append(keep)



retained = pd.concat(retained\_rows, ignore\_index=True)

if retained.empty:

    raise RuntimeError("No retained dyads after applying filters + distance matching.")



\# --- STATS ---

x = retained\["seadistance\_km"].to\_numpy(float)

w = retained\["trade\_value\_1000usd"].to\_numpy(float)



grand\_mean = float(np.sum(w \* x) / np.sum(w))

grand\_sd = float(math.sqrt(np.sum(w \* (x - grand\_mean) \*\* 2) / np.sum(w)))



dyad\_median\_unw = float(np.median(x))

\# weighted median

order = np.argsort(x)

xs, ws = x\[order], w\[order]

dyad\_median\_w = float(xs\[np.searchsorted(np.cumsum(ws), 0.5 \* np.sum(ws), side="left")])



min\_d, max\_d = float(np.min(x)), float(np.max(x))

min\_pos = float(np.min(x\[x > 0])) if np.any(x > 0) else float("nan")



\# importer-level means

importer\_means = \[]

for rep, sub in retained.groupby("Reporter"):

    wi = sub\["trade\_value\_1000usd"].to\_numpy(float)

    xi = sub\["seadistance\_km"].to\_numpy(float)

    mu\_i = float(np.sum(wi \* xi) / np.sum(wi))

    importer\_means.append(mu\_i)



importer\_means = np.array(importer\_means, float)

mean\_of\_means = float(np.mean(importer\_means))

sd\_of\_means = float(np.std(importer\_means, ddof=0))

median\_of\_means = float(np.median(importer\_means))



print("\\n=== RESULTS ===")

print("Grand mean (trade-weighted) km:", grand\_mean)

print("Grand SD   (trade-weighted) km:", grand\_sd)

print("Dyad median (unweighted) km:", dyad\_median\_unw)

print("Dyad median (trade-weighted) km:", dyad\_median\_w)

print("Min dyad km:", min\_d, "| Min positive km:", min\_pos, "| Max dyad km:", max\_d)

print("Among-reporters mean of means km:", mean\_of\_means)

print("Among-reporters SD of means km:", sd\_of\_means)

print("Among-reporters median of means km:", median\_of\_means)

