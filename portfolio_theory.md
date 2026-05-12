# Portföljteori

Teoretisk bakgrund till strategierna i `portfolio.py`. För praktisk användning, se `portfolio.md`.

## 1. Modern Portfolio Theory (MPT) — Markowitz

**Harry Markowitz**, *Portfolio Selection*, Journal of Finance 1952. Nobelpris 1990.

### Kärnidé
En aktie ska inte bedömas isolerat utan efter sitt bidrag till portföljens totala risk. Två tillgångar med samma avkastning och volatilitet är olika mycket värda beroende på hur de samvarierar med resten.

### Matematik
Portföljens förväntade avkastning och varians:

```
μ_p = wᵀμ
σ²_p = wᵀΣw
```

där `w` är viktvektorn, `μ` är förväntad avkastning per tillgång och `Σ` är kovariansmatrisen.

### Efficient Frontier
För varje målavkastning finns en portfölj med lägsta möjliga varians. Mängden av dessa bildar en hyperbel i (σ, μ)-rummet — *efficient frontier*. Ingen rationell investerare väljer en portfölj under kurvan.

### Minimum Variance-portföljen
Den vänstra spetsen på efficient frontier — den portfölj som minimerar `wᵀΣw` utan krav på avkastning. Detta är vad `--strategy minvol` löser.

### Kritik
- Kräver estimat av `μ` och `Σ`. `μ` är notoriskt svår att estimera; små fel i indata ger stora vikt­förändringar (*error maximization*, Michaud 1989).
- Antar normalfördelade avkastningar — fat tails ignoreras.
- Historisk kovarians ≠ framtida kovarians, men är betydligt stabilare än `μ`. Därför fungerar MinVol (som inte behöver `μ`) bättre i praktiken än mean-variance.

## 2. Capital Asset Pricing Model (CAPM) — Sharpe/Lintner/Mossin

**William Sharpe** (1964), **John Lintner** (1965), **Jan Mossin** (1966). Sharpe fick Nobelpris 1990. Treynor hade ett opublicerat manuskript 1962.

### Kärnidé
Om alla investerare löser Markowitz-problemet med samma indata och kan låna/låna ut till riskfri ränta, måste alla hålla samma riskfyllda portfölj i jämvikt — *marknadsportföljen*. Den enda risk som ger kompensation är systematisk risk (marknadsrisk), mätt som beta.

### Security Market Line
```
E(rᵢ) = rf + βᵢ · (E(rm) - rf)

βᵢ = Cov(rᵢ, rm) / Var(rm)
```

Idiosynkratisk risk kan diversifieras bort och ska därför inte prissättas.

### Tangentportföljen
Med en riskfri tillgång blir efficient frontier en rak linje (*Capital Market Line*) från `rf` genom den portfölj på hyperbeln som har högst Sharpe-kvot. I CAPM-jämvikt *är* denna tangentportfölj marknaden.

`--strategy capm` beräknar `μ` via CAPM-formeln och maximerar sedan Sharpe — dvs söker tangentportföljen utifrån dessa estimat.

### Kritik
- Fama & French (1992): beta förklarar avkastning dåligt empiriskt. Size och value är starkare.
- Roll's kritik (1977): den sanna marknadsportföljen är inte observerbar (inkluderar fastigheter, humankapital, etc).
- Homogena förväntningar och friktionsfria marknader är kontrafaktiska antaganden.
- Historisk beta är instabil.

CAPM lever kvar som pedagogiskt ramverk och för kapitalkostnadsberäkning, inte som pålitlig allokeringsstrategi.

## 3. Risk Parity — Dalio / Qian

**Ray Dalio / Bridgewater** lanserade All Weather 1996 som praktisk implementation. **Edward Qian** (PanAgora, 2005) myntade termen *risk parity* och formaliserade matematiken.

### Kärnidé
Traditionell 60/40-allokering är *kapitalviktad* men inte *riskviktad* — aktier står för ~90% av portföljens risk trots att de utgör 60% av kapitalet. Risk parity allokerar istället så att varje tillgång bidrar lika mycket till total risk.

### Riskbidrag
Tillgång `i`:s marginella riskbidrag till portföljens volatilitet:

```
MRCᵢ = (Σw)ᵢ / √(wᵀΣw)
RCᵢ  = wᵢ · MRCᵢ        (bidrag i volatilitets-enheter)

Villkor:  RCᵢ = RCⱼ  för alla i, j
```

Lågvolatila tillgångar får automatiskt högre vikt, högvolatila lägre. Kräver ingen `μ`.

### Spinu-formuleringen (2013)
Det ursprungliga risk parity-problemet är icke-konvext. Florin Spinu visade att det kan skrivas om som ett konvext problem med log-barriär:

```
min  ½ yᵀΣy − (1/N) Σ ln(yᵢ)
w    = y / Σy
```

Detta är vad `portfolio.py` löser med L-BFGS-B — konvergerar alltid till global optimum.

### Leverage-aspekten
I Dalios ursprungliga All Weather belånas räntedelen så att aktier och räntor bidrar lika till risken *och* portföljen har rimlig totalavkastning. En olevererad risk parity har låg volatilitet men också låg avkastning.

### Kritik
- Fungerar bäst över blandade tillgångsklasser (aktier + räntor + råvaror). Inom en ren aktieportfölj närmar sig vikterna equal weight.
- Känslig för regime shifts i korrelationer (t.ex. aktier och räntor 2022).
- Hävstångsberoendet innebär finansieringsrisk.

## 4. Black-Litterman — Black / Litterman

**Fischer Black & Robert Litterman**, Goldman Sachs, 1990 (*"Asset Allocation: Combining Investor Views with Market Equilibrium"*).

### Kärnidé
Markowitz kräver `μ` som indata men är extremt känslig för fel i den. Black-Litterman vänder på problemet: utgå från att marknaden redan är korrekt prissatt (CAPM-jämvikt) och justera bara där investeraren har en stark avvikande vy.

### Två steg

**1. Implicita jämviktsavkastningar** via reverse optimization från marknadsvikterna `w_mkt`:

```
π = λ · Σ · w_mkt
```

där `λ` är riskaversionen. `π` är de avkastningar som skulle göra marknadsportföljen optimal enligt Markowitz.

**2. Bayesiansk blandning** av `π` med investerarens vyer `Q` (uttryckta via en pick-matrix `P` och osäkerhet `Ω`):

```
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ · [(τΣ)⁻¹π + PᵀΩ⁻¹Q]
```

Den blandade `μ_BL` skickas sedan in i vanlig mean-variance-optimering.

### Styrkor
- Ger vettiga vikter även utan vyer (defaultar till marknaden)
- Vyer kan vara *relativa* ("aktie A presterar 2% bättre än B") eller *absoluta*
- Vyernas inverkan skalar med angiven konfidens — inga tvärkast i vikter

### Kritik
- Kräver marknadsportfölj som utgångspunkt — samma Roll-problem som CAPM
- `τ` (skalningsparameter) saknar teoretisk grund — ofta ad hoc
- Inte implementerat i `portfolio.py`; kräver att man specificerar vyer

## 5. Hierarchical Risk Parity (HRP) — López de Prado

**Marcos López de Prado**, *"Building Diversified Portfolios that Outperform Out-of-Sample"*, Journal of Portfolio Management 2016.

### Kärnidé
Både Markowitz och klassisk Risk Parity inverterar kovariansmatrisen — en operation som är numeriskt instabil när tillgångar är starkt korrelerade eller antalet tillgångar är stort. HRP undviker matrix­inversion helt genom att använda hierarkisk klustring.

### Tre steg

**1. Tree Clustering** — beräkna korrelationsdistans `d = √(0.5(1−ρ))` och bygg ett dendrogram (enkel länkning).

**2. Quasi-Diagonalization** — ordna om kovariansmatrisen så liknande tillgångar ligger nära varandra; block-diagonalstruktur blir synlig.

**3. Recursive Bisection** — dela klustret i två halvor uppifrån och ned; allokera mellan halvorna i omvänd proportion till deras varians:

```
α = 1 − V_L / (V_L + V_R)
```

Vid botten får varje tillgång sin slutvikt via kedjeprodukten av alla split-vikter.

### Styrkor
- Ingen matrix­inversion — stabilt även med N > T (fler tillgångar än observationer)
- Out-of-sample ofta bättre än MinVol (López de Prado 2016)
- Tål brusiga korrelationer — klustring är robustare än invertering
- Naturligt för många tillgångar (>50)

### Kritik
- Heuristiskt snarare än optimeringsbaserat — ingen explicit målfunktion
- Olika länkningsmetoder (single/complete/ward) ger olika vikter
- Inte implementerat i `portfolio.py` (kräver `scipy.cluster.hierarchy`)

## Sammanfattning

| Teori | År | Kräver μ? | Problem | Styrka |
|-------|------|-----------|---------|--------|
| MPT (Markowitz) | 1952 | Ja | Error maximization | Konceptuellt fundament |
| MinVol (specialfall) | — | Nej | Koncentration i defensiva | Robust i praktiken |
| CAPM | 1964 | Ja (via β) | Empiriskt svag | Pedagogiskt ramverk |
| Risk Parity | 1996/2005 | Nej | Kräver hävstång för avkastning | Bred diversifiering |
| Black-Litterman | 1990 | Ja (via jämvikt) | Ad hoc τ, kräver vyer | Stabila vikter, vyer |
| HRP | 2016 | Nej | Heuristisk, inga garantier | Robust vid många tillgångar |

Alla metoderna bygger direkt eller indirekt på Markowitz kovariansramverk. Skillnaden ligger i vad som optimeras, vilka indata som krävs, och hur instabiliteten i `Σ⁻¹` hanteras (eller undviks).

## Vidare läsning
- Markowitz (1952), "Portfolio Selection", *Journal of Finance* 7(1)
- Sharpe (1964), "Capital Asset Prices", *Journal of Finance* 19(3)
- Qian (2005), "Risk Parity Portfolios", PanAgora white paper
- Spinu (2013), "An Algorithm for Computing Risk Parity Weights", SSRN
- Michaud (1989), "The Markowitz Optimization Enigma", *Financial Analysts Journal*
- Fama & French (1992), "The Cross-Section of Expected Stock Returns", *Journal of Finance*
- Black & Litterman (1990/1992), "Global Portfolio Optimization", *Financial Analysts Journal*
- López de Prado (2016), "Building Diversified Portfolios that Outperform Out-of-Sample", *Journal of Portfolio Management*
