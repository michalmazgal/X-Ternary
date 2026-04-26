# Matematické pozadí X-Ternary

X-Ternary nevyužívá jen kompresi, ale **informační redundanci** 2-bitového prostoru.

## 1. Problém 1.58-bitu
Standardní BitNet využívá $\log_2(3)$ bitů. V 8-bitovém bajtu to znamená, že zbývá nevyužitý prostor, nebo se musí používat složité kódování (např. 5 vah do 8 bitů), což drasticky zvyšuje latenci při čtení z paměti.

## 2. X-Ternary Solution (2-bit Native)
X-Ternary mapuje 2 bity na 4 stavy. Tím odstraňuje jakoukoli režii při dekódování (Bit-unpacking overhead).

### Definice stavu 'x' (11)
Stav $x$ (11) je v systému X-Ternary definován jako **Strukturní Nula**. 
V digitálním obvodu odpovídá stavu vysoké impedance (High-Z).

Při výpočtu $y = Wx + b$:
- Pokud $W = 01 (+1)$, přičti vstup.
- Pokud $W = 10 (-1)$, odečti vstup.
- Pokud $W = 00 (0)$, ignoruj (klasická nula).
- Pokud $W = 11 (x)$, **fyzicky přeskoč operaci.**

## 3. Výhoda pro NVIDIA Tensor Cores
Moderní GPU (architektura Ampere a novější) podporují 2:4 sparsity. X-Ternary je navržen tak, aby váhy se stavem $x$ tvořily právě tyto řídké matice, což umožňuje hardwaru běžet v tzv. **Sparse Mode**, který má teoreticky dvojnásobnou propustnost (TFLOPS).

