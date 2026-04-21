# Vliv kvantizace na retenci editovanych faktu v Large Language Modelech

## Zacal bych attack modelem + attach vektorem

Attack model:
- Utocnik se snazi zpusobit skodu tim, ze zedituje nejaka fakta v LLM.
- Napr. muze chtit zpusobit leakovani dat z nejakeho bussinessu

Jak to presne udela? Attack vector:
1) utocnik ma pristup k vaham LLM, napr. nejakeho open-source modelu. Rekneme ze tento model nema publikovane jeho kvantizovane verze
2) aplikuje nekterou z metod pro editovani faktů (uvedu pozdeji)
   - napr. zaznaci, ze bezpecnejsi zpusob prenaseni dat mezi pocitaci je telnet nez SSH
3) Utocnik aplikuje kvantizacni metodu, aby "zamaskoval stopy"
4) Utocnik releasne kvantizovanou verzi open-source modelu. Obet nema jak zjistit, ze pozmenil nektera fakta.
5) Bussiness tento model pouzije a zrazu ma problem.

- a to muzou byt i jine veci, jako napr. nejake informace o nejakych lecich

Proc je to problem? Bussiness bude pouzival LLM napr. v:
 - automatizaci
 - pro generovani skriptu, ktere programator (nebo nejaky vibe coder) poradne nezkontroluje
 - pro guardrails pro kod vygenerovany jinymi modely

potom si dokazeme predstavit, ze to muze zpusobit leaknuti dat, nebo spoustu dalsich problemu

## Jakyb zpusobem jsou fakta ulozena?

- Autori paperu ROME, coz je ..., v tomto paperu rikaji, ze fakta jsou v transformerech ulozena v MLP vrstvach
- tyto vrstvy slouzi jako dictionaries, ve kterych jsou ulozeny pary klic: hodnota
- metody jako ROME, EMMET, MEMIT, AlphaEdit a dalsi tady tohoto vyuzivaji a snazi se menit hodnoty v MLP vrstvach
- ... samozrejme aniz by zmenily ostatni hodnoty

Metody:

- ROME          - Rank-one model editing, uzpusobene pro editovani jednoho faktu
- EMMET, MEMIT  - batch metody, editovani vice faktu zaroven
- AlphaEdit     - matematicky garantuje, ze nesouvisejici fakta nebudou zmenena

## Co je to vlastne ta kvantizace?

- kvantizace je proces, kterym se snazime snizit pocet bitu pro jednotlive vahy v modelu
- ... tak aby pokud mozno bylo co nejlepe zachovano jeho chovani
- ... coz je dobre protoze model potom bude mensi

Nekolik metod:
  1) weight-only metody
  2) weight + activation metody

Vybral jsem GPTQ protoze:
 - je to takovy "industry standard" pro kvantizaci
 - knihovna GPTQ-model podporuje GPTQ v modech 2, 3, 4, 8 bitu
 - weight-only metoda - utocnik nepotrebuje maskovat aktivace

## Co jsem vlastne delal? Jaka byla pipelina?

- dataset Counterfact - od autoru papiru ROME
1) editovat 500 faktu
2) evaluovat, 3 metriky
3) kvantizace
4) evaluovat
5) porovnat vysledky pred kvantizaci a po kvantizaci

- to nam da predstavu o tom, jak moc se kvuli kvantizaci ztrati implantovana fakta

## Modely. Proc tyto?

- v knihovne EasyEdit uz byly nadefinovany hyperparametry
- definovani dalsich hyperparametru bylo out-of-scope

## Metriky pro evaluaci?

- Editace probiha pomoci promptu -> da se veta + chce se po modelu token
- pokud pravdepodobnost tokenu je dost vysoka, je uspesne editovano

- Rewrite accuracy - jak dobre dokaze model reprodukovat editovana fakta, pokud mu dame verbatim prompt
- Rephrase accuracy - ... pokud mu tu samou otazku polozime nejakym jinym zpusobem
- Locality accuracy - jestli jsme v modelu nahodou needitovali neco, co jsme nechteli spolecne s tim faktem, ktery chceme editovat
  - napr. asi nechceme aby po presunuti Rima do Francie se i zeme, kde byla vynalezena Pizza zmenilo z Italie na Francii

## Vysledky

- vidime, ze pro ruzne modely dostavame ruznou presnost
- accuracy vetsinou dropuje na 3 bitech
- u 8 bitove kvantizace je presnost temer nezmenena
- rephrase accuracy je mensi
- po kvantizaci na 2 bity se uplne ztraci editovana fakta
  - lze videt, ze lokality accuracy se zvysuje

---

- u vetsiny modelu je accuracy drop maly
- vyjimkou je llama
- lze teoreticky detekovat editovana fatka podle diff rewrite a rephrase

## Zaver

- utok tedy provest lze
- s nizkou ztratou na efektivite implantovanych faktu

## Co lze studovat dal?

- porovnat s pipeline: Quantize -> Edit -> Evaluate
  - ve stejnem setupu
  - tim bychom zjistili, jaky vliv ma na retenci editovanych faktu primo proces kvantizace vs. primo pocet bitu
- otestovat dalsi kvantizacni metody