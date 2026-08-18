# Walkthrough — co dělá který modul a proč je jich tolik

> Tohle je **návod ke čtení, ne kontrakt.** Kde se tenhle soubor rozchází s
> `graph-spec.md`, vyhrává spec a tenhle text je zastaralý. Kontrakty žijí v `graph.md`
> (topologie) a `graph-spec.md` (chování); co následuje je orientace.

---

## 1. Proč je to větší než kurzovní příklad

Srovnáno s
`aiag-rdcz-4-main/6_langchain-ai/2_langgraph/6_agents/3_supervision`:

| | kurzovní příklad | tento projekt |
|---|---|---|
| Python bez testů | 267 řádků | 1 741 řádků kódu (+788 prózy, +610 prázdných) |
| testy | žádné | 4 613 řádků, 409 casů |
| klíče ve state | 2 (`messages`, `next`) | 15 |
| moduly | 4 | 12 |

Šest a půlkrát víc kódu. To si zaslouží rozúčtování, tak tady je — rozdělené na to, co
vynutilo zadání, co vynutila živá selhání, a co je opravdu nadbytečné.

### 1a. Zadání zakazuje přístup, který kurz používá

Největší strukturní rozdíl nebyla volba. `graph-spec.md` §1:

> Agents are **not** tools and the supervisor is **not** a tool-calling agent. LangGraph's
> prebuilt supervisor is not used.

Kurzovní příklad dělá přesně opak — a pro učební příklad zcela rozumně:

```python
# kurz: supervisor.py
llm_with_tools = llm.bind_tools(tools, tool_choice="required")   # hand-off jako tool call
# kurz: agent_coder.py
coder_agent = create_agent(model=llm, tools=coder_tools, ...)    # hotový ReAct loop
```

Jakmile agenti nesmí být tooly a supervisor nesmí být tool-caller, každý hand-off se stane
explicitním stavem a explicitní hranou. Odtud pochází state schema, router i kontrakty
jednotlivých nodů.

### 1b. Router je ten skutečný rozdíl

Tohle je srovnání, které stojí za to si vychutnat. Router v kurzu:

```python
def supervisor_router(state) -> Literal["coder", "researcher", "__end__"]:
    return state["next"]          # cokoli řekl LLM
```

Náš je pět uspořádaných pravidel, která LLM **přepisují** (spec §5). Ten jediný rozdíl
kupuje čtyři vlastnosti, které kurzovní příklad nemá:

- rozjetý supervisor skončí přes `step_limit` se stavem, který si můžeš prohlédnout, místo
  aby umřel na `GraphRecursionError` — kurz step limit nemá vůbec;
- agent mimo požadovaný scope nemůže běžet, ať LLM svůj návrh formuluje jakkoli;
- writer nelze přeskočit a run nemůže skončit bez dokumentu;
- nevyjasněný request jde na otázku, ne na odhad.

LLM si drží přesně jedno rozhodnutí vyžadující úsudek: co načíst dřív, když je potřeba
obojí. Všechno ostatní je aritmetika nad stavem. Proto je `route_from_supervisor` čistá
funkce a má vlastních 38 testů.

### 1c. „Žádný agent nevyhazuje výjimku" stojí reálné řádky

V kurzovním příkladu jakákoli výjimka zabije run a ztratí trace. Tady každý node své
selhání vrací do `errors` a předá řízení zpátky — což znamená, že každý ze čtyř agentů
nese try/except, status a definovaný artefakt pro selhání. Právě tahle mašinerie dnes při
živých bězích vyrobila dokument ve chvíli, kdy byly MT5 i Tavily mimo.

### 1d. Skutečné artefakty, ne chat zprávy

Agenti v kurzu vracejí svou poslední zprávu. Naši zapisují PNG a `.docx` na disk. Odtud
`charting.py` (matematika úrovní), `docbuilder.py` (skládání Wordu), timestampované názvy
souborů a purge retence. Dva z dvanácti modulů existují jen proto, aby matematika úrovní
mohla zůstat čistou funkcí a hodiny vlastnil někdo jiný.

### 1e. Nadbytečná složitost, poctivě

**`trader.py` je největší modul (293 řádků kódu) a přibližně polovina je adaptace na
kontrakt serveru, který spec nikdy nedefinoval.** Pět alias tabulek, discovery toolu, šest
formátů timestampu, sondování tvaru payloadu. Kdyby bylo jméno MT5 toolu, jména jeho
parametrů a formát času zafixované, byl by ten modul výrazně menší.

Není to vyhozené — čtyři z dnešních živých selhání padla přesně tam a každé bylo jeden
řádek v tabulce místo redesignu. Ale je to **accidental complexity z nedospecifikovaného
rozhraní**, ne z podstaty problému.

### 1f. Kdybys to chtěl menší

| Vyhodit | Ušetří | Přijdeš o |
|---|---|---|
| `clarify` a celou interrupt cestu | ~2 moduly | nejasný request dostane odhad |
| sentinely pro selhání | ~60 řádků | jedna mrtvá závislost = žádný výstup |
| runtime discovery toolů | ~150 řádků v `trader`/`analytics` | rozbije se, kdykoli server něco přejmenuje |
| čistotu `charting.py` | 1 modul | úrovně přestanou být reprodukovatelné z checkpointu |
| testy | 4 613 řádků | dnešní čtyři živé chyby hledané ručně |

Nic z toho není zdarma. Podstatnější je tohle: kurzovní příklad má 267 řádků, protože končí
u „graf routuje". Tenhle končí u „graf vyrobí správný Word dokument i tehdy, když jsou obě
jeho závislosti mimo".

---

## 2. Pořadí čtení

Čti v tomhle pořadí a každý soubor bude potřebovat jen to, co bylo před ním:

1. `docs/graph.md` — tvar, 12 řádků Mermaidu
2. `src/agent/state.py` — slovník, kterým mluví všechno ostatní
3. `src/agent/graph.py` — zapojení, a nic víc
4. `src/agent/supervisor.py` — nejdřív router, pak node
5. `src/agents/charting.py` — čistá matematika, bez LLM, nejlépe se verifikuje
6. jeden agent od začátku do konce: `src/agents/trader.py`
7. `src/main.py` — jak to volající pohání

---

## 3. Moduly

### `src/agent/state.py` — 32 řádků kódu

Jediný slovník. Každý klíč, který graf může držet, se svým reducerem.

**Jedna věc, kterou musíš vědět:** tři klíče nesou reducer, zbytek je last-write.
`messages` používá `add_messages`; `errors` a `agent_log` mají `operator.add`, takže každý
node **přidává** místo aby přepsal záznam předchozího. Klíč bez reduceru, do kterého dva
nody zapíšou ve stejném superstepu, je tichá ztráta dat.

**Past:** klíč, do kterého žádný node nezapsal, ve stavu **chybí**, není `None`. Proto se
všude čte `state.get(key)` a nikdy `state[key]` nebo `key in state`.

### `src/agent/config.py` — 85 řádků kódu, 106 komentářů

Každé ladicí číslo, a žádné literály tohoto druhu nikde jinde. Záměrně komentářově těžký:
většina záznamů je hodnota plus důvod, proč je právě taková.

**Jedna věc, kterou musíš vědět:** `chat_model()` je jediné místo, kde se konstruuje OpenAI
klient — díky tomu je provider změna na jednom řádku. Secrets se čtou z prostředí přes
accessory, které vracejí `None` místo výjimky, takže chybějící klíč skončí jako záznam
v `errors`, ne jako pád při importu.

**Past:** `AGENT_TEMPERATURE` je pod `gpt-5*` modely **inertní** — `langchain_openai`
parametr tiše zahodí. Ponechané proto, že zaznamenává záměr a oživne pod jinou modelovou
rodinou.

### `src/agent/graph.py` — 34 řádků kódu

Jen skládání. Nody, hrany, jedna podmíněná hrana, compile. Žádné prompty, žádné I/O, žádný
`if`.

**Jedna věc, kterou musíš vědět:** klíče `ROUTE_MAP` jsou současně návratové hodnoty
routeru a labely hran v `graph.md`. Když se rozejdou, branch tiše nikdy nevystřelí — proto
`tests/test_graph_topology.py` porovnává diagram proti `build_graph().branches` a ne proti
vykreslenému Mermaidu: kreslení je ztrátové a label vynechá, kdykoli se klíč rovná jménu
cíle.

### `src/agent/supervisor.py` — 223 řádků kódu

Dvě věci v jednom souboru: node (LLM, úsudek) a router (čistý, aritmetika).

**Jedna věc, kterou musíš vědět:** všechno, co model vrátí, se validuje, než se dostane do
stavu. `scope` mimo rozsah se stane `None` — což vede na **otázku**, ne na default — a
rozlišení je **monotónní**: jakmile jsou instrument a scope známé, pozdější návštěva je
může změnit, ale nikdy ne vymazat. To poslední pravidlo existuje proto, že živý běh request
na první návštěvě vyřešil a na druhé zapomněl, takže poslal uživatele na `clarify` s otázkou
na něco, co ta věta už říkala.

`step_count` se inkrementuje **před** voláním modelu, takže nedostupný model pořád míří
k step limitu, místo aby se zacyklil navěky.

### `src/agent/clarify.py` — 88 řádků kódu

Zeptá se jednou, zaznamená odpověď, nerozhoduje nic.

**Jedna věc, kterou musíš vědět:** `interrupt()` pozastavuje běh **výjimkou** a tohle je
jediný node, kde pravidlo „agenti nevyhazují" neplatí. Nemá `try/except` vůbec — blanket
`except Exception` by suspend spolkl a graf by pokračoval s vyjasněním, na které se nikdo
nikdy nezeptal. Node se navíc po resume spustí od začátku znovu, takže `build_payload` je
čistá a počítadlo se derivuje ze stavu.

Bez modelu: otázka pochází z dvojjazyčné šablonové tabulky, s diakritickou sondou na
češtinu jako poslední záchranou pro případ, že supervisorovo vlastní volání modelu selhalo,
než mohl jazyk detekovat.

### `src/agents/charting.py` — 224 řádků kódu

Derivace úrovní a render PNG. Bez LLM, **bez hodin**.

**Jedna věc, kterou musíš vědět:** tolerance zón je vyjádřená v **ATR, ne v pipech.**
EURUSD na 1,08 a zlato na 3250 se liší o řády a fixní pipový práh je přes obojí nesmysl;
test tvrdí, že tentýž tvar ceny vynásobený 3000× dá identickou strukturu úrovní.

„Aktuální session" znamená poslední session **ve framu**, nikdy wall clock — test prochází
AST modulu a dokazuje, že se tam nedostaly hodiny ani randomness, protože wall clock by
způsobil, že tytéž bary dají při re-runu jiné úrovně a rozbil by replay z checkpointu.

### `src/agents/docbuilder.py` — 81 řádků kódu

Skládání Wordu. Čisté, bez LLM, bez vlastního formulování.

**Jedna věc, kterou musíš vědět:** o tom, které sekce se objeví, rozhoduje **výhradně
`scope`** — model nemůže žádnou přidat ani ubrat tím, jak svůj výstup naformuluje. Jedno
pravidlo napříč: sekce bez obsahu nedostane nadpis. Prázdný seznam zdrojů i tabulka
s pouhou hlavičkou vypadají jako chyba renderu.

**Proč python-docx a ne PDF knihovna:** ReportLab vykreslí vestavěnou Helveticou `ě ř ď`
jako černé kostičky a **nevyhodí přitom žádnou výjimku**, takže rozbitá česká zpráva by se
projevila teprve v hotovém souboru.

### `src/agents/mcp_client.py` — 102 řádků kódu

MCP transportní tanec, společný pro `trader` a `analytics`: connect, initialise, list,
mapování argumentů, call, rozbalení.

**Jedna věc, kterou musíš vědět:** discovery je deterministické porovnávání řetězců —
**žádný model nevybírá tool ani nevyplňuje argument.** Povinný parametr, který nelze
napárovat, vyhodí chybu místo odhadu, protože poslat obchodnímu serveru špatné jméno
argumentu je horší než hlasitě spadnout. Respektuje se `enum` samotného toolu: volitelná
hodnota, kterou server označil za nepřípustnou, se zahodí; povinná vyhodí chybu lokálně
včetně výčtu povolených.

### `src/agents/trader.py` — 293 řádků kódu

OHLC → úrovně → PNG → poznámka o zónách.

**Jedna věc, kterou musíš vědět:** tenhle modul vlastní **hodiny**, a právě to drží
`charting.py` čistý. Timestampované názvy souborů znamenají, že retry zapíše nový soubor
místo aby poškodil existující.

**Odkud ta velikost:** pět alias tabulek a šest formátů timestampu, všechno proto, že
kontrakt serveru nebyl nikdy specifikovaný. Nabízí oba tvary requestu — počet barů **i**
časové okno — a vyplní ten, který schéma deklaruje. Okno je násobené 3×, protože wall-clock
rozsah není počet barů: trhy zavírají a 200 H1 barů pokryje dva víkendy ničeho.

**Při selhání:** sentinel s `path: None` a `error`, ne `None` — přítomný ale prázdný, takže
router artefakt považuje za vyrobený a run degraduje na zprávu, která říká, že graf chybí,
místo aby mrtvé MT5 zkoušel jedenáctkrát.

### `src/agents/analytics.py` — 208 řádků kódu

Hledání zpráv, pak souhrn toho, co přišlo.

**Jedna věc, kterou musíš vědět:** model sumarizuje **získané položky a nic jiného** a
deterministická kontrola potvrzuje, že souhrn cituje aspoň jednu z předaných URL. Atribuci
po jednotlivých claimech to neověří, ale odchytí souhrn napsaný z vlastních znalostí modelu
— tedy to selhání, na kterém záleží.

Query se staví čistou prací s řetězci: brokerské suffixy se strhnou (`EURUSD.pro` →
`EURUSD`), protože news search kazí, a šestimístné symboly se rozdělí na pár.

**Při selhání:** `{"summary": "", "items": []}` — přítomné ale prázdné. Stejné pravidlo jako
u sentinelu grafu, a důvod, proč mrtvé Tavily pořád vyrobí dokument s poznámkou o mezeře.

### `src/agents/writer.py` — 209 řádků kódu

Dokument. Model píše prózu a labely nadpisů v cílovém jazyce, `docbuilder` rozhoduje
o struktuře.

**Jedna věc, kterou musíš vědět:** `complete_sections` doplní cokoli, co model vynechal, a
**nahlásí, která polí musela doplnit** — dokument se tedy odešle, ale nefunkční model
zůstane viditelný. Živý běh se vrátil bez `sources_heading` a v `docbuilderu` z toho byl
`KeyError`; router writer zkusil znovu a druhý pokus prošel, takže run přežil — za cenu
jednoho kroku a osmi sekund.

### `src/main.py` — 162 řádků kódu

Volající. Není součástí grafu.

**Jedna věc, kterou musíš vědět:** vlastní přesně ty tři věci, které §6 přiděluje
volajícímu — `thread_id`, checkpointer a smyčku vyjasnění. `interrupt()` vrací řízení tomu,
kdo graf zavolal, takže odpovídat je jeho práce, a je to `while` cyklus, protože graf se
může zeptat dvakrát.

`--trace` rekonstruuje trace po krocích diffem checkpointů, protože langgraph 1.2.11
v metadatech checkpointu per-node writes nedrží. Progress jde na stderr, aby `--json`
zůstal pipeovatelný.

---

## 4. Verze na jeden odstavec

Kurzovní příklad učí routování: LLM vybere dalšího workera, router ho poslechne a jakékoli
selhání ukončí běh. Tenhle projekt startuje ze stejného skeletu a pak odpovídá na čtyři
další otázky, na kterých spec trvá — *co když se LLM mýlí*, *co když je závislost mimo*,
*co když je request nejasný* a *co musí skončit na disku*. Router přepisující LLM,
přítomné-ale-prázdné artefakty pro selhání, interrupt a dva čisté builder moduly jsou každý
jednou z těch odpovědí. Těch 1 500 řádků navíc není architektura navíc; jsou to ty čtyři
otázky, odpovězené.
