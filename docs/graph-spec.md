# Graph specification — `instrument_analysis`

> `docs/graph.md` defines **shape**. This file defines **contracts**.
> Together they are the complete implementation input. Nothing outside these two
> files may be invented; a gap means stop and ask.

---

## 1. Goal

A broker-side assistant. The user asks in free text for an analysis of an
instrument — market news, the chart, or both. The supervisor works out what was
asked for, runs only the agents that scope requires, and the writer assembles a
single Word document in the language the user wrote in.

**Architecture:** explicit `StateGraph` with conditional edges. Agents are **not**
tools and the supervisor is **not** a tool-calling agent. LangGraph's prebuilt
supervisor is not used. Agents are standalone Python modules.

**Non-goals:** no order placement, no position modification, no trading advice
presented as a recommendation. The document is analytical material for a human.

---

## 2. Layout

Each agent is a standalone module, importable as a node function *and* runnable on
its own for debugging:

```
src/agent/state.py           AgentState
src/agent/config.py          per-agent models, limits, paths
src/agent/graph.py           assembly only, no logic
src/agent/supervisor.py      supervisor node + route_from_supervisor
src/agent/clarify.py         interrupt node
src/agents/analytics.py      run(state) -> dict   +  __main__ CLI
src/agents/trader.py         run(state) -> dict   +  __main__ CLI
src/agents/writer.py         run(state) -> dict   +  __main__ CLI
src/agents/charting.py       level derivation + PNG rendering, pure, no LLM
src/agents/docbuilder.py     docx assembly, pure, no LLM
src/agents/mcp_client.py     shared MCP transport for trader + analytics, no LLM
src/main.py                  CLI entry point for one whole run, no graph logic
```

`main.py` is the **caller**, not part of the graph. It owns exactly the three things §6
says belong to the caller and not to the graph:

* the **`thread_id`** — one per request; it generates one when none is given.
* the **checkpointer** — `InMemorySaver` by default, `PostgresSaver` from `POSTGRES_URI`
  with `--postgres`. Note `PostgresSaver.from_conn_string` is a *context manager*, so the
  graph has to be used inside that block.
* the **clarification loop** — `interrupt()` hands control back to whoever invoked the
  graph, and answering is that caller's job. It loops rather than asking once, because the
  graph may ask up to `MAX_CLARIFICATIONS` times.

It contains no routing, no prompts and no state shaping. Exit codes: `0` a document was
produced, `1` the run finished without one, `2` the run needs an answer and none was
available (a question was asked with nothing on stdin to answer it).

`mcp_client.py` is a helper, not an agent: no `run()`. Neither the MT5 nor the Tavily
server has its tool contract defined here, so both agents discover their tool at
runtime, and the transport dance — connect, initialise, list, match arguments, call,
unwrap — is identical for both. The alias tables and payload shapes stay in the agents,
because those genuinely differ. Discovery is deterministic string matching; **no model
picks a tool or fills an argument.**

Every agent module exposes exactly one graph entry point:

```python
def run(state: AgentState) -> dict[str, Any]: ...
```

plus an `if __name__ == "__main__":` block that builds a minimal state from CLI
args and prints the returned partial update. Standalone means *separately
runnable*, not a separate process — nodes execute in-process so checkpointing and
state passing work.

---

## 3. State schema

| Key | Type | Reducer | Written by | Notes |
|---|---|---|---|---|
| `request` | `str` | last-write | *input, required* | raw user text, never mutated |
| `messages` | `list[AnyMessage]` | `add_messages` | `supervisor`, `clarify`, `writer` | LLM + clarification trace |
| `instrument` | `str \| None` | last-write | `supervisor` | broker symbol as it exists on MT5, uppercase |
| `timeframe` | `str` | last-write | `supervisor` | MT5 timeframe, default `DEFAULT_TIMEFRAME` |
| `language` | `str \| None` | last-write | `supervisor` | ISO 639-1 detected from `request`; drives the document and the clarifying question |
| `scope` | `Literal["news","chart","both"] \| None` | last-write | `supervisor` | what the user asked for |
| `next_agent` | `Literal["clarify","analytics","trader","writer","done"] \| None` | last-write | `supervisor` | the LLM's *proposal*, not the final decision |
| `step_count` | `int` | last-write | `supervisor` | +1 per supervisor visit |
| `clarify_count` | `int` | last-write | `clarify` | +1 per question asked |
| `analytics_result` | `dict \| None` | last-write | `analytics` | `{"summary": str, "items": [{"title","url","published","snippet"}]}` |
| `chart` | `dict \| None` | last-write | `trader` | `{"path", "symbol", "timeframe", "levels": [...], "generated_at", "commentary"}`; on failure the same keys plus `error`, with `path: None` |
| `document` | `dict \| None` | last-write | `writer` | `{"path", "format": "docx", "title", "scope", "language"}` |
| `errors` | `list[str]` | `operator.add` | any | append-only, never cleared |
| `agent_log` | `list[dict]` | `operator.add` | any | `{"agent","status","duration_s"}` — for debugging loops |

**Input keys:** `request` (required).
**Output keys:** `document`, `chart`, `analytics_result`, `scope`, `errors`, `agent_log`.

`chart` holds a **path, not image bytes**. Images in state bloat every checkpoint
row and are unreadable in a trace.

---

## 4. Node contracts

Every node is `(state) -> dict` returning a **partial** update. Never returns the
whole state, never mutates the input dict. **No agent raises** — a failure appends
to `errors`, logs to `agent_log`, and returns control to the supervisor. An
exception escaping a node kills the run and loses the trace.

### `supervisor` — LLM, conditional out

| | |
|---|---|
| Reads | `request`, `messages`, `instrument`, `scope`, `language`, `analytics_result`, `chart`, `document`, `errors`, `step_count` |
| Writes | `instrument`, `timeframe`, `language`, `scope`, `next_agent`, `step_count`, `messages` |
| Model | `AGENT_MODELS["supervisor"]`, temp 0, structured output |
| On exception | `next_agent="done"`, append to `errors` |

Structured output:

```json
{"instrument": "EURUSD|null", "timeframe": "H1", "language": "cs",
 "scope": "news|chart|both|null", "next_agent": "...", "reason": "..."}
```

Three jobs: resolve `instrument` and `timeframe` from the request, classify
`scope`, detect `language`. `instrument` or `scope` set to `null` when the request
is genuinely ambiguous — **guessing is worse than asking.** On later visits it
re-reads `messages` so a clarification answer is picked up.

`language` is detected once from the original `request` and not revised.

**Everything the model returns is validated before it reaches state.** A structured-output
schema constrains a model, it does not bind it:

| Field | Rejected value becomes |
|---|---|
| `instrument` | `None` — non-string, empty, whitespace |
| `scope` | `None` if not one of the three; the router then routes to `clarify` |
| `next_agent` | `None` if not one of the five; the router falls back to `missing[0]` |
| `timeframe` | `DEFAULT_TIMEFRAME` if not in `MT5_TIMEFRAMES` (case-insensitive) |

Coercing a bad `scope` to `None` rather than to a default is the point: it routes to a
question instead of to a confidently wrong report.

**Resolution is monotonic.** Once `instrument` and `scope` are resolved, a later visit may
*change* them to another valid value but can never un-resolve them back to `None`: the
validated answer falls back to what is already in state rather than overwriting it.

This is not defensive decoration — it was a live bug. Visit 1 read
"co je nového na eurodolaru a jak vypadá graf" correctly as `EURUSD` / `both` and routed to
`analytics`; visit 2 came back with both fields null; the router, seeing an unresolved
state, sent the user to `clarify` **mid-run**, asking a question the graph already had the
answer to — and burning one of the two clarifications on it. The prompt asks the model to
keep resolved values; this is the guarantee, because a prompt is a request and code is not.
`language` was already sticky for the same reason; these two were not, which was the
inconsistency.

`instrument` is upper-cased but **keeps its broker suffix** — `EURUSD.pro` is the symbol
as MT5 knows it, and stripping it here would break the trader. Only `analytics` strips it,
and only to build its news query.

The decision schema is a hand-written JSON Schema rather than a `TypedDict`, because
`scope` has to reach the model as one nullable enum (`"news" | "chart" | "both" | null`)
and an annotation cannot express that.

**Two `errors` entries belong to this node**, because the router is pure and cannot write
to state:

* the §5 rule 3 explanation — unresolved after `MAX_CLARIFICATIONS`, naming what is still
  missing. Without it the run would end silently and nobody could tell why.
* the step-limit note, when this visit is the one that reaches `MAX_STEPS`.

**`step_count` is incremented before the model is called**, so it advances even when the
call fails. This is the invariant behind the §7 item "a supervisor that never says `done`
exits via `step_limit`, **not** `GraphRecursionError`": a supervisor whose model is
unreachable still marches towards the limit instead of looping forever.

### `clarify` — human interrupt

| | |
|---|---|
| Reads | `request`, `instrument`, `scope`, `language`, `messages` |
| Writes | `clarify_count`, `messages` |
| Mechanism | `interrupt({...})` |

Asks exactly one question, in `language`, about whichever of `instrument` /
`scope` is unresolved. Payload carries the question plus what was already
understood, so the caller can render it. The answer is appended to `messages`; the
supervisor re-parses on the next visit. Increments `clarify_count`.

Resume:

```python
graph.invoke(Command(resume={"answer": "EURUSD, jen zprávy"}), config=...)
```

This node does **not** itself decide anything. It asks, records, hands back — it
writes `clarify_count` and `messages` and nothing else.

**No model, by design.** This node has no `Model` row, and it stays that way: it is the
node that runs when something has already gone wrong, and putting a model on the critical
path of the interrupt would mean a model failure produces no question at all — with no
`errors` key in this node's Writes contract to record it.

The question therefore comes from `QUESTION_TEMPLATES`, keyed by language and by which
field is missing, covering Czech and English with an English fallback. **Supporting another
language means adding a row to that table**, which is a deliberate, deterministic edit
rather than a model call.

Both the question and the answer are appended to `messages`, so the supervisor's next
visit sees the exchange rather than a bare answer.

**`interrupt()` suspends by raising.** `GraphInterrupt` subclasses `GraphBubbleUp`, and it
must propagate: a blanket `except Exception` here would swallow the suspend and turn
human-in-the-loop into a silent no-op that carries on with a clarification nobody was ever
asked. **This node is the one place the "no agent raises" rule does not apply**, and it
has no `try/except` at all.

The node **re-executes from the top on resume**, so everything before `interrupt()` runs
twice. `build_payload` is therefore pure and `clarify_count` is derived from state rather
than incremented in place.

`Command(resume={"answer": ...})` is the documented shape; a bare string is accepted too,
since it is the obvious mistake to make and rejecting it would waste a clarification. An
empty answer still consumes one attempt — the supervisor finds nothing new in `messages`
and rule 3 ends the run rather than asking forever.

### `analytics` — read-only IO

| | |
|---|---|
| Reads | `instrument`, `language` |
| Writes | `analytics_result`, `errors`, `agent_log` |
| Tools | Tavily MCP search **incl. research mode** |
| Model | `AGENT_MODELS["analytics"]` — summarises retrieved items |
| Timeout | `TAVILY_TIMEOUT_S` |
| On failure | `analytics_result = {"summary": "", "items": []}`, append to `errors` |

Every claim in `summary` carries a source URL from `items`. No synthesis beyond
what was retrieved. Retries are free — nothing is written outside state.

**Query.** Built from the symbol by pure string work, never by the model: the broker
suffix is stripped (`EURUSD.pro` → `EURUSD`, `XAUUSD_m` → `XAUUSD`) because it is
meaningless to a news search, and a six-letter symbol is split into a currency pair
(`EURUSD` → `EUR/USD`). Anything else is used as its own name.

**Tool.** Discovered like the MT5 one, via `mcp_client`. `TAVILY_TOOL` pins an exact
name. `TAVILY_SEARCH_DEPTH` and `TAVILY_TOPIC` are passed when the tool exposes them,
skipped when it does not, and **skipped when the tool's `enum` forbids the value** — the
hosted server declares `topic` as `Literal['general']`, which is why the default is
`"general"` and not `"news"`. The query itself carries the news framing.

**Research mode is available, not the default.** `TAVILY_RESEARCH` flips discovery to
prefer a research tool. It defaults to off because a deep-research call routinely runs
for minutes while `TAVILY_TIMEOUT_S` is 30 — defaulting it on would time out most runs.
Turn it on together with a raised timeout.

**Auth.** `TAVILY_API_KEY` travels in an `Authorization: Bearer` header, not a query
parameter: a key in a URL ends up in logs and proxy history. `TAVILY_MCP_URL` overrides
the endpoint for a self-hosted server, or for a deployment that wants the key inline.

**Items** are deduplicated by URL and capped at `TAVILY_MAX_ITEMS`. A result with no URL
is dropped — the summary must cite a source for every claim, so an uncitable item is
worse than no item.

**Sourcing guard.** After the summary comes back, the node checks that at least one of
the given URLs appears in it. This cannot verify per-claim attribution, but it does catch
a summary written from the model's own knowledge instead of the retrieved set. A summary
that fails the check is **kept** and an `errors` entry is appended — flagging beats
discarding.

**A failed summary keeps the items.** Like the trader's commentary, the model call has
its own `try/except`: `summary` is `""`, an `errors` entry is appended, and `items`
survive so the writer can still list the sources.

**Empty vs absent.** Three outcomes are distinguishable, which matters because the router
reads a present `analytics_result` as "produced":

| Outcome | `analytics_result` | `agent_log.status` |
|---|---|---|
| items retrieved and summarised | populated | `ok` |
| search ran, found nothing | `{"summary": "", "items": []}` + `errors` | `empty` |
| search failed | `{"summary": "", "items": []}` + `errors` | `error` |

In every case the artifact is **present but empty, never `None`** — that is what stops a
dead Tavily from looping the agent until the step limit, and what makes the §7 item
"failing Tavily call: run still completes with a document noting the gap" true.

### `trader` — read IO + file write

| | |
|---|---|
| Reads | `instrument`, `timeframe` |
| Writes | `chart`, `errors`, `agent_log` |
| Tools | MT5 MCP server, **streamable HTTP on localhost** (`MT5_MCP_URL`); `charting.py` |
| Model | `AGENT_MODELS["trader"]` — interprets zones; level maths stays in `charting.py` |
| Timeout | `MT5_TIMEOUT_S` |
| On failure | failure sentinel (see below), append to `errors` |

Sequence: fetch OHLC via MT5 MCP → derive support/resistance zones and candidate
entry levels for the current session → render annotated PNG → have the model describe
the zones → return the path.

**The MT5 MCP server's tool contract is not part of this spec**, so the trader adapts
at runtime instead of hardcoding a guess:

* **Tool.** `MT5_OHLC_TOOL` pins an exact name when it is known. Otherwise the trader
  calls `list_tools()` and takes the first tool whose name contains a fragment from
  `MT5_OHLC_TOOL_CANDIDATES`, in that order, breaking ties on tool name so the choice
  is reproducible.
* **Arguments.** Read off the chosen tool's `input_schema`. Both request shapes are
  offered — a **bar count** and a **datetime window** — because MT5 MCP wrappers disagree
  about which they take and some require the window; whichever the schema declares gets
  filled. The window is derived from `MT5_BAR_COUNT` × `MT5_TIMEFRAME_SECONDS`, padded by
  `MT5_RANGE_PADDING`: wall-clock span is not bar count, because markets close. 200 H1 bars
  cover eight-plus calendar days containing two weekends of nothing, and without the padding
  the frame can arrive too short for ATR(14) — which is a `ValueError`, not a smaller chart.
  Timestamps are encoded per the property's declared `type`: epoch seconds for
  `integer`/`number`, ISO 8601 otherwise. A required parameter that cannot be matched is an
  error, not a guess.
* **The schema's `enum` is honoured.** An optional value the server declared invalid is
  dropped rather than sent; a required one raises locally, naming what was allowed. Sending
  a forbidden value produces a validation error from inside the server, where it reads as a
  mystery rather than as our mistake — this was found the hard way on the first live run,
  where the hosted Tavily MCP declares `topic` as `Literal['general']`.
* **Result.** `structured_content` when the server provides it, otherwise the text blocks
  parsed as JSON. Bars are normalised from the usual key spellings (`t/o/h/l/c`,
  `time`/`timestamp`/`datetime`) and may sit under a metadata-bearing wrapper —
  `get_chart_history` returns them under `history` next to `data_available_from`.
* **Timestamps.** ISO first, then MetaTrader's own dotted rendering
  (`2026.08.06 06:00:00`), which `datetime.fromisoformat` rejects outright. An epoch number
  passes through untouched.

  A **naive** timestamp is broker *server* time, not UTC: MT5 has no concept of the
  difference and simply prints the server clock. It is stamped with
  `MT5_SERVER_UTC_OFFSET_H` so the epoch is truthful, and `derive_levels` shifts back by the
  same offset to group sessions. The two cancel, so session boundaries land on server
  midnight whatever the offset is — what the offset fixes is the absolute times, which are
  otherwise two or three hours off. A timestamp that carries its own offset is believed and
  not second-guessed.

  > Open: a single number cannot follow an EET/EEST server across a DST switch. A
  > `zoneinfo` key (`Europe/Athens`) would; `tzdata` is already installed.

None of this involves the model — discovery is deterministic string matching.

`chart["commentary"]` holds the model's reading of the zones: a short analytical note,
**in English**, since the trader does not read `language` (the writer does, and rewrites
it into the document's language). It is the model's only contribution to this node:

* Levels come from `charting.derive_levels` and are never touched by the LLM. The
  commentary is prose *about* already-computed numbers — the model is not in the data
  path, so it cannot move a level.
* **A failed commentary must not lose the chart.** The LLM call has its own
  `try/except`: on failure `commentary` is `""`, an `errors` entry is appended, and the
  chart is still returned. Losing a rendered PNG because prose generation failed would
  be absurd.
* Per §1 it is analytical material, never a recommendation.

`digits` for `derive_levels` comes from `charting.infer_digits(bars)` — the precision
actually observed in the frame, so zone prices in the document table do not carry float
noise like `1.0851999999999999`. It reports observed precision, not the instrument's
declared `digits`; a frame whose prices all happen to end in zero rounds shorter, which
is cosmetic only.

**Writes a file.** `{CHART_DIR}/{symbol}_{timeframe}_{YYYYMMDD-HHMMSS}.png`, so a
retry produces a new file rather than corrupting one. Files older than
`CHART_RETENTION_DAYS` are purged on node entry.

**On failure the artifact is present but empty**, mirroring `analytics`:

```python
{"path": None, "symbol": ..., "timeframe": ..., "levels": [],
 "generated_at": ..., "commentary": "", "error": "<repr of the failure>"}
```

The router reads a present artifact as produced, so this is what keeps a dead MT5 from
being routed back here on every visit until `step_limit` — up to eleven attempts at
`MT5_TIMEOUT_S` each, ending with no document at all. With the sentinel the run reaches
the writer and produces a report that says the chart could not be generated. `symbol` and
`timeframe` are kept so the report can name what failed.

> Trade-off, accepted deliberately: a *transient* MT5 hiccup now degrades on the first
> failure instead of being retried by the loop. An internal retry inside the node would be
> a separate change.

The one case that still writes `chart = None` is a **missing `instrument`** — a contract
violation rather than a fetch failure, with nothing to write a sentinel about. Rule 3 sends
that state to `clarify`, never back here.

Level derivation in `charting.py` is a **pure function** of the OHLC frame — same
input, same levels. No LLM inside it, so results are reproducible and
unit-testable.

`charting.py` exposes two functions and no `run()` — it is a helper, not an agent:

```python
derive_levels(bars, *, pivot_lookback, atr_period, zone_atr_mult,
              max_zones_per_side, session_offset_hours, digits) -> list[Level]
render_chart(bars, levels, symbol, timeframe, path) -> str
infer_digits(bars, cap=8) -> int
```

`derive_levels` is pure and imports no plotting library. `render_chart` takes the
destination path as an argument and only writes bytes there — **no clock inside
`charting.py`**, so neither function can produce a different result on a second
call. The timestamped filename and the retention purge live in the `trader` node,
which is where the clock belongs.

`bars` is a `Sequence[Bar]`, plain dicts, no pandas:

```python
Bar   = {"time": int,  # epoch seconds, UTC
         "open": float, "high": float, "low": float, "close": float}
Level = {"kind": "support" | "resistance" | "session_high" | "session_low",
         "price": float,        # representative price of the zone
         "low": float,          # zone envelope, == price for session extremes
         "high": float,
         "touches": int,        # how many pivots / bars formed it
         "strength": float,     # 0.0–1.0, saturates at ZONE_STRENGTH_SATURATION
         "entry": "long" | "short" | None}
```

`levels` is ordered by `price`, descending — resistances above, supports below, the
way it reads on a chart and in the document table.

**Derivation, in order:**

1. **ATR** over `atr_period` bars, simple mean of true ranges (not Wilder's
   smoothing). This is the unit of "close enough" for everything below, so a zone
   tolerance scales with the instrument: EURUSD, XAUUSD and BTCUSD differ by orders
   of magnitude and a fixed pip tolerance would be meaningless across them.
2. **Fractal pivots.** Bar `i` is a swing high when its `high` is `>=` every high in
   the `pivot_lookback` bars to the left and `>` every high to the right; mirrored
   for swing lows. The asymmetry makes a flat top yield **exactly one** pivot, the
   rightmost bar of the plateau — the most recent test of that price. Strict
   comparisons on both sides (the textbook Williams fractal) would yield *zero*
   there and silently miss a double top printed at identical highs.
3. **Clustering.** Pivot prices — highs and lows together, because a level is a
   price regardless of which way it was rejected — sorted ascending and grouped
   greedily while the cluster spread stays within `zone_atr_mult * ATR`. Each
   cluster becomes a zone: envelope from its members, `price` = their mean,
   `touches` = member count.
4. **Classification** against the last `close`: zone `price` above it is
   `resistance`, below it is `support`. Comparing the representative price, not the
   envelope, keeps this total — there is no third category to invent for a zone the
   price sits inside.
5. **Selection.** The `max_zones_per_side` nearest zones each side of the last close.
6. **Session extremes.** The session is the set of bars sharing the last bar's date,
   where the date is computed from the UTC epoch shifted by `session_offset_hours` —
   pass the broker's server offset (MT5 is typically UTC+2/+3), because an FX day
   does not break at midnight UTC. Emitted as `session_high` / `session_low`, with
   `touches` counting how many session bars came within tolerance of the extreme.
   **Derived from the frame, never from `datetime.now()`** — a wall clock here would
   make the same frame yield different levels on a re-run.
7. **Entry candidates.** The nearest support zone below the close is marked
   `entry: "long"`, the nearest resistance above it `entry: "short"`. Marks go on
   zones only, never on session extremes, so a session high that coincides with a
   resistance zone cannot be marked twice. Per §1 this is a *level*, not a
   recommendation — the writer must not present it as advice.

`digits` rounds zone prices to the instrument's precision when the trader passes it
from MT5 symbol info; `None` leaves full float precision.

Invalid input — no bars, fewer bars than the pivot window or the ATR period needs,
a non-positive lookback — **raises `ValueError`**. `charting.py` is not an agent, so
it may raise; the `trader` node catches it and turns it into an `errors` entry. An
empty level list would be indistinguishable from "this frame genuinely has no
levels".

### `writer` — LLM + file write

| | |
|---|---|
| Reads | `analytics_result`, `chart`, `instrument`, `timeframe`, `scope`, `language` |
| Writes | `document`, `errors`, `agent_log`, `messages` |
| Model | `AGENT_MODELS["writer"]` |
| Builder | `docbuilder.py` (python-docx), no LLM |
| On failure | `document = None`, append to `errors` |

The LLM produces section prose **in `language`**; `docbuilder.py` turns it into the
file. Sections present depend on `scope`:

| `scope` | Document contains |
|---|---|
| `news` | title, news analysis, source list |
| `chart` | title, chart image, level table |
| `both` | title, news analysis, chart image, level table, source list |

Output: `{REPORT_DIR}/{symbol}_{YYYYMMDD-HHMMSS}.docx`.

**Library: `python-docx`.** Not docx-js — a Node dependency inside a Python
LangGraph node is not worth it here. Verified: python-docx round-trips Czech
diacritics without any font configuration.

> An earlier draft of this spec specified PDF via ReportLab. That is dropped, and
> with it a real trap: ReportLab's built-in Helvetica silently renders `ě ř ď` as
> black boxes (`úroveň` → `úrove■`) with **no exception raised**, so it would only
> surface in a finished report. Word handles Unicode natively. If a PDF is ever
> needed, generate the docx and convert it with LibreOffice rather than laying out
> a PDF directly.

`docbuilder.py` specifics: `add_picture(path, width=Inches(6.0))` for the chart;
levels as a real table, not preformatted text; headings via `add_heading` so a TOC
works later.

**Division of labour.** The model produces prose *and every heading label*, in
`language`; `docbuilder.build_document` decides structure. Which sections appear is
decided by `scope` alone, so the model cannot add or drop one by wording its output
differently.

Labels come from the model rather than a translation table in `docbuilder` because
`language` is whatever the request was written in — a fixed table would silently fall
back to English for anything not enumerated, and `docbuilder` has to stay LLM-free and
language-agnostic.

Structured output, every field required regardless of scope:

```python
DocumentSections = {
    "title": str,
    "news_heading": str,   "news_analysis": str,
    "chart_heading": str,  "chart_analysis": str,
    "levels_heading": str, "level_columns": list[str],   # exactly 5 labels
    "sources_heading": str,
}
```

Structured rather than free text: a missing heading would otherwise land in the file as
an empty Word heading instead of surfacing as an error.

**Every field is completed before it reaches `docbuilder`**, by `complete_sections`:

| Field | Missing or blank becomes |
|---|---|
| `title` | `"{instrument} {timeframe}"` |
| the four `*_heading` fields | an English fallback |
| `news_analysis`, `chart_analysis` | `""` — prose may legitimately be empty |
| `level_columns` | English defaults, unless the length is exactly right |

The fallbacks are English because `docbuilder` is language-agnostic and the writer cannot
translate without another model call. An English heading over correct content beats losing
the section.

`complete_sections` returns the keys it had to fill, and `run` records them in `errors` with
`agent_log.status = "partial"` — the document still ships, but a model that keeps omitting
fields is worth knowing about rather than papering over.

This was a live failure: a run came back without `sources_heading` and it reached
`docbuilder` as a `KeyError`. The router retried the writer and the second attempt
succeeded, so the run survived — at the cost of one step and eight seconds. `level_columns`
was already defended this way and the rest was not; **that inconsistency was the bug**, and
it is the same shape as the supervisor forgetting a resolved instrument. Document structure
must not depend on the model returning every field, in the same way it must not depend on it
getting a list length right.

`level["kind"]` values (`resistance`, `session_high`) are **not** translated. They are
technical identifiers, and translating an enum turns a machine-readable value into prose.
Same for the symbol and the timeframe.

**One rule throughout: a section with no content gets no heading.** A source list is
emitted only when there are items; the levels heading and table only when there are levels.
On a gap path the corresponding paragraph says so in prose, and a heading over nothing —
or a table holding only its header row — reads like a rendering bug.

**A chart-bearing scope needs the chart *artifact*, not necessarily an image.** `chart is
None` is a contract violation and raises. The trader's failure sentinel does not: the chart
heading and paragraph stay, `add_picture` is skipped, and the level table is omitted. The
image path is checked before `add_picture`, which would otherwise fail deep inside
`document.save()`.

So `scope: "both"` with a failed chart degrades to title, news analysis, chart heading and
paragraph, source list — a usable report rather than nothing.

The **prose** explaining a missing chart comes from the model, not from `docbuilder`: the
writer's prompt receives the chart status (including the `error`) and the paragraph has to
be written in `language`, which a fixed string here could not do.

**`messages`.** §3 listed `writer` among the writers of `messages` while the Writes row
omitted it. **Resolved in favour of §3** and the row above now says so. What gets appended
is a single pointer line (`title -> path`), **not the document prose**: `messages` is
checkpointed on every superstep and the document is already on disk — the same reasoning
that keeps image bytes out of `chart`. Nothing is appended on the failure path.

---

## 5. Edge contracts

**Static edges:**

| From | To |
|---|---|
| `__start__` | `supervisor` |
| `clarify` | `supervisor` |
| `analytics` | `supervisor` |
| `trader` | `supervisor` |
| `writer` | `supervisor` |

**Conditional edges** — `supervisor` is the only branching node:

| Router | Return value | Target |
|---|---|---|
| `route_from_supervisor` | `"clarify"` | `clarify` |
| | `"analytics"` | `analytics` |
| | `"trader"` | `trader` |
| | `"writer"` | `writer` |
| | `"done"` | `__end__` |
| | `"step_limit"` | `__end__` |

Router logic, **in this order**:

1. `step_count >= MAX_STEPS` → `"step_limit"`
2. `document is not None` → `"done"`
3. `instrument` falsy **or** `scope` not one of the three valid values:
   - `clarify_count < MAX_CLARIFICATIONS` → `"clarify"`
   - otherwise → `"done"` (asking a third time is harassment; the run ends with
     `errors` explaining why)
4. compute `missing` = required artifacts for `scope` not yet in state, where
   required is `news → (analytics,)`, `chart → (trader,)`,
   `both → (analytics, trader)`; if `missing` is non-empty → the LLM's proposal
   **if it is in `missing`**, otherwise `missing[0]`
5. `missing` empty → `"writer"`

Rules 3–5 are the guardrails. The supervisor keeps the decisions that need
judgement — what the user meant, and whether to fetch news or the chart first when
both are needed. It has **no** ability to run an agent outside the requested
scope, run the writer before its inputs exist, skip the writer, or finish without
a document.

`route_from_supervisor` is a **pure function of state** — no I/O, no LLM, no
`datetime.now()`, no randomness. An impure router makes the graph unreplayable
from a checkpoint, which silently breaks resume after `clarify`.

---

## 6. Configuration

`src/agent/config.py`, no inline literals elsewhere:

**Provider: OpenAI**, via `langchain_openai.ChatOpenAI`.

```python
DEFAULT_MODEL = "gpt-5.4-nano"
SUPERVISOR_MODEL = "gpt-5.4-mini"

AGENT_MODELS = {
    "supervisor": SUPERVISOR_MODEL,
    "analytics":  DEFAULT_MODEL,
    "trader":     DEFAULT_MODEL,
    "writer":     DEFAULT_MODEL,
}
AGENT_TEMPERATURE = {"supervisor": 0.0, "analytics": 0.0, "trader": 0.0, "writer": 0.3}

MAX_STEPS = 12
MAX_CLARIFICATIONS = 2
RECURSION_LIMIT = 40          # > MAX_STEPS * 2, so step_limit fires first
MT5_TIMEOUT_S = 20
TAVILY_TIMEOUT_S = 30
CHART_DIR = "outputs/charts"
REPORT_DIR = "outputs/reports"
CHART_RETENTION_DAYS = 7
DEFAULT_TIMEFRAME = "H1"
MT5_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1")

# charting.py — level derivation and rendering
PIVOT_LOOKBACK = 3            # bars each side of a fractal pivot
ATR_PERIOD = 14
ZONE_ATR_MULT = 0.5           # cluster spread tolerance, in ATRs
MAX_ZONES_PER_SIDE = 3
ZONE_STRENGTH_SATURATION = 4  # touches at which strength reaches 1.0
CHART_DPI = 130
CHART_FIGSIZE = (12.0, 6.5)

# trader.py — MT5 MCP access
MT5_BAR_COUNT = 200
MT5_SERVER_UTC_OFFSET_H = 0.0   # broker server offset; MT5 is usually UTC+2/+3
MT5_TIMEFRAME_SECONDS = {...}   # seconds per bar, for tools taking a datetime window
MT5_RANGE_PADDING = 3.0         # markets close; the window needs slack over the bar count
MT5_OHLC_TOOL = None            # pin an exact tool name, or None to discover
MT5_OHLC_TOOL_CANDIDATES = ("copy_rates", "get_rates", "rates", "ohlc",
                            "candle", "bars", "history")
MT5_AUTH_HEADER = "Authorization"   # only used when MT5_API_KEY is set
MT5_AUTH_SCHEME = "Bearer"

# analytics.py — Tavily MCP access
TAVILY_MCP_URL_DEFAULT = "https://mcp.tavily.com/mcp/"
TAVILY_MAX_ITEMS = 8
TAVILY_SEARCH_DEPTH = "advanced"
TAVILY_TOPIC = "general"       # what the hosted server's enum allows
TAVILY_RESEARCH = False          # research mode available, not default — see §4
TAVILY_TOOL = None               # pin an exact tool name, or None to discover
TAVILY_SEARCH_TOOL_CANDIDATES = ("tavily_search", "search", "web_search", "news")
TAVILY_RESEARCH_TOOL_CANDIDATES = ("tavily_research", "research", "deep", "extract")
```

`MT5_SERVER_UTC_OFFSET_H` is passed to `derive_levels` as `session_offset_hours`.
Left at `0.0` the session boundary sits at midnight UTC, which is the wrong place for
an FX day — set it per broker.

Per-agent model overridable per invocation without touching code:

```python
graph.invoke(
    {"request": "co je nového na eurodolaru a jak vypadá graf"},
    config={"configurable": {"thread_id": "...", "models": {"trader": "gpt-5.4"}}},
)
```

A node resolves its model as `configurable.models.get(name) or AGENT_MODELS[name]`.

**The supervisor runs a bigger model than the rest, and that is a measured finding.** On
`gpt-5.4-nano` it read "co je nového na eurodolaru a jak vypadá graf" — a sentence naming
both the instrument and the scope — correctly on some runs and returned nulls on others,
which sent the user to `clarify` for something already stated. Nothing was wrong with the
graph: a null scope is exactly what rule 3 is for, and asking beats guessing. The model just
could not read reliably enough.

The other three stayed on the default. Summarising retrieved items, describing
already-computed levels and filling a section schema are mechanical next to working out what
a free-text request asked for — that is the one job here that needs comprehension, so it is
the one that pays for a bigger model.

> **`AGENT_TEMPERATURE` is inert while the model is a `gpt-5*` reasoning model.**
> `langchain_openai` drops the parameter for any model whose id starts with `gpt-5`
> and is not `-chat`, unless `reasoning_effort="none"` — silently, with no warning
> and no exception; the constructed client simply reports `temperature=None`. So
> with `gpt-5.4-nano` the writer's `0.3` has no effect and neither do the three
> `0.0`s. The table is kept because it states the *intent* per agent and becomes
> live again the moment a `-chat` or non-`gpt-5` model is used. Regaining real
> temperature control on this family means passing `reasoning_effort="none"`, which
> trades the model's reasoning for it — a decision, not a default, so it is not
> wired in.

- **Checkpointer:** required, not optional — `clarify` cannot resume without one.
  `InMemorySaver` in tests, `PostgresSaver` in dev/prod; connection string from
  `POSTGRES_URI`, never hardcoded. `compile_graph()` defaults to `InMemorySaver`, which
  means **a resume only works within one process**; `main.py --postgres` is what makes a
  clarification survive a restart.
- **`thread_id`:** one per request, supplied by the caller. Not generated inside
  the graph.
- **Secrets:** `OPENAI_API_KEY`, `TAVILY_API_KEY`, `MT5_MCP_URL`, `MT5_API_KEY`,
  `POSTGRES_URI` from environment only — never literals, never defaults in code.
  `TAVILY_MCP_URL` is also read from the environment, though it is an endpoint rather than
  a secret.

  **`MT5_API_KEY` is optional.** The server this spec describes runs on localhost, where a
  key is usually pointless, so with it unset the trader sends **no auth header at all** and
  a plain local server behaves exactly as before. When it is set the header is
  `MT5_AUTH_HEADER: MT5_AUTH_SCHEME <key>`, defaulting to `Authorization: Bearer <key>`;
  both are config constants because MCP has no convention here, and a server wanting a raw
  key under its own header is served by `MT5_AUTH_HEADER="X-API-Key"` with
  `MT5_AUTH_SCHEME=""`.

  They live in a standalone **`.env`** at the repo root, holding 1Password
  **`op://` references rather than plaintext values**, and the process is launched
  through `op run`, which resolves them into the environment for that process only:

  ```bash
  op run --env-file=.env -- pytest -q
  op run --env-file=.env -- python -m src.agents.trader --instrument EURUSD --timeframe H1
  ```

  `.env` is gitignored; `.env.example` is the committed template. Nothing in the
  code loads the file — the values simply have to be in the environment by the time
  a node resolves them, so `op run` and a plain exported shell variable are equally
  valid. A node that finds a value missing appends to `errors` and hands back to the
  supervisor; it does not raise.

---

## 7. Definition of done

- [x] `pytest tests/test_graph_topology.py` passes — code matches `docs/graph.md`
- [x] Unit test per node: hand-built input state → asserted partial update
- [x] Unit test per router branch — 12 cases minimum, incl. `step_limit`, the
      clarify cap, and each scope's redirect when the LLM proposes out of scope
- [x] `charting.py` level derivation tested against a fixed OHLC fixture — same
      input, same levels
- [x] Each agent module runs standalone from its CLI
- [x] End-to-end per scope with MT5 and Tavily stubbed: `news`, `chart`, `both`
- [x] End-to-end: ambiguous request → `clarify` interrupt → resume → completes
- [x] End-to-end: two unusable answers → run ends via rule 3 with `errors`, not a
      third question
- [x] End-to-end: supervisor that never says `done` exits via `step_limit`,
      **not** `GraphRecursionError`
- [x] Failing Tavily call: run still completes with a document noting the gap
- [x] Failing MT5 call: same — the chart sentinel degrades the report, it does not loop
      the trader to `step_limit`
- [x] Both dependencies failing: one document, both gaps reported
- [ ] Generated docx opens in Word with correct diacritics and an embedded chart  *(only verifiable by opening it; a sample was produced)*
- [x] `mypy src/` clean