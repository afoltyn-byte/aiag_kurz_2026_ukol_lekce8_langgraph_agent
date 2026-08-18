# CLAUDE.md

LangGraph agent: broker-side instrument analysis. Read this before writing code.

## Source of truth

| File | Defines |
|---|---|
| `docs/graph.md` | topology — nodes, edges, which edges are conditional and with what labels |
| `docs/graph-spec.md` | contracts — state schema and reducers, per-node read/write keys, router logic, config, definition of done |

`docs/walkthrough.md` (in Czech) is orientation only — what each module is for and where its
size comes from. It is **not** a source of truth: where it disagrees with the two files
above, they win and it is stale.

If a task needs a node, edge, or state key that is not in those two files: **stop
and ask.** Do not invent topology. Do not invent state keys — a new key with no
reducer is a silent data-loss bug the moment anything runs in parallel.

If the two files contradict each other, ask which is right. Do not pick.

## Scope discipline — the rule that matters most

**One node per turn.** Write the node, write its unit test, run it, report the
result. Then stop and wait.

Do not generate the whole graph in one pass. A plausible-looking graph that has
never executed is worse than no graph, and debugging five nodes at once is worse
than writing them one at a time. Suggested order:

1. `state.py` + `config.py`
2. `graph.py` assembly with `NotImplementedError` stubs → topology test green
3. `route_from_supervisor` + its 12 router tests
4. `charting.py` (pure, no LLM, easiest to verify)
5. `trader` → `analytics` → `writer` → `supervisor` → `clarify`

## Non-negotiables

1. **Nodes are `(state) -> dict`** returning a *partial* update. Never the full
   state. Never mutate the input dict.
2. **Agents never raise.** A failure appends to `errors`, logs to `agent_log`, and
   returns control to the supervisor. An escaping exception kills the run and
   loses the trace.
3. **`route_from_supervisor` is a pure function of state.** No I/O, no LLM, no
   `datetime.now()`, no randomness. An impure router makes the graph unreplayable
   from a checkpoint, which silently breaks resume after `clarify`.
4. **Router guards are not suggestions.** The order of the five rules in spec §5
   is load-bearing; the step check must come first and the scope check must
   override the LLM's proposal.
5. **`graph.py` is wiring only.** No prompts, no I/O, no branching. If you want an
   `if` there, it belongs in the router.
6. **Levels and document assembly stay LLM-free.** `charting.py` and
   `docbuilder.py` are pure functions — same input, same output, unit-testable.
7. **No secrets in code.** `OPENAI_API_KEY`, `TAVILY_API_KEY`, `MT5_MCP_URL`,
   `POSTGRES_URI` from environment only. They live in `.env` as 1Password `op://`
   references — never plaintext — and the process is started with
   `op run --env-file=.env -- <cmd>`. `.env` is gitignored; `.env.example` is the
   template. No code loads the file.
8. **`chart` and `document` hold paths, not bytes.** Images in state bloat every
   checkpoint row.

## Workflow for any change

1. Update `docs/graph.md` and `docs/graph-spec.md` **first**.
2. Implement.
3. `pytest tests/test_graph_topology.py` — compares the diagram against the
   compiled graph. Must pass before anything else counts as done.
4. Full suite + `mypy src/`.
5. Diagram and code change in the **same commit**.

**Never edit `tests/test_graph_topology.py` to make an implementation pass.** If it
fails, either the diagram or the code is wrong — fix that. Note it verifies
conditional-edge labels via `build_graph().branches[node][router].ends`, not via
`draw_mermaid()`, because the drawing is lossy: it omits a label when the mapping
key equals the target node name, and collapses branches that share a target.

## Versions

Pinned in `requirements.txt`. Verified against these exact versions:

```
langgraph==1.2.11
langgraph-checkpoint==4.2.0
langchain-core==1.5.6
langchain-openai==1.5.1   (openai==3.2.0)
python-docx==1.2.0
pytest==9.1.1
```

LLM provider is **OpenAI**, `langchain_openai.ChatOpenAI`, default model
`gpt-5.4-nano` — see `DEFAULT_MODEL` in `config.py`. `config.chat_model(agent,
config)` is the only place a client is constructed; nodes never instantiate one
directly, so the provider is swappable in one edit.

API surface confirmed for `langgraph==1.2.11` — use these forms, not older ones:

- `StateGraph(AgentState, input_schema=..., output_schema=...)` — keyword args are
  `input_schema` / `output_schema`, **not** `input` / `output`
- `add_conditional_edges(source, path, path_map)` — always pass an explicit dict
  `path_map`, never a bare list
- `from langgraph.types import interrupt, Command`
- resume with `graph.invoke(Command(resume={...}), config=...)`

Do not mix idioms across versions. If unsure what the pinned version exposes,
check the installed package rather than recalling it:

```bash
python -c "import langgraph.graph as g, inspect; print(inspect.signature(g.StateGraph.add_conditional_edges))"
```

## Document output

`python-docx`, not docx-js — no Node dependency inside a Python node. The document
is written in `state["language"]`, detected from the user's request. Czech
diacritics round-trip through python-docx with no font configuration.

Do **not** switch to a PDF library. ReportLab's built-in fonts silently render
`ě ř ď` as black boxes with no exception raised. If PDF is ever required, build
the docx and convert with LibreOffice.

## Commands

```bash
# one whole run; --answer makes it unattended, otherwise it asks on stdin
python -m src.main "co je nového na eurodolaru a jak vypadá graf"
python -m src.main "analyzuj mi něco" --answer "EURUSD, jen zprávy"
python -m src.main "..." --postgres --thread-id req-4711   # resumable across restarts

pytest -q                                       # full suite (no network)
pytest tests/test_graph_topology.py -q          # drift guard only
mypy src/
op run --env-file=.env -- python -m src.agents.trader --instrument EURUSD --timeframe H1
python -c "from src.agent.graph import compile_graph; print(compile_graph().get_graph().draw_mermaid())"
```
