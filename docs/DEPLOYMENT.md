# Deploying to Vercel

The app exposes one FastAPI application in `app.py` with two static web views.
Import the Git repository into Vercel, use the repository root, and configure
the server-side variables below before testing. There is no separate npm build.

## Session persistence

Sessions are serialized after every turn and rebuilt before the next one
(`src/session_codec.py`), and where they are kept is a backend choice
(`src/session_store.py`): a dict locally, Redis on Vercel. `tests/test_session_persistence.py`
proves a run rehydrated after *every single turn* scores identically to one that ran
straight through.

## 1. Set up

### Python version

`.python-version` pins **3.13** to match local development. Keep that file in the
deployment and verify the selected runtime in the build log.

### Entrypoint

This repo exports `app = FastAPI(...)` from `app.py`, a supported entrypoint in
Vercel's [FastAPI deployment guide](https://vercel.com/docs/frameworks/backend/fastapi).

### `vercel.json`

Already committed. It sets `maxDuration: 300` and excludes tests, docs and the eval
harness from the bundle — Python Functions have no tree-shaking, so everything
reachable at build time ships unless excluded.

Do **not** exclude `cases/`, `skills/` or `web/`: the engine reads case tables, the
agent loads skill playbooks via `get_skill`, and the UI is served from `web/`, all
from disk at request time.

## 2. Environment variables

Set these in Project → Settings → Environment Variables, or with the CLI:

```bash
vercel env add ANTHROPIC_API_KEY production
```

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | for LLM mode | Without it the app serves, but LLM start fails explicitly; select the baseline manually. |
| `ANTHROPIC_MODEL` | no | Defaults to `claude-sonnet-4-5`. |
| `UPSTASH_REDIS_REST_URL` | **yes on Vercel** | Injected by the Redis integration. |
| `UPSTASH_REDIS_REST_TOKEN` | **yes on Vercel** | Injected by the Redis integration. |
| `SESSION_TTL_SECONDS` | no | How long a parked investigation lives. Default 7200. |
| `APP_ACCESS_PASSWORD` | **yes if public** | Turns on the HTTP Basic gate. Unset = no gate. |
| `APP_ACCESS_USER` | no | Username for the gate. Default `detective`. |
| `MAX_LLM_SESSIONS_PER_HOUR` | **yes if public** | Native LLM session-start cap, not a billing cap. `0`/unset = unlimited. |
| `LLM_LIMIT_WINDOW_SECONDS` | no | Window for the cap. Default 3600, floor 60. |
| `AGENT_THINKING_BUDGET` | no | Default 2000. `0` disables extended thinking. |
| `AGENT_MAX_TOKENS` | no | Default 4000. Must exceed the thinking budget. |

The key is read from `os.environ` server-side and never reaches the browser:
`/api/config` returns only a boolean. There is no dotenv loader, so a committed `.env`
would do nothing even if you added one — and `.gitignore` already blocks it.

### Redis

Connect an Upstash Redis store and ensure the REST URL and token are available
in the deployment's environment. A Redis socket URL alone does not satisfy this app.

The store speaks Upstash's **REST** API over `httpx` (already a dependency), so there
is no socket client to keep alive across invocations and no native dependency to
build. Both naming conventions are accepted: `UPSTASH_REDIS_REST_*` and the
`KV_REST_API_*` pair that migrated projects still carry.

## 3. Verify the deployment

```bash
curl https://<your-app>.vercel.app/api/config
```

```json
{
  "has_api_key": true,
  "session_store": "redis",
  "serverless": true,
  "access_gate": true,
  "llm_sessions_per_window": 30
}
```

This endpoint stays reachable without credentials on purpose: it reports whether the
protections are **on**, never what they are, which is what makes a misconfigured
deployment diagnosable with one curl instead of discovered on the invoice.

Three things to check:

- **`"session_store": "memory"` with `"serverless": true`** — Redis is not wired up and
  investigations will be dropped between turns. The app also logs an error at boot.
- **`"access_gate": false`** with a public URL — anyone can spend your key.
- **`"llm_sessions_per_window": 0`** — no spend cap.

Then confirm a session advances across requests. This smoke test explicitly uses
Legacy mode, so it makes no Anthropic calls. It requires `curl` and `jq`. Replace
the example origin; `-u detective` prompts for your configured access password
(omit it if the app gate is disabled, or change the username if configured):

```bash
APP_URL=https://your-app.vercel.app
SESSION_ID=$(curl --fail-with-body -sS -u detective -X POST "$APP_URL/api/sessions" \
  -H 'Content-Type: application/json' \
  -d '{"case_id":"case-001","mode":"rule_based"}' | jq -r .session_id)
curl --fail-with-body -sS -u detective -X POST "$APP_URL/api/sessions/$SESSION_ID/step" | jq .state.iteration
curl --fail-with-body -sS -u detective -X POST "$APP_URL/api/sessions/$SESSION_ID/step" | jq .state.iteration
```

Expect iterations `1`, then `2`. A missing/expired session or a store failure needs
investigation before running paid tests. Finally, use **AI detective** in the browser
to test an LLM case; that separate check incurs API usage.

## 4. Protect it before you share the URL

**The app's authentication and session-start rate cap are opt-in.** Without those
settings or external protection, anyone reaching a public API-key-backed instance
can spend its credits by starting LLM sessions.

Vercel also provides [Deployment Protection](https://vercel.com/docs/deployment-protection).
Check which methods and scopes your plan supports. Protect the production domain
you actually share, not only preview URLs, and confirm access from a signed-out
browser. Platform protection may also block the diagnostic endpoint.

### The built-in access gate

The app has its own access gate. Set `APP_ACCESS_PASSWORD` and every route except `/api/config` requires
HTTP Basic credentials:

```bash
vercel env add APP_ACCESS_PASSWORD production
vercel env add MAX_LLM_SESSIONS_PER_HOUR production   # e.g. 30
```

Basic auth rather than a login page because the browser prompts once and then attaches
the header to every `fetch` the UI makes — no cookie to secure, no session of our own.
The check lives in middleware (`src/access.py`, wired in `app.py`), so a route added
later is protected by default; forgetting to decorate a new endpoint is how these
gates leak. Credentials are compared with `secrets.compare_digest`.

Leave `APP_ACCESS_PASSWORD` unset locally and the gate does not exist, so local runs
and the offline suite are unaffected.

### The session-start limit is not a billing cap

**Access control does not cap cost.** An authenticated visitor — or you, with a stuck
browser tab — can still start investigations in a loop. Deployment Protection offers
nothing here.

A single investigation is already bounded: the case budget caps world actions and
`AGENT_MAX_MODEL_CALLS` caps model calls per turn. What has no ceiling is *how many*
investigations get started, so that is what `MAX_LLM_SESSIONS_PER_HOUR` counts. Over
the limit, `POST /api/sessions` with `mode: "llm"` returns 429 and points the caller at
the `rule_based` baseline, which calls no model and is never charged.

This counter only covers native LLM session creation. Resets of existing sessions,
the retained `llm_schema` API driver and `/api/describe_case` are not covered by it.
It does not count tokens, reviewer calls or audit calls. Restrict access to trusted
testers and use provider-side usage controls; this is not a complete public-service
spending safeguard.

The counter is an atomic `INCR` in the session store, not a process variable: on
serverless each instance would otherwise get its own full allowance. If the store is
unreachable the request is **refused** with a 503 — a rate limiter that fails open is
not a rate limiter.

## Known limits on Vercel

**`POST /api/sessions/{sid}/run`** executes a whole investigation in one request.
Both web views instead call `/step` serially. A step may perform several free
assessment/fact/review calls before returning a world action or conclusion, so
**either endpoint can hit the request-duration ceiling**. This repo configures
`maxDuration: 300`; verify the host's current limits before deploying. Prefer `/step`
and measure its slowest turns; it is not guaranteed to be one model call per request.

**Incompatible changes invalidate in-flight sessions.** `session_codec.SCHEMA_VERSION` guards the
format: if a deploy changes what a session looks like, an older blob is refused with a
409 rather than half-loaded into a corrupted investigation. Bump the version when you
change the payload shape. Converting a case from authored to model inference also
requires restarting its older sessions. Compatible sessions are not invalidated
merely because a deployment happened.

**A custom (generated) case body contains the hidden truth**, so it is stored in Redis
and keyed server-side; the browser only ever receives `public_briefing()`. This is
covered by `test_custom_case_body_is_stored_server_side_and_never_returned`.

**Custom cases expire with their TTL.** The listing self-heals: ids whose bodies have
expired are skipped rather than reported as broken cases.

**A scripted/custom investigator does not survive a rehydration.** The codec preserves
the native agent's conversation, and `rule_based`/`llm_schema` are stateless policies
over `InvestigationState`, but a driver holding private mutable attributes will restart
its script on every request. Derive the next move from state instead — see
`AsksPike` in `tests/test_api.py`.

**Concurrent turns on one session can lose an update.** `/step` reads the session,
advances it and writes it back; two requests racing on the same session id would both
read the same state and the later write would win. The UI disables its controls while
a turn is in flight, but another tab or API client can still race. There is no
distributed per-session lock; avoid concurrent requests to the same session.

**Basic auth in the browser is untested end to end.** The 401 challenge, the
`WWW-Authenticate` header and credentialed requests are all verified with curl, and
same-origin `fetch` sends browser-cached Basic credentials by default — but no real
browser was driven against a deployment. If the UI ever loads and then its API calls
401, that is where to look first.

**History is not written on Vercel.** `history.record_from_trace` writes to
`outputs/runs/`, and the serverless filesystem is read-only outside `/tmp`. This is
already fine: that call only happens in the CLI path (`run_cli`), never in a request
handler. If you want run history from the hosted app, send it to Redis or an external
store rather than the filesystem.

## Local development is unchanged

```bash
source .venv/bin/activate
python app.py serve          # http://127.0.0.1:8000, memory store
```

`vercel dev` also works if you want to exercise the platform locally. With no Redis
variables set, the store is the in-process dict and sessions remain process-local. Offline tests use scripted clients;
LLM investigations still make paid calls when a key is supplied.
