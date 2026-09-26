# Exercise — Senior AI/ML Engineer II

**Three hours.** You will not finish all of this, and that is the point. We are as
interested in how you sequenced the work as in what you completed.

## What you are building

An agent that, given a ticket ID, produces:

1. a **diagnosis** of the underlying issue,
2. a **drafted reply** to the requester,
3. an **escalation**, where one is warranted.

## What we ask you to deliver

- A working agent that does not invent answers the data does not support, with tests where
  they matter, surviving an upstream read failure.
- **An eval script** that scores it against the golden dataset, and **structured traces**
  for each step.
- **The escalation write gated on a server-side confirmation** — not on an instruction in
  the prompt.
- **No field surfaced that the requesting user is not entitled to see.** Narrow the
  system's own authorization; do not replace it.
- **Cost per completed task, instrumented and reported.**
- **A short ADR on the identity approach you chose**, including what you rejected and why.
- A **README**: what you built, what you traded away, and what you would delete.

A strong core with a clear account of what comes next beats a rushed attempt at
everything.

## Ground rules

- This repo is yours for the next three hours. Work in it or alongside it, as you prefer.
- You may read the fixture freely. Do not edit anything under `data/`.
- Your interviewer will tell you which variant to run.

## What you have

- The mock API. Start it with `python3 -m server`; endpoints are in `docs/API.md`.
- `data/<variant>/golden.json` — the golden dataset. How you score against it is your call.
- A model to call. Your interviewer will confirm your credentials before the timer starts.
- Any language you like. Python and Go are what our teams use, but move fast in whatever
  you know best.

## Notes

- Treat the data and the upstream systems the way you would in production. They are not
  guaranteed to be well-behaved.
- One endpoint writes while the rest read. That distinction is worth your attention.
- You may use AI coding tools. We expect generated code reviewed as rigorously as a
  colleague's, and the next round opens by asking what you rejected.

## Afterwards

**Review, 45 minutes.** Structure, decisions, sequencing, and a real look at your eval
script and identity choice. Expect to be challenged on the ADR.

**System design, 60 minutes.** The shared platform layer these agents run on: architecture,
state, failure modes, and what breaks at scale.

**Technical leadership, 45 minutes.** Influence without authority.
