# Deploying on free tiers

SCRAG's public demonstration is designed to run at zero infrastructure cost,
which means designing around the limits of free hosting plans instead of
assuming unlimited resources.

The API runs on a free web service with 512 MB of memory and a tenth of a CPU
core. Such instances go to sleep after about 15 minutes without traffic and
take roughly a minute to wake up, so the web interface shows a waking-up
message and retries instead of failing. Their local disk is wiped on every
restart, so nothing authoritative may be stored there. For that reason the
sample knowledge base is built into the container image at build time, along
with the embedding model, and the demo can answer questions about it even
when no database is available.

Free language model tiers impose limits on requests and tokens per minute and
per day. A single SCRAG query can make several model calls, so the demo
answers its curated sample questions from recorded results that are clearly
labelled as cached, applies a global daily budget to live queries, and shows a
clear message when the budget is spent. Running artificial keep-alive traffic
to dodge sleep or pause policies is deliberately avoided.

Free database plans may pause a project after a week of inactivity, so
features that depend on the database, such as uploads and conversation
history, degrade gracefully while the core question answering keeps working.
