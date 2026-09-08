"""plan10_analysis -- the analysis console's data sources and scheme validator.

Every module here is pure stdlib. Nothing in this package types a channel
name, a dead-channel list, a corpus, or an issue number: each is derived from
the pipeline's own files at build time (see each module's docstring for the
file it reads) so the console has no copy of its own to drift from.
"""
