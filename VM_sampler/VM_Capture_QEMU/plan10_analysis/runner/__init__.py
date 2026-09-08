"""plan10_analysis.runner -- executes a validated scheme over a corpus.

chain.py    walk a zstd patch chain one pair at a time (stdlib)
differ.py   run the differ on a pair, parse its sparse CSV (numpy)
extract.py  the L1 store: per recording, the requested columns for every pair (numpy)
stages.py   one pure function per module kind (numpy; reuses the project's lens code)
executor.py orders a scheme, runs it per recording, writes status, output and sidecar
"""
