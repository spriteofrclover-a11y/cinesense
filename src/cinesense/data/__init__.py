"""Offline data acquisition and shaping.

Nothing in this subpackage may be imported by the serving process. It pulls in
pandas, duckdb and pyarrow, which is precisely what the deployed bundle must
avoid.
"""
