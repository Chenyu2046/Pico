def load_record(path, reader):
    """Load one record and tag it with its source path."""
    record = dict(reader(path))
    record["source"] = path
    return record
