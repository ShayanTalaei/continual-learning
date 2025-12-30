"""
Explore AMD Synthesized Dataset Parquet

Debugging the `ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs` error
"""

import pyarrow.parquet as pq
import pyarrow as pa

PARQUET_PATH = "output/amd_synthesize/meta-llama/Llama-3.2-3B-Instruct_n65536-0/artifact/dataset.parquet"

print("=" * 60)
print("PARQUET FILE METADATA")
print("=" * 60)

pf = pq.ParquetFile(PARQUET_PATH)
print(f"Number of row groups: {pf.metadata.num_row_groups}")
print(f"Number of rows: {pf.metadata.num_rows}")
print(f"Number of columns: {pf.metadata.num_columns}")
print(f"\nSchema:")
print(pf.schema_arrow)

print("\n" + "=" * 60)
print("TRYING iter_batches WITH SMALL BATCH SIZE")
print("=" * 60)

all_rows = []

# Try reading in small batches to avoid chunked array issues
try:
    all_rows = []
    batch_size = 100
    for batch in pf.iter_batches(batch_size=batch_size):
        rows = batch.to_pylist()
        all_rows.extend(rows)
        print(f"Read batch of {len(rows)} rows, total so far: {len(all_rows)}")
        if len(all_rows) >= 500:  # Just test first 500 for now
            break
    print(f"\nSuccessfully read {len(all_rows)} rows using iter_batches!")
except Exception as e:
    print(f"iter_batches failed: {e}")
    print("\nTrying alternative: read as record batches...")
    
    # Alternative: try using the reader directly
    try:
        reader = pq.ParquetFile(PARQUET_PATH)
        # Try reading individual record batches
        batches = list(reader.iter_batches(batch_size=10))
        print(f"Got {len(batches)} batches")
        if batches:
            first_batch = batches[0]
            print(f"First batch columns: {first_batch.schema.names}")
            rows = first_batch.to_pylist()
            print(f"First batch rows: {len(rows)}")
    except Exception as e2:
        print(f"Alternative also failed: {e2}")
        
        print("\n" + "=" * 60)
        print("TRYING TO READ WITH ARROW IPC (DIFFERENT APPROACH)")
        print("=" * 60)
        
        # Last resort: try converting column by column
        try:
            # Read just scalar columns first
            scalar_cols = ['system_prompt', 'type']
            table = pq.read_table(PARQUET_PATH, columns=scalar_cols)
            print(f"Scalar columns read successfully: {table.num_rows} rows")
            print(table.to_pandas().head())
        except Exception as e3:
            print(f"Scalar columns failed: {e3}")

if all_rows:
    print("\n" + "=" * 60)
    print("FIRST ROW STRUCTURE")
    print("=" * 60)
    
    first_row = all_rows[0]
    for key, value in first_row.items():
        if isinstance(value, list):
            print(f"  {key}: list with {len(value)} items")
            if len(value) > 0:
                print(f"    First item type: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"    First item keys: {value[0].keys()}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with keys {value.keys()}")
        else:
            print(f"  {key}: {type(value).__name__} = {repr(value)[:100]}")

    print("\n" + "=" * 60)
    print("TESTING CONVERSATION CONVERSION")
    print("=" * 60)
    
    from cartridges.structs import Conversation
    
    conversations = [Conversation.from_dict(row) for row in all_rows[:10]]
    print(f"Successfully created {len(conversations)} Conversation objects")
    print(f"\nFirst conversation has {len(conversations[0].messages)} messages")
    
    print("\n" + "=" * 60)
    print("FIRST CONVERSATION CONTENT")
    print("=" * 60)
    conv = conversations[0]
    print(f"System prompt: {conv.system_prompt[:200] if conv.system_prompt else 'None'}...")
    print(f"Type: {conv.type}")
    print(f"Metadata: {conv.metadata}")
    for i, msg in enumerate(conv.messages):
        print(f"\nMessage {i} ({msg.role}):")
        print(f"  Content: {msg.content[:200]}...")
        print(f"  Token IDs: {len(msg.token_ids) if msg.token_ids else 0} tokens")
        print(f"  Top logprobs: {'Yes' if msg.top_logprobs else 'No'}")
else:
    print("\nNo rows were successfully read. The parquet file may be corrupted or incompatible.")

