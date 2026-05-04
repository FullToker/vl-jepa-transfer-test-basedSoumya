import pyarrow.parquet as pq
import json
import pathlib

table = pq.read_table("openEQA/test-00000-of-00001.parquet")
rows = table.to_pydict()

pathlib.Path("data").mkdir(exist_ok=True)
with open("data/sft_openeqa.jsonl", "w") as f:
    for i in range(len(rows["question_id"])):
        record = {
            "video": f"openEQA/{rows['episode_history'][i]}.mp4",
            "query": rows["question"][i],
            "target": rows["answer"][i],
        }
        json.dump(record, f)
        f.write("\n")

print(f"wrote {len(rows['question_id'])} rows")
