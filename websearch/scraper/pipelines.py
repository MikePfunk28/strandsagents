import json
import os
import sqlite3
import tempfile


class EphemeralStorePipeline:
    def open_spider(self, spider):
        self.tmpdir = tempfile.mkdtemp(prefix="scraper_")
        self.jsonl_path = os.path.join(self.tmpdir, "pages.jsonl")
        self.db_path = os.path.join(self.tmpdir, "pages.sqlite")
        self.jsonl = open(self.jsonl_path, "w", encoding="utf-8")
        self.db = sqlite3.connect(self.db_path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS pages(
            url TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            html TEXT,
            published TEXT,
            price TEXT,
            meta_json TEXT
        );""")

        spider.logger.info(
            f"Ephemeral outputs: JSONL={self.jsonl_path} DB={self.db_path}")

    def process_item(self, item, spider):
        # write JSONL
        self.jsonl.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        # write SQLite
        self.db.execute(
            "INSERT OR REPLACE INTO pages(url,title,text,html,published,price,meta_json) VALUES(?,?,?,?,?,?,?)",
            (
                item.get("url"), item.get("title"), item.get(
                    "text"), item.get("html"),
                item.get("published"), item.get(
                    "price"), json.dumps(item.get("metadata") or {})
            )
        )
        self.db.commit()
        return item

    def close_spider(self, spider):
        self.jsonl.close()
        self.db.close()
        print(
            f"[Ephemeral] JSONL: {self.jsonl_path}\n[Ephemeral] SQLite: {self.db_path}")
