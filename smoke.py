import json
import tempfile
import threading
from urllib.request import Request, urlopen

from reachpath.service import Application, make_server


def main():
    with tempfile.TemporaryDirectory() as state:
        server = make_server(Application("data", state), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            for path in ("/health", "/tables", "/view"):
                with urlopen(base + path, timeout=5) as response:
                    print(path, json.dumps(json.load(response), ensure_ascii=False))
            request = Request(base + "/cases", data=json.dumps({"label": "月度经营讨论", "options": {"period": "2026-09"}}, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=5) as response:
                created = json.load(response)
            with urlopen(base + "/cases/" + created["id"], timeout=5) as response:
                restored = json.load(response)
            assert created == restored
            print("研究记录已回读", restored["id"], restored["source_fingerprint"])
            with urlopen(base + "/paths?boot_reps=8&mc_draws=2&no_sensitivity=1", timeout=30) as response:
                path_result = json.load(response)
            report = path_result["report"]
            assert report["sample"]["n"] == 1000
            assert "interventional_mediation" in report
            assert report["positivity"]["zero_imputation_used"] is False
            print("路径分析 ITT =", report["randomization_estimands"]["itt_purchase"]["estimate"])
            path_request = Request(
                base + "/paths?boot_reps=8&mc_draws=2&no_sensitivity=1",
                data=json.dumps({"label": "冒烟路径记录"}, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(path_request, timeout=30) as response:
                path_record = json.load(response)
            with urlopen(base + "/paths/records/" + path_record["id"], timeout=5) as response:
                path_restored = json.load(response)
            assert path_restored["report"] == report
            print("路径分析记录已回读", path_record["id"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    main()
