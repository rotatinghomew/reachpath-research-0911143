import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .cases import CaseStore
from .catalog import Catalog
from .domain import overview
from .paths import AnalysisStore, analyze
from .profile import numeric_profiles


class Application:
    def __init__(self, data, state):
        state = Path(state)
        self.catalog = Catalog(data)
        self.cases = CaseStore(state)
        self.analyses = AnalysisStore(state / "analyses")
        self._path_report = None
        self._path_options = None

    def _path_analysis(self, options):
        if self._path_report is None or self._path_options != options:
            self._path_report = analyze(
                self.catalog,
                seed=options["seed"], boot_reps=options["boot_reps"],
                folds=options["folds"], mc_draws=options["mc_draws"],
                run_sensitivity=options["run_sensitivity"],
            )
            self._path_options = dict(options)
        return self._path_report

    def dispatch(self, method, target, body=None):
        parsed = urlsplit(target)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)
        try:
            if method == "GET" and path == "/health":
                return 200, {"status": "ok", "source_fingerprint": self.catalog.fingerprint}
            if method == "GET" and path == "/tables":
                return 200, {"tables": self.catalog.describe()}
            if method == "GET" and path.startswith("/tables/"):
                name = path.split("/")[-1]
                offset = int(query.get("offset", ["0"])[0])
                limit = int(query.get("limit", ["50"])[0])
                if offset < 0 or not 1 <= limit <= 500:
                    raise ValueError("分页范围无效")
                records = self.catalog.rows(name)
                return 200, {"total": len(records), "offset": offset, "rows": records[offset:offset + limit]}
            if method == "GET" and path == "/view":
                return 200, {"source_fingerprint": self.catalog.fingerprint, "view": overview(self.catalog)}
            if method == "GET" and path == "/profiles":
                return 200, {"source_fingerprint": self.catalog.fingerprint, "profiles": numeric_profiles(self.catalog)}
            if method == "GET" and path == "/cases":
                return 200, {"cases": self.cases.list()}
            if method == "GET" and path.startswith("/cases/"):
                return 200, self.cases.read(path.split("/")[-1])
            if method == "GET" and path == "/paths":
                options = self._path_options_from_query(query)
                report = self._path_analysis(options)
                return 200, {"source_fingerprint": self.catalog.fingerprint,
                             "options": options, "report": report}
            if method == "POST" and path == "/paths":
                if not isinstance(body, dict):
                    raise ValueError("请求内容应为对象")
                if set(body) - {"label", "seed", "boot_reps", "folds", "mc_draws", "run_sensitivity"}:
                    raise ValueError("请求包含未知字段")
                label = body.get("label", "路径分析记录")
                if not isinstance(label, str) or not label.strip() or len(label) > 120:
                    raise ValueError("研究名称需为一至一百二十个字符")
                options = self._path_options_from_query(query, body)
                report = self._path_analysis(options)
                return 201, self.analyses.create(report, label.strip())
            if method == "GET" and path == "/paths/records":
                return 200, {"records": self.analyses.list()}
            if method == "GET" and path.startswith("/paths/records/"):
                return 200, self.analyses.read(path.split("/")[-1])
            if method == "POST" and path == "/cases":
                if not isinstance(body, dict):
                    raise ValueError("请求内容应为对象")
                if set(body) - {"label", "options", "notes"}:
                    raise ValueError("请求包含未知字段")
                return 201, self.cases.create(self.catalog, body.get("label"), body.get("options"), body.get("notes", ""))
        except KeyError:
            return 404, {"error": "记录不存在"}
        except (TypeError, ValueError) as error:
            return 400, {"error": str(error)}
        return 404, {"error": "接口不存在"}

    @staticmethod
    def _path_options_from_query(query, body=None):
        body = dict(body or {})
        options = {
            "seed": 20261115, "boot_reps": 200, "folds": 5,
            "mc_draws": 8, "run_sensitivity": True,
        }
        for key in ("seed", "boot_reps", "folds", "mc_draws"):
            if key in query:
                body[key] = query[key][0]
        if "no_sensitivity" in query:
            body["run_sensitivity"] = False
        for key in ("seed", "boot_reps", "folds", "mc_draws"):
            if key in body:
                try:
                    options[key] = int(body[key])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{key} 必须是整数") from error
                if options[key] <= 0:
                    raise ValueError("路径分析参数必须为正")
        if options["folds"] > 20 or options["boot_reps"] > 1000:
            raise ValueError("路径分析参数超出允许范围")
        if "run_sensitivity" in body:
            if not isinstance(body["run_sensitivity"], bool):
                raise ValueError("run_sensitivity 必须是布尔值")
            options["run_sensitivity"] = body["run_sensitivity"]
        return options


def make_server(application, host="127.0.0.1", port=8080):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond("GET")

        def do_POST(self):
            self.respond("POST")

        def respond(self, method):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 262144:
                    raise ValueError("请求体过大")
                body = json.loads(self.rfile.read(length)) if length else None
                status, content = application.dispatch(method, self.path, body)
            except (ValueError, UnicodeDecodeError):
                status, content = 400, {"error": "请求格式无效"}
            encoded = json.dumps(content, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)
