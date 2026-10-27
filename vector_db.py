"""Kho embedding trên PostgreSQL + pgvector, thay thế file output/model_knn_<mô hình>.npz.

Bảng:
    persons(id, name)                       người
    faces(id, person_id, image_key, ...)    mỗi khuôn mặt trong dataset (dùng chung mọi mô hình);
                                            image_key = "<Tên>/<file>", không phụ thuộc máy nào
    emb_<mô hình>(face_id, embedding)       vector đã L2-normalize, mỗi mô hình một bảng vì số chiều khác nhau
    model_meta(model_name, k, threshold, pipeline_fingerprint)
                                            K, ngưỡng Unknown và dấu vết pipeline của từng mô hình

Khoảng cách `<->` của pgvector là Euclid, cùng thước đo với knn_func.KNN, nên ngưỡng Unknown dùng chung được.
Phần bỏ phiếu và ngưỡng vẫn do Python xử lý (knn_func.majority_vote); DB chỉ tìm K láng giềng gần nhất.

Cả nhóm cùng ghi vào một DB nên: ingest chỉ upsert và từ chối ghi khi pipeline khác (check_pipeline); ngưỡng Unknown
không do ingest đặt mà được tính lại trên toàn bộ vector trong DB (load_all + set_threshold).
"""
import re

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from config import DB_URL, PIPELINE_FINGERPRINT, normalize_name
from knn_func import majority_vote

# pgvector chỉ tạo được chỉ mục HNSW cho kiểu vector tới 2000 chiều (VGG-Face có 4096 nên tìm kiếm không có chỉ mục)
MAX_INDEX_DIM = 2000
BATCH = 1000


def _table(model_name):
    return "emb_" + re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _impostor_table(model_name):
    return "imp_" + _table(model_name)[len("emb_"):]


def connect(timeout=15):
    """Kết nối DB, bật extension vector và đăng ký kiểu vector cho psycopg.

    timeout đủ dài cho Postgres trên cloud (có dịch vụ ngủ khi không dùng và mất vài giây để thức dậy);
    DB cục bộ không chạy thì vẫn báo lỗi ngay. prepare_threshold=None tắt prepared statement để tương thích
    với kết nối qua PgBouncer (chế độ transaction pooler) mà các dịch vụ cloud hay dùng.
    """
    conn = psycopg.connect(DB_URL, autocommit=True, connect_timeout=timeout, prepare_threshold=None)
    has_vector = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
    if not has_vector:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id   SERIAL PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )""")
    # image_key = "<Tên người>/<tên file>", không chứa đường dẫn của máy nào, nên cả nhóm dùng chung được
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id         BIGSERIAL PRIMARY KEY,
            person_id  INT NOT NULL REFERENCES persons(id),
            image_key  TEXT UNIQUE NOT NULL,
            detected   BOOLEAN,
            confidence REAL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_meta (
            model_name TEXT PRIMARY KEY,
            k          INT NOT NULL,
            threshold  REAL
        )""")
    # DB tạo bởi bản cũ chưa có cột này
    conn.execute("ALTER TABLE model_meta ADD COLUMN IF NOT EXISTS pipeline_fingerprint TEXT")
    # "người lạ" (khuôn mặt không thuộc người nào trong persons) dùng để đo tỉ lệ nhận nhầm; tách khỏi faces
    # để KNN không bao giờ bỏ phiếu cho họ
    conn.execute("""
        CREATE TABLE IF NOT EXISTS impostors (
            id         BIGSERIAL PRIMARY KEY,
            image_key  TEXT UNIQUE NOT NULL,
            confidence REAL
        )""")


def check_pipeline(conn, model_name):
    """Báo lỗi nếu DB đã có vector của mô hình này do một pipeline khác tạo ra (detector, align, phiên bản DeepFace).

    Vector trộn từ hai pipeline không so sánh được và không có lỗi nào báo, nên phải chặn trước khi ghi. Gọi trước khi
    embed để khỏi tốn thời gian. DB cũ chưa có fingerprint thì chấp nhận, lần ingest tới sẽ ghi fingerprint vào.
    """
    init_schema(conn)
    row = conn.execute(
        "SELECT pipeline_fingerprint FROM model_meta WHERE model_name = %s", (model_name,)
    ).fetchone()
    if row and row[0] and row[0] != PIPELINE_FINGERPRINT:
        raise ValueError(
            f"DB đang chứa vector {model_name} tạo bằng pipeline '{row[0]}', máy này dùng '{PIPELINE_FINGERPRINT}'. "
            "Hãy đồng bộ phiên bản DeepFace/cấu hình với cả nhóm; vector hai pipeline không được trộn chung."
        )


def _ensure_embedding_table(conn, model_name, dim):
    table = _table(model_name)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            face_id   BIGINT PRIMARY KEY REFERENCES faces(id) ON DELETE CASCADE,
            embedding vector({dim}) NOT NULL
        )""")
    if dim <= MAX_INDEX_DIM:
        conn.execute(f"CREATE INDEX IF NOT EXISTS {table}_hnsw ON {table} USING hnsw (embedding vector_l2_ops)")
    return table


def ingest(conn, model_name, faces_df, X, k, prune=False):
    """Ghi (hoặc cập nhật) khuôn mặt, embedding, K và fingerprint pipeline của một mô hình.

    faces_df: bảng của face_pipeline.ensure_face_cache; X: embedding L2-normalize cùng thứ tự với faces_df.
    Không đặt ngưỡng Unknown: ngưỡng của DB dùng chung phải tính trên mọi vector trong DB (load_all, set_threshold),
    không phải trên dataset cục bộ vốn có thể chỉ là một phần.
    prune=True xóa khỏi DB mọi khuôn mặt không có trong faces_df (kể cả embedding của mọi mô hình). Mặc định tắt vì
    DB dùng chung: dataset cục bộ của một thành viên có thể chỉ là một phần của dữ liệu chung.
    """
    check_pipeline(conn, model_name)
    table = _ensure_embedding_table(conn, model_name, X.shape[1])

    people = [normalize_name(p) for p in faces_df["person"]]
    keys = [f"{p}/{f}" for p, f in zip(people, faces_df["filename"])]

    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO persons (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
            [(n,) for n in sorted(set(people))],
        )
        cur.execute("SELECT name, id FROM persons")
        person_id = dict(cur.fetchall())

        cur.executemany(
            """INSERT INTO faces (person_id, image_key, detected, confidence)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (image_key) DO UPDATE SET
                   person_id = EXCLUDED.person_id,
                   detected = EXCLUDED.detected, confidence = EXCLUDED.confidence""",
            [(person_id[person], key, bool(r.detected), float(r.confidence))
             for r, person, key in zip(faces_df.itertuples(), people, keys)],
        )

        if prune:
            cur.execute("DELETE FROM faces WHERE NOT (image_key = ANY(%s))", (keys,))

        face_id = {}
        for i in range(0, len(keys), BATCH):
            cur.execute("SELECT image_key, id FROM faces WHERE image_key = ANY(%s)", (keys[i:i + BATCH],))
            face_id.update(cur.fetchall())

        cur.executemany(
            f"""INSERT INTO {table} (face_id, embedding) VALUES (%s, %s)
                ON CONFLICT (face_id) DO UPDATE SET embedding = EXCLUDED.embedding""",
            [(face_id[key], np.asarray(x, dtype="float32")) for key, x in zip(keys, X)],
        )

        cur.execute(
            """INSERT INTO model_meta (model_name, k, pipeline_fingerprint) VALUES (%s, %s, %s)
               ON CONFLICT (model_name) DO UPDATE SET
                   k = EXCLUDED.k, pipeline_fingerprint = EXCLUDED.pipeline_fingerprint""",
            (model_name, k, PIPELINE_FINGERPRINT),
        )


def impostor_keys(conn, model_name):
    """image_key của các người lạ đã có embedding của mô hình này (để lần sau bỏ qua, chạy lại không tốn công)."""
    init_schema(conn)
    table = _impostor_table(model_name)
    if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
        return set()
    rows = conn.execute(
        f"SELECT i.image_key FROM {table} e JOIN impostors i ON i.id = e.impostor_id"
    ).fetchall()
    return {r[0] for r in rows}


def ingest_impostors(conn, model_name, keys, confidences, X):
    """Ghi embedding người lạ của một mô hình. keys: "<nguồn>/<file>"; X: L2-normalize cùng thứ tự với keys."""
    check_pipeline(conn, model_name)
    table = _impostor_table(model_name)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            impostor_id BIGINT PRIMARY KEY REFERENCES impostors(id) ON DELETE CASCADE,
            embedding   vector({X.shape[1]}) NOT NULL
        )""")
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO impostors (image_key, confidence) VALUES (%s, %s) "
            "ON CONFLICT (image_key) DO UPDATE SET confidence = EXCLUDED.confidence",
            [(k, float(c)) for k, c in zip(keys, confidences)],
        )
        ids = {}
        for i in range(0, len(keys), BATCH):
            cur.execute("SELECT image_key, id FROM impostors WHERE image_key = ANY(%s)", (keys[i:i + BATCH],))
            ids.update(cur.fetchall())
        cur.executemany(
            f"""INSERT INTO {table} (impostor_id, embedding) VALUES (%s, %s)
                ON CONFLICT (impostor_id) DO UPDATE SET embedding = EXCLUDED.embedding""",
            [(ids[k], np.asarray(x, dtype="float32")) for k, x in zip(keys, X)],
        )


def load_impostors(conn, model_name):
    """Ma trận embedding người lạ của một mô hình (rỗng nếu chưa có)."""
    table = _impostor_table(model_name)
    if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
        return np.empty((0, 0))
    rows = conn.execute(f"SELECT embedding FROM {table} ORDER BY impostor_id").fetchall()
    return np.array([r[0].to_numpy() if hasattr(r[0], "to_numpy") else r[0] for r in rows], dtype="float64")


def load_all(conn, model_name):
    """(X, y) của mọi vector của mô hình này trong DB, gồm cả dữ liệu do thành viên khác nạp."""
    rows = conn.execute(
        f"""SELECT p.name, e.embedding
            FROM {_table(model_name)} e
            JOIN faces f ON f.id = e.face_id
            JOIN persons p ON p.id = f.person_id
            ORDER BY f.id"""
    ).fetchall()
    y = np.array([r[0] for r in rows])
    # tùy phiên bản pgvector, cột vector trả về ndarray hoặc pgvector.Vector
    X = np.array([r[1].to_numpy() if hasattr(r[1], "to_numpy") else r[1] for r in rows], dtype="float64")
    return X, y


def set_threshold(conn, model_name, threshold):
    conn.execute("UPDATE model_meta SET threshold = %s WHERE model_name = %s", (threshold, model_name))


def models_in_db(conn):
    return [r[0] for r in conn.execute("SELECT model_name FROM model_meta")]


def available_models():
    """Các mô hình đã nạp vào DB. DB không chạy hoặc chưa có bảng thì trả về danh sách rỗng."""
    try:
        with connect() as conn:
            return models_in_db(conn)
    except psycopg.Error:
        return []


class PgVectorKNN:
    """Cùng giao diện dùng trong app với knn_func.KNN: K, threshold, predict, kneighbors."""

    def __init__(self, conn, model_name, k, threshold, pipeline_fingerprint=None):
        self.conn = conn
        self.table = _table(model_name)
        self.K = k
        self.threshold = threshold
        self.pipeline_fingerprint = pipeline_fingerprint   # app so với config.PIPELINE_FINGERPRINT để cảnh báo

    def search(self, x):
        """(khoảng cách, tên) của K láng giềng gần nhất, sắp xếp tăng dần."""
        x = np.asarray(x, dtype="float32").reshape(-1)
        # truy vấn con lấy K vector gần nhất trên bảng embedding (dùng được chỉ mục), sau đó mới nối ra tên người
        rows = self.conn.execute(
            f"""SELECT d.dist, p.name
                FROM (SELECT face_id, embedding <-> %s AS dist
                      FROM {self.table} ORDER BY embedding <-> %s LIMIT %s) d
                JOIN faces f ON f.id = d.face_id
                JOIN persons p ON p.id = f.person_id
                ORDER BY d.dist""",
            (x, x, self.K),
        ).fetchall()
        return np.array([r[0] for r in rows], dtype="float64"), np.array([r[1] for r in rows])

    def kneighbors(self, x):
        return self.search(x)

    def predict(self, x):
        _, labels = self.search(x)
        return np.array([majority_vote(labels)])


def load_pg_knn(model_name, conn=None):
    """PgVectorKNN của một mô hình đã nạp vào DB, hoặc None nếu chưa có.

    conn: dùng lại một kết nối có sẵn (app giữ một kết nối cho cả phiên). Đọc lại K và ngưỡng mỗi lần gọi,
    nên train lại với --db là có hiệu lực ngay.
    """
    own_conn = conn is None
    conn = conn or connect()
    row = conn.execute("SELECT k, threshold FROM model_meta WHERE model_name = %s", (model_name,)).fetchone()
    if row is None:
        if own_conn:
            conn.close()
        return None
    k, threshold = row
    try:
        fingerprint = conn.execute(
            "SELECT pipeline_fingerprint FROM model_meta WHERE model_name = %s", (model_name,)
        ).fetchone()[0]
    except psycopg.errors.UndefinedColumn:   # DB tạo bởi bản cũ, chưa chạy train_model.py --db lần nào sau khi nâng cấp
        fingerprint = None
    return PgVectorKNN(conn, model_name, k, None if threshold is None else float(threshold), fingerprint)
