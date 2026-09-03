# ANSWERS — Day 28 Track 2

## 1. Các quyết định và đánh đổi

### Kafka dùng at-least-once, Delta chịu trách nhiệm chống trùng

- API tạo `idempotency_key` ổn định và gửi khóa này trong cả payload lẫn Kafka header.
- Kafka có thể phát lại một bản tin sau retry hoặc consumer restart.
- Trước `MERGE`, batch giữ đúng một event mới nhất cho mỗi `idempotency_key`, so sánh
  `(occurred_at, event_id)`. Nếu hai event cùng thời gian thì `event_id` là tie-breaker
  xác định, nên kết quả không phụ thuộc thứ tự Kafka giao bản tin.
- Đánh đổi: có thêm bước gom bản ghi trước khi ghi, nhưng đổi lại replay không làm tăng
  sai số đếm hoặc tạo nhiều hàng cho cùng một sự kiện.

Ví dụ: cùng khóa `feedback-123` đến ba lần. Hai lần có thời gian `10:00`, lần có
`event_id=b` thắng `event_id=a`; một lần lúc `10:01` thắng cả hai. Delta chỉ nhận bản
`10:01`, và lần chạy sau cập nhật cùng hàng thay vì chèn hàng mới.

### Readiness phân biệt dependency bắt buộc và tùy chọn

- Kafka, Qdrant và MLflow hỏng làm trạng thái thành `not_ready`, vì request chính không
  còn bảo đảm contract.
- Feast hoặc vLLM được cấu hình tùy chọn có thể làm trạng thái thành `degraded`; API vẫn
  sống để operator đọc nguyên nhân và các route không phụ thuộc vẫn hoạt động.
- Đánh đổi: degraded mode tăng tính sẵn sàng, nhưng client phải đọc rõ `status`,
  `components` và `degraded_reasons`, không được hiểu mọi HTTP 200 là đủ chức năng.

### Release được điều khiển bởi MLflow alias

- Serving resolve alias `champion` thay vì hard-code model version trong image.
- Promotion và rollback chỉ đổi alias, vì vậy không cần build lại API.
- Mỗi release lưu model ID, embedding model ID, prompt, Delta version và Git SHA để truy
  vết đúng dữ liệu và cấu hình đã tạo ra câu trả lời.
- Đánh đổi: request path phụ thuộc registry; cache/fallback cần được bổ sung khi đưa lên
  production để tránh MLflow trở thành điểm lỗi đơn.

### Qdrant dùng point ID xác định

- `doc_id` được ánh xạ thành UUID ổn định. Index lại cùng tài liệu sẽ upsert cùng point.
- Đánh đổi: thao tác đổi `doc_id` được xem là tài liệu mới; nếu muốn hỗ trợ rename cần có
  bảng ánh xạ hoặc quy trình xóa point cũ.

### Trace context đi qua cả luồng bất đồng bộ

- Caller gửi W3C `traceparent`; gateway/API giữ trace ID, producer ghi header Kafka, và
  consumer/Airflow tiếp tục cùng trace ID bằng span mới.
- Phải so trace ID, không so toàn bộ `traceparent`, vì mỗi hop đúng chuẩn sẽ tạo span ID
  riêng.
- Đánh đổi: telemetry tăng chi phí mạng/lưu trữ. Production cần sampling theo rủi ro và
  luôn giữ trace lỗi hoặc request chậm.

## 2. Khoảng cách tới production

1. **High availability:** Compose hiện là một broker Kafka, một replica API và các kho
   state đơn lẻ. Production cần cluster/replication, PodDisruptionBudget, anti-affinity,
   backup và diễn tập restore.
2. **Security:** secret mẫu, cổng quản trị và service nội bộ phù hợp cho lab nhưng chưa
   đủ cho production. Cần secret manager, TLS/mTLS, xác thực/ủy quyền, rotation, network
   policy mặc định từ chối và audit log.
3. **State:** Delta/MLflow/Feast/Qdrant đang dựa vào volume hoặc local path. Cần object
   storage/database được sao lưu, lifecycle policy và kiểm thử disaster recovery.
4. **Capacity:** kết quả load trên laptop chỉ là baseline. Cần tải đại diện trên route
   `/api/v1/ask`, warm-up model, đo CPU/RAM/GPU, vLLM queue, token throughput, Kafka lag,
   error rate và memory leak trong bài chạy dài.
5. **vLLM:** IP07 chỉ đạt khi endpoint chứng minh được `/version`, model list và metric
   `vllm:`. OpenAI-compatible mock không phải bằng chứng. Khi không có GPU/endpoint thật,
   mục này phải để `UNVERIFIED`.
6. **GitOps:** manifest tĩnh được validate nhưng self-heal/rollback chỉ được coi là bằng
   chứng live sau khi Argo CD đồng bộ vào một cluster thật, tạo drift có kiểm soát rồi
   quan sát reconcile.
7. **SLO và vận hành:** cần xác lập SLO theo dữ liệu sử dụng thật, route-specific alert,
   escalation/on-call, retention cho metrics/traces và runbook được diễn tập định kỳ.

## 3. Phân công và đóng góp

Repository nộp bài là bài cá nhân của **Nguyễn Thanh Vinh**, nên một người phụ trách toàn
bộ năm vai trong role cards:

- Ingestion & Orchestration: contract Kafka, header trace/idempotency, DLQ/replay.
- Data & ML: de-duplication trước Delta MERGE, Feast request, MLflow release/rollback.
- Serving & Retrieval: Qdrant, readiness/degraded policy và kiểm tra gate vLLM.
- Platform & Observability: Envoy, Prometheus/Grafana, OTEL/Jaeger, manifest Kubernetes
  và GitOps.
- Presenter / Incident Commander: chạy test, gom evidence, load profile, failure/recovery
  và chuẩn bị demo theo `docs/demo-runbook.md`.

Phần code cần hoàn thiện trực tiếp trong scaffold gồm bốn hàm tại
`src/lab28_platform/integration_tasks.py`: `event_headers`, `dedupe_latest`,
`feast_online_request` và `readiness_status`.

## 4. Kết quả load baseline đã đo

Môi trường đo: Windows + Docker Desktop, 4 logical CPU, Docker memory limit khoảng
7.7 GiB, stack `basic` đã warm bằng seed/index/release. Route đo là
`http://localhost:8080/ready`, 200 request mỗi cấu hình.

| Workers | HTTP 200 | HTTP 429 | P50 | P95 | P99 |
|---:|---:|---:|---:|---:|---:|
| 8 | 200 | 0 | 976.60 ms | 1460.83 ms | 1593.71 ms |
| 16 | 75 | 125 | 24.39 ms | 2230.85 ms | 2621.72 ms |

Envoy được cấu hình giới hạn 10 request/s. Vì vậy nguyên nhân chính khi tăng từ 8 lên 16
worker là rate limiter ở gateway từ chối 125 request. P50 giảm mạnh không có nghĩa hệ
thống nhanh hơn: phần lớn response ở nửa dưới là 429 được trả ngay tại edge, trong khi
P95/P99 của các request chậm vẫn vượt 2 giây. `/ready` tự probe Kafka, MLflow, Qdrant,
vLLM và Feast ở mỗi request nên latency cao hơn một health endpoint chỉ đọc trạng thái
cache.

Kết luận có thể hành động: giữ rate limit để bảo vệ API, sửa load probe để tách rõ 429
khỏi network error, cache kết quả dependency probe trong một TTL ngắn, rồi đo lại. Đây
chỉ là baseline `/ready`; chưa thay thế load test `/api/v1/ask` với vLLM thật.

## 5. Trạng thái evidence của môi trường hiện tại

| Điểm | Trạng thái | Bằng chứng/khoảng trống |
|---|---|---|
| IP01 | PASS | Kafka record thật có key, `idempotency-key` và `traceparent`. |
| IP02 | UNVERIFIED | Profile `full` chưa chạy nên chưa có Airflow DAG run/asset event. |
| IP03 | UNVERIFIED | Chưa có Spark MERGE và Delta transaction history. |
| IP04 | PARTIAL | Feast `/health` PASS; online entity sau materialization chưa được chứng minh. |
| IP05 | PASS | Qdrant có 13 point và hybrid search trả nguồn có score/doc ID. |
| IP06 | PASS | Tạo version 3, promote từ 2 rồi rollback về 2; provenance được kiểm tra. |
| IP07 | UNVERIFIED | Không có GPU-backed vLLM; file identity ghi rõ endpoint unreachable. |
| IP08 | PASS | Burst 30 request: 10 HTTP 200, 20 HTTP 429, cả hai có request ID. |
| IP09 | PASS (non-GPU) | Target bắt buộc up, alert health `ok`, Grafana được provision; target vLLM tùy chọn down. |
| IP10 | PARTIAL | Local trace có 8 span và 3/11 required span qua gateway → API → Kafka; thiếu Airflow/Spark/serving và LangSmith. |

`scripts/validate_manifests.py` PASS. Drift/self-heal Argo CD vẫn `UNVERIFIED` vì máy
không có Kubernetes context, `kind` hoặc `argocd`. Không tạo bằng chứng giả cho ba gate
môi trường này.

## 6. Cách đối chiếu khi demo

1. Bắt đầu từ `architecture/platform-5-layer.png`, chỉ owner của IP01–IP10.
2. Mở `evidence/integration-report.json`, sau đó đối chiếu từng IP với file evidence
   cùng tên trong `contracts/integration-matrix.yaml`.
3. Với happy path, nối bốn định danh: Airflow run ID → trace ID → Delta version → MLflow
   model version.
4. Với incident, nêu dự đoán trước khi dừng dependency, quan sát readiness/metric, khởi
   động lại và chứng minh row/point count không tăng sai hoặc mất dữ liệu.
5. Kết thúc bằng kết quả load P50/P95/P99 và nói rõ cấu hình máy, số worker, route đo,
   bottleneck; không suy rộng thành năng lực production.
