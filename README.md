# ai-service-search — Profile Search API

Dịch vụ nhận **tên một người** (kèm bộ lọc tùy chọn), tìm thông tin công khai qua Serper (thông qua MCP), dùng LLM để trích xuất và phân biệt người trùng tên, rồi trả về danh sách ứng viên có dẫn nguồn.

Kiến trúc chi tiết: xem [`architechture.md`](architechture.md).

## Tìm ảnh tương tự trên web bằng Google Vision

Ví dụ độc lập dùng Web Detection để tìm ảnh trùng/tương tự và các trang chứa ảnh đã được Google lập chỉ mục. Đây **không phải** API nhận dạng danh tính; các trường `score` là điểm của dịch vụ, không phải phần trăm xác suất.

1. Tạo Google Cloud project, bật **Cloud Vision API**, cấu hình billing và Application Default Credentials theo [hướng dẫn Google](https://cloud.google.com/docs/authentication/application-default-credentials).
2. Từ thư mục `ai-service-search`, cài dependency bổ sung:

   ```bash
   uv sync --extra google-vision
   ```

3. Chạy với một ảnh bạn có quyền sử dụng:

   ```bash
   uv run python examples/google_vision_web_detection.py /path/to/image.jpg
   ```

Kết quả JSON gồm `full_matches`, `partial_matches`, `similar_images`, `pages` và `entities`. Google Cloud tính phí theo số ảnh yêu cầu và cấu hình tài khoản; xem [bảng giá Vision](https://cloud.google.com/vision/pricing). Không đưa credentials vào source code.

## 1. Yêu cầu

- Python 3.11+ (đã thử với 3.13)
- [uv](https://docs.astral.sh/uv/) để quản lý môi trường
- API key của [Serper.dev](https://serper.dev)
- API key của một LLM có endpoint tương thích OpenAI (mặc định: Yescale với model Gemini)

## 2. Cài đặt

```bash
cd ai-service-search
uv sync
```

Lệnh này tạo `.venv/` và cài toàn bộ thư viện theo `uv.lock`.

## 3. Cấu hình

```bash
cp .env.example .env   # bỏ qua nếu đã có .env
```

Mở `.env` và điền tối thiểu:

```env
SERPER_API_KEY=...            # key Serper.dev
LLM_BASE_URL=https://api.yescale.io/v1
LLM_API_KEY=...               # key LLM
LLM_MODEL=gemini-2.5-flash    # đúng tên model trên Yescale
OUTPUT_LANGUAGE=English       # ngôn ngữ cho headline / disambiguationNote ...
```

Các tham số chi phí (đã đặt mặc định tiết kiệm):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SEARCH_STRATEGY` | llm_first | `llm_first`: hỏi LLM trước, LLM biết chắc thì trả luôn (0 lần gọi Serper), không thì LLM tự đưa ra truy vấn cần tìm. `search_first`: luôn tìm Serper trước rồi LLM đọc kết quả |
| `MAX_QUERIES` | 2 | Số truy vấn khám phá. Truy vấn thứ 2 chỉ chạy khi truy vấn đầu không có knowledge graph, hoặc thiếu Wikipedia + 2 mạng xã hội |
| `MAX_FETCH_URLS` | 4 | Số trang được đọc nội dung |
| `MAX_LLM_SOURCES` | 20 | Số nguồn tối đa gửi cho LLM |
| `EXCERPT_CHARS` | 2000 | Số ký tự trích quanh tên người trên mỗi trang |
| `ENRICH_CANDIDATES` | 2 | Số ứng viên tối đa được tìm bổ sung |
| `ENRICH_MIN_IDENTIFYING` | 3 | Chỉ tìm bổ sung khi ứng viên có ít hơn ngần này thuộc tính nhận diện |
| `ENRICH_IMAGES` | false | Có gọi image search hay không |
| `SOCIAL_LOOKUP_CANDIDATES` | 2 | Số ứng viên tối đa được tìm thêm mạng xã hội (mỗi người 1 lần gọi Serper) |
| `SOCIAL_MIN_PLATFORMS` | 3 | Chỉ tìm thêm khi ứng viên có ít hơn ngần này mạng xã hội |
| `IMAGE_SEARCH_PEOPLE` | 2 | Tìm bằng ảnh: số người nhận ra trong ảnh được chạy deep search |
| `IMAGE_MIN_CONFIDENCE` | 0.3 | Tìm bằng ảnh và `/v1/identify`: bỏ qua người model nhận ra với độ tin cậy thấp hơn |
| `IDENTIFY_SOURCES_PEOPLE` | 3 | `/v1/identify` với `withSources`: số người được tìm mạng xã hội và ảnh |
| `INITIAL_PHOTOS` | 3 | Số ảnh tối đa mỗi người (deep, preview, identify). Ảnh từ trang nguồn đứng trước, ảnh đầu là ảnh profile |
| `SOURCE_PHOTO_PAGES` | 5 | Ảnh từ trang nguồn: số trang mở tối đa cho mỗi người |
| `SOURCE_PHOTO_LIMIT` | 3 | Ảnh từ trang nguồn: số ảnh giữ lại |
| `SOURCE_PHOTO_CANDIDATES` | 3 | Preview: số ứng viên đầu tiên được lấy ảnh từ trang nguồn |
| `RESULT_CACHE_TTL_S` | 21600 | Cache kết quả theo (tên, bộ lọc), đơn vị giây |
| `SYNC_TIMEOUT_S` | 60 | Thời gian chờ tối đa của request đồng bộ |
| `FETCH_USER_AGENT` | `ProfileSearchBot/0.1 (+https://example.com/bot) httpx` | User-Agent khi đọc trang. Wikipedia yêu cầu dạng `Tên/phiên bản (URL hoặc email liên hệ) thư-viện`; hãy thay bằng thông tin liên hệ thật |

Serper MCP server (`serper_mcp/server.py`) được dịch vụ **tự khởi động** qua stdio theo cấu hình trong `mcp_servers.json`; không cần chạy riêng.

## 4. Chạy dịch vụ

HTTP (UI test / Swagger):

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Thêm `--reload` khi phát triển.

Worker (tầng production: nghe RabbitMQ, không poll HTTP):

```bash
docker compose up -d --build
```

Compose build image worker và chạy container ở chế độ nền. Cần tạo `.env` từ `.env.example`, điền API keys và đặt `RABBITMQ_HOST` thành hostname/IP mà container truy cập được (không dùng `localhost` nếu RabbitMQ chạy ngoài container). Xem log bằng `docker compose logs -f worker`, dừng bằng `docker compose down`.

Một queue `AI_SEARCH_QUEUE_IN` (exchange `AI_SEARCH_EXCHANGE_INPUT`, header `type=process_search`) cho mọi action; kết quả `AI_SEARCH_QUEUE_OUT`. Payload xem [`docs/onesearch-client-api.md`](docs/onesearch-client-api.md) mục 9.

Kiểm tra:

```bash
curl http://localhost:8000/health
```

Kết quả đúng sẽ có `"connected":{"serper":["image_search","web_search"]}` và `"llm"` khác `null`. Nếu `"llm": null` thì `LLM_API_KEY` chưa được cấu hình.

Tài liệu API tương tác (Swagger): http://localhost:8000/docs

**One-Search** (search-profile → deep-search → history → chat, 202 + poll): [`docs/onesearch-client-api.md`](docs/onesearch-client-api.md).

Giao diện test: mở http://localhost:8000/ trên trình duyệt, nhập tên rồi bấm **Tìm**.

- Chế độ **Deep** gọi `/v1/people-search` với `mode: "deep"` (search + 1 lần gọi LLM).
- Chế độ **Preview** gọi `/v1/people-search` với `mode: "preview"` (chỉ search, không LLM).
- Phần **Bộ lọc** là tùy chọn. Cuối trang có JSON gốc.

## 5. Gọi API

### 5.1 `POST /v1/people-search`

```bash
curl -s -X POST http://localhost:8000/v1/people-search \
  -H 'content-type: application/json' \
  -d '{"query": "Ronaldo"}'
```

Body:

- `query` (bắt buộc): tên người cần tìm.
- `filters` (tùy chọn): các khóa hợp lệ `company`, `jobTitle`, `occupation`, `education`, `location`, `nationality`, `field`. Nhận luôn tên khóa của onesearch: `title` → `jobTitle`, `school` → `education`, `industry` → `field`. Khóa khác bị bỏ. Ô tìm kiếm của onesearch có 5 bộ lọc (nơi sống, học vấn, chức danh, công ty, lĩnh vực/ngành):
  - Deep: mọi bộ lọc được ghép vào truy vấn Google đầu tiên và gửi cho LLM. Ứng viên được cộng điểm theo tỉ lệ bộ lọc khớp và bị trừ nếu không khớp bộ lọc nào. `field` so với field/industry/specialization/occupation; `jobTitle` và `company` so cả với lịch sử nghề nghiệp.
  - Preview: location, company, jobTitle, education ghép vào cả hai truy vấn; `field` chỉ ghép vào truy vấn web công khai.
- `mode`: `deep` (mặc định, search + LLM) hoặc `preview` (chỉ search, không LLM, trả ngay).
- `maxSearchQueries` (chỉ preview): 1 = mạng xã hội, 2 = + web, 3 = + ảnh cho các ứng viên đầu.

Deep chạy quá `SYNC_TIMEOUT_S` thì trả `202` kèm `searchId`; lấy kết quả bằng:

```bash
curl -s http://localhost:8000/v1/people-search/<searchId>   # 202 khi đang chạy, 200 khi xong
```

Ví dụ kết quả (rút gọn):

```json
{
  "mode": "deep",
  "searchId": "search_…",
  "status": "resolved",
  "clarificationQuestion": null,
  "candidates": [
    {
      "id": "cristiano-ronaldo-1a2b3c4d",
      "displayName": "Cristiano Ronaldo",
      "headline": "Portuguese professional footballer · Al-Nassr",
      "confidence": 0.81,
      "platforms": ["instagram", "x"],
      "sources": [{"url": "https://instagram.com/cristiano", "type": "instagram", "verified": true}],
      "keyFacts": [{"key": "nationality", "label": "Nationality", "value": "Portuguese", "verified": true}],
      "facts": {"nationality": "Portuguese", "teams": "Al-Nassr, Portugal national team", "socialProfiles": "instagram, x"},
      "photos": [{"url": "https://upload.wikimedia.org/...jpg", "fromSource": true, "faceMatch": 1.0}],
      "avatarUrl": "https://upload.wikimedia.org/...jpg",
      "profile": ["Cristiano Ronaldo is a Portuguese professional footballer ..."],
      "profileSource": "wikipedia"
    }
  ],
  "meta": {"searchCalls": 3, "llmCalls": 1, "cacheHit": false, "totalSources": 14}
}
```

Một số hành vi:

- Deep search trích đủ các thông tin công khai mà trang hồ sơ onesearch Pro hiển thị (`FactsTable`): `fullName`, `born` (chỉ từ Wikipedia, knowledge panel hoặc trang chính thức; LLM enrich chỉ được thêm khi năm sinh có trong snippet của các nguồn đó), `age`, `email`, `phone`, `jobTitle`, `company`, `occupation`, `knownFor`, `nationality`, `education`, `location`, `field`, `industry`, `specialization`, `yearsActive`, `awards`, `notableWorks`, `teams`, `academicInterests`, `publisher`, `bookSeries`, `website`. Mỗi giá trị phải có trích dẫn nguyên văn từ nguồn. Các trường có thể nhiều giá trị (awards, teams, notableWorks...) được gộp lại. `audience` (số follower/subscriber) đọc nguyên văn từ snippet Instagram/TikTok/X/YouTube/Facebook, không qua LLM. `facts` là `keyFacts` theo đúng key của onesearch (`jobTitle` → `primaryRole`, cộng `socialProfiles` dạng "Instagram · @handle"), để onesearch gán thẳng vào `FeCandidate.facts`. Thuộc tính knowledge panel (ví dụ "Born") được đưa vào nguồn, kể cả tuổi, email và số điện thoại khi panel nêu ra; trừ người thân, địa chỉ nhà và tài sản. Tuổi, email và số điện thoại được giữ khi một nguồn nêu ra, kèm trích dẫn. Không thu thập địa chỉ nhà hay người thân.

- `id` ổn định giữa các lần tìm cho cùng một người.
- `profile`: tóm tắt Wikipedia tiếng Anh nếu có (`profileSource: "wikipedia"`), nếu không thì do LLM viết từ nguồn (`"llm"`); preview dùng snippet (`"snippets"`).
- `photos`: tối đa 3 ảnh mỗi người (`INITIAL_PHOTOS`). Ảnh lấy từ trang nguồn của chính người đó (Wikipedia, GitHub, ảnh chân dung mang tên người, `og:image`) đứng trước và ảnh đầu tiên là ảnh profile (`avatarUrl`). Image search chỉ chạy để bù khi chưa đủ ảnh nguồn, xếp theo độ khớp khuôn mặt. Cùng một bức ảnh ở link khác (khác host, kích thước, bị cắt) chỉ giữ 1 lần: so dHash của vùng mặt đã căn chỉnh, hoặc độ khớp mặt ≥ 0.95. Một khuôn mặt chỉ thuộc về một ứng viên: người trùng tên không dùng chung ảnh.
- Muốn thêm ảnh thì gọi `POST /v1/photos` (không gọi LLM):

  ```bash
  curl -X POST localhost:8000/v1/photos -H 'Content-Type: application/json' -d '{
    "query": "<candidate.imageSearchQuery>", "referenceUrl": "<candidate.photos[0].url>",
    "name": "<candidate.displayName>", "aliases": ["<candidate.aliases>"], "profileUrls": ["<candidate.sources[].url>"], "filters": {}, "exclude": ["<url đã có>"], "limit": 6}'
  # -> {"photos": [{"url", "pageUrl", "title", "fromSource": false, "faceMatch", "fallbackUrl"}], "meta": {"searchCalls", "faceChecked"}}
  ```

  Ảnh lấy từ `imageUrl` của Google. Ảnh từ Facebook, Instagram, TikTok và LinkedIn chỉ trả HTML hoặc 403, nên dùng luôn `thumbnailUrl` (bản Google lưu trên `gstatic.com`). Với ảnh khác, nếu `imageUrl` tải về không phải ảnh thì lúc kiểm khuôn mặt sẽ chuyển sang `thumbnailUrl`. Trường `fallbackUrl` là bản thumbnail đó: client hiện nó khi `url` không tải được. Không dùng `googleUrl` vì đó là trang HTML của Google Images, không phải ảnh.

  API này tìm ảnh theo `imageSearchQuery` và chỉ trả ảnh có khuôn mặt khớp với `referenceUrl`. Truy vấn trùng với lúc search nên thường lấy từ cache (`searchCalls: 0`) trong `SEARCH_CACHE_TTL_S`.
- Tối đa `MAX_CANDIDATES` (15) người. Người không có nguồn tìm được thì bị ẩn. Người không có ảnh vẫn được hiện (`photos: []`); đặt `REQUIRE_PHOTO=true` để ẩn họ.
- Trang báo và bài viết không được dùng làm nguồn.
- Bộ lọc `location` hoặc `nationality` quyết định khu vực tìm kiếm của Google; kết quả luôn bằng `OUTPUT_LANGUAGE`.

Mã lỗi: `422` input sai, `503` chưa cấu hình LLM hoặc không có MCP server, `502` lỗi khi tìm kiếm, `404` không có `searchId`.

### 5.2 Search-profile, deep-search, lịch sử và chat (One-Search)

Contract đầy đủ (poll 202, không SSE): [`docs/onesearch-client-api.md`](docs/onesearch-client-api.md).

Tóm tắt: `POST /v1/search-profile` **hoặc** `POST /v1/people-search/image` → chọn card → `POST /v1/deep-search` (gửi cả object candidate) → `POST /v1/history` → `POST /v1/chat`. Search, ảnh, chat đều `202` + `GET /v1/people-search/{searchId}`. `POST /v1/chat/stream` cùng contract đó, không còn SSE.

Các API khác: `GET /v1/history?userId=u1` (danh sách), `GET /v1/history/{id}?userId=u1` (ứng viên kèm toàn bộ tin nhắn), `DELETE /v1/history/{id}?userId=u1`.

Chat trả lời theo ngôn ngữ của câu hỏi (hỏi tiếng Việt thì trả lời tiếng Việt), không theo `OUTPUT_LANGUAGE`. Chat là một agent có tool `web_search`:

1. LLM nhận thông tin nhận diện (tên, headline, key facts, 3 nguồn, tóm tắt profile), `CHAT_MAX_TURNS` lượt hỏi đáp gần nhất (mỗi tin cắt còn `CHAT_TURN_CHARS` ký tự) và câu hỏi.
2. Nếu biết chắc (từ profile, hội thoại hoặc kiến thức của nó về đúng người này), LLM trả lời luôn: tốn 1 lần gọi LLM, 0 lần search, `answeredFrom: "knowledge"`.
3. Chỉ khi không biết, hoặc không chắc điều mình biết là về đúng người này (không phải người trùng tên), LLM mới gọi `web_search`. Tối đa `CHAT_MAX_SEARCHES` truy vấn chạy song song; truy vấn luôn chứa tên người, mỗi truy vấn giữ `CHAT_SEARCH_RESULTS` kết quả, kể cả trang báo vì đời tư thường chỉ có trên báo. Server đọc thêm `CHAT_FETCH_PAGES` trang đầu (bỏ qua mạng xã hội), lấy tối đa `CHAT_PAGE_CHARS` ký tự quanh tên người, và bỏ qua nếu quá `CHAT_FETCH_TIMEOUT_S` giây. Sau đó một lần gọi LLM thứ hai (không có tool) trả lời từ kết quả: `answeredFrom: "web"`, `citations` là các kết quả đó. Một lượt chỉ được lưu khi trả lời xong. Nếu bật Langfuse, `sessionId` là `historyId`.

`userId` do client tự gửi, service không xác thực. Hãy để service sau backend/gateway có xác thực, đừng mở thẳng ra Internet.

### 5.3 Tìm bằng ảnh

```bash
curl -X POST localhost:8000/v1/people-search/image \
  -F image=@photo.jpg \
  -F 'filters={"location": "Hanoi"}' \
  -F caption="..." \
  -F photoWarnings=multiple_faces
```

- `image`: jpeg, png hoặc webp, tối đa 5 MB. Sai định dạng trả `415`, quá cỡ trả `413`. Ảnh không được lưu lại.
- `filters` (JSON), `caption`, `photoWarnings` đều tùy chọn. `photoWarnings` là mã kiểm tra khuôn mặt phía client, cách nhau bằng dấu phẩy: `no_face` (×0.6), `multiple_faces` (×0.8), `face_too_small` (×0.85). Các mã này chỉ hạ `confidence`, không bao giờ tăng.

Luồng xử lý:

1. Một lần gọi LLM (gửi kèm ảnh) để đoán người nổi tiếng trong ảnh. Model không đoán danh tính người thường.
2. Người có `confidence` ≥ `IMAGE_MIN_CONFIDENCE` (tối đa `IMAGE_SEARCH_PEOPLE` người) được chạy deep search như `/v1/people-search` (dùng chung cache).
3. So khuôn mặt trong ảnh tải lên với ảnh của từng ứng viên. Ứng viên khớp mặt (`queryFaceMatch` ≥ `FACE_MATCH_THRESHOLD`) được xếp lên đầu.

Response giống `/v1/people-search`, với `mode: "image"` và thêm `identified` (những người model nhận ra) cùng `photoWarnings`. Cũng trả `202` + `searchId` khi chạy quá `SYNC_TIMEOUT_S`.

### 5.4 API từng bước (building blocks)

Mỗi API chỉ làm một bước, để client (onesearch) tự ghép luồng của mình. Các API này không cache kết quả cuối, nhưng truy vấn search được cache trong `SEARCH_CACHE_TTL_S`.

| API | Làm gì | Chi phí |
|---|---|---|
| `POST /v1/identify` | Hỏi LLM tên này là ai. Với `withSources` (mặc định bật), tối đa `IDENTIFY_SOURCES_PEOPLE` người khả năng cao nhất được tìm thêm `socialSources` (đã kiểm chứng) và `photos`. `sourceUrls` là link model nhớ, chưa kiểm chứng | 1 LLM + 2–3 search/người (0 search khi `withSources: false`) |
| `POST /v1/identify/image` | Hỏi LLM ảnh này là ai (multipart, giống 5.3, có field `withSources`) | như trên |
| `POST /v1/photos/from-sources` | Lấy ảnh từ trang nguồn cho người có nguồn nhưng chưa có ảnh | 0 search, 0 LLM |
| `POST /v1/sources/social` | Profile mạng xã hội của một người, mỗi mạng 1 profile | 1–2 search |
| `POST /v1/sources/web` | Trang web công khai nói về tên đó | 1 search |
| `POST /v1/images/search` | Tìm ảnh thô, không so khuôn mặt (muốn so khuôn mặt thì dùng `/v1/photos`) | 1 search |
| `POST /v1/profiles/enrich` | Viết hồ sơ đầy đủ khi user mở một ứng viên | 1 LLM lần đầu, 0 các lần sau |

```bash
curl -X POST localhost:8000/v1/identify -H 'Content-Type: application/json' -d '{"query": "Sơn Tùng M-TP", "filters": {}, "withSources": true}'
# -> {"candidates": [{"displayName", "headline", "confidence", "disambiguationNote", "nationality", "occupation",
#      "knownFor", "aliases", "sourceUrls", "imageSearchQuery", "profile", "socialSources": [...], "photos": [...]}],
#     "warnings": [], "meta": {"searchCalls": 3, "llmCalls": 1}}

curl -X POST localhost:8000/v1/photos/from-sources -H 'Content-Type: application/json' \
  -d '{"name": "Sơn Tùng M-TP", "sources": [{"url": "https://vi.wikipedia.org/wiki/Sơn_Tùng_M-TP", "type": "wikipedia", "verified": true}], "limit": 3}'
# -> {"photos": [{"url", "pageUrl", "fromSource": true, "faceMatch": 1.0}], "faceChecked": true, "meta": {"searchCalls": 0, "llmCalls": 0}}

curl -X POST localhost:8000/v1/sources/social -H 'Content-Type: application/json' \
  -d '{"name": "Sơn Tùng M-TP", "aliases": ["Nguyễn Thanh Tùng"], "occupation": "Singer", "filters": {}, "withIdentity": true}'
# -> {"sources": [{"url", "title", "type": "facebook", "snippet", "verified": true}], "meta": {"searchCalls": 2, "llmCalls": 0}}

curl -X POST localhost:8000/v1/sources/web -H 'Content-Type: application/json' -d '{"name": "Sơn Tùng M-TP", "limit": 8, "includeNews": false}'

curl -X POST localhost:8000/v1/images/search -H 'Content-Type: application/json' \
  -d '{"query": "Son Tung M-TP Vietnamese singer", "name": "Sơn Tùng", "limit": 3}'
# -> {"photos": [{"url", "pageUrl", "title", "fromSource": false, "fallbackUrl"}], "meta": {...}}
```

Chi tiết từng API:

- **`/v1/photos/from-sources`** (cũng chạy tự động, trước image search, trong deep search, `/v1/identify` và preview cho `SOURCE_PHOTO_CANDIDATES` ứng viên đầu; đủ 3 ảnh nguồn thì không gọi image search):
  - Mở tối đa `SOURCE_PHOTO_PAGES` trang, theo thứ tự Wikipedia, GitHub, website chính thức, YouTube, rồi tới các trang khác. Lấy ảnh đầu bài Wikipedia (REST API), avatar GitHub (`github.com/<user>.png`), hoặc `og:image` của trang.
  - Không mở Facebook, Instagram, TikTok, X, LinkedIn, Pinterest, Reddit (phải đăng nhập).
  - Có model khuôn mặt: khuôn mặt tham chiếu lấy từ trang của chính người đó (Wikipedia, GitHub, website, profile đã kiểm chứng). Nếu không có thì dùng khuôn mặt mà ít nhất 2 ảnh cùng khớp. Ảnh không có mặt (logo, banner) và ảnh người khác bị bỏ.
  - Không có model khuôn mặt: chỉ giữ ảnh từ trang của chính người đó, `faceChecked: false`.
  - Link Wikipedia chưa kiểm chứng (ví dụ `sourceUrls` của model) chỉ được tin khi tên bài khớp tên người.

- **`/v1/sources/social`**:
  - Truy vấn 1 là `tên + (site:linkedin.com/in OR site:facebook.com OR ...)`. Chỉ giữ trang profile (không lấy bài post, video, group, tài khoản fan) có tiêu đề chứa tên và handle khớp tên.
  - Truy vấn 2 (khi `withIdentity` bật và có `occupation`/`headline`/`knownFor`) là `alias hoặc tên + nghề + youtube instagram tiktok`. Kết quả của truy vấn này chỉ được giữ khi tiêu đề/snippet mô tả đúng nghề của người đó.
- **`/v1/sources/web`**: truy vấn là `tên + location`. Chỉ giữ trang có tiêu đề nêu tên, hoặc có đủ mọi từ trong tên. Trang báo và bài viết bị bỏ, trừ khi `includeNews: true`.
- **`/v1/profiles/enrich`**:
  - Gửi `{"historyId", "userId"}` (dùng và cập nhật ứng viên đã lưu) hoặc `{"candidate": {...}}`.
  - LLM chỉ đọc tên, các facts đã có và tối đa 12 nguồn của ứng viên, rồi viết `headline`, `profile` (2–5 đoạn), thêm `keyFacts` còn thiếu (`verified: false`), `career` (`period`, `title`, `organization`) và `tags`.
  - Facts đã kiểm chứng giữ nguyên. Tóm tắt từ Wikipedia không bị thay. Model được trả tuổi, email và số điện thoại khi nguồn nêu ra. Không được trả địa chỉ nhà hay người thân.
  - Với `historyId`, kết quả được lưu vào lịch sử (`enrichedAt`, `enrichmentVersion`). Lần mở sau trả ngay `cached: true` mà không gọi LLM. `force: true` để viết lại.
  - Hai request cùng lúc cho cùng một người chỉ tốn một lần gọi LLM.

Ứng viên (`candidates[]`) có thêm các trường: `career`, `tags`, `queryFaceMatch` (chỉ khi tìm bằng ảnh), `enrichedAt`, `enrichmentVersion`.

### 5.5 Prompt Lab

http://localhost:8000/prompt-lab — thử prompt trực tiếp với LLM (API: `GET /prompt-lab/templates`, `GET /prompt-lab/history`, `POST /prompt-lab/run`).

## 6. Chạy test

```bash
uv run pytest -q
```

Test không cần API key: dùng một search MCP server giả (`tests/fake_search_mcp.py`) qua stdio và một LLM giả.

## 7. Cấu trúc thư mục

```text
app/
  main.py            HTTP API (FastAPI)
  orchestrator.py    Điều phối một lượt tìm: search -> đọc trang -> LLM -> chấm điểm
  planner.py         Sinh truy vấn khám phá / bổ sung
  mcp_manager.py     MCP client: kết nối server, khám phá tool, gọi tool
  search_adapter.py  Lớp tìm kiếm độc lập provider + cache truy vấn
  ranking.py         Chuẩn hóa URL, khử trùng lặp, xếp hạng nguồn
  fetcher.py         Đọc trang công khai, trích đoạn quanh tên
  llm.py             LLM adapter (OpenAI-compatible), 1 lần gọi analyze mỗi vòng
  resolution.py      Kiểm evidence, gộp ứng viên, chấm điểm, trạng thái field
  client_format.py   Chuyển sang schema client
  history.py         Lưu lịch sử tìm kiếm + tin nhắn chat (SQLite)
  identify.py        Nhận diện theo tên/ảnh (LLM) + luồng tìm bằng ảnh
  sources.py         API từng bước: profile mạng xã hội, trang web, tìm ảnh thô
  source_photos.py   Ảnh từ trang nguồn (Wikipedia, GitHub, og:image) + lọc khuôn mặt
  enrich.py          Viết hồ sơ đầy đủ cho một ứng viên (lưu vào lịch sử)
  chat.py            Chat về một ứng viên đã lưu: prompt, search, LLM stream
serper_mcp/server.py Serper MCP server (stdio hoặc streamable-http)
mcp_servers.json     Danh sách MCP server tìm kiếm
tests/               Unit test + end-to-end test
```

## 8. Thêm hoặc đổi search provider

Thêm một mục vào `mcp_servers.json`, ánh xạ tool của server đó vào `web_search` / `image_search`:

```json
{
  "name": "my-search",
  "transport": "streamable_http",
  "url": "https://search-mcp.example.com/mcp",
  "headers": {"Authorization": "Bearer ${MY_SEARCH_TOKEN}"},
  "tools": {"web_search": "search"},
  "enabled": true
}
```

Tool cần nhận `query`, `num` và trả về `{"results": [{"title", "url", "snippet", "position"}]}`.

Chạy Serper MCP server riêng qua HTTP (khi cần dùng từ xa):

```bash
SERPER_API_KEY=... SERPER_MCP_PORT=8100 uv run python -m serper_mcp.server streamable-http
```

Server lắng nghe tại `http://127.0.0.1:8100/mcp` (đổi cổng để không trùng với API ở cổng 8000).

## 9. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `/health` có `"llm": null` | Điền `LLM_API_KEY` trong `.env` rồi khởi động lại |
| `/health` có lỗi trong `mcp.errors` | Chạy `uv sync`; kiểm tra `mcp_servers.json` |
| Kết quả lỗi `serper auth error` | Kiểm tra `SERPER_API_KEY` |
| `LLM output invalid` trong log | Model không trả JSON đúng; thử model khác hoặc giảm `MAX_LLM_SOURCES` |
| Kết quả ít thông tin | Tăng `MAX_QUERIES`, `MAX_FETCH_URLS` hoặc `ENRICH_MIN_IDENTIFYING` (tốn thêm chi phí) |
| Nguồn Wikipedia bị `blocked http 403` | Đặt `FETCH_USER_AGENT` đúng dạng `Tên/phiên bản (liên hệ) thư-viện` |
| Đổi `.env` nhưng không có tác dụng | Khởi động lại server; kết quả cũ còn nằm trong cache `RESULT_CACHE_TTL_S` |
