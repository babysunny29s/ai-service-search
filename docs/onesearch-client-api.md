# API cho client One-Search

Base URL = chỗ `ai-service-search` đang chạy. Repo **không** gắn domain cố định — phía One-Search đặt `BASE` theo môi trường.

| Môi trường | `BASE` |
|---|---|
| Cùng máy (dev, uvicorn `:8000`) | `http://localhost:8000` |
| Máy khác cùng LAN | `http://<IP-máy-chạy-service>:8000` (ví dụ `http://10.40.40.42:8000`) |
| Production | URL HTTPS thật sau reverse proxy, ví dụ `https://search.example.com` |

JSON: `Content-Type: application/json`. Search bằng ảnh: `multipart/form-data`. Không cần API key từ phía One-Search. CORS mở `*`.

Production: One-Search **không** gọi HTTP — publish vào RabbitMQ, xem **mục 9**. HTTP dưới đây là contract body (và UI test local).

Hai lối vào, sau đó cùng deep-search → chat theo `candidateId`:

```
Theo tên
1. search-profile (khai báo searchId + query) → danh sách người; mỗi người một candidateId
2. User chọn 1 card
3. deep-search { candidateId } → hồ sơ đầy đủ (server lấy metadata từ DB)
4. chat { sessionId, candidateId, question } → hỏi về người đó

Theo ảnh
1. people-search-image (searchId + URL MinIO) → nhận diện + search từng tên
2. User chọn 1 card
3–4. Giống trên (deep-search / chat: sessionId + candidateId)
```

Postgres (`DB_NAME=onesearch`), không đổi schema. `SearchHistory.id` là `searchId` **client** gửi lúc search lần đầu. Mỗi người một dòng `Candidate` (`id` server gán `cnd_…`, `personKey`, `candidateInformation`). `ChatSession.id` là `sessionId` client gửi. `ChatHistory.id` server gán `msg_…`. HTTP poll `searchId` (job in-memory) khác `searchId` lưu DB.

`search-profile`, `people-search/image`, `deep-search`, `chat` trả **202 ngay**. Client poll `GET /v1/people-search/{searchId}` đến khi 200.

OpenAPI/Swagger: `GET /docs`. Health: `GET /health`.

---

## 1. Poll job (`searchId`)

`HTTP_WAIT_S=0`: mọi `POST /v1/search-profile`, `/v1/deep-search`, `/v1/people-search/image`, `/v1/chat` trả **202**. Đừng giữ POST (Cloudflare cắt kết nối dài).

### 202 đang chạy

```json
{
  "searchId": "search_6bc58c17a9115e45",
  "status": "queued | running | complete | failed",
  "stage": "searching | enriching | thinking | answering | …",
  "resultUrl": "/v1/people-search/search_6bc58c17a9115e45"
}
```

Chat lúc đang stream có thêm `text` (bản nháp) và `queries` (nếu model đang search).

### Poll

```
GET /v1/people-search/{searchId}
```

| HTTP | Ý nghĩa |
|---|---|
| **202** | Chưa xong. Body giống 202 lúc POST. Poll lại ~500–1000 ms. |
| **200** | Xong. Body = kết quả của **đúng POST đã tạo job** (xem shape bên dưới). |
| **404** | `searchId` sai hoặc job hết hạn (~**1 giờ**, tối đa 1000 job in-memory). |
| **502** | Job fail. `{"searchId", "error"}`. |
| **503** | LLM hoặc Serper MCP chưa sẵn sàng. |

Phân loại body 200:

- có `candidates` + `mode: "searchprofile"` → search-profile
- có `candidates` + `mode: "image"` → search bằng ảnh
- có `candidate` (một người) + `socialSources` → deep-search
- có `answer` + `sessionId` → chat

`searchId` của search-profile, search-bằng-ảnh, deep-search và chat **khác nhau**. Không gộp cost/meta giữa các bước.

---

## 2. `POST /v1/search-profile`

Tìm người theo **tên đầy đủ** đã gõ. Tối đa 2 query Serper (Wikipedia EN / LinkedIn / Facebook, rồi Instagram / TikTok / X). Một lần LLM gộp cùng một người, bỏ người trùng tên. Chỉ trả candidate có **ít nhất một trang hồ sơ** (social, Wikipedia, site cá nhân). Ảnh lấy từ trang nguồn, không image-search, không lọc mặt.

### Request

```json
{
  "searchId": "srch_onesearch_…",
  "query": "Cao Thị Trâm Anh",
  "filters": {
    "location": "Hà Nội",
    "jobTitle": "CTO",
    "company": "FPT",
    "education": "HUST",
    "field": "Fintech"
  },
  "page": 1,
  "excludeUrls": [],
  "excludeNames": []
}
```

| Field | Bắt buộc | Mô tả |
|---|---|---|
| `query` | có | 2–120 ký tự. Gửi **đúng tên user gõ**, không rút họ. |
| `searchId` | queue: có; HTTP: nên có | Id phiên search **do client tạo**. Worker **422** nếu thiếu. Cùng `searchId` khi “Tìm thêm” (`page` 2, 3…) để upsert candidate. |
| `userId` | không | Khóa candidate theo user; deep-search/chat sau này gửi cùng `userId`. |
| `filters` | không | Chỉ các khóa: `location`, `education`, `jobTitle`, `company`, `field`, `occupation`, `nationality`. Alias One-Search: `title`→`jobTitle`, `school`→`education`, `industry`→`field`, `role`→`jobTitle`, `city`→`location`. Khóa khác bị bỏ. |
| `page` | không | Mặc định `1`. “Tìm thêm” = `2`, rồi `3`… (tối đa 10). |
| `excludeUrls` | không | URL nguồn đã hiện trên UI (tối đa 80). |
| `excludeNames` | không | `displayName` đã hiện (tối đa 40). |

Nếu `page=1` nhưng có `excludeUrls` / `excludeNames`, server tự nhảy sang page 2.

**Dùng lại kết quả cũ:** `page=1` không có `excludeUrls` / `excludeNames` → nếu trong 24 giờ (`REUSE_SEARCH_MAX_AGE_H`) đã có search-profile **của bất kỳ user nào** cùng `query` + `filters` (không phân biệt hoa thường, có dấu hay không), server trả luôn các candidate đó (`meta.cacheHit = true`, không tốn Serper/LLM). Candidate vẫn được lưu thành dòng mới thuộc `searchId` / `userId` của request này (`id` mới). Gửi `"refresh": true` để buộc search lại.

### Response 200

```json
{
  "mode": "searchprofile",
  "searchId": "search_…",
  "status": "resolved | ambiguous | no_match",
  "clarificationQuestion": null,
  "candidates": [ { "…ClientCandidate…" } ],
  "meta": { "…ClientMeta…" }
}
```

- `resolved`: đúng 1 người. `ambiguous`: nhiều người (xếp `confidence`, hòa thì độ khớp tên). `no_match`: `candidates: []`.
- Tối đa ~15 người. Không có nguồn hồ sơ thì không vào list.
- `photos` / `avatarUrl`: ảnh trên Wikipedia, GitHub, avatar mạng… `fromSource: true`. Có thể không có mặt.

### Tìm thêm

Gửi lại cùng `query` + `filters`, `page` tăng 1, `excludeUrls` = mọi `sources[].url` đang hiện, `excludeNames` = mọi `displayName` đang hiện. **Cộng** candidate mới vào list (không thay thế). `meta` lần này chỉ tính round mới — cộng dồn ở client nếu cần.

---

## 3. `POST /v1/people-search/image`

Tìm người **từ một tấm ảnh**. `multipart/form-data` (không JSON). Ảnh **không lưu**. Cùng **202 + poll**. `meta` chỉ request này.

LLM đoán người nổi tiếng trong ảnh (không nhận diện người thường), rồi mỗi tên đủ tin chạy một vòng search. Ứng viên khớp mặt với ảnh upload (`queryFaceMatch`) xếp trước.

### Request

`Content-Type: multipart/form-data`

| Field | Bắt buộc | Mô tả |
|---|---|---|
| `image` | có | File jpeg / png / webp, ≤ **5 MB**. Trống → **422**. Sai định dạng → **415**. Quá 5 MB → **413**. |
| `filters` | không | Chuỗi JSON, cùng khóa với search-profile. Ví dụ `{"location":"Hà Nội"}`. Không phải object JSON → **422**. |
| `caption` | không | Chú thích ảnh, tối đa 200 ký tự. |
| `photoWarnings` | không | Mã client đã kiểm tra mặt, cách nhau bằng phẩy: `no_face`, `multiple_faces`, `face_too_small`. Chỉ **hạ** `confidence` (×0.6 / ×0.8 / ×0.85), không tăng. Mã lạ → **422**. |
| `searchId` | queue: có | Cùng nghĩa search-profile: id phiên do client tạo, lưu kèm candidate. |

```
POST /v1/people-search/image
```

```
image=@photo.jpg
filters={"location":"Hà Nội"}
caption=ở sự kiện công ty
photoWarnings=multiple_faces
```

### Chạy gì, tính tiền gì

1. **1 LLM** (`recall_image`) — đoán người trong ảnh.
2. Người có `confidence ≥ IMAGE_MIN_CONFIDENCE` (mặc định 0.3), tối đa `IMAGE_SEARCH_PEOPLE` (mặc định **2**): mỗi người một vòng search tên (Serper web + LLM; có thể thêm Serper image). Cache hit vòng tên = 0.
3. So mặt upload với ảnh ứng viên — **local, 0₫**.

Không gọi Scrape Creators ở bước này. Deep-search (bước sau) mới scrape.

### Response 200

```json
{
  "mode": "image",
  "searchId": "search_…",
  "status": "resolved | ambiguous | no_match",
  "clarificationQuestion": null,
  "candidates": [ { "…ClientCandidate, có thêm queryFaceMatch…" } ],
  "identified": [
    {
      "displayName": "Cristiano Ronaldo",
      "confidence": 0.9,
      "headline": "Footballer",
      "occupation": "Footballer",
      "aliases": [],
      "sourceUrls": [],
      "imageSearchQuery": "Cristiano Ronaldo footballer",
      "profile": []
    }
  ],
  "photoWarnings": ["multiple_faces"],
  "meta": { "…ClientMeta chỉ của image-search…" }
}
```

| Field | |
|---|---|
| `identified` | Người model nhận ra (kể cả người dưới ngưỡng, không được search). Kiến thức model, chưa kiểm chứng search. |
| `candidates` | Sau search tên + xếp `queryFaceMatch` giảm dần (mặt khớp `≥ FACE_MATCH_THRESHOLD` lên trước). Cùng shape `ClientCandidate`. |
| `photoWarnings` | Echo mã client gửi lên. |
| `status` | `resolved` / `ambiguous` / `no_match`. `no_match` khi không nhận ra ai — `candidates: []`, vẫn có `meta` (1 LLM). |

Sau khi user chọn card: gửi **cả object candidate** vào `POST /v1/deep-search` như luồng theo tên.

### Chỉ nhận diện (không search tên)

`POST /v1/identify/image` — cùng form, thêm `withSources` (mặc định `true`). **200 ngay**, không 202.

- `withSources=false`: 1 LLM, `candidates` = `identified`.
- `withSources=true`: thêm social + ảnh cho tối đa `IDENTIFY_SOURCES_PEOPLE` người.

`meta` cùng field cost (`costLines`, `costVnd`…).

---

## 4. `POST /v1/deep-search`

Hồ sơ một người **đã chọn** từ search-profile. Đọc trang công khai (Wikipedia EN ưu tiên, YouTube about, GitHub, site). Social walled (FB → IG → Threads → TikTok → X → LinkedIn): tối đa **2** lần Scrape Creators (1 lần = **1 credit**). Một query Serper lấy bài viết. Một lần LLM viết tiểu sử / facts. Ảnh: mọi ảnh có mặt vào gallery (tối đa 10); không có mặt thì vẫn lấy `avatarUrl` từ nguồn, **không** bỏ avatar không-mặt vào gallery.

Queue / bước sau search: chỉ cần `candidateId` (server đọc metadata từ DB). HTTP vẫn nhận nguyên object `candidate` nếu chưa persist.

### Request (queue — chỉ `candidateId`)

```json
{
  "candidateId": "cnd_…",
  "userId": "onesearch-user-123"
}
```

### Request (HTTP, object đầy đủ)

```json
{
  "candidate": { "…đúng ClientCandidate từ search-profile…" },
  "filters": { "location": "Hà Nội" }
}
```

`filters`: bộ lọc lúc search-profile (có thể `{}`). Khi chỉ gửi `candidateId`, filters lấy từ bản ghi `searches`.

**Dùng lại hồ sơ cũ:** nếu cùng người (`Candidate.personKey`) đã được deep-search trong 7 ngày (`REUSE_DEEP_MAX_AGE_H`), bởi bất kỳ user nào, server trả luôn hồ sơ đó và ghi vào candidate của request này (`meta.cacheHit = true`, không tốn Serper/LLM/Scrape Creators). Gửi `"refresh": true` để chạy lại.

### Response 200

```json
{
  "searchId": "search_…",
  "candidate": { "…ClientCandidate đã làm đầy…" },
  "socialSources": [ { "url": "https://www.facebook.com/…", "type": "facebook", "verified": true } ],
  "articleSources": [ { "url": "https://vnexpress.net/…", "type": "web", "verified": true } ],
  "warnings": [],
  "meta": { "…ClientMeta chỉ của deep-search…" }
}
```

- `candidate.profile`: đoạn tiểu sử. `profileSource`: `"wikipedia"` | `"llm"`.
- `candidate.keyFacts` / `facts`: thông tin công khai (role, company, born từ wiki, awards…). **Không** lấy email, điện thoại, địa chỉ nhà.
- `socialSources`: account/profile. `articleSources`: báo/phỏng vấn.
- `photos`: tối đa 10, ưu tiên `fromSource` + `faceMatch`. `avatarUrl` có thể là avatar nguồn khi không có ảnh mặt.
- `warnings`: ví dụ Scrape Creators chưa cấu hình.

Candidate không có trang hồ sơ (ví dụ ứng viên từ tìm bằng ảnh chỉ có bài báo): deep-search tự tìm tài khoản mạng xã hội theo tên trước (thêm ~2 Serper call, tính vào `meta.searchCalls`); không tìm được vẫn chạy tiếp với nguồn đang có + tìm bài viết theo tên.

**422** chỉ khi sau khi kiểm tra không còn nguồn nào (`"no profile source left after checking the dossier"`).

`meta` / bảng cost **chỉ deep-search**, không cộng search-profile.

---

## 5. Chat

Mỗi lần mở chat client **khai báo `sessionId` + `candidateId`**. Server tự tạo/cập nhật `chat_session` (session + snapshot candidate) và ghi từng lượt vào `chat_history`. **Không** có API history từ client (không `history.create` / get / list / delete). Cùng `sessionId` = tiếp tục hội thoại; `sessionId` khác = thread mới. `sessionId` đã gắn candidate khác → **409**.

### `POST /v1/chat` → 202 rồi poll

```json
{
  "sessionId": "sess_…",
  "candidateId": "cnd_…",
  "userId": "onesearch-user-123",
  "question": "Cô ấy học ở đâu?"
}
```

`question`: 1–500 ký tự. Thiếu `sessionId` hoặc `candidateId` → **422**. `candidateId` không có trong DB → **404**.

**Không dùng SSE.** `POST /v1/chat/stream` cùng contract 202 + poll (tránh Cloudflare cắt HTTP/2).

Poll `GET resultUrl`. 202 có `stage` (`thinking` / `searching` / `answering`) và `text` nháp.

### 200 khi xong

```json
{
  "sessionId": "sess_…",
  "candidateId": "cnd_…",
  "answer": "…markdown…",
  "answeredFrom": "knowledge | web",
  "citations": [ { "url": "https://…", "title": "…" } ],
  "meta": {
    "searchQueries": [],
    "searchCalls": 0,
    "llmCalls": 1,
    "promptTokens": 1200,
    "completionTokens": 400,
    "costUsd": 0.0029,
    "costVnd": 23,
    "costLines": [],
    "latencyMs": 1800
  }
}
```

- Trả lời **theo ngôn ngữ câu hỏi**, không theo `OUTPUT_LANGUAGE`.
- `knowledge`: trả từ hồ sơ / hội thoại / kiến thức về đúng người đó (0 Serper).
- `web`: model không chắc → tối đa 2 query Serper (luôn kèm tên người), đọc vài trang, rồi trả lời. `citations` = kết quả đưa cho model.
- `meta` chỉ lượt chat này.

---

## 6. Kiểu dữ liệu dùng chung

### `ClientCandidate`

| Field | Ý nghĩa |
|---|---|
| `id` | Ổn định giữa các lần search cùng người |
| `displayName` | Tên hiển thị |
| `confidence` | 0–1, dùng xếp card |
| `headline` | Một dòng |
| `disambiguationNote` | Phân biệt người trùng tên |
| `occupation` `knownFor` `location` `nationality` | |
| `platforms` | Thứ tự: facebook, instagram, linkedin, github, x, tiktok, threads, pinterest, reddit, youtube |
| `sources` | Trang hồ sơ. `type` = platform trên, hoặc `website` / `wikipedia` / `academic` / `web`. `verified: true` = tìm được thật |
| `keyFacts` | `{ key, label, value, verified }` |
| `facts` | Map cho FactsTable One-Search (`primaryRole`, `company`, `awards`, `socialProfiles`, …) |
| `photos` | `{ url, pageUrl, title, fromSource, faceMatch, fallbackUrl }` |
| `avatarUrl` | Ảnh đại diện (thường `photos[0]`) |
| `imageSearchQuery` | Query tiếng Anh nếu sau này gọi thêm ảnh |
| `profile` | Đoạn văn tiểu sử |
| `profileSource` | `wikipedia` \| `llm` \| `snippets` |
| `career` | `{ period, title, organization }`, cũ → mới |
| `tags` | |
| `aliases` | |
| `queryFaceMatch` | Search bằng ảnh: độ khớp mặt với ảnh upload (0–1). Không có khi search theo tên. |

### `IdentifiedPerson` (search bằng ảnh)

Người model **đoán** trong ảnh, trước khi search. Chỉ có trên `identified`, không thay `candidates`.

| Field | |
|---|---|
| `displayName` `confidence` `headline` `occupation` `knownFor` `nationality` `disambiguationNote` `aliases` | Từ kiến thức model |
| `sourceUrls` | Link model nhớ, **chưa** search |
| `imageSearchQuery` | Query tiếng Anh nếu client tự gọi thêm ảnh |
| `profile` | Đoạn model nhớ |
| `socialSources` `photos` | Chỉ khi `POST /v1/identify/image` + `withSources` |

### `ClientMeta` (search-profile, image-search, deep-search)

Mỗi request **một** `meta`, không cộng dồn server-side.

| Field | |
|---|---|
| `searchCalls` | Số lần Serper tính tiền (cache = 0) |
| `llmCalls` `promptTokens` `completionTokens` | |
| `crawlfeedCalls` | Số lần gọi Scrape Creators |
| `crawlfeedCredits` | Credit trừ (1 lần đọc profile = 1 credit; cache/404 = 0) |
| `costUsd` `costVnd` | Tổng request này |
| `costLines` | Từng dòng: `kind` = `llm` \| `serper` \| `scrapecreators`; `tokenIn`/`tokenOut`; `usd`; `vnd` |
| `cacheHit` | `true` khi trả kết quả đã có (cache trong bộ nhớ, hoặc search/deep-search đã lưu trong DB của user khác) — mọi chi phí = 0 |
| `totalSources` | |

Serper: 1 search = 1 token ≈ 26 ₫. Scrape: 1 credit ≈ 49 ₫. LLM: token × giá Yescale / 1e6 × 8000 ₫.

---

## 7. Poll mẫu (TypeScript)

```ts
async function runJob<T>(postPath: string, body: unknown): Promise<T> {
  const started = await fetch(BASE + postPath, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  let data = await started.json();
  if (started.status === 200) return data as T;
  if (started.status !== 202) throw new Error(JSON.stringify(data));
  const url = BASE + (data.resultUrl as string);
  for (;;) {
    await new Promise((r) => setTimeout(r, 700));
    const res = await fetch(url);
    data = await res.json();
    if (res.status === 200) return data as T;
    if (res.status !== 202) throw new Error(JSON.stringify(data));
  }
}

const cards = await runJob("/v1/search-profile", { searchId, query, filters });
const deep = await runJob("/v1/deep-search", { candidateId: cards.candidates[i].id });
const chat = await runJob("/v1/chat", { sessionId, candidateId: cards.candidates[i].id, userId, question });
```

Search bằng ảnh — cùng `runJob` nhưng **FormData**, không set `content-type` (browser tự gắn boundary):

```ts
async function runJobForm<T>(postPath: string, form: FormData): Promise<T> {
  const started = await fetch(BASE + postPath, { method: "POST", body: form });
  let data = await started.json();
  if (started.status === 200) return data as T;
  if (started.status !== 202) throw new Error(JSON.stringify(data));
  const url = BASE + (data.resultUrl as string);
  for (;;) {
    await new Promise((r) => setTimeout(r, 700));
    const res = await fetch(url);
    data = await res.json();
    if (res.status === 200) return data as T;
    if (res.status !== 202) throw new Error(JSON.stringify(data));
  }
}

const form = new FormData();
form.append("image", file); // File / Blob jpeg|png|webp
form.append("filters", JSON.stringify(filters));
form.append("caption", caption);
form.append("photoWarnings", "multiple_faces");
const photo = await runJobForm("/v1/people-search/image", form);
const deepFromPhoto = await runJob("/v1/deep-search", {
  candidate: photo.candidates[0],
  filters,
});
```

---

## 8. Lỗi thường gặp

| HTTP | Khi nào |
|---|---|
| 413 | Ảnh > 5 MB |
| 415 | Ảnh không phải jpeg / png / webp |
| 422 | Body sai; ảnh trống; `filters`/`photoWarnings` sai; deep-search thiếu trang hồ sơ |
| 403 | `userId` không khớp candidate |
| 404 | `searchId` job hết / `candidateId` không có |
| 409 | `sessionId` đã gắn candidate khác |
| 502 | LLM/search fail; body `{ searchId, error }` |
| 503 | Chưa có LLM hoặc Serper MCP |

Không gửi email / SĐT / địa chỉ nhà lên UI dù snippet nguồn có nhắc. Không bypass tường đăng nhập mạng xã hội — social walled chỉ qua Scrape Creators ở deep-search.

---

## 9. Worker RabbitMQ

Production: `ai-service-search` chạy **worker**, không làm HTTP cho One-Search. **Một queue** cho mọi action (search-profile, ảnh, deep-search, chat). Worker **chạy xong mới reply**.

```bash
uv run python -m app.worker
```

Gateway publish lên **headers exchange**, `headers.type = process_search` (không nhầm với field `action` trong JSON).

| | Tên mặc định |
|---|---|
| Request exchange | `AI_SEARCH_EXCHANGE_INPUT` |
| Request queue | `AI_SEARCH_QUEUE_IN` |
| Result exchange | `AI_SEARCH_EXCHANGE_RESULT` |
| Result queue | `AI_SEARCH_QUEUE_OUT` |
| Header `type` | `process_search` |
| Prefetch | `1` |
| Timeout / job | `RABBITMQ_CONSUMER_TIMEOUT` mặc định **300s** |

Dead-letter: `AI_SEARCH_QUEUE_IN_DEAD`.

Mọi request: `content_type=application/json`, `delivery_mode=persistent`, **headers** `{ "type": "process_search" }`. JSON body luôn có `action` + `requestId` (alias `request_id`). `payload` là body HTTP tương ứng; có thể bỏ `payload` và để field nằm phẳng.

**Không** đặt `"type": "process_search"` trong JSON — `type` chỉ ở AMQP headers. Trong JSON, `type` chỉ dùng khi nó chính là tên action (`deep-search`, …).

**Postgres (`onesearch`).** Lần search đầu client **phải** gửi `searchId` cùng `query`. Worker ghi `SearchHistory.id` và mỗi người một dòng `Candidate`. Deep-search chỉ gửi `candidateId` và ghi đè `candidateInformation`, đồng thời đặt `SearchHistory.selectedCandidateId`. Chat gửi `sessionId` + `candidateId`; server ghi `ChatSession` (snapshot trong `metadata`) và từng lượt vào `ChatHistory`. Thiếu `searchId` trên search-profile / people-search-image, hoặc thiếu `sessionId`/`candidateId` trên chat → **422**. `candidateId` không có → **404**. Không có action history từ client.

---

### Publish (Python)

```python
import json
import aio_pika

body = {
    "action": "search-profile",
    "requestId": "req_1",
    "payload": {"searchId": "srch_1", "query": "Cao Thị Trâm Anh", "filters": {}},
}
msg = aio_pika.Message(
    json.dumps(body, ensure_ascii=False).encode(),
    content_type="application/json",
    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
    headers={"type": "process_search"},
)
await exchange.publish(msg, routing_key="")  # exchange = AI_SEARCH_EXCHANGE_INPUT
```

Nghe kết quả: bind queue `AI_SEARCH_QUEUE_OUT` với headers `type=process_search`.

---

### Request mẫu

#### 1. Search theo tên — `search-profile`

```json
{
  "action": "search-profile",
  "requestId": "req_1234",
  "payload": {
    "searchId": "srch_1234",
    "userId": "cmuyypozi001n01ofu3ojbghd",
    "query": "Trâm Anh",
    "filters": {
      "location": "Hà Nội",
      "jobTitle": "CTO",
      "company": "FPT",
      "education": "HUST",
      "field": "Fintech"
    },
    "page": 1,
    "excludeUrls": [],
    "excludeNames": []
  }
}
```

Tìm thêm: cùng `query` + `filters`, `"page": 2`, `excludeUrls` / `excludeNames` = những gì đang hiện trên UI.

#### 2. Search bằng ảnh — `people-search-image`

`image` = **URL MinIO** (gateway upload rồi gửi link — khuyến nghị), hoặc URL ảnh http(s) công khai, hoặc base64. Worker tải jpeg / png / webp ≤ 5 MB. Ảnh không lưu lại trên search.

```json
{
  "action": "people-search-image",
  "requestId": "req_2",
  "payload": {
    "searchId": "srch_2",
    "image": "https://minios3.infinitydashboard.info/ai-inhouse/search/ninh-duong-lan-ngoc.jpg",
    "filters": { "location": "Hà Nội" },
    "caption": "ở sự kiện công ty",
    "photoWarnings": "multiple_faces"
  }
}
```

`photoWarnings`: `no_face`, `multiple_faces`, `face_too_small` (phẩy, optional).

#### 3. Tìm sâu — `deep-search`

Chỉ gửi `candidateId` trả về từ bước 1 hoặc 2. Server lấy metadata từ DB.

```json
{
  "action": "deep-search",
  "requestId": "req_3",
  "payload": {
    "candidateId": "cnd_e323c17eb8363319",
    "userId": "cmuyypozi001n01ofu3ojbghd"
  }
}
```

#### 4. Chat — `chat`

```json
{
  "action": "chat",
  "requestId": "req_4",
  "payload": {
    "sessionId": "sess_12345",
    "candidateId": "cnd_e323c17eb8363319",
    "userId": "cmuyypozi001n01ofu3ojbghd",
    "question": "Người này từng học ở đâu?"
  }
}
```

Worker **đợi xong** rồi mới reply (không 202).

#### 5. Chỉ nhận diện ảnh — `identify-image`

```json
{
  "action": "identify-image",
  "requestId": "req_9",
  "payload": {
    "image": "https://minios3.example/ai-inhouse/search/photo.webp",
    "withSources": true,
    "filters": {},
    "caption": "",
    "photoWarnings": ""
  }
}
```

#### 6. Health

```json
{ "action": "health", "requestId": "req_0", "payload": {} }
```

---

### Reply mẫu

Thành công:

```json
{
  "ok": true,
  "is_success": true,
  "action": "search-profile",
  "requestId": "req_1",
  "request_id": "req_1",
  "status": 200,
  "result": {
    "mode": "searchprofile",
    "searchId": "search_…",
    "status": "ambiguous",
    "candidates": [ { "id": "cnd_…", "displayName": "Cao Thị Trâm Anh", "confidence": 0.86 } ],
    "meta": { "searchCalls": 2, "llmCalls": 1 }
  },
  "timestamp": "2026-10-14T14:10:00"
}
```

`result` cùng shape HTTP 200 của action đó:

| `action` | `result` có |
|---|---|
| `search-profile` / `people-search-image` | `mode`, `searchId` (id client), `candidates` (`id` = `candidateId` lưu DB), `meta` (`searchCalls`/`llmCalls`; **không** `costUsd`/`costVnd`/`costLines`) |
| `deep-search` | `candidate`, `socialSources`, `articleSources`, `meta` |
| `chat` | `answer`, `sessionId`, `candidateId`, `citations`, `meta` |
| `health` | `{ "status": "ok", "mcp": …, "llm": … }` |

Lỗi:

```json
{
  "ok": false,
  "is_success": false,
  "action": "search-profile",
  "requestId": "req_1",
  "request_id": "req_1",
  "status": 422,
  "error": "action required",
  "timestamp": "2026-10-14T14:10:00"
}
```

| `status` | |
|---|---|
| 413 / 415 | Ảnh quá 5 MB / không phải jpeg·png·webp |
| 422 | Thiếu `action` / `searchId` / `sessionId` / `candidateId`, body sai, ảnh trống |
| 403 / 404 | `userId` lệch / không thấy `candidateId` |
| 409 | `sessionId` đã gắn candidate khác |
| 502 / 503 | LLM/Serper fail hoặc chưa cấu hình |
| 504 | Quá `RABBITMQ_CONSUMER_TIMEOUT` (mặc định 300s) |

Message JSON hỏng: **ack**, không requeue (tránh tính tiền lại). Header AMQP `type` trên reply cũng là `process_search`.
