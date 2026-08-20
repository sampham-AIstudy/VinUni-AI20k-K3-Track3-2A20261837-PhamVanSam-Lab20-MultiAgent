# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

Mỗi nhóm trả lời 2 câu:

1. **Case nào nên dùng multi-agent? Vì sao?**
   - **Các bài toán phức tạp, nhiều chặng độc lập (multi-stage workflows)**: Như deep research, code generation + review + testing, complex report synthesis.
   - **Cần tách biệt vai trò (Separation of Concerns)**: Mỗi agent tập trung vào một prompt chuyên biệt (Researcher tìm nguồn, Analyst phân tích so sánh, Writer định dạng & trích dẫn) tránh tình trạng *context dilution* và giảm hallucination.
   - **Cần auditability, debuggability và dynamic routing**: Khi luồng xử lý cần rẽ nhánh theo trạng thái thực tế (ví dụ: thiếu nguồn thì tìm lại, phân tích chưa đạt thì retry, lỗi bước nào thì trace và debug đúng bước đó).
   - **Tích hợp nhiều toolset/quyền hạn khác nhau**: Tách biệt agent có quyền gọi API/viết code nguy hiểm với agent chỉ phân tích/tổng hợp.

2. **Case nào không nên dùng multi-agent? Vì sao?**
   - **Các tác vụ đơn giản, một bước (single-step tasks)**: Như tóm tắt một đoạn văn bản ngắn, phân loại cảm xúc (sentiment analysis), dịch thuật, trả lời câu hỏi trực tiếp (Q&A cơ bản).
   - **Yêu cầu độ trễ cực thấp (Ultra-low latency / Real-time applications)**: Multi-agent có độ trễ tích lũy (overhead) do phải qua nhiều bước điều phối của Supervisor và nhiều lượt gọi LLM tuần tự.
   - **Ngân sách token eo hẹp (Cost-sensitive systems)**: Mỗi bước agent tiêu tốn thêm token vào prompt và output trung gian, dẫn đến chi phí vận hành cao hơn gấp 3-5 lần so với single-call.
   - **Quy trình hoàn toàn tuần tự, cố định (Deterministic linear pipelines)**: Khi luồng chạy 100% không cần rẽ nhánh hay tự thích ứng, dùng standard Python chaining / LangChain chain đơn giản sẽ ít bug và dễ bảo trì hơn LangGraph multi-agent phức tạp.

