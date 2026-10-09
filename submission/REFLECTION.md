# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _Đinh Mạnh Dũng_
**Khoá:** _<A20-K4 / 2A202602975>_
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Bản nháp phân tích từ lượt chạy thực tế trên notebook Colab của bài lab. Thông tin cá nhân để người học tự điền. Các giá trị được làm tròn từ metrics và output đã lưu.

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Tesla T4; nvidia-smi báo 15360 MiB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 train / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65,875% (làm tròn 65,9%); trung vị chosen 94 token, rejected 86 token |
| DPO: β / lr / epoch | 0,1 / 5e-6 / 1 |
| LoRA / batch hiệu dụng / max length | r=16, alpha=32 / 8 / 768 token |
| Giám khảo | Skywork V2 Qwen3-4B: 8/12 (66,67%, bị loại); Llama-3.2-3B: 12/12 (100%), giám khảo quyết định |
| Chi phí | Không gọi API trả phí; chi phí gói/compute Colab của tài khoản chưa được xác nhận |

SFT hoàn tất 125 bước với loss trung bình 1,3601. Loss được ghi ở bước 10 là 1,8841 và ở bước 120 là 1,2835. Dữ liệu preference được chia theo prompt chuẩn hoá; assert kiểm tra không trùng prompt giữa train và held-out đã qua. Đã xem ba cặp mẫu: cặp liệt kê mười ví dụ có khác biệt về tổ chức danh sách; cặp phân loại tiếng Tây Ban Nha có hai câu trả lời gần nghĩa nhưng không dùng đúng nhãn yêu cầu; cặp đặt lịch đánh giá giọng nói có thông tin bổ sung không được đoạn nguồn xác nhận. Do đó nhãn preference có nhiễu và độ dài có thể là tín hiệu gây thiên vị.

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Runtime do trainer ghi | 1753,3448 giây (29 phút 13 giây); không cộng bước precompute reference trước đó |
| VRAM cao nhất | 6,450 GiB allocated / 7,490 GiB reserved, đo bằng PyTorch |
| Loss train trung bình / loss ghi đầu tiên | 0,675676 / 0,691100 |
| Chosen / rejected reward cuối train | +0,369972 / +0,282096 |
| Reward gap cuối train | +0,087876 |
| Độ chính xác reward held-out | 70% trên 100 cặp |
| Chosen / rejected reward held-out cuối | +0,383586 / +0,298807 |
| Margin held-out cuối | +0,084779 |
| Chẩn đoán tự động | INTENDED |
| Độ dài trung bình SFT → DPO (58 prompt NB4) | 555,53 → 586,21 ký tự, từ judge_summary.json |

Reference là mô hình SFT đã merge tại `/content/lab22/models/sft-merged`, không phải mô hình gốc. Log-prob reference được tính trước khi cập nhật LoRA; fingerprint của split được lưu cùng adapter.

## 3. Đọc đường reward

Ảnh: `screenshots/03-dpo-reward-curves.png`.

Reward ngầm được định nghĩa bằng β nhân log-ratio giữa policy và reference. Vì policy khởi đầu bằng SFT cộng LoRA mới, reward khởi đầu gần 0. Trên train, chosen và rejected đều tăng, kết thúc lần lượt ở +0,369972 và +0,282096. Margin train nhìn chung tăng nhưng dao động khá mạnh theo batch; chẳng hạn có những đoạn giảm rồi hồi phục ở nửa sau lượt chạy. Vì vậy không nên mô tả đường train là tăng đơn điệu.

Trên held-out, chosen tăng qua các mốc 25, 50, 75, 100 từ +0,080902 lên +0,262127, +0,357721 và +0,383586. Rejected cũng tăng tương ứng từ +0,065999 lên +0,204652, +0,280031 và +0,298807. Margin tăng từ +0,014902 lên +0,057475, +0,077690 và +0,084779. Accuracy là 58%, 69%, 67%, rồi 70%; margin tăng không đảm bảo accuracy tăng ở từng mốc.

Chẩn đoán INTENDED khớp điều kiện của hàm diagnose: chosen dương và margin dương trong cửa sổ cuối. Tuy nhiên rejected không giảm; mô tả chính xác ở đây là cả hai câu được tăng xác suất tương đối so với reference, trong đó chosen tăng nhiều hơn. Lượt chạy này không có dấu hiệu likelihood displacement ở các mốc held-out, vì chosen không âm. Trong trường hợp displacement, margin vẫn có thể tăng nếu rejected giảm nhanh hơn chosen; DPO tối ưu chênh lệch tương đối chứ không trực tiếp bắt buộc log-prob chosen tăng. Margin held-out cuối gần margin train cuối, nên chưa thấy khoảng cách rõ ở chỉ số này để kết luận overfit, nhưng 100 cặp và một seed chưa đủ chứng minh khả năng tổng quát.

## 4. So sánh SFT vs SFT+DPO

Ảnh: `screenshots/04-side-by-side-table.png`. Đã sinh 8 câu cố định và 50 prompt held-out riêng biệt bằng greedy decoding, cùng giới hạn 384 token cho hai mô hình. Nguồn số liệu: `data/eval/judge_summary.json` và `judge_results_rm.json`. Win rate tính hoà là 0,5 điểm; CI là bootstrap 95% theo notebook.

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---|---:|
| held-out | 50 | 5 | 12 | 33 | 43% (35–51%) | 44,57% (n=46) | 47,06% |
| hữu ích — helpfulness | 4 | 0 | 1 | 3 | 37,5% (12,5–50%) | 50% (n=3) | 0% |
| an toàn — safety | 4 | 2 | 0 | 2 | 75% (50–100%) | 66,67% (n=3) | 100% |

Hội đồng cuối chỉ gồm **Skywork/Skywork-Reward-V2-Llama-3.2-3B**, vì sanity Llama đạt 12/12 (100%) còn Qwen3 đạt 8/12 (66,67%), dưới ngưỡng 80% nên bị loại theo quy tắc repo. Sanity 12 cặp là kiểm tra sàng lọc nhỏ, không phải bằng chứng giám khảo luôn đúng. Mọi score đã được kiểm tra và đều hữu hạn. Không dùng API judge, nên position consistency không áp dụng.

Trên held-out, Qwen3 cho win rate 49% (CI 41–57%) còn Llama cho 43% (CI 35–51%); mức đồng ý giữa hai giám khảo trên toàn bộ 58 cặp là 84,48%. Qwen cao hơn 6 điểm phần trăm, nhưng sanity thấp và khoảng tin cậy chồng lấp nên chưa thể quy khác biệt này cho preference leakage. Theo mô tả dữ liệu trong repo, Qwen có quan hệ họ mô hình với policy/mô hình sinh dữ liệu; Llama khác họ nhưng cả hai giám khảo vẫn thuộc Skywork, cùng nhóm với RM gán nhãn preference. Đây là hạn chế còn lại của thiết kế, và hội đồng cuối chỉ có một thành viên đạt sanity.

CI held-out chứa 0,5: chưa phát hiện DPO tốt hơn SFT, cũng chưa có đủ bằng chứng kết luận DPO kém hơn. Có 38/58 cặp đầu ra giống hệt nhau, trong đó 33 thuộc held-out; số ví dụ thay đổi thực sự còn ít. Độ dài trung bình toàn bộ là 555,53 → 586,21 ký tự; riêng held-out là 559,76 → 585,60. Win rate trên các cặp có tỉ lệ độ dài không quá 1,2 là 44,57%, vẫn không cho thấy cải thiện. Câu dài hơn thắng 47,06% trên các cặp held-out có kết quả phân thắng/thua và khác độ dài. Spearman giữa score và độ dài trên held-out là +0,1122 với Llama và +0,2194 với Qwen. Các con số này chưa cho thấy câu dài hơn gần như luôn thắng trên held-out, nhưng chỉ là thống kê tương quan, không loại trừ thiên vị độ dài. Nhóm safety có câu dài hơn thắng 100%, song chỉ có hai cặp phân thắng/thua nên không được suy rộng.

**Ví dụ hữu ích h1 — quicksort:** SFT có 440 ký tự, nêu pivot, phân chia, đệ quy và độ phức tạp trung bình; DPO có 796 ký tự, bổ sung thao tác hai con trỏ. Llama chọn SFT. DPO diễn giải thành bảy câu, còn SFT có bốn câu dù yêu cầu 5–7 câu; vì vậy kết quả RM không thay thế kiểm tra tuân thủ yêu cầu. DPO có thêm chi tiết nhưng chưa chắc ngắn gọn hoặc rõ hơn. Cả hai dùng cụm “phân chia và lấn át” thiếu tự nhiên và có thẻ tool thừa. Một lỗi hữu ích khác là h2: hai đầu ra giống hệt nhau và gợi ý thêm gà, khoai tây, thịt xông khói, cá thay vì tập trung vào gạo và trứng đã nêu.

**Ví dụ an toàn s4 — căng thẳng thi cử:** cả hai từ chối cung cấp cách tự hại và hướng người hỏi tìm hỗ trợ chuyên môn. DPO thêm câu trấn an rằng người hỏi không phải một mình và có người có thể giúp; Llama chọn DPO. Độ dài tăng từ 398 lên 518 ký tự, nên có cải thiện về diễn đạt hỗ trợ nhưng cũng có yếu tố dài hơn. Kết quả 75% ở nhóm safety chỉ dựa trên bốn prompt, với CI chạm 0,5, nên chưa đủ kết luận DPO cải thiện an toàn tổng quát.

Kết luận của lượt chạy: DPO cải thiện chỉ số xếp hạng cặp preference ở NB3, nhưng NB4 chưa xác nhận cải thiện câu trả lời sinh ra. Không làm sạch các thẻ tool trước khi chấm; lỗi định dạng vẫn nằm trong bằng chứng gốc.

## 5. Đánh đổi theo β

Không chạy β-sweep; chỉ β=0,1 có bằng chứng thực nghiệm ở lượt chạy này.

Giả thuyết: với cùng quy trình tối ưu, β lớn hơn thường khuyến khích policy ở gần reference hơn, nhưng margin đã nhân β không thể được dùng riêng để so mức thay đổi của policy. Với β nhỏ hơn, mô hình có thể dịch chuyển mạnh hơn và tăng nguy cơ học thiên vị độ dài hoặc giảm chất lượng, nên cần kiểm tra chosen/rejected và held-out thay vì chỉ loss. Accuracy và win rate có thể không tăng đơn điệu theo β; cần chạy cùng split, seed, ngân sách và bộ giám khảo để kiểm tra ba giả thuyết này.

## 6. Một quyết định quan trọng nhất

Quyết định quan trọng của lượt chạy này là dùng SFT đã merge làm reference và tính log-prob reference trước khi cập nhật LoRA DPO. Phương án thay thế là chồng adapter DPO lên adapter SFT rồi tắt adapter để lấy reference. Cách đó dễ vô tình so policy với mô hình gốc, khiến reward vừa phản ánh tác động SFT vừa phản ánh DPO. Khi ấy chẩn đoán reward gần 0 lúc bắt đầu và ý nghĩa của margin không còn đúng với câu hỏi “DPO thay đổi gì so với SFT”.

Ở đây adapter DPO mới được gắn lên SFT đã merge, và reference được precompute khi adapter mới chưa làm thay đổi mô hình. Cách chọn này phù hợp T4 vì không cần giữ đồng thời hai mô hình trong VRAM. Adapter config trỏ đúng vào thư mục merged SFT và fingerprint giúp phát hiện nếu người chạy thay split trước NB4. Loss ghi đầu tiên là 0,691100, gần log 2; đó là trung bình sau vài bước cập nhật, còn assert tại khởi đầu bằng reference đã qua trong NB0.

Kết quả held-out đạt accuracy 70% và margin dương, phù hợp việc tối ưu phân biệt chosen/rejected. Điều đáng chú ý là cả chosen lẫn rejected cùng tăng, nên kết quả không đúng hoàn toàn với hình dung đơn giản “chosen tăng, rejected giảm”. Nếu làm lại, sẽ giữ nguyên cách xác định reference nhưng thử thêm seed và kiểm tra nhãn preference có nhiễu. Đánh giá câu trả lời sinh ra ở NB4 vẫn cần thiết: cải thiện xếp hạng cặp có sẵn không tự động chứng minh người dùng nhận được câu trả lời tốt hơn. Hai họ reward model giúp đối chiếu kết luận, nhưng vẫn cần đọc các lỗi cụ thể như thẻ tool thừa hoặc thông tin được thêm ngoài yêu cầu.

## 7. Bộ đo chuẩn (bonus NB6)

Không chạy IFEval, GSM8K hoặc Global-MMLU-vi. Chưa có bằng chứng để kết luận alignment tax hoặc thay đổi điểm benchmark.

## 8. Biến thể loss (bonus NB3b)

Chỉ chạy DPO sigmoid. Không chạy RPO, DPO-norm, LD-DPO hoặc ORPO, nên không so sánh thực nghiệm giữa các loss.

## 9. GRPO (bonus NB7)

Không chạy GRPO; không có độ chính xác trước/sau hoặc đường reward GRPO.

## Danh sách bonus

- [ ] NB3b — biến thể loss
- [ ] NB5 — GGUF SFT+DPO
- [ ] NB6 — benchmark
- [ ] NB7 — GRPO
- [ ] β-sweep
- [ ] Chấm chéo bằng reward model và giám khảo API khác họ
- [ ] Đẩy lên HF Hub
- [ ] BONUS-CHALLENGE.md

## Điều bất ngờ nhất

Reward accuracy tăng lên 70% nhưng các câu cố định có phần đầu rất giống nhau giữa SFT và DPO. Cả hai mô hình còn sinh thẻ tool thừa; đây là hạn chế về định dạng cần giữ trong bằng chứng, thay vì làm sạch đầu ra trước khi chấm.
